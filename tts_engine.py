"""
tts_engine.py
=============
Text-to-Speech engine for the Accessible Document Reader.

pyttsx3 has been intentionally excluded — it calls runAndWait() which
spins a blocking event loop and hangs unpredictably inside FastAPI's
thread-pool workers.

Engine priority
---------------
1. gTTS  (pip install gTTS)
   Uses the Google TTS *library* — no API key, no authentication.
   It does make an HTTP request to translate.google.com to synthesise
   audio, so the server needs outbound internet access.
   Output is MP3; we convert to WAV using pydub/ffmpeg so the rest of
   the stack always receives uniform WAV files.

2. espeak-ng  (apt install espeak-ng)
   Runs as a subprocess with a hard timeout.  Fully offline, instant,
   no hanging possible.  Quality is robotic but reliable.

3. Silent WAV placeholder
   Written when both engines fail so the caller never crashes.

Long texts are split into sentence-aware chunks (≤ _CHUNK_CHARS chars)
and the WAV fragments are concatenated in order.
"""

import io
import logging
import os
import re
import shutil
import struct
import subprocess
import tempfile
import wave
from pathlib import Path

log = logging.getLogger(__name__)

# ── Tunable constants ─────────────────────────────────────────────────────────
_CHUNK_CHARS    = 500    # gTTS is stable up to ~500 chars per request
_ESPEAK_TIMEOUT = 30     # seconds before we kill an espeak subprocess
_SAMPLE_RATE    = 22050  # output WAV sample rate
_LANGUAGE       = os.getenv("TTS_LANGUAGE", "en")  # BCP-47 language tag


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def synthesize_text(text: str, output_wav: str) -> None:
    """
    Convert *text* to speech and write a 16-bit PCM WAV to *output_wav*.
    Never raises — falls back to a silent WAV on total failure.
    """
    text = text.strip()
    if not text:
        _write_silent_wav(output_wav)
        return

    chunks = _split_text(text)

    for engine in (_engine_gtts, _engine_espeak):
        try:
            wav_fragments = [engine(chunk) for chunk in chunks]
            _concat_wavs(wav_fragments, output_wav)
            for f in wav_fragments:
                try:
                    os.unlink(f)
                except OSError:
                    pass
            if os.path.exists(output_wav) and os.path.getsize(output_wav) > 44:
                return
        except Exception as exc:
            log.warning("TTS engine %s failed: %s", engine.__name__, exc)

    log.error("All TTS engines failed — writing silent placeholder.")
    _write_silent_wav(output_wav)


# ─────────────────────────────────────────────────────────────────────────────
# Engine 1 — gTTS
# ─────────────────────────────────────────────────────────────────────────────

def _engine_gtts(text: str) -> str:
    """
    Synthesise *text* with gTTS and return a path to a temporary WAV file.
    gTTS returns MP3; we convert to WAV for a uniform output format.
    """
    from gtts import gTTS  # ImportError → caught by caller

    tts = gTTS(text=text, lang=_LANGUAGE, slow=False)
    mp3_buffer = io.BytesIO()
    tts.write_to_fp(mp3_buffer)
    mp3_buffer.seek(0)
    return _mp3_buffer_to_wav(mp3_buffer)


def _mp3_buffer_to_wav(mp3_buffer: io.BytesIO) -> str:
    """Convert an in-memory MP3 BytesIO to a temp WAV file. Returns WAV path."""
    # Try pydub first (cleanest)
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_mp3(mp3_buffer)
        audio = audio.set_frame_rate(_SAMPLE_RATE).set_channels(1).set_sample_width(2)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        audio.export(tmp.name, format="wav")
        tmp.close()
        return tmp.name
    except Exception:
        pass

    # Fallback: write MP3 to disk, call ffmpeg subprocess
    mp3_tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    mp3_tmp.write(mp3_buffer.read())
    mp3_tmp.flush()
    mp3_tmp.close()

    wav_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wav_tmp.close()

    try:
        ffmpeg = shutil.which("ffmpeg") or shutil.which("avconv")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg not found — install ffmpeg or pydub")
        subprocess.run(
            [
                ffmpeg, "-y", "-i", mp3_tmp.name,
                "-ar", str(_SAMPLE_RATE),
                "-ac", "1",
                "-acodec", "pcm_s16le",
                wav_tmp.name,
            ],
            check=True,
            capture_output=True,
            timeout=30,
        )
        return wav_tmp.name
    finally:
        try:
            os.unlink(mp3_tmp.name)
        except OSError:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Engine 2 — espeak-ng (subprocess, hard timeout — can never hang)
# ─────────────────────────────────────────────────────────────────────────────

def _engine_espeak(text: str) -> str:
    """
    Synthesise *text* with espeak-ng via subprocess.
    Returns path to a temporary WAV file.
    Raises RuntimeError if espeak-ng is not installed.
    """
    espeak = shutil.which("espeak-ng") or shutil.which("espeak")
    if espeak is None:
        raise RuntimeError("espeak-ng / espeak not found on PATH.")

    txt_tmp = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
    txt_tmp.write(text)
    txt_tmp.flush()
    txt_tmp.close()

    wav_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wav_tmp.close()

    try:
        subprocess.run(
            [
                espeak,
                "-f", txt_tmp.name,
                "-w", wav_tmp.name,
                "-s", "150",    # words per minute
                "-a", "100",    # amplitude
                "--ipa=0",
            ],
            check=True,
            capture_output=True,
            timeout=_ESPEAK_TIMEOUT,
        )
    finally:
        try:
            os.unlink(txt_tmp.name)
        except OSError:
            pass

    return wav_tmp.name


# ─────────────────────────────────────────────────────────────────────────────
# WAV utilities
# ─────────────────────────────────────────────────────────────────────────────

def _concat_wavs(wav_paths: list[str], output_path: str) -> None:
    """Concatenate multiple PCM WAV fragments into a single file."""
    valid = [p for p in wav_paths if p and os.path.exists(p) and os.path.getsize(p) > 44]
    if not valid:
        _write_silent_wav(output_path)
        return

    if len(valid) == 1:
        shutil.copy2(valid[0], output_path)
        return

    all_frames = b""
    params = None
    for wp in valid:
        try:
            with wave.open(wp, "rb") as wf:
                if params is None:
                    params = wf.getparams()
                all_frames += wf.readframes(wf.getnframes())
        except Exception as exc:
            log.warning("Could not read WAV fragment %s: %s", wp, exc)

    if params is None or not all_frames:
        _write_silent_wav(output_path)
        return

    with wave.open(output_path, "wb") as out:
        out.setparams(params)
        out.writeframes(all_frames)


def _write_silent_wav(path: str, duration_sec: float = 0.5) -> None:
    """Write a short silent WAV — used as a safe no-op placeholder."""
    n = int(_SAMPLE_RATE * duration_sec)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(_SAMPLE_RATE)
        wf.writeframes(struct.pack("<" + "h" * n, *([0] * n)))


# ─────────────────────────────────────────────────────────────────────────────
# Text chunking (sentence-aware)
# ─────────────────────────────────────────────────────────────────────────────

def _split_text(text: str) -> list[str]:
    """
    Split text into chunks of at most _CHUNK_CHARS characters, preferring
    sentence boundaries so synthesis sounds natural.
    """
    if len(text) <= _CHUNK_CHARS:
        return [text]

    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        if len(sentence) > _CHUNK_CHARS:
            if current:
                chunks.append(current.strip())
                current = ""
            for i in range(0, len(sentence), _CHUNK_CHARS):
                chunks.append(sentence[i : i + _CHUNK_CHARS].strip())
            continue

        if len(current) + len(sentence) + 1 > _CHUNK_CHARS:
            if current:
                chunks.append(current.strip())
            current = sentence
        else:
            current = (current + " " + sentence).strip() if current else sentence

    if current:
        chunks.append(current.strip())

    return [c for c in chunks if c]
