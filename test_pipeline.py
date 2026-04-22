"""
test_pipeline.py
================
Standalone test script — runs the full Accessible Document Reader pipeline
directly in Python WITHOUT needing the FastAPI server to be running.

Usage
-----
  # Basic test (uses fallback TTS + no captioner):
  python test_pipeline.py --doc path/to/your.pdf

  # Full test with trained captioner checkpoint:
  python test_pipeline.py --doc path/to/your.pdf --checkpoint checkpoints/best.pt

  # Save combined audio to a specific path:
  python test_pipeline.py --doc your.pdf --checkpoint checkpoints/best.pt --output out.wav

  # Dry-run: parse only, no TTS (fast check of document parsing):
  python test_pipeline.py --doc your.pdf --dry-run

What it does
------------
  1. Parses the document (PDF or DOCX) into sequential blocks
  2. For each text block  → synthesises speech → saves block_N.wav
  3. For each image block → runs captioner → synthesises caption → saves block_N.wav
  4. Concatenates all per-block WAVs into a single combined_output.wav
  5. Prints a full summary of every block and its audio path
"""

import argparse
import os
import sys
import wave
import shutil
import logging
import tempfile
import struct
from pathlib import Path

# ── Make sure local packages are importable ───────────────────────────────────
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "chart_captioner"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger("test_pipeline")

# ── ANSI colours (skipped on Windows if no colour support) ───────────────────
_USE_COLOR = sys.stdout.isatty() and os.name != "nt"
def _c(code, text): return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text
GREEN  = lambda t: _c("32", t)
YELLOW = lambda t: _c("33", t)
CYAN   = lambda t: _c("36", t)
BOLD   = lambda t: _c("1",  t)
DIM    = lambda t: _c("2",  t)
RED    = lambda t: _c("31", t)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _wav_duration(path: str) -> float:
    """Return duration in seconds of a PCM WAV file."""
    try:
        with wave.open(path, "rb") as wf:
            return wf.getnframes() / wf.getframerate()
    except Exception:
        return 0.0


def _concat_wavs(wav_paths: list[str], output_path: str) -> None:
    """Concatenate multiple PCM WAV files into one output file."""
    valid = [p for p in wav_paths if p and os.path.exists(p) and os.path.getsize(p) > 44]
    if not valid:
        raise ValueError("No valid WAV files to concatenate.")
    if len(valid) == 1:
        shutil.copy2(valid[0], output_path)
        return
    all_frames = b""
    params = None
    for wp in valid:
        with wave.open(wp, "rb") as wf:
            if params is None:
                params = wf.getparams()
            all_frames += wf.readframes(wf.getnframes())
    with wave.open(output_path, "wb") as out:
        out.setparams(params)
        out.writeframes(all_frames)


def _print_separator(char="─", width=70):
    print(DIM(char * width))


def _print_block_summary(idx: int, btype: str, content: str, audio_path: str | None):
    label = GREEN("TEXT ") if btype == "text" else YELLOW("IMAGE")
    prefix = BOLD(f"[{idx:03d}]") + f" {label}"
    preview = content[:120].replace("\n", " ")
    if len(content) > 120:
        preview += "…"
    print(f"{prefix}  {preview}")
    if audio_path and os.path.exists(audio_path):
        dur = _wav_duration(audio_path)
        size_kb = os.path.getsize(audio_path) / 1024
        print(DIM(f"       audio → {audio_path}  [{dur:.1f}s, {size_kb:.0f} KB]"))
    elif audio_path is None:
        print(DIM("       audio → (skipped)"))


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run(doc_path: str, checkpoint: str | None, output_wav: str, dry_run: bool):

    doc_path = os.path.abspath(doc_path)
    if not os.path.exists(doc_path):
        print(RED(f"Error: file not found: {doc_path}"))
        sys.exit(1)

    print()
    print(BOLD("═" * 70))
    print(BOLD("  Accessible Document Reader — Pipeline Test"))
    print(BOLD("═" * 70))
    print(f"  Document  : {doc_path}")
    print(f"  Checkpoint: {checkpoint or '(none — fallback captions)'}")
    print(f"  Output WAV: {output_wav}")
    print(f"  Dry run   : {dry_run}")
    print(BOLD("═" * 70))
    print()

    # ── Step 1: Load captioner ────────────────────────────────────────────────
    captioner = None
    if checkpoint:
        checkpoint = os.path.abspath(checkpoint)
        if not os.path.exists(checkpoint):
            print(RED(f"  Warning: checkpoint not found at {checkpoint}. Using fallback."))
        else:
            print(CYAN("→ Loading ChartCaptioner model…"))
            try:
                from captioner import CaptionerService
                captioner = CaptionerService(checkpoint)
                print(GREEN("  Model loaded.\n"))
            except Exception as exc:
                print(RED(f"  Failed to load model: {exc}"))
                print("  Continuing with fallback captions.\n")

    # ── Step 2: Parse document ────────────────────────────────────────────────
    print(CYAN("→ Parsing document…"))
    from document_parser import parse_document
    raw_blocks = parse_document(doc_path)
    print(f"  Found {BOLD(str(len(raw_blocks)))} blocks.\n")

    if dry_run:
        print(CYAN("DRY RUN — skipping TTS. Block types:"))
        for i, b in enumerate(raw_blocks):
            prefix = GREEN("TEXT ") if b["type"] == "text" else YELLOW("IMAGE")
            preview = (b.get("content") or b.get("image_path") or "")[:100]
            print(f"  [{i:03d}] {prefix}  {preview}")
        print()
        print(GREEN("Dry run complete."))
        return

    # ── Step 3: Process each block ────────────────────────────────────────────
    from tts_engine import synthesize_text
    from PIL import Image

    tmpdir = tempfile.mkdtemp(prefix="adr_test_")
    print(CYAN(f"→ Temp directory: {tmpdir}\n"))
    _print_separator()

    processed_blocks = []
    total = len(raw_blocks)

    for idx, raw in enumerate(raw_blocks):
        btype = raw["type"]
        content = ""
        audio_path = None

        if btype == "text":
            content = raw["content"].strip()
            if not content:
                continue
            wav_path = os.path.join(tmpdir, f"block_{idx:04d}.wav")
            try:
                print(DIM(f"  TTS block {idx+1}/{total}…"), end="\r")
                synthesize_text(content, wav_path)
                audio_path = wav_path
            except Exception as exc:
                log.warning("TTS failed for block %d: %s", idx, exc)

        elif btype == "image":
            img_path = raw["image_path"]
            # Caption
            try:
                if captioner is not None:
                    caption = captioner.caption(img_path)
                else:
                    img = Image.open(img_path)
                    w, h = img.size
                    caption = f"An image of size {w}×{h} pixels."
            except Exception as exc:
                log.warning("Captioning failed for block %d: %s", idx, exc)
                caption = "An image is present in the document."

            content = f"Image description: {caption}"
            wav_path = os.path.join(tmpdir, f"block_{idx:04d}.wav")
            try:
                synthesize_text(content, wav_path)
                audio_path = wav_path
            except Exception as exc:
                log.warning("TTS failed for image block %d: %s", idx, exc)

        block = {"index": idx, "type": btype, "content": content, "audio_path": audio_path}
        processed_blocks.append(block)
        _print_block_summary(idx, btype, content, audio_path)
        _print_separator()

    # ── Step 4: Combine audio ─────────────────────────────────────────────────
    print()
    print(CYAN("→ Combining all audio blocks…"))

    wav_paths = [
        b["audio_path"]
        for b in sorted(processed_blocks, key=lambda b: b["index"])
        if b["audio_path"] and os.path.exists(b["audio_path"])
    ]

    if not wav_paths:
        print(RED("  No audio blocks to combine. Exiting."))
        return

    output_wav = os.path.abspath(output_wav)
    _concat_wavs(wav_paths, output_wav)

    total_duration = _wav_duration(output_wav)
    total_size_mb  = os.path.getsize(output_wav) / (1024 * 1024)

    print()
    print(BOLD("═" * 70))
    print(BOLD("  Done!"))
    print(f"  Blocks processed  : {len(processed_blocks)}")
    print(f"  Audio blocks      : {len(wav_paths)}")
    print(f"  Total duration    : {total_duration:.1f} seconds")
    print(f"  Output file       : {output_wav}")
    print(f"  Output size       : {total_size_mb:.2f} MB")
    print(BOLD("═" * 70))
    print()
    play_cmd = 'python -c "import webbrowser; webbrowser.open(' + repr(output_wav) + ')"'
    print(f"  Play with:  {CYAN(play_cmd)}")
    print(f"  Or open:    {CYAN(output_wav)}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Test the full Accessible Document Reader pipeline without the server.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--doc", "-d", required=True,
        help="Path to the PDF or DOCX file to process.",
    )
    p.add_argument(
        "--checkpoint", "-c", default=None,
        help="Path to the trained ChartCaptioner .pt checkpoint (optional).",
    )
    p.add_argument(
        "--output", "-o", default="combined_output.wav",
        help="Output WAV file path (default: combined_output.wav).",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Parse only — list blocks without running TTS or captioner.",
    )
    args = p.parse_args()

    run(
        doc_path   = args.doc,
        checkpoint = args.checkpoint,
        output_wav = args.output,
        dry_run    = args.dry_run,
    )
