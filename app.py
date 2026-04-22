"""
Accessible Document Reader — FastAPI Backend
============================================
Parses PDF / DOCX files sequentially, converts text blocks to speech,
and captions image blocks using the user-supplied ChartCaptioner model.

Endpoints
---------
POST /process          — Upload a document; returns a session_id
GET  /status/{id}      — Poll processing status + completed blocks
GET  /audio/{id}/{idx} — Stream an audio chunk
GET  /export/{id}      — Download single combined WAV of full document
GET  /blocks/{id}      — Return all blocks (text + captions) as JSON
DELETE /session/{id}   — Clean up temporary files
"""

import io
import os
import sys
import uuid
import asyncio
import logging
import tempfile
import threading
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# ── Local imports ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from document_parser import parse_document
from tts_engine import synthesize_text
from captioner import CaptionerService

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Accessible Document Reader",
    description="Converts PDFs / DOCX documents to speech + image captions "
                "for visually impaired students.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory session store ──────────────────────────────────────────────────
# session_id → {
#   "status":  "processing" | "done" | "error",
#   "error":   str | None,
#   "blocks":  [{"index": int, "type": "text"|"image", "content": str,
#                "audio_path": str | None}],
#   "tmpdir":  str,
# }
SESSIONS: dict[str, dict] = {}
SESSIONS_LOCK = threading.Lock()

# ── Caption service (singleton) ──────────────────────────────────────────────
CAPTIONER: Optional[CaptionerService] = None

CHECKPOINT_ENV = os.getenv("CHART_CAPTIONER_CHECKPOINT", "")


@app.on_event("startup")
async def startup():
    global CAPTIONER
    if CHECKPOINT_ENV and Path(CHECKPOINT_ENV).exists():
        log.info("Loading ChartCaptioner from %s …", CHECKPOINT_ENV)
        try:
            CAPTIONER = CaptionerService(CHECKPOINT_ENV)
            log.info("ChartCaptioner ready.")
        except Exception as exc:
            log.warning("ChartCaptioner failed to load: %s. "
                        "Image captioning will use fallback descriptions.", exc)
    else:
        log.warning("No CHART_CAPTIONER_CHECKPOINT set or file not found. "
                    "Image captioning uses fallback descriptions. "
                    "Set the env var to a trained .pt checkpoint to enable it.")


# ────────────────────────────────────────────────────────────────────────────
# Helper
# ────────────────────────────────────────────────────────────────────────────

def _new_session(tmpdir: str) -> str:
    sid = str(uuid.uuid4())
    with SESSIONS_LOCK:
        SESSIONS[sid] = {
            "status": "processing",
            "error": None,
            "blocks": [],
            "tmpdir": tmpdir,
        }
    return sid


def _update_session(sid: str, **kwargs):
    with SESSIONS_LOCK:
        SESSIONS[sid].update(kwargs)


def _append_block(sid: str, block: dict):
    with SESSIONS_LOCK:
        SESSIONS[sid]["blocks"].append(block)


# ────────────────────────────────────────────────────────────────────────────
# Background processing pipeline
# ────────────────────────────────────────────────────────────────────────────

def _process_document(sid: str, doc_path: str, tmpdir: str):
    """
    Runs in a thread-pool thread.
    1. Parse document → sequential blocks [{type, content/image_path}]
    2. For each block:
       - text  → TTS → save .wav → record block
       - image → caption model → TTS description → save .wav → record block
    """
    try:
        log.info("[%s] Parsing document …", sid)
        raw_blocks = parse_document(doc_path)
        log.info("[%s] %d blocks found.", sid, len(raw_blocks))

        for idx, raw in enumerate(raw_blocks):
            btype = raw["type"]
            block = {"index": idx, "type": btype, "content": "", "audio_path": None}

            if btype == "text":
                text = raw["content"].strip()
                if not text:
                    continue
                block["content"] = text
                try:
                    audio_path = os.path.join(tmpdir, f"block_{idx}.wav")
                    synthesize_text(text, audio_path)
                    block["audio_path"] = audio_path
                except Exception as exc:
                    log.warning("[%s] TTS failed for block %d: %s", sid, idx, exc)

            elif btype == "image":
                img_path = raw["image_path"]
                # Generate caption
                try:
                    if CAPTIONER is not None:
                        caption = CAPTIONER.caption(img_path)
                    else:
                        caption = _fallback_caption(img_path)
                except Exception as exc:
                    log.warning("[%s] Captioning failed for block %d: %s", sid, idx, exc)
                    caption = "An image is present in the document."

                description = f"Image description: {caption}"
                block["content"] = description
                try:
                    audio_path = os.path.join(tmpdir, f"block_{idx}.wav")
                    synthesize_text(description, audio_path)
                    block["audio_path"] = audio_path
                except Exception as exc:
                    log.warning("[%s] TTS failed for image block %d: %s", sid, idx, exc)

            _append_block(sid, block)
            log.info("[%s] Block %d/%d done (%s).", sid, idx + 1, len(raw_blocks), btype)

        _update_session(sid, status="done")
        log.info("[%s] Processing complete.", sid)

    except Exception as exc:
        log.error("[%s] Fatal error: %s", sid, exc, exc_info=True)
        _update_session(sid, status="error", error=str(exc))


def _fallback_caption(img_path: str) -> str:
    """Basic size/mode description when no model checkpoint is available."""
    try:
        img = Image.open(img_path)
        w, h = img.size
        mode = img.mode
        return f"An image of size {w}×{h} pixels ({mode} mode)."
    except Exception:
        return "An image is present in the document."



# ────────────────────────────────────────────────────────────────────────────
# WAV concat helper (used by /export)
# ────────────────────────────────────────────────────────────────────────────

def _concat_wavs(wav_paths: list[str], output_path: str) -> None:
    """Merge multiple PCM WAV files into a single file in order."""
    import wave, shutil as _sh
    valid = [p for p in wav_paths if p and os.path.exists(p) and os.path.getsize(p) > 44]
    if not valid:
        raise ValueError("No valid WAV fragments to merge.")
    if len(valid) == 1:
        _sh.copy2(valid[0], output_path)
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

# ────────────────────────────────────────────────────────────────────────────
# Routes
# ────────────────────────────────────────────────────────────────────────────

@app.post("/process", summary="Upload a document for processing")
async def process_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="PDF or DOCX file"),
):
    """
    Upload a PDF or DOCX file. Processing runs asynchronously.
    Returns a `session_id` to poll for status and retrieve results.
    """
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".pdf", ".docx", ".doc"}:
        raise HTTPException(400, "Only PDF and DOCX files are supported.")

    tmpdir = tempfile.mkdtemp(prefix="adr_")
    doc_path = os.path.join(tmpdir, file.filename)
    content = await file.read()
    with open(doc_path, "wb") as f:
        f.write(content)

    sid = _new_session(tmpdir)
    background_tasks.add_task(_process_document, sid, doc_path, tmpdir)

    return {"session_id": sid, "message": "Processing started."}


@app.get("/status/{session_id}", summary="Get processing status")
async def get_status(session_id: str):
    """
    Poll this endpoint until `status` is `"done"` or `"error"`.
    Returns the number of completed blocks so far.
    """
    session = _get_session(session_id)
    return {
        "session_id": session_id,
        "status": session["status"],
        "error": session["error"],
        "blocks_ready": len(session["blocks"]),
    }


@app.get("/blocks/{session_id}", summary="Get all processed blocks")
async def get_blocks(session_id: str):
    """
    Returns every block processed so far (incrementally available).
    Each block contains `index`, `type`, `content`, and `has_audio`.
    """
    session = _get_session(session_id)
    blocks = [
        {
            "index": b["index"],
            "type": b["type"],
            "content": b["content"],
            "has_audio": b["audio_path"] is not None,
        }
        for b in session["blocks"]
    ]
    return {"session_id": session_id, "status": session["status"], "blocks": blocks}


@app.get("/audio/{session_id}/{block_index}", summary="Stream audio for a block")
async def get_audio(session_id: str, block_index: int):
    """
    Download the WAV audio file for a specific block index.
    """
    session = _get_session(session_id)
    matching = [b for b in session["blocks"] if b["index"] == block_index]
    if not matching:
        raise HTTPException(404, f"Block {block_index} not found yet.")
    block = matching[0]
    if block["audio_path"] is None or not os.path.exists(block["audio_path"]):
        raise HTTPException(404, f"No audio available for block {block_index}.")
    return FileResponse(
        block["audio_path"],
        media_type="audio/wav",
        filename=f"block_{block_index}.wav",
    )



@app.get("/export/{session_id}", summary="Download full combined audio")
async def export_audio(session_id: str):
    """
    Concatenates every block WAV in document order into a single file and streams it.
    Only available once status is \'done\'. Subsequent calls return the cached file.
    """
    session = _get_session(session_id)

    if session["status"] != "done":
        raise HTTPException(400, "Document is still processing. Wait until status is \'done\'.")

    combined_path = os.path.join(session["tmpdir"], "combined_output.wav")

    # Return cached combined file if already built
    if os.path.exists(combined_path) and os.path.getsize(combined_path) > 44:
        return FileResponse(combined_path, media_type="audio/wav", filename="document_audio.wav")

    # Collect all audio paths in block order
    blocks_sorted = sorted(session["blocks"], key=lambda b: b["index"])
    wav_paths = [
        b["audio_path"]
        for b in blocks_sorted
        if b["audio_path"] and os.path.exists(b["audio_path"])
    ]

    if not wav_paths:
        raise HTTPException(404, "No audio blocks found for this session.")

    try:
        _concat_wavs(wav_paths, combined_path)
    except Exception as exc:
        log.error("[%s] Failed to build combined audio: %s", session_id, exc, exc_info=True)
        raise HTTPException(500, f"Audio merge failed: {exc}")

    return FileResponse(combined_path, media_type="audio/wav", filename="document_audio.wav")

@app.delete("/session/{session_id}", summary="Delete session and temp files")
async def delete_session(session_id: str):
    """
    Clean up all temporary files and remove the session from memory.
    """
    session = _get_session(session_id)
    import shutil
    try:
        shutil.rmtree(session["tmpdir"], ignore_errors=True)
    except Exception:
        pass
    with SESSIONS_LOCK:
        SESSIONS.pop(session_id, None)
    return {"message": f"Session {session_id} deleted."}


@app.get("/health", summary="Health check")
async def health():
    return {
        "status": "ok",
        "captioner_loaded": CAPTIONER is not None,
    }


# ────────────────────────────────────────────────────────────────────────────
# Internal
# ────────────────────────────────────────────────────────────────────────────

def _get_session(session_id: str) -> dict:
    with SESSIONS_LOCK:
        session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(404, f"Session '{session_id}' not found.")
    return session
