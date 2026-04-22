# Accessible Document Reader — Backend

A Python backend designed for **visually impaired students** that:

1. Accepts a **PDF or DOCX** document upload
2. **Parses the document sequentially** — preserving reading order
3. **Converts text blocks to speech** (offline, no API required)
4. **Captions image blocks** using your trained **ChartCaptioner** model (CNN + ViT + Transformer decoder)
5. Returns audio files (WAV) and text captions for each block

---

## Project Structure

```
backend/
├── app.py                  ← FastAPI application (endpoints + session management)
├── document_parser.py      ← PDF (PyMuPDF) + DOCX (python-docx) sequential parser
├── tts_engine.py           ← Offline TTS: pyttsx3 → espeak-ng → silent fallback
├── captioner.py            ← Wrapper around ChartCaptioner model
├── chart_captioner/        ← Your original model files (copied as-is)
│   ├── model.py
│   ├── inference.py
│   ├── dataset.py
│   ├── train.py
│   ├── evaluate.py
│   └── requirements.txt
├── requirements.txt
├── start.sh
└── README.md
```

---

## Setup

### 1. Install system dependencies (Linux/Ubuntu)

```bash
sudo apt install espeak-ng   # TTS fallback engine
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. (Optional) Install pyttsx3 voices

On Linux, pyttsx3 uses espeak under the hood:
```bash
sudo apt install espeak espeak-data
```

---

## Running the Server

### With your trained model checkpoint

```bash
export CHART_CAPTIONER_CHECKPOINT="checkpoints/best.pt"
bash start.sh
```

### Without a checkpoint (fallback descriptions)

```bash
bash start.sh
```

The server starts at **http://localhost:8000**.  
Interactive API docs: **http://localhost:8000/docs**

---

## API Reference

### `POST /process`
Upload a PDF or DOCX file to begin processing.

```bash
curl -X POST http://localhost:8000/process \
  -F "file=@lecture_notes.pdf"
```

Response:
```json
{
  "session_id": "a1b2c3d4-...",
  "message": "Processing started."
}
```

---

### `GET /status/{session_id}`
Poll until `status` is `"done"` or `"error"`.

```bash
curl http://localhost:8000/status/a1b2c3d4-...
```

Response:
```json
{
  "session_id": "a1b2c3d4-...",
  "status": "processing",
  "error": null,
  "blocks_ready": 5
}
```

---

### `GET /blocks/{session_id}`
Retrieve all processed blocks (available incrementally during processing).

```bash
curl http://localhost:8000/blocks/a1b2c3d4-...
```

Response:
```json
{
  "session_id": "a1b2c3d4-...",
  "status": "done",
  "blocks": [
    {
      "index": 0,
      "type": "text",
      "content": "Chapter 1: Introduction to Machine Learning",
      "has_audio": true
    },
    {
      "index": 1,
      "type": "image",
      "content": "Image description: A bar chart showing training accuracy over 30 epochs, rising from 60% to 94%.",
      "has_audio": true
    }
  ]
}
```

---

### `GET /audio/{session_id}/{block_index}`
Download the WAV audio file for a block.

```bash
curl -O http://localhost:8000/audio/a1b2c3d4-.../0
# saves block_0.wav
```

---

### `DELETE /session/{session_id}`
Clean up all temporary files for a session.

```bash
curl -X DELETE http://localhost:8000/session/a1b2c3d4-...
```

---

### `GET /health`
Check if the server is running and whether the model is loaded.

```json
{
  "status": "ok",
  "captioner_loaded": true
}
```

---

## Architecture: Processing Pipeline

```
Upload (PDF/DOCX)
       │
       ▼
┌─────────────────────────┐
│   document_parser.py    │  PyMuPDF or python-docx
│                         │  → sequential [block, block, ...]
│  text block ────────────┼──────────────────────────┐
│  image block ───────────┼──────────────────┐       │
└─────────────────────────┘                  │       │
                                             ▼       ▼
                                    ┌─────────────┐  ┌──────────────┐
                                    │ captioner.py│  │ tts_engine.py│
                                    │ ChartCaptioner  │ pyttsx3 /   │
                                    │ CNN+ViT model   │ espeak-ng   │
                                    └──────┬──────┘  └──────┬───────┘
                                           │ caption         │ .wav
                                           └────────┬────────┘
                                                    ▼
                                           tts_engine.py
                                           (caption → .wav)
                                                    │
                                                    ▼
                                           Session store
                                           blocks[{type, content, audio_path}]
```

---

## Model Integration

The `chart_captioner/` folder is used **exactly as provided** — no modifications.  
`captioner.py` imports `inference.load_model` and `inference.caption_image` directly:

```python
from inference import load_model, caption_image

model, tokenizer = load_model("checkpoints/best.pt", device)
caption = caption_image(model, tokenizer, image_path="chart.png", device=device)
```

The model runs **100% offline** with no API calls.

---

## TTS Engines (Offline Only)

| Priority | Engine | Notes |
|----------|--------|-------|
| 1st | **pyttsx3** | Cross-platform offline TTS; uses system voices |
| 2nd | **espeak-ng** | Linux CLI; `apt install espeak-ng` |
| 3rd | Silent WAV | Placeholder so the server never crashes |

---

## Frontend Integration Tips

For a frontend aimed at visually impaired users:

- Poll `/status/{id}` every 1–2 seconds
- As `blocks_ready` increases, fetch new blocks from `/blocks/{id}` and begin playing audio immediately (don't wait for full processing)
- Use `GET /audio/{id}/{index}` as the `src` of an `<audio>` element
- Announce block type before playing: *"Text"* or *"Image description"*
