# DocReader AI — Accessible Document Reader

An AI-powered web application for visually impaired students. Converts PDFs and Word documents to speech with AI-generated image descriptions.

**Backend:** Python FastAPI + ChartCaptioner model  
**Frontend:** React + Tailwind CSS + Web APIs (SpeechSynthesis, SpeechRecognition)

---

## Project Structure

```
.
├── frontend/                ← React app (deployed to Vercel)
│   ├── src/
│   │   ├── pages/           ← Home, Upload, Reader, Assistant
│   │   ├── components/      ← Button, AudioPlayer, etc.
│   │   ├── services/        ← API integration (getBlocks, getStatus, etc.)
│   │   ├── utils/           ← speech.js (TTS + announcements)
│   │   ├── App.jsx
│   │   └── index.css        ← Tailwind + animations
│   ├── package.json
│   └── vite.config.js
├── app.py                   ← FastAPI backend
├── document_parser.py       ← PDF/DOCX parsing
├── tts_engine.py            ← Text-to-speech
├── captioner.py             ← Image descriptions
├── requirements.txt
├── package.json             ← Root build config
├── vercel.json              ← Vercel deployment
├── .nvmrc                   ← Node version (18)
└── README.md
```

---

## Frontend Setup

### Install & Build

```bash
npm install       # Installs frontend deps
npm run build     # Builds React app → frontend/dist/
npm run dev       # Local dev server (port 5173)
```

### Environment Variables

Create `frontend/.env.local`:

```
VITE_API_BASE_URL=http://localhost:8000
```

For production (Vercel), set:

```
VITE_API_BASE_URL=https://your-api.example.com
```

### Pages

- **Home** (`/`) — Intro, features, call-to-action buttons
- **Upload** (`/upload`) — Drag-drop file upload, live progress polling
- **Reader** (`/reader`) — Block-by-block navigation, audio playback, image descriptions
- **Assistant** (`/assistant`) — Voice Q&A with mic input & speech output

### Accessibility

- **Global Speech:** Every button/link speaks its label on hover or focus
- **Keyboard Navigation:** Tab, Enter, Space, Arrow keys
- **Voice Toggle:** "Voice On/Off" button in navbar
- **ARIA:** Full semantic markup, live regions, roles
- **High Contrast:** Dark theme, large buttons, focus rings

---

## Backend API

### Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/process` | Upload a PDF or DOCX file |
| `GET` | `/status/{session_id}` | Poll processing status |
| `GET` | `/blocks/{session_id}` | Get all processed blocks |
| `GET` | `/audio/{session_id}/{block_index}` | Stream WAV audio |
| `GET` | `/export/{session_id}` | Download full document audio |
| `DELETE` | `/session/{session_id}` | Clean up session |
| `GET` | `/health` | Health check |

### Running Backend

```bash
# Install Python dependencies
pip install -r requirements.txt

# Start server
python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

API docs: **http://localhost:8000/docs**

---

## Deployment (Vercel)

The app is configured for **Vercel** deployment.

### Deploy Frontend Only

```bash
# Push to GitHub
git add .
git commit -m "Add DocReader AI frontend"
git push

# Import in Vercel dashboard:
# Build Command: npm run build
# Output Directory: frontend/dist
# Environment: VITE_API_BASE_URL=<your-api-url>
```

### Deploy Backend Separately

For production backend, use Render, Railway, or Heroku:

```bash
# Example: Render
# Connect your repo → Select Python → Start command:
# uvicorn app:app --host 0.0.0.0 --port $PORT
```

Then set `VITE_API_BASE_URL` to your production backend URL in Vercel.

---

## Development Workflow

### Local Testing

```bash
# Terminal 1: Start backend
python -m uvicorn app:app --reload

# Terminal 2: Start frontend dev server
cd frontend && npm run dev
```

Visit **http://localhost:5173** — frontend automatically proxies API calls to backend.

### Building for Production

```bash
npm run build

# Output: frontend/dist/
# → Ready to deploy to Vercel, Netlify, GitHub Pages, etc.
```

---

## Features

### Document Processing

- Accepts PDF and DOCX files
- Extracts text blocks sequentially
- Automatically captions images using AI
- Converts all content to speech (offline)

### Reader

- Play/Pause/Next/Previous controls
- Click to jump to any block
- Highlights currently-spoken block
- Per-block audio playback
- Download full document as WAV

### Voice Assistant

- Speech recognition input (browser SpeechRecognition API)
- Navigation help, keyboard shortcuts, usage tips
- All responses spoken aloud

### Accessibility

- Voice announcements on button hover/focus
- Full keyboard navigation
- High-contrast dark theme
- Large touch targets
- Screen reader support (ARIA)

---

## Technology Stack

**Frontend:**
- React 19 (functional components)
- Vite (fast build)
- Tailwind CSS 4 (styling)
- React Router (navigation)
- Web Speech API (SpeechSynthesis + SpeechRecognition)

**Backend:**
- Python 3.8+
- FastAPI (async HTTP)
- PyMuPDF (PDF parsing)
- python-docx (Word parsing)
- pyttsx3 (offline TTS)
- ChartCaptioner (image captioning)

---

## Troubleshooting

### API Connection Failed

Check `VITE_API_BASE_URL` environment variable and backend status:

```bash
curl http://localhost:8000/health
```

### No Speech Output

Ensure SpeechSynthesis is available:

```javascript
console.log(window.speechSynthesis ? 'Available' : 'Not available');
```

### Build Errors

Clear cache and reinstall:

```bash
rm -rf frontend/node_modules frontend/.next
npm install
npm run build
```

---

## License

MIT
