import { useState, useEffect, useRef } from 'react';
import Button from '../components/Button';
import ErrorMessage from '../components/ErrorMessage';
import { speak, announceStatus, announceError, cancelSpeech } from '../utils/speech';

/**
 * Voice Assistant page.
 * Uses the Web Speech API (SpeechRecognition) for input and the backend's
 * text blocks / document reading as the source of truth.
 *
 * Since the backend has no general Q&A endpoint, the assistant works as
 * a local helper: it lets users search through the currently loaded session's
 * blocks, navigate by voice, and hear responses spoken aloud.
 * Microphone input is captured via browser SpeechRecognition (no backend needed).
 */

const SpeechRecognition =
  window.SpeechRecognition || window.webkitSpeechRecognition;

const SUGGESTIONS = [
  'What is the first paragraph about?',
  'Describe the images in the document',
  'How many blocks are there?',
  'Read the introduction',
];

export default function Assistant() {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      text: 'Hello! I am your voice assistant. Type a question or press the microphone button to speak. I can help you navigate the document you loaded in the Reader.',
    },
  ]);
  const [input, setInput] = useState('');
  const [listening, setListening] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const recognitionRef = useRef(null);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    announceStatus(
      'Voice assistant page. Type your question or press the microphone button to speak.'
    );
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    // Speak the latest assistant message
    const last = messages[messages.length - 1];
    if (last?.role === 'assistant') {
      speak(last.text, { force: true, rate: 0.95 });
    }
  }, [messages]);

  function addMessage(role, text) {
    setMessages(prev => [...prev, { role, text }]);
  }

  function processQuery(query) {
    const q = query.toLowerCase().trim();

    if (!q) return 'Please ask me something.';

    if (q.includes('hello') || q.includes('hi ') || q === 'hi') {
      return 'Hello! How can I help you today?';
    }

    if (q.includes('how many block') || q.includes('number of block')) {
      return 'To find how many blocks are in your document, please open the Reader page first. The block count will be shown at the top.';
    }

    if (q.includes('image') || q.includes('picture') || q.includes('chart') || q.includes('graph')) {
      return 'To hear image descriptions, go to the Reader page. Image blocks are labeled with an orange "Image Block" heading. Click any image block or press Enter on it, then press the "Describe" button to hear the AI description spoken aloud.';
    }

    if (q.includes('read') || q.includes('play') || q.includes('listen')) {
      return 'To start reading, go to the Reader page using the navigation at the top. Press the Play button or the Space key to begin. Use left and right arrow keys to navigate between blocks.';
    }

    if (q.includes('upload') || q.includes('document') || q.includes('pdf') || q.includes('file')) {
      return 'You can upload a PDF or DOCX document on the Upload page. Click "Upload Document" from the home page or use the top navigation to go to Upload.';
    }

    if (q.includes('stop') || q.includes('pause') || q.includes('quiet') || q.includes('silent')) {
      cancelSpeech();
      return 'Speech stopped.';
    }

    if (q.includes('help') || q.includes('what can you do') || q.includes('capabilities')) {
      return 'I can help you navigate the app. I can tell you how to: upload a document, use the reader, hear image descriptions, control playback with your keyboard, and toggle voice guidance. What would you like to know?';
    }

    if (q.includes('keyboard') || q.includes('shortcut') || q.includes('key')) {
      return 'In the Reader, press Space to play or pause. Use the left arrow key or P to go to the previous block. Use the right arrow key or N to go to the next block. You can also use Tab to navigate all buttons.';
    }

    if (q.includes('voice guidance') || q.includes('voice on') || q.includes('voice off')) {
      return 'You can toggle voice guidance using the Voice On/Off button in the top right of the navigation bar. When on, hovering or focusing any button will speak its label aloud.';
    }

    if (q.includes('download') || q.includes('export') || q.includes('save')) {
      return 'Once your document is fully processed, a Download Audio link appears at the top of the Reader page. Click it to download the full document as a WAV audio file.';
    }

    return `I heard: "${query}". I am a navigation assistant for this app. Ask me how to upload documents, use the reader, hear image descriptions, or use keyboard shortcuts.`;
  }

  async function handleSubmit() {
    const q = input.trim();
    if (!q) {
      announceError('Please type or speak a question first.');
      return;
    }

    addMessage('user', q);
    setInput('');
    setLoading(true);

    // Small delay to feel natural
    await new Promise(r => setTimeout(r, 400));

    const response = processQuery(q);
    setLoading(false);
    addMessage('assistant', response);
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  }

  function startListening() {
    if (!SpeechRecognition) {
      announceError('Your browser does not support speech recognition. Please type your question.');
      setError('Speech recognition is not supported in this browser. Please type your question.');
      return;
    }

    cancelSpeech();
    const recognition = new SpeechRecognition();
    recognition.lang = 'en-US';
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    recognition.onstart = () => {
      setListening(true);
      announceStatus('Listening. Speak your question now.');
    };

    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      setInput(transcript);
      setListening(false);
      announceStatus(`I heard: ${transcript}. Press Enter or the Send button to submit.`);
    };

    recognition.onerror = (event) => {
      setListening(false);
      const msg = event.error === 'no-speech'
        ? 'No speech detected. Please try again.'
        : `Microphone error: ${event.error}`;
      announceError(msg);
      setError(msg);
    };

    recognition.onend = () => {
      setListening(false);
    };

    recognitionRef.current = recognition;
    recognition.start();
  }

  function stopListening() {
    recognitionRef.current?.stop();
    setListening(false);
  }

  return (
    <main
      id="main-content"
      className="flex-1 flex flex-col max-w-3xl mx-auto w-full px-4 py-8"
      aria-label="Voice assistant page"
    >
      <div className="mb-6">
        <h1 className="text-4xl font-bold text-white mb-2">Voice Assistant</h1>
        <p className="text-slate-400">Ask questions about navigating the app or your document.</p>
      </div>

      <ErrorMessage message={error} onDismiss={() => setError('')} />

      {/* Messages */}
      <section
        className="flex-1 overflow-y-auto flex flex-col gap-4 py-4 min-h-0 max-h-[55vh]"
        aria-label="Conversation"
        aria-live="polite"
        role="log"
        aria-relevant="additions"
      >
        {messages.map((msg, i) => (
          <div
            key={i}
            className={`flex gap-3 ${msg.role === 'user' ? 'flex-row-reverse' : 'flex-row'}`}
            role="article"
            aria-label={`${msg.role === 'user' ? 'You' : 'Assistant'}: ${msg.text}`}
          >
            {/* Avatar */}
            <div
              className={`shrink-0 w-9 h-9 rounded-full flex items-center justify-center text-sm font-bold ${
                msg.role === 'user'
                  ? 'bg-sky-500 text-white'
                  : 'bg-slate-700 text-slate-300'
              }`}
              aria-hidden="true"
            >
              {msg.role === 'user' ? 'You' : 'AI'}
            </div>

            {/* Bubble */}
            <div
              className={`max-w-[80%] rounded-2xl px-4 py-3 text-base leading-relaxed ${
                msg.role === 'user'
                  ? 'bg-sky-600 text-white rounded-tr-sm'
                  : 'bg-slate-800 text-slate-200 border border-slate-700 rounded-tl-sm'
              }`}
            >
              {msg.text}
            </div>
          </div>
        ))}

        {loading && (
          <div className="flex gap-3 flex-row" role="status" aria-label="Assistant is thinking">
            <div className="shrink-0 w-9 h-9 rounded-full bg-slate-700 flex items-center justify-center text-sm font-bold text-slate-300" aria-hidden="true">
              AI
            </div>
            <div className="bg-slate-800 border border-slate-700 rounded-2xl rounded-tl-sm px-4 py-3 flex items-center gap-1">
              {[1, 2, 3].map(n => (
                <span
                  key={n}
                  className="wave-bar w-1.5 rounded-full bg-sky-400"
                  style={{ animationDelay: `${(n - 1) * 0.15}s` }}
                  aria-hidden="true"
                />
              ))}
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </section>

      {/* Suggestions */}
      <div className="flex flex-wrap gap-2 mt-4 mb-3" aria-label="Suggested questions">
        {SUGGESTIONS.map(s => (
          <button
            key={s}
            onClick={() => { setInput(s); inputRef.current?.focus(); }}
            aria-label={`Use suggestion: ${s}`}
            onMouseEnter={() => announceStatus(`Suggestion: ${s}`)}
            onFocus={() => announceStatus(`Suggestion: ${s}`)}
            className="text-xs px-3 py-1.5 rounded-full bg-slate-700 text-slate-300 hover:bg-slate-600 hover:text-white transition-colors border border-slate-600 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400"
          >
            {s}
          </button>
        ))}
      </div>

      {/* Input area */}
      <div className="flex gap-3 items-end" role="form" aria-label="Type or speak your question">
        {/* Microphone */}
        <button
          onClick={listening ? stopListening : startListening}
          aria-label={listening ? 'Stop listening' : 'Start voice input — press to speak'}
          aria-pressed={listening}
          onMouseEnter={() => announceStatus(listening ? 'Stop listening button' : 'Microphone button. Press to speak.')}
          onFocus={() => announceStatus(listening ? 'Stop listening button' : 'Microphone button. Press to speak.')}
          disabled={loading}
          className={[
            'w-12 h-12 rounded-full flex items-center justify-center shrink-0',
            'transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400',
            listening
              ? 'bg-red-500 hover:bg-red-400 animate-pulse'
              : 'bg-slate-700 hover:bg-slate-600 text-slate-300',
          ].join(' ')}
        >
          {listening ? (
            <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2} aria-hidden="true">
              <rect x="6" y="4" width="4" height="16" />
              <rect x="14" y="4" width="4" height="16" />
            </svg>
          ) : (
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2} aria-hidden="true">
              <path strokeLinecap="round" strokeLinejoin="round" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
            </svg>
          )}
        </button>

        {/* Text input */}
        <div className="flex-1 relative">
          <textarea
            ref={inputRef}
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={listening ? 'Listening…' : 'Type your question here…'}
            aria-label="Question input. Press Enter to send."
            disabled={loading}
            rows={2}
            className="w-full bg-slate-800 border border-slate-600 text-white rounded-xl px-4 py-3 text-base resize-none focus:outline-none focus:ring-2 focus:ring-sky-500 focus:border-sky-500 placeholder-slate-500 disabled:opacity-50"
          />
        </div>

        {/* Send */}
        <Button
          size="md"
          onClick={handleSubmit}
          loading={loading}
          disabled={!input.trim() || loading}
          label="Send message"
          aria-label="Send message"
          className="h-12 w-12 p-0 !rounded-full"
        >
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2} aria-hidden="true">
            <path strokeLinecap="round" strokeLinejoin="round" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
          </svg>
        </Button>
      </div>

      {listening && (
        <div
          className="flex items-center justify-center gap-2 mt-4 text-red-400 text-sm font-medium"
          role="status"
          aria-live="assertive"
        >
          <span className="w-2 h-2 bg-red-400 rounded-full animate-ping" aria-hidden="true" />
          Listening… speak now
        </div>
      )}
    </main>
  );
}
