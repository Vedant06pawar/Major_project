import { useState, useEffect, useRef, useCallback } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import Button from '../components/Button';
import AudioPlayer from '../components/AudioPlayer';
import StatusBadge from '../components/StatusBadge';
import ErrorMessage from '../components/ErrorMessage';
import ProgressBar from '../components/ProgressBar';
import { getBlocks, getStatus, getAudioUrl, getExportUrl, deleteSession } from '../services/api';
import { announceStatus, announceError, speak, cancelSpeech } from '../utils/speech';

const POLL_INTERVAL_MS = 2000;

export default function Reader() {
  const location = useLocation();
  const navigate = useNavigate();

  const sessionId = location.state?.sessionId;

  const [blocks, setBlocks] = useState([]);
  const [docStatus, setDocStatus] = useState('processing');
  const [currentIdx, setCurrentIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [error, setError] = useState('');
  const [activeAudioSrc, setActiveAudioSrc] = useState(null);

  const pollRef = useRef(null);
  const blockRefs = useRef({});
  const audioRef = useRef(null);

  // Redirect if no session
  useEffect(() => {
    if (!sessionId) {
      announceError('No document session found. Please upload a document first.');
      navigate('/upload');
    }
  }, [sessionId, navigate]);

  // Fetch blocks and poll
  useEffect(() => {
    if (!sessionId) return;
    announceStatus('Reader page. Loading your document.');
    fetchBlocks();
    pollRef.current = setInterval(fetchBlocks, POLL_INTERVAL_MS);
    return () => clearInterval(pollRef.current);
  }, [sessionId]);

  async function fetchBlocks() {
    try {
      const data = await getBlocks(sessionId);
      setBlocks(data.blocks || []);
      setDocStatus(data.status);
      if (data.status === 'done' || data.status === 'error') {
        clearInterval(pollRef.current);
        if (data.status === 'done') {
          announceStatus(`Document fully loaded. ${data.blocks.length} blocks available.`);
        } else {
          announceError('Document processing failed.');
        }
      }
    } catch (err) {
      clearInterval(pollRef.current);
      setError('Could not load document blocks.');
      announceError('Could not load document blocks.');
    }
  }

  // Scroll current block into view
  useEffect(() => {
    const el = blockRefs.current[currentIdx];
    if (el) el.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }, [currentIdx]);

  function playBlock(idx) {
    const block = blocks[idx];
    if (!block) return;
    cancelSpeech();
    setCurrentIdx(idx);
    setPlaying(true);

    if (block.has_audio) {
      setActiveAudioSrc(getAudioUrl(sessionId, block.index));
    } else {
      // Fall back to browser TTS
      speak(block.content, { force: true, rate: 0.95 });
      setPlaying(false);
    }

    announceStatus(
      block.type === 'image'
        ? `Image block ${idx + 1} of ${blocks.length}: ${block.content}`
        : `Text block ${idx + 1} of ${blocks.length}`
    );
  }

  function handlePlayPause() {
    if (!blocks.length) return;
    if (playing && audioRef.current) {
      audioRef.current.pause();
      setPlaying(false);
      announceStatus('Paused.');
    } else {
      playBlock(currentIdx);
    }
  }

  function handleNext() {
    const next = currentIdx + 1;
    if (next < blocks.length) {
      playBlock(next);
    } else {
      announceStatus('End of document.');
    }
  }

  function handlePrev() {
    const prev = currentIdx - 1;
    if (prev >= 0) {
      playBlock(prev);
    } else {
      announceStatus('Beginning of document.');
    }
  }

  function onAudioEnded() {
    setPlaying(false);
    // Auto-advance
    const next = currentIdx + 1;
    if (next < blocks.length) {
      setTimeout(() => playBlock(next), 600);
    } else {
      announceStatus('End of document reached.');
    }
  }

  function handleDescribeImage(block) {
    speak(block.content, { force: true, rate: 0.9 });
    announceStatus(`Describing image: ${block.content}`);
  }

  async function handleCleanup() {
    try {
      await deleteSession(sessionId);
    } catch (_) {}
    navigate('/upload');
    announceStatus('Session ended. Returning to upload page.');
  }

  // Keyboard controls
  useEffect(() => {
    function onKey(e) {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
      if (e.key === ' ') { e.preventDefault(); handlePlayPause(); }
      if (e.key === 'ArrowRight' || e.key === 'n') handleNext();
      if (e.key === 'ArrowLeft' || e.key === 'p') handlePrev();
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [currentIdx, playing, blocks]);

  if (!sessionId) return null;

  const currentBlock = blocks[currentIdx];

  return (
    <main
      id="main-content"
      className="flex-1 flex flex-col max-w-4xl mx-auto w-full px-4 py-8 gap-6"
      aria-label="Document reader"
    >
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold text-white">Document Reader</h1>
          <p className="text-slate-400 text-sm mt-1">
            {blocks.length} block{blocks.length !== 1 ? 's' : ''} loaded
          </p>
        </div>
        <div className="flex items-center gap-3">
          <StatusBadge status={docStatus} />
          {docStatus === 'done' && (
            <a
              href={getExportUrl(sessionId)}
              download="document_audio.wav"
              className="text-sm text-sky-400 hover:text-sky-300 underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400 rounded"
              aria-label="Download full document audio as WAV file"
              onMouseEnter={() => announceStatus('Download full audio button')}
              onFocus={() => announceStatus('Download full audio button')}
            >
              Download Audio
            </a>
          )}
          <Button
            variant="ghost"
            size="sm"
            onClick={handleCleanup}
            label="End session and return to upload"
          >
            End Session
          </Button>
        </div>
      </div>

      <ErrorMessage message={error} onDismiss={() => setError('')} />

      {/* Playback controls */}
      <section
        className="bg-slate-800 rounded-2xl p-5 border border-slate-700"
        aria-label="Playback controls"
      >
        <div className="flex flex-wrap items-center justify-center gap-3 mb-4">
          <Button
            variant="secondary"
            size="md"
            onClick={handlePrev}
            disabled={currentIdx === 0}
            label="Previous block"
            aria-label="Previous block"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2} aria-hidden="true">
              <path strokeLinecap="round" strokeLinejoin="round" d="M15 19l-7-7 7-7" />
            </svg>
            Previous
          </Button>

          <Button
            size="lg"
            onClick={handlePlayPause}
            disabled={!blocks.length}
            label={playing ? 'Pause reading' : 'Play reading'}
            aria-label={playing ? 'Pause reading' : 'Play reading'}
            className="min-w-[120px]"
          >
            {playing ? (
              <>
                <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                  <rect x="6" y="4" width="4" height="16" />
                  <rect x="14" y="4" width="4" height="16" />
                </svg>
                Pause
              </>
            ) : (
              <>
                <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                  <polygon points="5,3 19,12 5,21" />
                </svg>
                Play
              </>
            )}
          </Button>

          <Button
            variant="secondary"
            size="md"
            onClick={handleNext}
            disabled={currentIdx >= blocks.length - 1}
            label="Next block"
            aria-label="Next block"
          >
            Next
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2} aria-hidden="true">
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
            </svg>
          </Button>
        </div>

        {/* Block position indicator */}
        {blocks.length > 0 && (
          <div className="text-center text-slate-400 text-sm" aria-live="polite" aria-atomic="true">
            Block {currentIdx + 1} of {blocks.length}
          </div>
        )}

        {/* Keyboard hints */}
        <div className="flex flex-wrap justify-center gap-4 mt-3 text-xs text-slate-600" aria-hidden="true">
          <span>Space — Play/Pause</span>
          <span>← → — Navigate blocks</span>
        </div>
      </section>

      {/* Audio player for current block */}
      {activeAudioSrc && (
        <AudioPlayer
          src={activeAudioSrc}
          label={currentBlock ? `Block ${currentIdx + 1}: ${currentBlock.type === 'image' ? 'Image description' : 'Text'}` : 'Audio'}
          autoPlay={playing}
          onEnded={onAudioEnded}
        />
      )}

      {/* Document blocks */}
      <section
        className="flex flex-col gap-3"
        aria-label="Document blocks"
        aria-live="polite"
      >
        {docStatus === 'processing' && blocks.length === 0 && (
          <div className="flex items-center justify-center py-12 text-slate-400" role="status">
            <svg className="spinner w-6 h-6 mr-3 text-sky-400" viewBox="0 0 24 24" fill="none" aria-hidden="true">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
            </svg>
            Loading document blocks…
          </div>
        )}

        {blocks.map((block, i) => {
          const isActive = i === currentIdx;
          const isImage = block.type === 'image';

          return (
            <article
              key={block.index}
              ref={el => { blockRefs.current[i] = el; }}
              id={`block-${i}`}
              tabIndex={0}
              role="article"
              aria-label={`Block ${i + 1}, ${isImage ? 'image' : 'text'}${isActive ? ', currently selected' : ''}`}
              aria-current={isActive ? 'true' : undefined}
              onClick={() => { setCurrentIdx(i); playBlock(i); }}
              onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setCurrentIdx(i); playBlock(i); } }}
              onFocus={() => announceStatus(`Block ${i + 1} of ${blocks.length}. ${isImage ? 'Image block' : 'Text block'}. Press Enter to play.`)}
              className={[
                'rounded-xl px-5 py-4 border cursor-pointer transition-all',
                'focus-visible:outline-none focus-visible:ring-3 focus-visible:ring-sky-400',
                isActive
                  ? 'border-sky-500 bg-sky-500/10 line-highlight'
                  : 'border-slate-700 bg-slate-800/60 hover:border-slate-500 hover:bg-slate-800',
              ].join(' ')}
            >
              <div className="flex items-start gap-3">
                {/* Type icon */}
                <span
                  className={`mt-0.5 shrink-0 text-xl ${isActive ? 'text-sky-400' : 'text-slate-500'}`}
                  aria-hidden="true"
                >
                  {isImage ? '🖼️' : '📝'}
                </span>

                <div className="flex-1 min-w-0">
                  {/* Type label */}
                  <span
                    className={`text-xs font-semibold uppercase tracking-wide ${isImage ? 'text-amber-400' : 'text-sky-400'} block mb-1`}
                  >
                    {isImage ? 'Image' : 'Text'} Block {i + 1}
                  </span>

                  {/* Content */}
                  <p
                    className={`text-base leading-relaxed ${isActive ? 'text-white' : 'text-slate-300'}`}
                  >
                    {block.content}
                  </p>
                </div>

                {/* Describe image button */}
                {isImage && (
                  <button
                    onClick={(e) => { e.stopPropagation(); handleDescribeImage(block); }}
                    aria-label={`Describe image at block ${i + 1} aloud`}
                    onMouseEnter={() => announceStatus('Describe image button')}
                    onFocus={() => announceStatus('Describe image button')}
                    className="shrink-0 px-2 py-1 text-xs font-medium bg-amber-500/10 text-amber-300 border border-amber-500/30 rounded-lg hover:bg-amber-500/20 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-amber-400"
                  >
                    Describe
                  </button>
                )}

                {/* Audio indicator */}
                {block.has_audio && (
                  <span
                    className="shrink-0 w-2 h-2 rounded-full bg-emerald-400 mt-2"
                    aria-label="Audio available"
                    title="Audio available"
                  />
                )}
              </div>
            </article>
          );
        })}
      </section>

      {docStatus === 'processing' && blocks.length > 0 && (
        <div className="mt-2" aria-live="polite">
          <p className="text-slate-500 text-sm text-center mb-2">Processing more blocks…</p>
          <ProgressBar value={blocks.length} max={blocks.length + 3} label="Loading more blocks" />
        </div>
      )}
    </main>
  );
}
