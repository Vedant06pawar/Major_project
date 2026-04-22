import { useRef, useState, useEffect } from 'react';
import { accessibleHandlers, announceStatus } from '../utils/speech';

export default function AudioPlayer({ src, label = 'Audio', autoPlay = false, onEnded }) {
  const audioRef = useRef(null);
  const [playing, setPlaying] = useState(false);
  const [progress, setProgress] = useState(0);
  const [duration, setDuration] = useState(0);

  useEffect(() => {
    const el = audioRef.current;
    if (!el || !src) return;

    el.src = src;
    if (autoPlay) {
      el.play().catch(() => {});
    }
  }, [src, autoPlay]);

  function toggle() {
    const el = audioRef.current;
    if (!el) return;
    if (playing) {
      el.pause();
      announceStatus('Paused');
    } else {
      el.play();
      announceStatus('Playing');
    }
  }

  function onTimeUpdate() {
    const el = audioRef.current;
    if (!el || !el.duration) return;
    setProgress(el.currentTime);
    setDuration(el.duration);
  }

  function seek(e) {
    const el = audioRef.current;
    if (!el || !el.duration) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const ratio = (e.clientX - rect.left) / rect.width;
    el.currentTime = ratio * el.duration;
  }

  function fmt(secs) {
    if (!secs || isNaN(secs)) return '0:00';
    const m = Math.floor(secs / 60);
    const s = Math.floor(secs % 60).toString().padStart(2, '0');
    return `${m}:${s}`;
  }

  if (!src) return null;

  return (
    <div
      className="bg-slate-800 rounded-xl p-4 flex flex-col gap-3"
      role="region"
      aria-label={`Audio player: ${label}`}
    >
      <audio
        ref={audioRef}
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        onEnded={() => { setPlaying(false); onEnded?.(); }}
        onTimeUpdate={onTimeUpdate}
        onLoadedMetadata={onTimeUpdate}
        className="sr-only"
      />

      <p className="text-slate-300 text-sm font-medium truncate">{label}</p>

      {/* Seek bar */}
      <div
        className="w-full bg-slate-600 rounded-full h-2 cursor-pointer group"
        role="slider"
        aria-label="Seek audio"
        aria-valuemin={0}
        aria-valuemax={duration || 1}
        aria-valuenow={progress}
        tabIndex={0}
        onClick={seek}
        onKeyDown={(e) => {
          const el = audioRef.current;
          if (!el) return;
          if (e.key === 'ArrowRight') el.currentTime = Math.min(el.currentTime + 5, el.duration);
          if (e.key === 'ArrowLeft') el.currentTime = Math.max(el.currentTime - 5, 0);
        }}
        {...accessibleHandlers('Seek bar. Use arrow keys to skip 5 seconds.')}
      >
        <div
          className="bg-sky-500 h-full rounded-full group-hover:bg-sky-400 transition-colors"
          style={{ width: duration > 0 ? `${(progress / duration) * 100}%` : '0%' }}
        />
      </div>

      <div className="flex items-center justify-between gap-3">
        <span className="text-slate-500 text-xs tabular-nums">{fmt(progress)}</span>

        <button
          onClick={toggle}
          aria-label={playing ? 'Pause audio' : 'Play audio'}
          {...accessibleHandlers(playing ? 'Pause audio button' : 'Play audio button')}
          className="w-10 h-10 rounded-full bg-sky-500 hover:bg-sky-400 flex items-center justify-center focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-300 transition-colors"
        >
          {playing ? (
            <svg className="w-4 h-4 text-white" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
              <rect x="6" y="4" width="4" height="16" />
              <rect x="14" y="4" width="4" height="16" />
            </svg>
          ) : (
            <svg className="w-4 h-4 text-white" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
              <polygon points="5,3 19,12 5,21" />
            </svg>
          )}
        </button>

        <span className="text-slate-500 text-xs tabular-nums">{fmt(duration)}</span>
      </div>
    </div>
  );
}
