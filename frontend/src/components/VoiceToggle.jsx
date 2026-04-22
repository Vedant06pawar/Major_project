import { useState, useEffect } from 'react';
import { isVoiceGuidanceEnabled, setVoiceGuidanceEnabled, speak } from '../utils/speech';

export default function VoiceToggle() {
  const [enabled, setEnabled] = useState(isVoiceGuidanceEnabled());

  function toggle() {
    const next = !enabled;
    setEnabled(next);
    setVoiceGuidanceEnabled(next);
    speak(next ? 'Voice guidance enabled' : 'Voice guidance disabled', { force: true });
  }

  return (
    <button
      onClick={toggle}
      aria-label={enabled ? 'Voice guidance is on. Click to disable.' : 'Voice guidance is off. Click to enable.'}
      aria-pressed={enabled}
      className={[
        'flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium',
        'border transition-colors duration-150 focus-visible:outline-none',
        'focus-visible:ring-2 focus-visible:ring-sky-400',
        enabled
          ? 'border-sky-500 bg-sky-500/10 text-sky-300'
          : 'border-slate-600 bg-slate-800 text-slate-400 hover:text-slate-200',
      ].join(' ')}
    >
      <svg
        className="w-4 h-4"
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
        strokeWidth={2}
        aria-hidden="true"
      >
        {enabled ? (
          <path strokeLinecap="round" strokeLinejoin="round" d="M15.536 8.464a5 5 0 010 7.072M12 6v12m-3.536-9.536a5 5 0 000 7.072M9 12H3m18 0h-6" />
        ) : (
          <path strokeLinecap="round" strokeLinejoin="round" d="M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z" />
        )}
      </svg>
      {enabled ? 'Voice On' : 'Voice Off'}
    </button>
  );
}
