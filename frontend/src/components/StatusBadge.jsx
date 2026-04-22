const CONFIGS = {
  processing: { color: 'bg-amber-500/10 text-amber-300 border-amber-500/30', dot: 'bg-amber-400', label: 'Processing' },
  done: { color: 'bg-emerald-500/10 text-emerald-300 border-emerald-500/30', dot: 'bg-emerald-400', label: 'Done' },
  error: { color: 'bg-red-500/10 text-red-300 border-red-500/30', dot: 'bg-red-400', label: 'Error' },
  idle: { color: 'bg-slate-500/10 text-slate-400 border-slate-500/30', dot: 'bg-slate-500', label: 'Idle' },
};

export default function StatusBadge({ status = 'idle' }) {
  const cfg = CONFIGS[status] || CONFIGS.idle;
  return (
    <span
      role="status"
      aria-live="polite"
      aria-label={`Status: ${cfg.label}`}
      className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-sm border ${cfg.color}`}
    >
      <span
        className={`w-2 h-2 rounded-full ${cfg.dot} ${status === 'processing' ? 'animate-pulse' : ''}`}
        aria-hidden="true"
      />
      {cfg.label}
    </span>
  );
}
