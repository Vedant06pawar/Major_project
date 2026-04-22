import { accessibleHandlers } from '../utils/speech';

const VARIANTS = {
  primary: 'bg-sky-500 hover:bg-sky-400 text-white shadow-lg hover:shadow-sky-500/30',
  secondary: 'bg-slate-700 hover:bg-slate-600 text-slate-100',
  danger: 'bg-red-600 hover:bg-red-500 text-white',
  ghost: 'bg-transparent hover:bg-slate-700 text-slate-300 hover:text-white border border-slate-600',
  success: 'bg-emerald-600 hover:bg-emerald-500 text-white',
};

const SIZES = {
  sm: 'px-3 py-2 text-sm',
  md: 'px-5 py-3 text-base',
  lg: 'px-7 py-4 text-lg',
  xl: 'px-8 py-5 text-xl',
};

export default function Button({
  children,
  variant = 'primary',
  size = 'md',
  label,
  disabled = false,
  loading = false,
  icon: Icon = null,
  className = '',
  onClick,
  ...props
}) {
  const ariaLabel = label || (typeof children === 'string' ? children : undefined);
  const handlers = accessibleHandlers(ariaLabel ? `${ariaLabel} button` : 'button');

  return (
    <button
      {...handlers}
      {...props}
      onClick={onClick}
      disabled={disabled || loading}
      aria-label={ariaLabel}
      aria-busy={loading}
      className={[
        'inline-flex items-center justify-center gap-2 rounded-xl font-semibold',
        'focus-visible:outline-none focus-visible:ring-3 focus-visible:ring-sky-400',
        'disabled:opacity-50 disabled:cursor-not-allowed',
        'cursor-pointer select-none',
        VARIANTS[variant] || VARIANTS.primary,
        SIZES[size] || SIZES.md,
        className,
      ].join(' ')}
    >
      {loading ? (
        <svg className="spinner w-5 h-5" viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
        </svg>
      ) : Icon ? (
        <Icon className="w-5 h-5" aria-hidden="true" />
      ) : null}
      {children}
    </button>
  );
}
