import { Link, useLocation } from 'react-router-dom';
import VoiceToggle from './VoiceToggle';
import { accessibleHandlers } from '../utils/speech';

const LINKS = [
  { to: '/', label: 'Home' },
  { to: '/upload', label: 'Upload' },
  { to: '/reader', label: 'Reader' },
  { to: '/assistant', label: 'Assistant' },
];

export default function Navbar() {
  const { pathname } = useLocation();

  return (
    <header
      className="bg-slate-900 border-b border-slate-700 sticky top-0 z-50"
      role="banner"
    >
      <nav
        className="max-w-5xl mx-auto px-4 py-3 flex items-center justify-between"
        aria-label="Main navigation"
      >
        <Link
          to="/"
          className="text-sky-400 font-bold text-lg tracking-tight focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400 rounded"
          aria-label="DocReader AI — go to home"
          {...accessibleHandlers('DocReader AI, home')}
        >
          DocReader AI
        </Link>

        <ul className="flex items-center gap-1 list-none m-0 p-0" role="list">
          {LINKS.map(({ to, label }) => {
            const active = pathname === to;
            return (
              <li key={to}>
                <Link
                  to={to}
                  aria-label={`${label}${active ? ', current page' : ''}`}
                  aria-current={active ? 'page' : undefined}
                  {...accessibleHandlers(`${label} navigation link`)}
                  className={[
                    'px-3 py-2 rounded-lg text-sm font-medium transition-colors',
                    'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400',
                    active
                      ? 'bg-sky-500/20 text-sky-300'
                      : 'text-slate-400 hover:text-slate-100 hover:bg-slate-700',
                  ].join(' ')}
                >
                  {label}
                </Link>
              </li>
            );
          })}
        </ul>

        <VoiceToggle />
      </nav>
    </header>
  );
}
