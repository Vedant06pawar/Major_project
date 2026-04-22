import { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import Button from '../components/Button';
import { announceStatus, accessibleHandlers } from '../utils/speech';

const FEATURES = [
  {
    icon: '📄',
    title: 'PDF & DOCX Support',
    desc: 'Upload any PDF or Word document and have it read aloud line by line.',
  },
  {
    icon: '🖼️',
    title: 'Image Descriptions',
    desc: 'Charts and images are automatically described so nothing is missed.',
  },
  {
    icon: '🎙️',
    title: 'Voice Assistant',
    desc: 'Ask questions and get spoken responses hands-free.',
  },
  {
    icon: '♿',
    title: 'Built for Accessibility',
    desc: 'Full keyboard navigation, screen reader support, and voice guidance throughout.',
  },
];

export default function Home() {
  const navigate = useNavigate();

  useEffect(() => {
    announceStatus(
      'Welcome to DocReader AI. An accessible document reader for visually impaired students. Press Tab to navigate or use the Upload button to get started.'
    );
  }, []);

  return (
    <main
      id="main-content"
      className="flex-1 flex flex-col items-center px-4 py-16 max-w-4xl mx-auto w-full"
      aria-label="Home page"
    >
      {/* Hero */}
      <div className="text-center mb-16">
        <div className="text-7xl mb-6" role="img" aria-label="Open book with sound waves">
          📖
        </div>
        <h1 className="text-5xl font-bold text-white mb-4 leading-tight">
          DocReader{' '}
          <span className="text-sky-400">AI</span>
        </h1>
        <p className="text-xl text-slate-400 max-w-xl mx-auto leading-relaxed">
          An AI-powered document reader designed for visually impaired students. Upload a PDF, sit back, and listen.
        </p>

        <div className="flex flex-col sm:flex-row gap-4 justify-center mt-10">
          <Button
            size="xl"
            onClick={() => navigate('/upload')}
            aria-label="Upload a PDF or DOCX document to get started"
            className="min-w-[200px]"
          >
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2} aria-hidden="true">
              <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
            </svg>
            Upload Document
          </Button>

          <Button
            size="xl"
            variant="ghost"
            onClick={() => navigate('/assistant')}
            aria-label="Open the voice assistant"
            className="min-w-[200px]"
          >
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2} aria-hidden="true">
              <path strokeLinecap="round" strokeLinejoin="round" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
            </svg>
            Voice Assistant
          </Button>
        </div>
      </div>

      {/* Features */}
      <section aria-label="Features" className="w-full">
        <h2 className="text-2xl font-bold text-white text-center mb-8">What DocReader AI can do</h2>
        <ul
          className="grid grid-cols-1 sm:grid-cols-2 gap-5 list-none p-0 m-0"
          role="list"
        >
          {FEATURES.map(({ icon, title, desc }) => (
            <li
              key={title}
              className="bg-slate-800/60 border border-slate-700 rounded-2xl p-6 hover:border-sky-500/40 hover:bg-slate-800 transition-all"
              {...accessibleHandlers(`${title}: ${desc}`)}
            >
              <div className="text-4xl mb-3" role="img" aria-label={title} aria-hidden="true">{icon}</div>
              <h3 className="text-lg font-semibold text-white mb-2">{title}</h3>
              <p className="text-slate-400 text-sm leading-relaxed">{desc}</p>
            </li>
          ))}
        </ul>
      </section>

      {/* How to use */}
      <section aria-label="How to use" className="w-full mt-16">
        <h2 className="text-2xl font-bold text-white text-center mb-8">How it works</h2>
        <ol className="flex flex-col sm:flex-row gap-4 list-none p-0 m-0">
          {[
            { step: '1', title: 'Upload', desc: 'Upload your PDF or DOCX file' },
            { step: '2', title: 'Process', desc: 'AI extracts text and describes images' },
            { step: '3', title: 'Listen', desc: 'Navigate and listen block by block' },
          ].map(({ step, title, desc }) => (
            <li
              key={step}
              className="flex-1 flex flex-col items-center text-center bg-slate-800/40 border border-slate-700 rounded-2xl p-6"
              aria-label={`Step ${step}: ${title} — ${desc}`}
            >
              <div
                className="w-12 h-12 rounded-full bg-sky-500 text-white font-bold text-xl flex items-center justify-center mb-3"
                aria-hidden="true"
              >
                {step}
              </div>
              <h3 className="text-white font-semibold mb-1">{title}</h3>
              <p className="text-slate-400 text-sm">{desc}</p>
            </li>
          ))}
        </ol>
      </section>
    </main>
  );
}
