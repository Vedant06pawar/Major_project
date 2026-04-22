import { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import UploadZone from '../components/UploadZone';
import ProgressBar from '../components/ProgressBar';
import Button from '../components/Button';
import ErrorMessage from '../components/ErrorMessage';
import StatusBadge from '../components/StatusBadge';
import { uploadDocument, getStatus } from '../services/api';
import { announceStatus, announceError } from '../utils/speech';

const POLL_INTERVAL_MS = 1500;

export default function Upload() {
  const navigate = useNavigate();
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const [status, setStatus] = useState('idle'); // idle | uploading | processing | done | error
  const [blocksReady, setBlocksReady] = useState(0);
  const [totalEstimate, setTotalEstimate] = useState(0);
  const [error, setError] = useState('');
  const pollRef = useRef(null);

  useEffect(() => {
    announceStatus('Upload page. Select a PDF or DOCX file to begin.');
  }, []);

  useEffect(() => {
    return () => clearInterval(pollRef.current);
  }, []);

  function handleFile(f) {
    setFile(f);
    setError('');
    setStatus('idle');
    announceStatus(`File selected: ${f.name}. Press Upload to start processing.`);
  }

  async function handleUpload() {
    if (!file) {
      announceError('Please select a file first.');
      return;
    }
    setError('');
    setUploading(true);
    setStatus('uploading');
    announceStatus('Uploading your document. Please wait.');

    try {
      const { session_id } = await uploadDocument(file);
      setSessionId(session_id);
      setStatus('processing');
      announceStatus('Upload successful. Your document is being processed. Please wait.');
      startPolling(session_id);
    } catch (err) {
      const msg = err.message || 'Upload failed.';
      setError(msg);
      setStatus('error');
      announceError(msg);
    } finally {
      setUploading(false);
    }
  }

  function startPolling(sid) {
    clearInterval(pollRef.current);
    pollRef.current = setInterval(async () => {
      try {
        const data = await getStatus(sid);
        setBlocksReady(data.blocks_ready);

        if (data.blocks_ready > totalEstimate) {
          setTotalEstimate(prev => Math.max(prev, data.blocks_ready + 2));
        }

        if (data.status === 'done') {
          clearInterval(pollRef.current);
          setStatus('done');
          announceStatus(`Processing complete. ${data.blocks_ready} blocks ready. Navigating to reader.`);
          setTimeout(() => navigate('/reader', { state: { sessionId: sid } }), 1500);
        } else if (data.status === 'error') {
          clearInterval(pollRef.current);
          setStatus('error');
          const msg = data.error || 'Processing failed on the server.';
          setError(msg);
          announceError(msg);
        }
      } catch (err) {
        clearInterval(pollRef.current);
        const msg = 'Lost connection to server.';
        setError(msg);
        setStatus('error');
        announceError(msg);
      }
    }, POLL_INTERVAL_MS);
  }

  const isProcessing = status === 'processing';
  const isDone = status === 'done';
  const canUpload = !!file && status === 'idle';

  return (
    <main
      id="main-content"
      className="flex-1 max-w-2xl mx-auto w-full px-4 py-12"
      aria-label="Upload document page"
    >
      <h1 className="text-4xl font-bold text-white mb-2">Upload Document</h1>
      <p className="text-slate-400 mb-8">
        Upload a PDF or DOCX file to have it read aloud with image descriptions.
      </p>

      <div className="flex flex-col gap-6">
        <UploadZone
          onFile={handleFile}
          disabled={isProcessing || isDone || uploading}
        />

        {file && (
          <div
            className="flex items-center gap-3 bg-slate-800 rounded-xl px-4 py-3 border border-slate-700"
            role="status"
            aria-label={`Selected file: ${file.name}, size ${(file.size / 1024).toFixed(1)} kilobytes`}
          >
            <span className="text-2xl" aria-hidden="true">📎</span>
            <div className="flex-1 min-w-0">
              <p className="text-white font-medium truncate">{file.name}</p>
              <p className="text-slate-500 text-sm">{(file.size / 1024).toFixed(1)} KB</p>
            </div>
            {status === 'idle' && (
              <button
                onClick={() => { setFile(null); announceStatus('File removed.'); }}
                aria-label="Remove selected file"
                className="text-slate-500 hover:text-red-400 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-red-400 rounded p-1"
              >
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2} aria-hidden="true">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            )}
          </div>
        )}

        <ErrorMessage message={error} onDismiss={() => setError('')} />

        {isProcessing && (
          <div className="flex flex-col gap-3" role="status" aria-live="polite">
            <div className="flex items-center justify-between">
              <span className="text-slate-300 text-sm">Processing document…</span>
              <StatusBadge status="processing" />
            </div>
            <ProgressBar
              value={blocksReady}
              max={Math.max(totalEstimate, blocksReady + 1)}
              label={`${blocksReady} block${blocksReady !== 1 ? 's' : ''} ready`}
            />
            <p className="text-slate-500 text-sm">
              Text blocks are converted to speech. Images are described by AI.
            </p>
          </div>
        )}

        {isDone && (
          <div
            className="flex items-center gap-3 bg-emerald-900/30 border border-emerald-500/30 rounded-xl px-4 py-3 text-emerald-300"
            role="status"
            aria-live="polite"
          >
            <svg className="w-5 h-5 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2} aria-hidden="true">
              <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
            </svg>
            <span>Done! Redirecting to reader…</span>
          </div>
        )}

        {canUpload && (
          <Button
            size="lg"
            onClick={handleUpload}
            loading={uploading}
            label="Upload and process document"
            className="w-full"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2} aria-hidden="true">
              <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
            </svg>
            Upload & Process
          </Button>
        )}

        {status === 'error' && (
          <Button
            size="lg"
            variant="secondary"
            onClick={() => { setStatus('idle'); setError(''); setFile(null); }}
            label="Try again — upload a new file"
            className="w-full"
          >
            Try Again
          </Button>
        )}
      </div>
    </main>
  );
}
