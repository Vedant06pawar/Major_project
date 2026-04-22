import { useState, useRef } from 'react';
import { announceLabel, announceError } from '../utils/speech';

const ACCEPTED = '.pdf,.docx,.doc';
const ACCEPTED_TYPES = ['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'application/msword'];

export default function UploadZone({ onFile, disabled = false }) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef(null);

  function validate(file) {
    if (!file) return false;
    const ext = file.name.toLowerCase().split('.').pop();
    const validExt = ['pdf', 'docx', 'doc'].includes(ext);
    if (!validExt) {
      announceError('Only PDF and DOCX files are supported.');
      return false;
    }
    return true;
  }

  function handleFile(file) {
    if (validate(file)) onFile(file);
  }

  function onDrop(e) {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    handleFile(file);
  }

  function onDragOver(e) {
    e.preventDefault();
    setDragging(true);
  }

  function onDragLeave() {
    setDragging(false);
  }

  function onInputChange(e) {
    handleFile(e.target.files[0]);
  }

  function onKeyDown(e) {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      inputRef.current?.click();
    }
  }

  return (
    <div
      role="button"
      tabIndex={disabled ? -1 : 0}
      aria-label="Upload zone. Press Enter or Space to browse files, or drag and drop a PDF or DOCX file."
      aria-disabled={disabled}
      onDrop={onDrop}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onMouseEnter={() => announceLabel('Upload zone. Drag and drop or press Enter to browse files.')}
      onFocus={() => announceLabel('Upload zone. Press Enter or Space to select a PDF or DOCX file.')}
      onClick={() => !disabled && inputRef.current?.click()}
      onKeyDown={onKeyDown}
      className={[
        'relative flex flex-col items-center justify-center gap-4',
        'rounded-2xl border-2 border-dashed p-12 cursor-pointer',
        'transition-all duration-200 text-center',
        'focus-visible:outline-none focus-visible:ring-3 focus-visible:ring-sky-400',
        disabled
          ? 'opacity-50 cursor-not-allowed border-slate-700 bg-slate-800/50'
          : dragging
          ? 'border-sky-400 bg-sky-500/10 scale-[1.01]'
          : 'border-slate-600 bg-slate-800/50 hover:border-sky-500 hover:bg-slate-800',
      ].join(' ')}
    >
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED}
        onChange={onInputChange}
        disabled={disabled}
        className="sr-only"
        aria-hidden="true"
        tabIndex={-1}
      />

      <div className={`text-5xl transition-transform ${dragging ? 'scale-110' : ''}`} aria-hidden="true">
        {dragging ? '📂' : '📄'}
      </div>

      <div>
        <p className="text-slate-200 font-semibold text-lg">
          {dragging ? 'Drop your file here' : 'Drag & drop or click to browse'}
        </p>
        <p className="text-slate-500 text-sm mt-1">Supports PDF and DOCX files</p>
      </div>
    </div>
  );
}
