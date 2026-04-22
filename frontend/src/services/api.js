/**
 * API service — maps to the FastAPI backend endpoints.
 * Base URL auto-detects from environment or falls back to localhost:8000.
 */

const BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

async function request(path, options = {}) {
  const res = await fetch(`${BASE_URL}${path}`, options);
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(text || `HTTP ${res.status}`);
  }
  return res;
}

async function json(path, options = {}) {
  const res = await request(path, options);
  return res.json();
}

/**
 * POST /process — upload a PDF or DOCX file.
 * Returns { session_id, message }
 */
export async function uploadDocument(file) {
  const formData = new FormData();
  formData.append('file', file);
  return json('/process', { method: 'POST', body: formData });
}

/**
 * GET /status/{session_id}
 * Returns { session_id, status, error, blocks_ready }
 */
export async function getStatus(sessionId) {
  return json(`/status/${sessionId}`);
}

/**
 * GET /blocks/{session_id}
 * Returns { session_id, status, blocks: [{ index, type, content, has_audio }] }
 */
export async function getBlocks(sessionId) {
  return json(`/blocks/${sessionId}`);
}

/**
 * GET /audio/{session_id}/{block_index}
 * Returns the audio URL (used as <audio src> or for programmatic playback).
 */
export function getAudioUrl(sessionId, blockIndex) {
  return `${BASE_URL}/audio/${sessionId}/${blockIndex}`;
}

/**
 * GET /export/{session_id}
 * Returns the combined audio download URL.
 */
export function getExportUrl(sessionId) {
  return `${BASE_URL}/export/${sessionId}`;
}

/**
 * DELETE /session/{session_id}
 * Cleans up server-side temp files.
 */
export async function deleteSession(sessionId) {
  return json(`/session/${sessionId}`, { method: 'DELETE' });
}

/**
 * GET /health
 * Returns { status, captioner_loaded }
 */
export async function getHealth() {
  return json('/health');
}
