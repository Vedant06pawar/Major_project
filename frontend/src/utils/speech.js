/**
 * Speech utility using the Web Speech API (SpeechSynthesis).
 * Provides hover/focus announcements, error speech, and reading controls.
 */

let _voiceGuidanceEnabled = true;
let _currentUtterance = null;
let _lastSpokenText = '';
let _lastSpeakTime = 0;
const DEBOUNCE_MS = 400;

export function isVoiceGuidanceEnabled() {
  return _voiceGuidanceEnabled;
}

export function setVoiceGuidanceEnabled(val) {
  _voiceGuidanceEnabled = val;
  if (!val) cancelSpeech();
}

export function cancelSpeech() {
  if (window.speechSynthesis) {
    window.speechSynthesis.cancel();
  }
  _currentUtterance = null;
}

/**
 * Speak text immediately, cancelling any in-progress speech.
 */
export function speak(text, opts = {}) {
  if (!window.speechSynthesis || !text) return;

  const { force = false, rate = 1, pitch = 1 } = opts;

  // Debounce: don't re-speak the same text within DEBOUNCE_MS
  const now = Date.now();
  if (!force && text === _lastSpokenText && now - _lastSpeakTime < DEBOUNCE_MS) return;

  cancelSpeech();

  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = rate;
  utterance.pitch = pitch;
  utterance.volume = 1;

  _currentUtterance = utterance;
  _lastSpokenText = text;
  _lastSpeakTime = now;

  window.speechSynthesis.speak(utterance);
}

/**
 * Speak a label for accessibility guidance (only when voice guidance is on).
 */
export function announceLabel(label) {
  if (!_voiceGuidanceEnabled) return;
  speak(label);
}

/**
 * Speak an error message (always speaks regardless of guidance toggle).
 */
export function announceError(message) {
  speak(`Error: ${message}`, { force: true, rate: 0.95 });
}

/**
 * Speak a status/success message.
 */
export function announceStatus(message) {
  speak(message, { force: true });
}

/**
 * Returns event handlers for hover and focus accessibility announcements.
 * Use by spreading onto interactive elements:
 *   <button {...accessibleHandlers("Upload PDF button")} />
 */
export function accessibleHandlers(label) {
  return {
    onMouseEnter: () => announceLabel(label),
    onFocus: () => announceLabel(label),
  };
}
