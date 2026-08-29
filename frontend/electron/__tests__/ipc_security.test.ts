/* Stage 3 Electron IPC Security & Preload Bridge Unit Tests */

import { describe, it, expect } from 'vitest';
import * as fs from 'fs';
import * as path from 'path';
import {
  sanitizeText,
  sanitizeNotificationPayload,
} from '../sanitization';
import { WHITELISTED_IPC_CHANNELS } from '../channels';

describe('Electron IPC Security & Whitelist (Stage 3)', () => {
  const preloadTsPath = path.resolve(__dirname, '../preload.ts');
  const preloadContent = fs.readFileSync(preloadTsPath, 'utf-8');

  it('should contain all approved whitelisted channels and no extra channels', () => {
    const expectedChannels = [
      'ultron:window:minimize',
      'ultron:window:maximize',
      'ultron:window:close-to-tray',
      'ultron:hud:toggle',
      'ultron:notification:send',
      'ultron:shortcut:activated',
      'ultron:backend:status',
      'ultron:platform:get-info',
    ];

    expect([...WHITELISTED_IPC_CHANNELS].sort()).toEqual(expectedChannels.sort());
  });

  it('should use contextBridge.exposeInMainWorld to expose window.ultronNative', () => {
    expect(preloadContent).toContain("contextBridge.exposeInMainWorld('ultronNative'");
  });

  it('should not expose raw ipcRenderer directly in preload', () => {
    // Ensure we do not expose raw ipcRenderer object
    expect(preloadContent).not.toMatch(/exposeInMainWorld\s*\(\s*['"]ipcRenderer['"]/);
    expect(preloadContent).not.toContain('ipcRenderer: ipcRenderer');
  });

  it('should not import or expose dangerous Node.js modules in preload', () => {
    expect(preloadContent).not.toContain("require('child_process')");
    expect(preloadContent).not.toContain("require('fs')");
    expect(preloadContent).not.toContain("require('os')");
    expect(preloadContent).not.toContain("from 'child_process'");
    expect(preloadContent).not.toContain("from 'fs'");
  });

  it('should provide cleanup unsubscribe functions for event listeners', () => {
    expect(preloadContent).toContain('removeListener');
    expect(preloadContent).toContain('onGlobalShortcutTriggered');
    expect(preloadContent).toContain('onBackendStatusUpdate');
  });
});

describe('Notification Payload Sanitization Engine (Stage 3)', () => {
  it('should sanitize raw input strings and enforce maximum length', () => {
    const longText = 'A'.repeat(600);
    const sanitized = sanitizeText(longText, 500, 'fallback');
    expect(sanitized.length).toBe(500);
    expect(sanitized.endsWith('...')).toBe(true);
  });

  it('should redact JWT tokens from notifications', () => {
    const textWithJwt = 'Alert: user token eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.doNotLeakThisSecretSignature';
    const sanitized = sanitizeText(textWithJwt, 500, 'fallback');
    expect(sanitized).not.toContain('eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9');
    expect(sanitized).toContain('[REDACTED_TOKEN]');
  });

  it('should redact API keys and bearer tokens from notifications', () => {
    const textWithKey = 'System error with key sk-1234567890abcdef1234567890 and Bearer secret_bearer_token_12345';
    const sanitized = sanitizeText(textWithKey, 500, 'fallback');
    expect(sanitized).not.toContain('sk-1234567890abcdef1234567890');
    expect(sanitized).not.toContain('secret_bearer_token_12345');
    expect(sanitized).toContain('[REDACTED_CREDENTIAL]');
  });

  it('should redact base64 image data from notifications', () => {
    const textWithImage = 'Vision detection: data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg== in frame';
    const sanitized = sanitizeText(textWithImage, 500, 'fallback');
    expect(sanitized).not.toContain('data:image/png;base64');
    expect(sanitized).toContain('[REDACTED_RAW_FRAME]');
  });

  it('should redact passwords and secrets from notifications', () => {
    const textWithSecret = 'Error: password: SuperSecretPassword123! could not authenticate';
    const sanitized = sanitizeText(textWithSecret, 500, 'fallback');
    expect(sanitized).not.toContain('SuperSecretPassword123!');
    expect(sanitized).toContain('[REDACTED_SECRET]');
  });

  it('should redact internal chain-of-thought tags from notifications', () => {
    const textWithThought = 'Planner: <thought>Evaluating user authorization for tool execution</thought> Response ready.';
    const sanitized = sanitizeText(textWithThought, 500, 'fallback');
    expect(sanitized).not.toContain('Evaluating user authorization');
    expect(sanitized).toContain('[REDACTED_REASONING]');
  });

  it('should sanitize full notification payloads with proper level fallback', () => {
    const payload = {
      title: 'Alert with sk-1234567890abcdef1234567890',
      body: 'Status ok password: mysecret123',
      level: 'warning',
    };

    const sanitized = sanitizeNotificationPayload(payload);
    expect(sanitized.title).toContain('[REDACTED_CREDENTIAL]');
    expect(sanitized.body).toContain('[REDACTED_SECRET]');
    expect(sanitized.level).toBe('warning');
  });

  it('should handle non-object or missing payload fields safely', () => {
    const sanitized = sanitizeNotificationPayload({} as any);
    expect(sanitized.title).toBe('ULTRON Notification');
    expect(sanitized.body).toBe('System status update.');
    expect(sanitized.level).toBe('info');
  });
});
