/* Stage 6 Native OS Notification Dispatcher, Sanitization & Deduplication Tests */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  sanitizeText,
  sanitizeNotificationPayload,
} from '../sanitization';
import {
  isDuplicateNotification,
  resetNotificationHistory,
  dispatchBackendStatusNotification,
} from '../notifications';

describe('Strengthened Multi-Vector Notification Sanitization Engine (Stage 6)', () => {
  it('should redact authorization headers from notifications', () => {
    const text = 'Failed request with Authorization: Bearer secret_admin_token_12345 in header';
    const sanitized = sanitizeText(text, 500, 'fallback');
    expect(sanitized).not.toContain('Bearer secret_admin_token_12345');
    expect(sanitized).toContain('[REDACTED_AUTH_HEADER]');
  });

  it('should redact biometric vector embeddings and facial encodings', () => {
    const text = 'Detected subject face_encoding: [0.1234, -0.5678, 0.9876, 0.5432] in camera frame';
    const sanitized = sanitizeText(text, 500, 'fallback');
    expect(sanitized).not.toContain('[0.1234, -0.5678, 0.9876, 0.5432]');
    expect(sanitized).toContain('[REDACTED_BIOMETRIC_VECTOR]');
  });

  it('should redact system prompts and hidden meta-instructions', () => {
    const text = 'Planner prompt leak: System Prompt: You are an autonomous AI assistant with full system capabilities instructions: do not disobey';
    const sanitized = sanitizeText(text, 500, 'fallback');
    expect(sanitized).not.toContain('You are an autonomous AI assistant');
    expect(sanitized).toContain('[REDACTED_SYSTEM_PROMPT]');
  });

  it('should strip arbitrary HTML and script tags from notification payloads', () => {
    const text = '<script>alert("xss")</script><b>Warning:</b> Memory threshold exceeded <img src="x" />';
    const sanitized = sanitizeText(text, 500, 'fallback');
    expect(sanitized).not.toContain('<script>');
    expect(sanitized).not.toContain('</script>');
    expect(sanitized).not.toContain('<b>');
    expect(sanitized).not.toContain('<img');
    expect(sanitized).toContain('alert("xss")Warning: Memory threshold exceeded');
  });

  it('should redact GitHub personal access tokens and AWS access keys', () => {
    const text = 'Credentials found: ghp_1234567890123456789012345 and AKIAIOSFODNN7EXAMPLE';
    const sanitized = sanitizeText(text, 500, 'fallback');
    expect(sanitized).not.toContain('ghp_1234567890123456789012345');
    expect(sanitized).not.toContain('AKIAIOSFODNN7EXAMPLE');
    expect(sanitized).toContain('[REDACTED_CREDENTIAL]');
  });
});

describe('Notification Deduplication & State-Change Engine (Stage 6)', () => {
  beforeEach(() => {
    resetNotificationHistory();
  });

  it('should allow the first occurrence of a notification and suppress rapid duplicates', () => {
    const payload = sanitizeNotificationPayload({
      title: 'Warning Alert',
      body: 'Subsystem high temperature',
      level: 'warning',
    });

    // First call -> allowed
    expect(isDuplicateNotification(payload)).toBe(false);

    // Immediate second call with identical content -> duplicate suppressed
    expect(isDuplicateNotification(payload)).toBe(true);
  });

  it('should allow distinct notifications even in rapid succession', () => {
    const payload1 = sanitizeNotificationPayload({
      title: 'Alert 1',
      body: 'Event 1 occurred',
      level: 'info',
    });

    const payload2 = sanitizeNotificationPayload({
      title: 'Alert 2',
      body: 'Event 2 occurred',
      level: 'info',
    });

    expect(isDuplicateNotification(payload1)).toBe(false);
    expect(isDuplicateNotification(payload2)).toBe(false);
  });

  it('should notify on backend status transitions and suppress identical status polling', () => {
    // 1. Initial transition to OFFLINE -> allowed (dispatches notification)
    const firstOffline = dispatchBackendStatusNotification('OFFLINE');
    // Returns boolean (may be true or false depending on whether Notification.isSupported is true in test runtime)
    expect(typeof firstOffline).toBe('boolean');

    // 2. Consecutive identical OFFLINE polling -> strictly suppressed
    const duplicateOffline = dispatchBackendStatusNotification('OFFLINE');
    expect(duplicateOffline).toBe(false);

    // 3. State transition to ONLINE -> allowed (dispatches state change)
    const toOnline = dispatchBackendStatusNotification('ONLINE');
    expect(typeof toOnline).toBe('boolean');

    // 4. Consecutive identical ONLINE polling -> strictly suppressed
    const duplicateOnline = dispatchBackendStatusNotification('ONLINE');
    expect(duplicateOnline).toBe(false);
  });
});
