/* ULTRON Notification & IPC Payload Sanitization Engine (Phase 4G.4: Stage 6)
 *
 * Enforces strict security bounds on text payloads sent to native OS notifications.
 * Prevents accidental or intentional data exfiltration of credentials, tokens,
 * raw image data, biometric vectors, system prompts, or internal chain-of-thought reasoning.
 *
 * HARD SECURITY INVARIANTS:
 * - Notification != Authorization
 * - Zero credential, token, or chain-of-thought leakage to OS notifications.
 */

export interface RawNotificationPayload {
  title?: unknown;
  body?: unknown;
  level?: unknown;
}

export interface SanitizedNotification {
  title: string;
  body: string;
  level: 'info' | 'warning' | 'error';
}

const MAX_TITLE_LENGTH = 120;
const MAX_BODY_LENGTH = 500;

// Sensitive data patterns
const JWT_PATTERN = /eyJ[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}/gi;
const API_KEY_PATTERN = /(?:sk-[a-zA-Z0-9_-]{16,}|key-[a-zA-Z0-9_-]{16,}|ghp_[a-zA-Z0-9]{20,}|Bearer\s+[a-zA-Z0-9_.-]{16,}|AKIA[0-9A-Z]{16})/gi;
const AUTH_HEADER_PATTERN = /(?:Authorization|Proxy-Authorization)\s*:\s*[^\r\n,;]+/gi;
const BASE64_IMAGE_PATTERN = /data:(?:image|application)\/[a-zA-Z+.-]+;base64,[a-zA-Z0-9+/=]{30,}/gi;
const PASSWORD_PATTERN = /(?:password|passwd|secret|auth_token|access_key|private_key)\s*[:=]\s*[^\s,;]+/gi;
const BIOMETRIC_VECTOR_PATTERN = /\[\s*(?:-?\d+\.\d+,\s*){3,}-?\d+\.\d+\s*\]|(?:face_encoding|biometric_vector|embedding)\s*[:=]\s*\[[^\]]+\]/gi;
const THOUGHT_PATTERN = /<thought>[\s\S]*?<\/thought>|\[CHAIN_OF_THOUGHT\][\s\S]*?\[\/CHAIN_OF_THOUGHT\]|internal_thought\s*:\s*[^\r\n]+/gi;
const SYSTEM_PROMPT_PATTERN = /(?:System\s*Prompt\s*:|You are an AI assistant|instructions\s*:)\s*[\s\S]{20,}/gi;
const HTML_TAG_PATTERN = /<[^>]+>/g;

export function sanitizeText(input: unknown, maxLength: number, fallback: string): string {
  if (typeof input !== 'string') {
    return fallback;
  }

  let cleaned = input
    .replace(JWT_PATTERN, '[REDACTED_TOKEN]')
    .replace(API_KEY_PATTERN, '[REDACTED_CREDENTIAL]')
    .replace(AUTH_HEADER_PATTERN, '[REDACTED_AUTH_HEADER]')
    .replace(BASE64_IMAGE_PATTERN, '[REDACTED_RAW_FRAME]')
    .replace(PASSWORD_PATTERN, '[REDACTED_SECRET]')
    .replace(BIOMETRIC_VECTOR_PATTERN, '[REDACTED_BIOMETRIC_VECTOR]')
    .replace(THOUGHT_PATTERN, '[REDACTED_REASONING]')
    .replace(SYSTEM_PROMPT_PATTERN, '[REDACTED_SYSTEM_PROMPT]')
    .replace(HTML_TAG_PATTERN, '') // Strip arbitrary HTML tags
    .trim();

  if (cleaned.length === 0) {
    return fallback;
  }

  if (cleaned.length > maxLength) {
    cleaned = cleaned.substring(0, maxLength - 3) + '...';
  }

  return cleaned;
}

export function sanitizeNotificationPayload(payload: RawNotificationPayload): SanitizedNotification {
  const title = sanitizeText(payload?.title, MAX_TITLE_LENGTH, 'ULTRON Notification');
  const body = sanitizeText(payload?.body, MAX_BODY_LENGTH, 'System status update.');
  
  let level: 'info' | 'warning' | 'error' = 'info';
  if (payload?.level === 'warning' || payload?.level === 'error') {
    level = payload.level;
  }

  return { title, body, level };
}
