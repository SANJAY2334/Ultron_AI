/* ULTRON Native OS Notification Dispatcher & Deduplication Engine (Phase 4G.4: Stage 6)
 *
 * Dispatches sanitized OS notifications via Electron Notification API with state-change
 * tracking, rate-limiting, and click-to-focus navigation.
 *
 * HARD SECURITY INVARIANTS:
 * - Notification != Authorization
 * - Clicking a notification ONLY brings the interface to focus (Navigation ONLY).
 * - Zero privileged tool execution or safety interlock bypasses.
 */

import { BrowserWindow, Notification } from 'electron';
import { sanitizeNotificationPayload, RawNotificationPayload, SanitizedNotification } from './sanitization';

export interface NotificationHistoryEntry {
  key: string;
  timestamp: number;
}

// In-memory deduplication cache & state tracking
const recentNotifications: Map<string, number> = new Map();
const COOLDOWN_MS = 5000; // 5-second cooldown for identical notifications
let lastRecordedBackendStatus: 'ONLINE' | 'OFFLINE' | 'CONNECTING' | null = null;

export function isDuplicateNotification(sanitized: SanitizedNotification): boolean {
  const now = Date.now();
  const key = `${sanitized.title}|${sanitized.body}|${sanitized.level}`;
  const lastTime = recentNotifications.get(key);

  if (lastTime && now - lastTime < COOLDOWN_MS) {
    return true; // Duplicate suppressed
  }

  // Cleanup old entries (older than 30s)
  for (const [entryKey, timestamp] of recentNotifications.entries()) {
    if (now - timestamp > 30000) {
      recentNotifications.delete(entryKey);
    }
  }

  recentNotifications.set(key, now);
  return false;
}

export function resetNotificationHistory(): void {
  recentNotifications.clear();
  lastRecordedBackendStatus = null;
}

export function dispatchNativeNotification(
  rawPayload: RawNotificationPayload,
  getMainWindow?: () => BrowserWindow | null
): boolean {
  const sanitized = sanitizeNotificationPayload(rawPayload);

  if (isDuplicateNotification(sanitized)) {
    return false; // Suppressed by deduplication
  }

  if (Notification && Notification.isSupported && Notification.isSupported()) {
    const notification = new Notification({
      title: sanitized.title,
      body: sanitized.body,
      silent: sanitized.level === 'info',
    });

    // Notification click navigation only: restore and focus Command Center
    notification.on('click', () => {
      if (getMainWindow) {
        const win = getMainWindow();
        if (win && !win.isDestroyed()) {
          if (win.isMinimized()) win.restore();
          if (!win.isVisible()) win.show();
          win.focus();
        }
      }
    });

    notification.show();
    return true;
  }

  return false;
}

export function dispatchBackendStatusNotification(
  newStatus: 'ONLINE' | 'OFFLINE' | 'CONNECTING',
  getMainWindow?: () => BrowserWindow | null
): boolean {
  if (newStatus === lastRecordedBackendStatus) {
    return false; // State has not changed — suppress duplicate status notification
  }

  lastRecordedBackendStatus = newStatus;

  let title = 'ULTRON System Status';
  let body = `Backend kernel status transitioned to ${newStatus}.`;
  let level: 'info' | 'warning' | 'error' = 'info';

  if (newStatus === 'ONLINE') {
    title = 'ULTRON Connected';
    body = 'Zero-Trust backend kernel is active and responding.';
    level = 'info';
  } else if (newStatus === 'OFFLINE') {
    title = 'ULTRON Offline';
    body = 'Backend connection lost. Multimodal features are unavailable.';
    level = 'error';
  } else {
    title = 'ULTRON Connecting';
    body = 'Establishing link with local backend gateway...';
    level = 'warning';
  }

  return dispatchNativeNotification({ title, body, level }, getMainWindow);
}
