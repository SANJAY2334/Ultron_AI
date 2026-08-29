/* ULTRON Native Desktop API TypeScript Declarations (Phase 4G.4: Stage 3)
 *
 * Strongly typed interface for native desktop integration exposed via contextBridge.
 *
 * SECURITY INVARIANTS:
 * - Avatar != Authority
 * - Desktop Shell != ToolExecutor
 * - Frontend != PolicyEngine
 * - No direct execution, no shell access, no filesystem access.
 */

export interface NativeNotificationPayload {
  title: string;
  body: string;
  level?: 'info' | 'warning' | 'error';
}

export interface NativePlatformInfo {
  platform: string;
  arch: string;
  version: string;
  isNativeDesktop: boolean;
}

export interface BackendStatusNotification {
  online: boolean;
  kernelState?: string;
}

export interface UltronNativeAPI {
  minimizeWindow: () => void;
  maximizeWindow: () => void;
  closeToTray: () => void;
  toggleHUD: () => Promise<boolean>;
  sendNotification: (payload: NativeNotificationPayload) => void;
  onGlobalShortcutTriggered: (callback: () => void) => () => void;
  onBackendStatusUpdate: (callback: (status: BackendStatusNotification) => void) => () => void;
  getPlatformInfo: () => Promise<NativePlatformInfo>;
}

declare global {
  interface Window {
    ultronNative?: UltronNativeAPI;
  }
}
