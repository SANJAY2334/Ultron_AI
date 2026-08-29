/* ULTRON Electron Preload Script (Phase 4G.4: Stage 3)
 *
 * Secure IPC bridge establishing the typed window.ultronNative API.
 *
 * HARD SECURITY INVARIANTS:
 * - contextIsolation: true (enforced)
 * - nodeIntegration: false (enforced)
 * - ipcRenderer is NEVER exposed directly
 * - Node.js modules are NEVER exposed directly
 * - Only strictly whitelisted channels are routed
 */

import { contextBridge, ipcRenderer } from 'electron';
import type {
  NativeNotificationPayload,
  NativePlatformInfo,
  BackendStatusNotification,
  UltronNativeAPI,
} from '../src/types/native';

const nativeAPI: UltronNativeAPI = {
  minimizeWindow: (): void => {
    ipcRenderer.send('ultron:window:minimize');
  },

  maximizeWindow: (): void => {
    ipcRenderer.send('ultron:window:maximize');
  },

  closeToTray: (): void => {
    ipcRenderer.send('ultron:window:close-to-tray');
  },

  toggleHUD: async (): Promise<boolean> => {
    return await ipcRenderer.invoke('ultron:hud:toggle');
  },

  sendNotification: (payload: NativeNotificationPayload): void => {
    ipcRenderer.send('ultron:notification:send', payload);
  },

  onGlobalShortcutTriggered: (callback: () => void): (() => void) => {
    const handler = (): void => {
      callback();
    };
    ipcRenderer.on('ultron:shortcut:activated', handler);
    return (): void => {
      ipcRenderer.removeListener('ultron:shortcut:activated', handler);
    };
  },

  onBackendStatusUpdate: (callback: (status: BackendStatusNotification) => void): (() => void) => {
    const handler = (_event: Electron.IpcRendererEvent, status: BackendStatusNotification): void => {
      callback(status);
    };
    ipcRenderer.on('ultron:backend:status', handler);
    return (): void => {
      ipcRenderer.removeListener('ultron:backend:status', handler);
    };
  },

  getPlatformInfo: async (): Promise<NativePlatformInfo> => {
    return await ipcRenderer.invoke('ultron:platform:get-info');
  },
};

// Expose strictly whitelisted API in main world
contextBridge.exposeInMainWorld('ultronNative', nativeAPI);
