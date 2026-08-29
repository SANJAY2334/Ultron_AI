/* ULTRON Native Desktop Shell — Main Process (Phase 4G.4: Stage 7)
 *
 * Authoritative Electron main process managing desktop application lifecycle,
 * single-instance locking, System Tray, global hotkeys, HUD Overlay, Native OS Notifications,
 * and resilient shutdown semantics.
 *
 * HARD SECURITY INVARIANTS:
 * - nodeIntegration: false
 * - contextIsolation: true
 * - sandbox: true
 * - webSecurity: true
 * - allowRunningInsecureContent: false
 * - Desktop Shell != ToolExecutor (No shell automation, no direct FS operations)
 * - The Python backend remains the sole authoritative execution boundary.
 * - IPC is restricted strictly to known, whitelisted channels.
 * - Notification != Authorization (Notifications are presentation-only).
 */

import { app, BrowserWindow, ipcMain } from 'electron';
import * as path from 'path';
import { WHITELISTED_IPC_CHANNELS } from './channels';
import { createSystemTray, destroySystemTray, updateTrayBackendStatus } from './tray';
import { registerGlobalShortcuts, unregisterGlobalShortcuts, ULTRON_GLOBAL_ACCELERATOR } from './hotkey';
import { createHUDWindow, toggleHUDWindow, destroyHUDWindow, getHUDWindow } from './hud';
import { dispatchNativeNotification, dispatchBackendStatusNotification } from './notifications';
import type { NativePlatformInfo } from '../src/types/native';

// Primary Command Center window reference
let mainWindow: BrowserWindow | null = null;
let isQuitting = false;
let isCleanedUp = false;

export { WHITELISTED_IPC_CHANNELS, ULTRON_GLOBAL_ACCELERATOR };

/**
 * Authoritative, idempotent shutdown sequence.
 * Ensures hotkeys are unregistered, HUD and Tray are destroyed,
 * and no orphaned processes/timers remain.
 */
export function performIdempotentShutdown(): void {
  if (isCleanedUp) return;
  isCleanedUp = true;
  isQuitting = true;

  try {
    unregisterGlobalShortcuts();
  } catch {
    // Ignore errors during emergency cleanup
  }

  try {
    destroyHUDWindow();
  } catch {
    // Ignore errors during emergency cleanup
  }

  try {
    destroySystemTray();
  } catch {
    // Ignore errors during emergency cleanup
  }
}

// 1. Single Instance Lock Enforcement (Only execute if app is defined in Electron runtime)
if (typeof app !== 'undefined' && app.requestSingleInstanceLock) {
  const gotSingleInstanceLock = app.requestSingleInstanceLock();

  if (!gotSingleInstanceLock) {
    // Another instance is already running; terminate this secondary process immediately
    app.quit();
  } else {
    // Handle second-instance launch: restore and focus the existing primary window
    app.on('second-instance', () => {
      if (mainWindow) {
        if (mainWindow.isMinimized()) {
          mainWindow.restore();
        }
        if (!mainWindow.isVisible()) {
          mainWindow.show();
        }
        mainWindow.focus();
      }
    });

    // 2. Main Window Creation Function
    function createMainWindow(): BrowserWindow {
      const isDev = process.env.NODE_ENV === 'development' || !app.isPackaged;

      const win = new BrowserWindow({
        width: 1440,
        height: 900,
        minWidth: 1024,
        minHeight: 700,
        title: 'ULTRON — Robotic Command Center',
        backgroundColor: '#070a12',
        show: false, // Show once ready-to-show to avoid visual flicker
        autoHideMenuBar: true,
        webPreferences: {
          nodeIntegration: false,
          contextIsolation: true,
          sandbox: true,
          webSecurity: true,
          allowRunningInsecureContent: false,
          preload: path.join(__dirname, 'preload.js'),
        },
      });

      // Graceful display on render readiness
      win.once('ready-to-show', () => {
        win.show();
      });

      // Close-to-tray intercept: minimize/hide to tray unless explicit quit is requested
      win.on('close', (event) => {
        if (!isQuitting) {
          event.preventDefault();
          win.hide();
        }
      });

      // Load development dev server or production built static distribution
      if (isDev && process.env.VITE_DEV_SERVER_URL) {
        win.loadURL(process.env.VITE_DEV_SERVER_URL);
      } else if (isDev) {
        win.loadURL('http://localhost:5173');
      } else {
        win.loadFile(path.join(__dirname, '../dist/index.html'));
      }

      // Handle window closure lifecycle
      win.on('closed', () => {
        mainWindow = null;
      });

      return win;
    }

    // 3. Register Whitelisted IPC Handlers
    function registerIPCHandlers(): void {
      // Window control channels
      ipcMain.on('ultron:window:minimize', (event) => {
        const win = BrowserWindow.fromWebContents(event.sender);
        if (win && !win.isDestroyed()) {
          win.minimize();
        }
      });

      ipcMain.on('ultron:window:maximize', (event) => {
        const win = BrowserWindow.fromWebContents(event.sender);
        if (win && !win.isDestroyed()) {
          if (win.isMaximized()) {
            win.unmaximize();
          } else {
            win.maximize();
          }
        }
      });

      ipcMain.on('ultron:window:close-to-tray', (event) => {
        const win = BrowserWindow.fromWebContents(event.sender);
        if (win && !win.isDestroyed()) {
          win.hide();
        }
      });

      // HUD toggle channel
      ipcMain.handle('ultron:hud:toggle', async () => {
        return toggleHUDWindow(() => mainWindow);
      });

      // Sanitized native notification channel with deduplication and click-to-focus
      ipcMain.on('ultron:notification:send', (_event, rawPayload) => {
        dispatchNativeNotification(rawPayload, () => mainWindow);
      });

      // Platform metadata channel
      ipcMain.handle('ultron:platform:get-info', async (): Promise<NativePlatformInfo> => {
        return {
          platform: process.platform,
          arch: process.arch,
          version: app.getVersion(),
          isNativeDesktop: true,
        };
      });
    }

    // 4. Safe Application Quit Action
    function performSafeQuit(): void {
      performIdempotentShutdown();
      app.quit();
    }

    // 5. Application Lifecycle Event Listeners
    app.whenReady().then(() => {
      registerIPCHandlers();
      mainWindow = createMainWindow();

      // Initialize System Tray and Global Shortcut
      createSystemTray(
        () => mainWindow,
        () => toggleHUDWindow(() => mainWindow),
        performSafeQuit
      );
      registerGlobalShortcuts(() => mainWindow);

      app.on('activate', () => {
        // On macOS/Windows re-activate: re-create window if none exist
        if (BrowserWindow.getAllWindows().length === 0) {
          mainWindow = createMainWindow();
        } else if (mainWindow) {
          if (mainWindow.isMinimized()) mainWindow.restore();
          mainWindow.show();
          mainWindow.focus();
        }
      });
    });

    // Close-to-tray on window close; quit when explicit quit is invoked
    app.on('window-all-closed', () => {
      if (isQuitting) {
        app.quit();
      }
    });

    // Clean shutdown hooks to prevent zombie processes and unregister hotkeys
    app.on('before-quit', () => {
      performIdempotentShutdown();
    });

    app.on('will-quit', () => {
      performIdempotentShutdown();
    });
  }
}

// Export for unit testing and lifecycle inspection
export function getMainWindow(): BrowserWindow | null {
  return mainWindow;
}

export function isAppQuitting(): boolean {
  return isQuitting;
}

export {
  updateTrayBackendStatus,
  getHUDWindow,
  createHUDWindow,
  toggleHUDWindow,
  destroyHUDWindow,
  dispatchNativeNotification,
  dispatchBackendStatusNotification,
};
