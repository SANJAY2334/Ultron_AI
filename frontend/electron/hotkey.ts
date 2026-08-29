/* ULTRON Global Shortcut Manager (Phase 4G.4: Stage 4)
 *
 * Registers the global accelerator 'CommandOrControl+Shift+U' to summon/focus the application.
 *
 * HARD SECURITY INVARIANTS:
 * - Global shortcut ONLY restores and focuses the user interface.
 * - ZERO tool execution or privileged bypasses.
 * - All authorization continues to route exclusively through the backend.
 */

import { BrowserWindow, globalShortcut } from 'electron';

export const ULTRON_GLOBAL_ACCELERATOR = 'CommandOrControl+Shift+U';

export function registerGlobalShortcuts(
  getMainWindow: () => BrowserWindow | null
): boolean {
  try {
    // Unregister any previous registration to prevent duplicates
    if (globalShortcut.isRegistered(ULTRON_GLOBAL_ACCELERATOR)) {
      globalShortcut.unregister(ULTRON_GLOBAL_ACCELERATOR);
    }

    const success = globalShortcut.register(ULTRON_GLOBAL_ACCELERATOR, () => {
      const win = getMainWindow();
      if (!win || win.isDestroyed()) return;

      if (win.isMinimized()) {
        win.restore();
      }
      if (!win.isVisible()) {
        win.show();
      }
      win.focus();

      // Notify renderer via whitelisted IPC channel
      if (!win.webContents.isDestroyed()) {
        win.webContents.send('ultron:shortcut:activated');
      }
    });

    return success;
  } catch {
    return false;
  }
}

export function unregisterGlobalShortcuts(): void {
  try {
    globalShortcut.unregisterAll();
  } catch {
    // Graceful error handling on quit
  }
}
