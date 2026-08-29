/* ULTRON Native System Tray Manager (Phase 4G.4: Stage 5)
 *
 * Manages the desktop System Tray presence, context menu, and window state transitions.
 *
 * HARD SECURITY INVARIANTS:
 * - Desktop Shell != ToolExecutor
 * - System Tray possesses ZERO execution privileges.
 * - Context menu actions are strictly limited to window visibility and application quit.
 */

import { BrowserWindow, Menu, Tray } from 'electron';
import { createTrayIcon } from './icon';

let trayInstance: Tray | null = null;
let currentBackendStatus: 'ONLINE' | 'OFFLINE' | 'CONNECTING' | 'RECONNECTING' = 'CONNECTING';

export function createSystemTray(
  getMainWindow: () => BrowserWindow | null,
  onToggleHUD: () => void,
  onQuitRequested: () => void
): Tray {
  if (trayInstance && !trayInstance.isDestroyed()) {
    return trayInstance;
  }

  const icon = createTrayIcon();
  trayInstance = new Tray(icon);
  trayInstance.setToolTip(`ULTRON — Robotic Command Center (${currentBackendStatus})`);

  function updateContextMenu(): void {
    if (!trayInstance || trayInstance.isDestroyed()) return;

    const win = getMainWindow();
    const isWinVisible = win ? win.isVisible() && !win.isMinimized() : false;

    const contextMenu = Menu.buildFromTemplate([
      {
        label: 'Open Command Center',
        enabled: !isWinVisible,
        click: () => {
          const targetWin = getMainWindow();
          if (targetWin && !targetWin.isDestroyed()) {
            if (targetWin.isMinimized()) targetWin.restore();
            if (!targetWin.isVisible()) targetWin.show();
            targetWin.focus();
            updateContextMenu();
          }
        },
      },
      {
        label: 'Minimize to Tray',
        enabled: isWinVisible,
        click: () => {
          const targetWin = getMainWindow();
          if (targetWin && !targetWin.isDestroyed()) {
            targetWin.hide();
            updateContextMenu();
          }
        },
      },
      {
        label: 'Toggle HUD Overlay',
        click: () => {
          onToggleHUD();
        },
      },
      { type: 'separator' },
      {
        label: `Backend: ${currentBackendStatus}`,
        enabled: false,
      },
      { type: 'separator' },
      {
        label: 'Quit ULTRON',
        click: () => {
          onQuitRequested();
        },
      },
    ]);

    trayInstance.setContextMenu(contextMenu);
  }

  // Initial menu setup
  updateContextMenu();

  // Click & Double-click toggle handler
  const toggleWindow = (): void => {
    const win = getMainWindow();
    if (!win || win.isDestroyed()) return;

    if (!win.isVisible() || win.isMinimized()) {
      if (win.isMinimized()) win.restore();
      win.show();
      win.focus();
    } else {
      win.hide();
    }
    updateContextMenu();
  };

  trayInstance.on('click', toggleWindow);
  trayInstance.on('double-click', toggleWindow);

  return trayInstance;
}

export function updateTrayBackendStatus(
  status: 'ONLINE' | 'OFFLINE' | 'CONNECTING' | 'RECONNECTING'
): void {
  currentBackendStatus = status;
  if (trayInstance && !trayInstance.isDestroyed()) {
    trayInstance.setToolTip(`ULTRON — Robotic Command Center (${status})`);
  }
}

export function getSystemTray(): Tray | null {
  return trayInstance && !trayInstance.isDestroyed() ? trayInstance : null;
}

export function destroySystemTray(): void {
  if (trayInstance && !trayInstance.isDestroyed()) {
    trayInstance.destroy();
    trayInstance = null;
  }
}
