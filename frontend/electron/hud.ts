/* ULTRON Native HUD Overlay Window Manager (Phase 4G.4: Stage 5)
 *
 * Manages the dedicated lightweight, frameless, transparent desktop HUD overlay window.
 *
 * HARD SECURITY INVARIANTS:
 * - nodeIntegration: false
 * - contextIsolation: true
 * - sandbox: true
 * - webSecurity: true
 * - allowRunningInsecureContent: false
 * - HUD Overlay is strictly PRESENTATION-ONLY.
 * - HUD possesses ZERO tool execution or policy evaluation authority.
 */

import { app, BrowserWindow, screen } from 'electron';
import * as path from 'path';

let hudWindow: BrowserWindow | null = null;

export function createHUDWindow(_getMainWindow?: () => BrowserWindow | null): BrowserWindow {
  if (hudWindow && !hudWindow.isDestroyed()) {
    return hudWindow;
  }

  const isDev = process.env.NODE_ENV === 'development' || !app.isPackaged;
  const primaryDisplay = screen.getPrimaryDisplay();
  const { width: screenWidth } = primaryDisplay.workAreaSize;

  const hudWidth = 360;
  const hudHeight = 440;
  // Position near the top-right corner with 24px padding
  const xPos = Math.max(0, screenWidth - hudWidth - 24);
  const yPos = 24;

  hudWindow = new BrowserWindow({
    width: hudWidth,
    height: hudHeight,
    minWidth: 320,
    minHeight: 380,
    x: xPos,
    y: yPos,
    frame: false,
    transparent: true,
    alwaysOnTop: true,
    skipTaskbar: true,
    resizable: false,
    backgroundColor: '#00000000',
    title: 'ULTRON — HUD Overlay',
    show: false,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      sandbox: true,
      webSecurity: true,
      allowRunningInsecureContent: false,
      preload: path.join(__dirname, 'preload.js'),
    },
  });

  hudWindow.once('ready-to-show', () => {
    if (hudWindow && !hudWindow.isDestroyed()) {
      hudWindow.show();
    }
  });

  // Load HUD view in dev or prod mode
  if (isDev && process.env.VITE_DEV_SERVER_URL) {
    hudWindow.loadURL(`${process.env.VITE_DEV_SERVER_URL}?view=hud`);
  } else if (isDev) {
    hudWindow.loadURL('http://localhost:5173/?view=hud');
  } else {
    hudWindow.loadFile(path.join(__dirname, '../dist/index.html'), {
      query: { view: 'hud' },
    });
  }

  hudWindow.on('closed', () => {
    hudWindow = null;
  });

  return hudWindow;
}

export function toggleHUDWindow(getMainWindow: () => BrowserWindow | null): boolean {
  if (!hudWindow || hudWindow.isDestroyed()) {
    const win = createHUDWindow(getMainWindow);
    return win.isVisible();
  }

  if (hudWindow.isVisible()) {
    hudWindow.hide();
    return false;
  } else {
    hudWindow.show();
    hudWindow.focus();
    return true;
  }
}

export function getHUDWindow(): BrowserWindow | null {
  return hudWindow && !hudWindow.isDestroyed() ? hudWindow : null;
}

export function destroyHUDWindow(): void {
  if (hudWindow && !hudWindow.isDestroyed()) {
    hudWindow.destroy();
    hudWindow = null;
  }
}
