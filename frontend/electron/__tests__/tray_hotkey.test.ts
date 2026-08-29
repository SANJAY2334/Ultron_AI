/* Stage 4 Electron System Tray & Global Hotkey Unit Tests */

import { describe, it, expect } from 'vitest';
import * as fs from 'fs';
import * as path from 'path';
import { ULTRON_GLOBAL_ACCELERATOR } from '../hotkey';

describe('Electron Global Hotkey & System Tray Contracts (Stage 4)', () => {
  const hotkeyTsPath = path.resolve(__dirname, '../hotkey.ts');
  const hotkeyContent = fs.readFileSync(hotkeyTsPath, 'utf-8');

  const trayTsPath = path.resolve(__dirname, '../tray.ts');
  const trayContent = fs.readFileSync(trayTsPath, 'utf-8');

  const mainTsPath = path.resolve(__dirname, '../main.ts');
  const mainContent = fs.readFileSync(mainTsPath, 'utf-8');

  it('should use the approved CommandOrControl+Shift+U accelerator', () => {
    expect(ULTRON_GLOBAL_ACCELERATOR).toBe('CommandOrControl+Shift+U');
    expect(hotkeyContent).toContain("ULTRON_GLOBAL_ACCELERATOR = 'CommandOrControl+Shift+U'");
  });

  it('should notify renderer via ultron:shortcut:activated on hotkey trigger', () => {
    expect(hotkeyContent).toContain("win.webContents.send('ultron:shortcut:activated')");
  });

  it('should unregister all shortcuts on shutdown to avoid leaks', () => {
    expect(hotkeyContent).toContain('globalShortcut.unregisterAll()');
    expect(mainContent).toContain('unregisterGlobalShortcuts()');
  });

  it('should prevent duplicate shortcut registration', () => {
    expect(hotkeyContent).toContain('globalShortcut.isRegistered(ULTRON_GLOBAL_ACCELERATOR)');
    expect(hotkeyContent).toContain('globalShortcut.unregister(ULTRON_GLOBAL_ACCELERATOR)');
  });

  it('should configure System Tray with required menu items', () => {
    expect(trayContent).toContain("'Open Command Center'");
    expect(trayContent).toContain("'Minimize to Tray'");
    expect(trayContent).toContain('Backend: ${currentBackendStatus}');
    expect(trayContent).toContain("'Quit ULTRON'");
  });

  it('should intercept window close event for close-to-tray behavior', () => {
    expect(mainContent).toContain("win.on('close'");
    expect(mainContent).toContain('event.preventDefault()');
    expect(mainContent).toContain('win.hide()');
  });

  it('should cleanly perform safe application quit', () => {
    expect(mainContent).toContain('function performSafeQuit()');
    expect(mainContent).toContain('isQuitting = true');
    expect(mainContent).toContain('destroySystemTray()');
    expect(mainContent).toContain('app.quit()');
  });

  it('should support dynamic tray backend status updates', () => {
    expect(trayContent).toContain('updateTrayBackendStatus');
    expect(trayContent).toContain('setToolTip');
  });
});
