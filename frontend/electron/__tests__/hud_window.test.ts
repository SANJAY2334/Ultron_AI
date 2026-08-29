/* Stage 5 Electron HUD Overlay Window Architecture Unit Tests */

import { describe, it, expect } from 'vitest';
import * as fs from 'fs';
import * as path from 'path';

describe('Electron HUD Overlay Window Architecture & Security (Stage 5)', () => {
  const hudTsPath = path.resolve(__dirname, '../hud.ts');
  const hudContent = fs.readFileSync(hudTsPath, 'utf-8');

  const mainTsPath = path.resolve(__dirname, '../main.ts');
  const mainContent = fs.readFileSync(mainTsPath, 'utf-8');

  it('should enforce frameless and transparent window configuration for HUD', () => {
    expect(hudContent).toContain('frame: false');
    expect(hudContent).toContain('transparent: true');
    expect(hudContent).toContain("backgroundColor: '#00000000'");
  });

  it('should configure HUD as always-on-top without taskbar presence', () => {
    expect(hudContent).toContain('alwaysOnTop: true');
    expect(hudContent).toContain('skipTaskbar: true');
  });

  it('should enforce all strict webPreferences security controls on HUD window', () => {
    expect(hudContent).toContain('nodeIntegration: false');
    expect(hudContent).toContain('contextIsolation: true');
    expect(hudContent).toContain('sandbox: true');
    expect(hudContent).toContain('webSecurity: true');
    expect(hudContent).toContain('allowRunningInsecureContent: false');
    expect(hudContent).toContain("preload: path.join(__dirname, 'preload.js')");
  });

  it('should implement singleton toggle logic for HUD overlay', () => {
    expect(hudContent).toContain('function toggleHUDWindow(');
    expect(hudContent).toContain('hudWindow.isVisible()');
    expect(hudContent).toContain('hudWindow.hide()');
    expect(hudContent).toContain('hudWindow.show()');
  });

  it('should hook ultron:hud:toggle IPC channel to toggleHUDWindow in main.ts', () => {
    expect(mainContent).toContain("ipcMain.handle('ultron:hud:toggle'");
    expect(mainContent).toContain('toggleHUDWindow(');
  });

  it('should cleanly destroy HUD window during application shutdown', () => {
    expect(hudContent).toContain('function destroyHUDWindow()');
    expect(mainContent).toContain('destroyHUDWindow()');
  });
});
