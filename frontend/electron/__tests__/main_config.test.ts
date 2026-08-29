/* Stage 2 Electron Main Process Security & Configuration Tests */

import { describe, it, expect } from 'vitest';
import * as fs from 'fs';
import * as path from 'path';

describe('Electron Main Process Security & Configuration (Stage 2)', () => {
  const mainTsPath = path.resolve(__dirname, '../main.ts');
  const mainContent = fs.readFileSync(mainTsPath, 'utf-8');

  it('should enforce nodeIntegration = false in webPreferences', () => {
    expect(mainContent).toContain('nodeIntegration: false');
  });

  it('should enforce contextIsolation = true in webPreferences', () => {
    expect(mainContent).toContain('contextIsolation: true');
  });

  it('should enforce sandbox = true in webPreferences', () => {
    expect(mainContent).toContain('sandbox: true');
  });

  it('should enforce webSecurity = true in webPreferences', () => {
    expect(mainContent).toContain('webSecurity: true');
  });

  it('should enforce allowRunningInsecureContent = false in webPreferences', () => {
    expect(mainContent).toContain('allowRunningInsecureContent: false');
  });

  it('should enforce single instance lock using app.requestSingleInstanceLock()', () => {
    expect(mainContent).toContain('app.requestSingleInstanceLock()');
    expect(mainContent).toContain('app.on(\'second-instance\'');
  });

  it('should reference preload.js inside webPreferences', () => {
    expect(mainContent).toContain("preload: path.join(__dirname, 'preload.js')");
  });

  it('should configure minimum window dimensions for Command Center', () => {
    expect(mainContent).toContain('minWidth: 1024');
    expect(mainContent).toContain('minHeight: 700');
  });

  it('should have graceful lifecycle shutdown handlers', () => {
    expect(mainContent).toContain("app.on('window-all-closed'");
    expect(mainContent).toContain("app.on('before-quit'");
  });
});
