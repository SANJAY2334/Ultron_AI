/* Stage 7 Header Component & Native Window Controls Tests */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, cleanup } from '@testing-library/react';
import { Header } from '../Header';
import { INITIAL_STATE } from '../../state/useUltronState';

describe('Header Component & Native Desktop Window Controls (Stage 7)', () => {
  beforeEach(() => {
    // Default: electron context with window.ultronNative present
    window.ultronNative = {
      minimizeWindow: vi.fn(),
      maximizeWindow: vi.fn(),
      closeToTray: vi.fn(),
      toggleHUD: vi.fn().mockResolvedValue(true),
      sendNotification: vi.fn(),
      onGlobalShortcutTriggered: vi.fn().mockReturnValue(() => {}),
      onBackendStatusUpdate: vi.fn().mockReturnValue(() => {}),
      getPlatformInfo: vi.fn().mockResolvedValue({
        platform: 'win32',
        arch: 'x64',
        version: '0.1.0',
        isNativeDesktop: true,
      }),
    };
  });

  afterEach(() => {
    cleanup();
  });

  it('should render native window controls when running in Electron context', () => {
    render(
      <Header
        connectivity="ONLINE"
        systemTelemetry={INITIAL_STATE.systemTelemetry}
        avatarState="IDLE"
        backendVersion="0.1.0"
        backendKernelState="READY"
      />
    );

    expect(screen.getByTitle('Minimize Window')).toBeDefined();
    expect(screen.getByTitle('Maximize or Restore Window')).toBeDefined();
    expect(screen.getByTitle('Close to System Tray')).toBeDefined();
  });

  it('should invoke minimizeWindow on Minimize button click', () => {
    render(
      <Header
        connectivity="ONLINE"
        systemTelemetry={INITIAL_STATE.systemTelemetry}
        avatarState="IDLE"
        backendVersion="0.1.0"
        backendKernelState="READY"
      />
    );

    const minBtn = screen.getByTitle('Minimize Window');
    fireEvent.click(minBtn);
    expect(window.ultronNative?.minimizeWindow).toHaveBeenCalledTimes(1);
  });

  it('should invoke maximizeWindow on Maximize button click', () => {
    render(
      <Header
        connectivity="ONLINE"
        systemTelemetry={INITIAL_STATE.systemTelemetry}
        avatarState="IDLE"
        backendVersion="0.1.0"
        backendKernelState="READY"
      />
    );

    const maxBtn = screen.getByTitle('Maximize or Restore Window');
    fireEvent.click(maxBtn);
    expect(window.ultronNative?.maximizeWindow).toHaveBeenCalledTimes(1);
  });

  it('should invoke closeToTray on Close to Tray button click', () => {
    render(
      <Header
        connectivity="ONLINE"
        systemTelemetry={INITIAL_STATE.systemTelemetry}
        avatarState="IDLE"
        backendVersion="0.1.0"
        backendKernelState="READY"
      />
    );

    const closeBtn = screen.getByTitle('Close to System Tray');
    fireEvent.click(closeBtn);
    expect(window.ultronNative?.closeToTray).toHaveBeenCalledTimes(1);
  });

  it('should not render native window controls when running in standard browser context', () => {
    // Simulate standard browser by deleting window.ultronNative
    delete (window as { ultronNative?: unknown }).ultronNative;

    render(
      <Header
        connectivity="ONLINE"
        systemTelemetry={INITIAL_STATE.systemTelemetry}
        avatarState="IDLE"
        backendVersion="0.1.0"
        backendKernelState="READY"
      />
    );

    expect(screen.queryByTitle('Minimize Window')).toBeNull();
    expect(screen.queryByTitle('Maximize or Restore Window')).toBeNull();
    expect(screen.queryByTitle('Close to System Tray')).toBeNull();
  });

  it('should render RECONNECTING connectivity indicator correctly', () => {
    render(
      <Header
        connectivity="RECONNECTING"
        systemTelemetry={INITIAL_STATE.systemTelemetry}
        avatarState="OFFLINE"
        backendVersion={null}
        backendKernelState={null}
      />
    );

    expect(screen.getByText('RECONNECTING...')).toBeDefined();
  });
});
