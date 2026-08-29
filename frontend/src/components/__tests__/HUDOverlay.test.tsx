/* Stage 5 HUDOverlay React Component Unit Tests */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, cleanup } from '@testing-library/react';
import { HUDOverlay } from '../HUDOverlay';
import { INITIAL_STATE } from '../../state/useUltronState';
import type { UltronSystemState } from '../../state/UltronSystemState';

describe('HUDOverlay React Component (Stage 5)', () => {
  beforeEach(() => {
    // Setup window.ultronNative mock
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

  it('should render the HUD title and avatar state correctly', () => {
    render(<HUDOverlay state={INITIAL_STATE} />);
    expect(screen.getByText('ULTRON HUD')).toBeDefined();
    expect(screen.getAllByText('OFFLINE').length).toBeGreaterThan(0);
    expect(screen.getByText('SILENCE')).toBeDefined();
  });

  it('should reflect SPEECH_DETECTED state and active VAD telemetry', () => {
    const activeState: UltronSystemState = {
      ...INITIAL_STATE,
      connectivity: 'ONLINE',
      avatarState: 'SPEECH_DETECTED',
      audioState: {
        ...INITIAL_STATE.audioState,
        vadActive: true,
      },
    };

    render(<HUDOverlay state={activeState} />);
    expect(screen.getAllByText('SPEECH_DETECTED').length).toBeGreaterThan(0);
    expect(screen.getByText('SPEECH DETECTED')).toBeDefined();
  });

  it('should reflect CONFIRMATION_REQUIRED state', () => {
    const confirmationState: UltronSystemState = {
      ...INITIAL_STATE,
      connectivity: 'ONLINE',
      avatarState: 'CONFIRMATION_REQUIRED',
      securityState: {
        ...INITIAL_STATE.securityState,
        policyDecision: 'REQUIRES_CONFIRMATION',
      },
    };

    render(<HUDOverlay state={confirmationState} />);
    expect(screen.getAllByText('CONFIRMATION_REQUIRED').length).toBeGreaterThan(0);
  });

  it('should render CPU and RAM telemetry when values are available', () => {
    const telemetryState: UltronSystemState = {
      ...INITIAL_STATE,
      connectivity: 'ONLINE',
      systemTelemetry: {
        cpu_percent: 42,
        memory_used_gb: 3.5,
        memory_total_gb: 16.0,
        active_processes: 120,
        uptime_seconds: 3600,
      },
    };

    render(<HUDOverlay state={telemetryState} />);
    expect(screen.getByText('42%')).toBeDefined();
    expect(screen.getByText('3.5GB')).toBeDefined();
  });

  it('should invoke maximizeWindow on EXPAND COMMAND CENTER button click', () => {
    render(<HUDOverlay state={INITIAL_STATE} />);
    const expandButton = screen.getByText('EXPAND COMMAND CENTER');
    fireEvent.click(expandButton);
    expect(window.ultronNative?.maximizeWindow).toHaveBeenCalledTimes(1);
  });

  it('should invoke toggleHUD on Hide HUD button click', () => {
    render(<HUDOverlay state={INITIAL_STATE} />);
    const hideButton = screen.getByTitle('Hide HUD');
    fireEvent.click(hideButton);
    expect(window.ultronNative?.toggleHUD).toHaveBeenCalledTimes(1);
  });
});
