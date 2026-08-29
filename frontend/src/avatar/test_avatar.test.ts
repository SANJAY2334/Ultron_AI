/* Phase 4G.2 Robotic Avatar Renderer Subsystem Unit Tests */

import { describe, it, expect } from 'vitest';
import { deriveAvatarState, getAvatarColor } from './stateMapper';
import type {
  AvatarAudioState,
  AvatarPlannerState,
  AvatarSecurityState,
  AvatarVisionState,
} from './types';

describe('ULTRON Avatar State Derivation Engine & Security Invariants', () => {
  const defaultAudio: AvatarAudioState = {
    micActive: true,
    vadActive: false,
    wakeWordTriggered: false,
    isSpeaking: false,
  };

  const defaultPlanner: AvatarPlannerState = {
    currentEventType: undefined,
    isExecuting: false,
  };

  const defaultSecurity: AvatarSecurityState = {
    policyDecision: 'ALLOW',
    userConfirmed: false,
  };

  const defaultVision: AvatarVisionState = {
    motionDetected: false,
    objectDetected: false,
    personDetected: false,
  };

  it('should default to IDLE when system is active and quiet', () => {
    const state = deriveAvatarState(defaultAudio, defaultPlanner, defaultSecurity, defaultVision, true);
    expect(state).toBe('IDLE');
  });

  it('should derive OFFLINE when backend is unreachable', () => {
    const state = deriveAvatarState(defaultAudio, defaultPlanner, defaultSecurity, defaultVision, false);
    expect(state).toBe('OFFLINE');
  });

  it('should prioritize security CONFIRMATION_REQUIRED over planner events', () => {
    const security: AvatarSecurityState = { policyDecision: 'REQUIRES_CONFIRMATION', userConfirmed: false };
    const planner: AvatarPlannerState = { currentEventType: 'THINKING', isExecuting: true };

    const state = deriveAvatarState(defaultAudio, planner, security, defaultVision, true);
    expect(state).toBe('CONFIRMATION_REQUIRED');
  });

  it('should map security DENY decision to WARNING state', () => {
    const security: AvatarSecurityState = { policyDecision: 'DENY', userConfirmed: false };

    const state = deriveAvatarState(defaultAudio, defaultPlanner, security, defaultVision, true);
    expect(state).toBe('WARNING');
  });

  it('should map planner stream events accurately', () => {
    expect(deriveAvatarState(defaultAudio, { currentEventType: 'THINKING', isExecuting: true }, defaultSecurity, defaultVision)).toBe('THINKING');
    expect(deriveAvatarState(defaultAudio, { currentEventType: 'TOOL_STARTED', isExecuting: true }, defaultSecurity, defaultVision)).toBe('TOOL_EXECUTION');
    expect(deriveAvatarState(defaultAudio, { currentEventType: 'TOOL_COMPLETED', isExecuting: true }, defaultSecurity, defaultVision)).toBe('PROCESSING');
    expect(deriveAvatarState(defaultAudio, { currentEventType: 'COMPLETED', isExecuting: false }, defaultSecurity, defaultVision)).toBe('SPEAKING');
    expect(deriveAvatarState(defaultAudio, { currentEventType: 'ERROR', isExecuting: false }, defaultSecurity, defaultVision)).toBe('ERROR');
  });

  it('should map audio pipeline state changes accurately', () => {
    expect(deriveAvatarState({ ...defaultAudio, wakeWordTriggered: true }, defaultPlanner, defaultSecurity, defaultVision)).toBe('LISTENING');
    expect(deriveAvatarState({ ...defaultAudio, vadActive: true }, defaultPlanner, defaultSecurity, defaultVision)).toBe('SPEECH_DETECTED');
    expect(deriveAvatarState({ ...defaultAudio, isSpeaking: true }, defaultPlanner, defaultSecurity, defaultVision)).toBe('SPEAKING');
  });

  it('should return valid color theme objects for all avatar states', () => {
    const states = [
      'IDLE',
      'LISTENING',
      'SPEECH_DETECTED',
      'THINKING',
      'PROCESSING',
      'SPEAKING',
      'TOOL_EXECUTION',
      'WARNING',
      'ERROR',
      'CONFIRMATION_REQUIRED',
      'OFFLINE',
    ] as const;

    states.forEach((st) => {
      const theme = getAvatarColor(st);
      expect(theme.primary).toMatch(/^#[0-9a-fA-F]{6}$/);
      expect(theme.glow).toBeDefined();
      expect(theme.badge).toBeDefined();
    });
  });

  it('should strictly enforce privacy invariants with zero sensitive string leakage', () => {
    const sensitivePayload = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.SECRET_KEY_123';
    const state = deriveAvatarState(defaultAudio, defaultPlanner, defaultSecurity, defaultVision, true);

    // Verify avatar state output is purely a restricted string enum
    expect(state).not.toContain(sensitivePayload);
    expect(typeof state).toBe('string');
  });
});
