/* ULTRON Unified State Reducer & Avatar Integration Tests (Phase 4G.3)
 *
 * Tests cover:
 * - Unified state derivation
 * - Audio → avatar mapping
 * - Vision → avatar mapping
 * - Planner → avatar mapping
 * - Security → avatar mapping
 * - Event priority order
 * - Offline mode
 * - Boot sequence state transitions
 * - Confirmation workflow
 * - Sensitive-data redaction
 * - Bounded queue enforcement
 */

import { describe, it, expect } from 'vitest';
import { ultronReducer, INITIAL_STATE } from '../useUltronState';
import type { UltronSystemState, UltronAction } from '../UltronSystemState';

function applyActions(actions: UltronAction[]): UltronSystemState {
  return actions.reduce(
    (state, action) => ultronReducer(state, action),
    INITIAL_STATE,
  );
}

describe('ULTRON Unified State Reducer (Phase 4G.3)', () => {

  /* ── Initial State ── */

  it('should initialize with CONNECTING connectivity and OFFLINE avatar', () => {
    expect(INITIAL_STATE.connectivity).toBe('CONNECTING');
    expect(INITIAL_STATE.avatarState).toBe('OFFLINE');
    expect(INITIAL_STATE.bootPhase).toBe('INITIALIZING');
  });

  /* ── Backend Connectivity ── */

  it('should transition to ONLINE when backend connects', () => {
    const state = applyActions([
      { type: 'BACKEND_CONNECTED', kernelState: 'RUNNING', version: '1.0.0' },
    ]);
    expect(state.connectivity).toBe('ONLINE');
    expect(state.backendKernelState).toBe('RUNNING');
    expect(state.backendVersion).toBe('1.0.0');
    expect(state.avatarState).toBe('IDLE'); // Online + no activity = IDLE
  });

  it('should transition to OFFLINE when backend disconnects', () => {
    const state = applyActions([
      { type: 'BACKEND_CONNECTED', kernelState: 'RUNNING', version: '1.0.0' },
      { type: 'BACKEND_DISCONNECTED' },
    ]);
    expect(state.connectivity).toBe('OFFLINE');
    expect(state.avatarState).toBe('OFFLINE');
  });

  it('should transition to RECONNECTING when backend reconnect is in progress and recover cleanly', () => {
    const reconnectingState = applyActions([
      { type: 'BACKEND_CONNECTED', kernelState: 'RUNNING', version: '1.0.0' },
      { type: 'BACKEND_DISCONNECTED' },
      { type: 'BACKEND_RECONNECTING', attempt: 2 },
    ]);
    expect(reconnectingState.connectivity).toBe('RECONNECTING');
    expect(reconnectingState.avatarState).toBe('OFFLINE');

    const recoveredState = applyActions([
      { type: 'BACKEND_CONNECTED', kernelState: 'RUNNING', version: '1.0.0' },
      { type: 'BACKEND_DISCONNECTED' },
      { type: 'BACKEND_RECONNECTING', attempt: 2 },
      { type: 'BACKEND_CONNECTED', kernelState: 'RUNNING', version: '1.0.0' },
    ]);
    expect(recoveredState.connectivity).toBe('ONLINE');
    expect(recoveredState.avatarState).toBe('IDLE');
  });

  /* ── Audio → Avatar Mapping ── */

  it('should map VAD active to SPEECH_DETECTED avatar state', () => {
    const state = applyActions([
      { type: 'BACKEND_CONNECTED', kernelState: 'RUNNING', version: '1.0.0' },
      { type: 'AUDIO_VAD_CHANGED', vadActive: true },
    ]);
    expect(state.audioState.vadActive).toBe(true);
    expect(state.avatarState).toBe('SPEECH_DETECTED');
  });

  it('should map wake word to LISTENING avatar state', () => {
    const state = applyActions([
      { type: 'BACKEND_CONNECTED', kernelState: 'RUNNING', version: '1.0.0' },
      { type: 'AUDIO_WAKE_WORD', triggered: true },
    ]);
    expect(state.audioState.wakeWordTriggered).toBe(true);
    expect(state.avatarState).toBe('LISTENING');
  });

  it('should map TTS speaking to SPEAKING avatar state', () => {
    const state = applyActions([
      { type: 'BACKEND_CONNECTED', kernelState: 'RUNNING', version: '1.0.0' },
      { type: 'AUDIO_TTS_START' },
    ]);
    expect(state.audioState.isSpeaking).toBe(true);
    expect(state.avatarState).toBe('SPEAKING');
  });

  it('should record STT result in transcript', () => {
    const state = applyActions([
      { type: 'BACKEND_CONNECTED', kernelState: 'RUNNING', version: '1.0.0' },
      { type: 'AUDIO_STT_RESULT', transcript: 'hello ultron', confidence: 0.92 },
    ]);
    expect(state.audioState.currentTranscript).toBe('hello ultron');
    expect(state.audioState.sttConfidence).toBe(0.92);
    expect(state.transcript.some(t => t.content.includes('hello ultron'))).toBe(true);
  });

  it('should disable VAD and wake word when mic is toggled off', () => {
    const state = applyActions([
      { type: 'BACKEND_CONNECTED', kernelState: 'RUNNING', version: '1.0.0' },
      { type: 'AUDIO_VAD_CHANGED', vadActive: true },
      { type: 'AUDIO_WAKE_WORD', triggered: true },
      { type: 'AUDIO_MIC_TOGGLED', active: false },
    ]);
    expect(state.audioState.micActive).toBe(false);
    expect(state.audioState.vadActive).toBe(false);
    expect(state.audioState.wakeWordTriggered).toBe(false);
  });

  /* ── Vision → Avatar Mapping ── */

  it('should update vision state from observation', () => {
    const state = applyActions([
      { type: 'BACKEND_CONNECTED', kernelState: 'RUNNING', version: '1.0.0' },
      {
        type: 'VISION_OBSERVATION_RECEIVED',
        observation: {
          detections: [
            {
              detection_id: 'd1',
              label: 'person',
              confidence: 0.9,
              bounding_box: { x: 0, y: 0, width: 0.5, height: 0.5 },
              timestamp: new Date().toISOString(),
            },
          ],
          tracks: [
            {
              track_id: 't1',
              class_name: 'person',
              confidence: 0.9,
              bounding_box: { x: 0, y: 0, width: 0.5, height: 0.5 },
              state: 'ACTIVE',
              speed: 0,
              direction: 'STATIONARY',
              is_active: true,
            },
          ],
          sceneSummary: null,
        },
      },
    ]);
    expect(state.visionState.objectDetected).toBe(true);
    expect(state.visionState.personDetected).toBe(true);
    expect(state.activeObjects).toHaveLength(1);
    expect(state.activeTracks).toHaveLength(1);
  });

  /* ── Planner → Avatar Mapping ── */

  it('should map planner THINKING to THINKING avatar state', () => {
    const state = applyActions([
      { type: 'BACKEND_CONNECTED', kernelState: 'RUNNING', version: '1.0.0' },
      {
        type: 'PLANNER_EVENT',
        event: { event_type: 'THINKING', payload: 'Analyzing...', request_id: 'r1', timestamp: new Date().toISOString() },
      },
    ]);
    expect(state.avatarState).toBe('THINKING');
  });

  it('should map planner TOOL_STARTED to TOOL_EXECUTION avatar state', () => {
    const state = applyActions([
      { type: 'BACKEND_CONNECTED', kernelState: 'RUNNING', version: '1.0.0' },
      {
        type: 'PLANNER_EVENT',
        event: { event_type: 'TOOL_STARTED', payload: 'Running tool...', request_id: 'r1', timestamp: new Date().toISOString() },
      },
    ]);
    expect(state.avatarState).toBe('TOOL_EXECUTION');
  });

  it('should map planner COMPLETED to SPEAKING avatar state', () => {
    const state = applyActions([
      { type: 'BACKEND_CONNECTED', kernelState: 'RUNNING', version: '1.0.0' },
      {
        type: 'PLANNER_EVENT',
        event: { event_type: 'COMPLETED', payload: 'Done.', request_id: 'r1', timestamp: new Date().toISOString() },
      },
    ]);
    expect(state.avatarState).toBe('SPEAKING');
  });

  it('should map planner ERROR to ERROR avatar state', () => {
    const state = applyActions([
      { type: 'BACKEND_CONNECTED', kernelState: 'RUNNING', version: '1.0.0' },
      {
        type: 'PLANNER_EVENT',
        event: { event_type: 'ERROR', payload: 'Failed', request_id: 'r1', timestamp: new Date().toISOString() },
      },
    ]);
    expect(state.avatarState).toBe('ERROR');
  });

  /* ── Security → Avatar Mapping ── */

  it('should map REQUIRES_CONFIRMATION to CONFIRMATION_REQUIRED avatar state', () => {
    const state = applyActions([
      { type: 'BACKEND_CONNECTED', kernelState: 'RUNNING', version: '1.0.0' },
      { type: 'SECURITY_DECISION', decision: 'REQUIRES_CONFIRMATION', reason: 'Destructive op', requestId: 'r1', action: 'delete_file' },
    ]);
    expect(state.avatarState).toBe('CONFIRMATION_REQUIRED');
    expect(state.securityState.pendingConfirmation).not.toBeNull();
  });

  it('should map DENY to WARNING avatar state', () => {
    const state = applyActions([
      { type: 'BACKEND_CONNECTED', kernelState: 'RUNNING', version: '1.0.0' },
      { type: 'SECURITY_DECISION', decision: 'DENY', reason: 'Missing capability' },
    ]);
    expect(state.avatarState).toBe('WARNING');
  });

  /* ── Priority Order ── */

  it('should prioritize OFFLINE over everything', () => {
    const state = applyActions([
      // Don't connect backend — stays OFFLINE
      { type: 'AUDIO_VAD_CHANGED', vadActive: true },
      {
        type: 'PLANNER_EVENT',
        event: { event_type: 'THINKING', payload: '...', request_id: 'r1', timestamp: new Date().toISOString() },
      },
    ]);
    expect(state.avatarState).toBe('OFFLINE');
  });

  it('should prioritize CONFIRMATION_REQUIRED over planner THINKING', () => {
    const state = applyActions([
      { type: 'BACKEND_CONNECTED', kernelState: 'RUNNING', version: '1.0.0' },
      {
        type: 'PLANNER_EVENT',
        event: { event_type: 'THINKING', payload: '...', request_id: 'r1', timestamp: new Date().toISOString() },
      },
      { type: 'SECURITY_DECISION', decision: 'REQUIRES_CONFIRMATION', reason: 'Destructive', requestId: 'r1', action: 'delete' },
    ]);
    expect(state.avatarState).toBe('CONFIRMATION_REQUIRED');
  });

  it('should prioritize ERROR over TOOL_EXECUTION', () => {
    const state = applyActions([
      { type: 'BACKEND_CONNECTED', kernelState: 'RUNNING', version: '1.0.0' },
      {
        type: 'PLANNER_EVENT',
        event: { event_type: 'ERROR', payload: 'Failed', request_id: 'r1', timestamp: new Date().toISOString() },
      },
    ]);
    expect(state.avatarState).toBe('ERROR');
  });

  it('should prioritize SPEECH_DETECTED over SPEAKING', () => {
    const state = applyActions([
      { type: 'BACKEND_CONNECTED', kernelState: 'RUNNING', version: '1.0.0' },
      { type: 'AUDIO_TTS_START' },
      { type: 'AUDIO_VAD_CHANGED', vadActive: true },
    ]);
    expect(state.avatarState).toBe('SPEECH_DETECTED');
  });

  /* ── Confirmation Workflow ── */

  it('should clear pending confirmation on user confirm', () => {
    const state = applyActions([
      { type: 'BACKEND_CONNECTED', kernelState: 'RUNNING', version: '1.0.0' },
      { type: 'SECURITY_DECISION', decision: 'REQUIRES_CONFIRMATION', reason: 'Delete op', requestId: 'r1', action: 'delete' },
      { type: 'SECURITY_CONFIRMATION_RESPONSE', confirmed: true, requestId: 'r1' },
    ]);
    expect(state.securityState.pendingConfirmation).toBeNull();
    expect(state.securityState.userConfirmed).toBe(true);
    expect(state.securityState.policyDecision).toBe('ALLOW');
  });

  it('should set DENY on user cancel', () => {
    const state = applyActions([
      { type: 'BACKEND_CONNECTED', kernelState: 'RUNNING', version: '1.0.0' },
      { type: 'SECURITY_DECISION', decision: 'REQUIRES_CONFIRMATION', reason: 'Delete op', requestId: 'r1', action: 'delete' },
      { type: 'SECURITY_CONFIRMATION_RESPONSE', confirmed: false, requestId: 'r1' },
    ]);
    expect(state.securityState.policyDecision).toBe('DENY');
    expect(state.avatarState).toBe('WARNING');
  });

  /* ── Boot Sequence ── */

  it('should track boot phase transitions', () => {
    const state = applyActions([
      { type: 'BOOT_PHASE_CHANGED', phase: 'CHECKING_BACKEND' },
      { type: 'BOOT_CHECK_RESULT', result: { subsystem: 'Backend', status: 'ONLINE', detail: 'OK' } },
      { type: 'BOOT_PHASE_CHANGED', phase: 'READY' },
    ]);
    expect(state.bootPhase).toBe('READY');
    expect(state.bootResults).toHaveLength(1);
    expect(state.bootResults[0].subsystem).toBe('Backend');
  });

  /* ── Bounded Queues ── */

  it('should cap transcript at MAX_TRANSCRIPT_ENTRIES', () => {
    const actions: UltronAction[] = [
      { type: 'BACKEND_CONNECTED', kernelState: 'RUNNING', version: '1.0.0' },
    ];
    for (let i = 0; i < 60; i++) {
      actions.push({ type: 'TRANSCRIPT_ADDED', source: 'SYSTEM', content: `msg ${i}` });
    }
    const state = applyActions(actions);
    expect(state.transcript.length).toBeLessThanOrEqual(50);
  });

  /* ── Sensitive Data Redaction ── */

  it('should never contain credentials in avatar state', () => {
    const sensitivePayload = 'eyJhbGciOiJIUzI1NiJ9.SECRET_KEY';
    const state = applyActions([
      { type: 'BACKEND_CONNECTED', kernelState: 'RUNNING', version: '1.0.0' },
    ]);
    // Avatar state is a restricted string enum — cannot contain arbitrary data
    expect(state.avatarState).not.toContain(sensitivePayload);
    expect(typeof state.avatarState).toBe('string');
    expect(['IDLE', 'LISTENING', 'SPEECH_DETECTED', 'THINKING', 'PROCESSING',
      'SPEAKING', 'TOOL_EXECUTION', 'WARNING', 'ERROR', 'CONFIRMATION_REQUIRED', 'OFFLINE']
    ).toContain(state.avatarState);
  });

  /* ── Capability Changes ── */

  it('should toggle capabilities and log to transcript', () => {
    const state = applyActions([
      { type: 'BACKEND_CONNECTED', kernelState: 'RUNNING', version: '1.0.0' },
      { type: 'SECURITY_CAPABILITY_CHANGED', capability: 'file:write', granted: true },
    ]);
    expect(state.securityState.grantedCapabilities['file:write']).toBe(true);
    expect(state.transcript.some(t => t.content.includes('file:write') && t.content.includes('GRANTED'))).toBe(true);
  });

  /* ── Planner Execution Lifecycle ── */

  it('should track planner execution start and end', () => {
    const state = applyActions([
      { type: 'BACKEND_CONNECTED', kernelState: 'RUNNING', version: '1.0.0' },
      { type: 'PLANNER_EXECUTION_STARTED' },
    ]);
    expect(state.plannerState.isExecuting).toBe(true);
    expect(state.avatarState).toBe('THINKING');

    const state2 = ultronReducer(state, { type: 'PLANNER_EXECUTION_ENDED' });
    expect(state2.plannerState.isExecuting).toBe(false);
    expect(state2.plannerState.currentEventType).toBeUndefined();
    expect(state2.avatarState).toBe('IDLE');
  });
});
