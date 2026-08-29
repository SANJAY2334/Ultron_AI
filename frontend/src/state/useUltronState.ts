/* ULTRON Unified State Reducer Hook (Phase 4G.3)
 *
 * Single useReducer managing all subsystem state transitions.
 * Calls deriveAvatarState() after every action to keep the
 * robotic avatar synchronized with real subsystem state.
 *
 * SECURITY INVARIANT: The reducer is a pure function with no
 * side effects. It never contacts the backend, executes tools,
 * or makes policy decisions.
 */

import { useReducer } from 'react';
import { deriveAvatarState } from '../avatar/stateMapper';
import type {
  UltronSystemState,
  UltronAction,
  TranscriptEntry,
  PlannerEventEntry,
} from './UltronSystemState';
import {
  MAX_TRANSCRIPT_ENTRIES,
  MAX_PLANNER_EVENTS,
} from './UltronSystemState';

/* ─── Initial State ─── */

export const INITIAL_STATE: UltronSystemState = {
  connectivity: 'CONNECTING',
  audioState: {
    status: 'UNAVAILABLE',
    micActive: false,
    vadActive: false,
    wakeWordTriggered: false,
    isSpeaking: false,
    currentTranscript: '',
    sttConfidence: 0,
    ttsStatus: 'UNAVAILABLE',
    latencyMs: 0,
  },
  visionState: {
    status: 'UNAVAILABLE',
    cameraOnline: false,
    motionDetected: false,
    objectDetected: false,
    personDetected: false,
  },
  plannerState: {
    status: 'OFFLINE',
    currentEventType: undefined,
    isExecuting: false,
  },
  securityState: {
    status: 'OFFLINE',
    policyDecision: undefined,
    userConfirmed: false,
    pendingConfirmation: null,
    grantedCapabilities: {
      'system:read': true,
      'file:read': true,
      'process:read': true,
      'file:write': false,
      'file:delete': false,
    },
  },
  avatarState: 'OFFLINE',
  transcript: [],
  sceneSummary: null,
  activeObjects: [],
  activeTracks: [],
  visionContext: null,
  plannerEvents: [],
  systemTelemetry: {
    cpu_percent: 0,
    memory_used_gb: 0,
    memory_total_gb: 0,
    active_processes: 0,
    uptime_seconds: 0,
  },
  bootPhase: 'INITIALIZING',
  bootResults: [],
  backendVersion: null,
  backendKernelState: null,
};

/* ─── Helpers ─── */

let transcriptCounter = 0;
let plannerEventCounter = 0;

function makeTranscriptEntry(source: TranscriptEntry['source'], content: string): TranscriptEntry {
  return {
    id: `tx_${++transcriptCounter}_${Date.now()}`,
    timestamp: new Date().toISOString(),
    source,
    content,
  };
}

function makePlannerEvent(
  eventType: PlannerEventEntry['eventType'],
  payload: string,
  requestId: string,
): PlannerEventEntry {
  return {
    id: `pe_${++plannerEventCounter}_${Date.now()}`,
    eventType,
    payload,
    requestId,
    timestamp: new Date().toISOString(),
  };
}

function boundedAppend<T>(arr: T[], item: T, max: number): T[] {
  const next = [...arr, item];
  return next.length > max ? next.slice(next.length - max) : next;
}

/* ─── Derive Avatar From Current State ─── */

function recomputeAvatar(state: UltronSystemState): UltronSystemState {
  const online = state.connectivity === 'ONLINE';
  const avatarState = deriveAvatarState(
    {
      micActive: state.audioState.micActive,
      vadActive: state.audioState.vadActive,
      wakeWordTriggered: state.audioState.wakeWordTriggered,
      isSpeaking: state.audioState.isSpeaking,
    },
    {
      currentEventType: state.plannerState.currentEventType,
      isExecuting: state.plannerState.isExecuting,
    },
    {
      policyDecision: state.securityState.policyDecision,
      userConfirmed: state.securityState.userConfirmed,
    },
    {
      motionDetected: state.visionState.motionDetected,
      objectDetected: state.visionState.objectDetected,
      personDetected: state.visionState.personDetected,
    },
    online,
  );
  return { ...state, avatarState };
}

/* ─── Reducer ─── */

function ultronReducer(state: UltronSystemState, action: UltronAction): UltronSystemState {
  let next: UltronSystemState;

  switch (action.type) {
    /* ── Boot ── */
    case 'BOOT_PHASE_CHANGED':
      next = { ...state, bootPhase: action.phase };
      break;

    case 'BOOT_CHECK_RESULT':
      next = {
        ...state,
        bootResults: [...state.bootResults, action.result],
        transcript: boundedAppend(
          state.transcript,
          makeTranscriptEntry('SYSTEM', `[BOOT] ${action.result.subsystem}: ${action.result.status} — ${action.result.detail}`),
          MAX_TRANSCRIPT_ENTRIES,
        ),
      };
      break;

    /* ── Connectivity ── */
    case 'BACKEND_CONNECTED':
      next = {
        ...state,
        connectivity: 'ONLINE',
        backendKernelState: action.kernelState,
        backendVersion: action.version,
        plannerState: { ...state.plannerState, status: 'ONLINE' },
        securityState: { ...state.securityState, status: 'ONLINE' },
        transcript: boundedAppend(
          state.transcript,
          makeTranscriptEntry('SYSTEM', `Backend connected (kernel: ${action.kernelState}, v${action.version})`),
          MAX_TRANSCRIPT_ENTRIES,
        ),
      };
      break;

    case 'BACKEND_DISCONNECTED':
      next = {
        ...state,
        connectivity: 'OFFLINE',
        plannerState: { ...state.plannerState, status: 'OFFLINE', isExecuting: false, currentEventType: undefined },
        securityState: { ...state.securityState, status: 'OFFLINE' },
        transcript: boundedAppend(
          state.transcript,
          makeTranscriptEntry('SYSTEM', 'Backend disconnected'),
          MAX_TRANSCRIPT_ENTRIES,
        ),
      };
      break;

    case 'BACKEND_RECONNECTING':
      next = {
        ...state,
        connectivity: 'RECONNECTING',
        plannerState: { ...state.plannerState, status: 'OFFLINE', isExecuting: false, currentEventType: undefined },
        securityState: { ...state.securityState, status: 'OFFLINE' },
      };
      break;

    /* ── Audio ── */
    case 'AUDIO_STATUS_RECEIVED':
      next = { ...state, audioState: { ...state.audioState, status: action.status } };
      break;

    case 'AUDIO_VAD_CHANGED':
      next = { ...state, audioState: { ...state.audioState, vadActive: action.vadActive } };
      break;

    case 'AUDIO_WAKE_WORD':
      next = {
        ...state,
        audioState: { ...state.audioState, wakeWordTriggered: action.triggered },
        transcript: action.triggered
          ? boundedAppend(state.transcript, makeTranscriptEntry('AUDIO', 'Wake word detected'), MAX_TRANSCRIPT_ENTRIES)
          : state.transcript,
      };
      break;

    case 'AUDIO_STT_RESULT':
      next = {
        ...state,
        audioState: {
          ...state.audioState,
          currentTranscript: action.transcript,
          sttConfidence: action.confidence,
        },
        transcript: boundedAppend(
          state.transcript,
          makeTranscriptEntry('AUDIO', `STT: "${action.transcript}" (${(action.confidence * 100).toFixed(0)}%)`),
          MAX_TRANSCRIPT_ENTRIES,
        ),
      };
      break;

    case 'AUDIO_TTS_START':
      next = { ...state, audioState: { ...state.audioState, isSpeaking: true, ttsStatus: 'SPEAKING' } };
      break;

    case 'AUDIO_TTS_END':
      next = { ...state, audioState: { ...state.audioState, isSpeaking: false, ttsStatus: 'IDLE' } };
      break;

    case 'AUDIO_MIC_TOGGLED':
      next = {
        ...state,
        audioState: {
          ...state.audioState,
          micActive: action.active,
          vadActive: action.active ? state.audioState.vadActive : false,
          wakeWordTriggered: action.active ? state.audioState.wakeWordTriggered : false,
        },
      };
      break;

    /* ── Vision ── */
    case 'VISION_STATUS_RECEIVED':
      next = {
        ...state,
        visionState: {
          ...state.visionState,
          status: action.status,
          cameraOnline: action.cameraOnline,
        },
      };
      break;

    case 'VISION_OBSERVATION_RECEIVED': {
      const obs = action.observation;
      const hasPersons = obs.detections.some((d) => d.label === 'person');
      const hasMotion = obs.tracks.some((t) => t.speed > 0 && t.is_active);
      next = {
        ...state,
        activeObjects: obs.detections,
        activeTracks: obs.tracks,
        sceneSummary: obs.sceneSummary,
        visionState: {
          ...state.visionState,
          objectDetected: obs.detections.length > 0,
          personDetected: hasPersons,
          motionDetected: hasMotion,
        },
      };
      break;
    }

    case 'VISION_CONTEXT_RECEIVED':
      next = { ...state, visionContext: action.context };
      break;

    /* ── Planner ── */
    case 'PLANNER_STATUS_RECEIVED':
      next = { ...state, plannerState: { ...state.plannerState, status: action.status } };
      break;

    case 'PLANNER_EXECUTION_STARTED':
      next = {
        ...state,
        plannerState: { ...state.plannerState, isExecuting: true, currentEventType: 'THINKING' },
      };
      break;

    case 'PLANNER_EXECUTION_ENDED':
      next = {
        ...state,
        plannerState: { ...state.plannerState, isExecuting: false, currentEventType: undefined },
        securityState: { ...state.securityState, policyDecision: undefined, pendingConfirmation: null },
      };
      break;

    case 'PLANNER_EVENT': {
      const evt = action.event;
      next = {
        ...state,
        plannerState: {
          ...state.plannerState,
          currentEventType: evt.event_type,
          isExecuting: evt.event_type !== 'COMPLETED' && evt.event_type !== 'ERROR',
        },
        plannerEvents: boundedAppend(
          state.plannerEvents,
          makePlannerEvent(evt.event_type, evt.payload, evt.request_id),
          MAX_PLANNER_EVENTS,
        ),
        transcript: boundedAppend(
          state.transcript,
          makeTranscriptEntry('PLANNER', `[${evt.event_type}] ${evt.payload}`),
          MAX_TRANSCRIPT_ENTRIES,
        ),
      };
      break;
    }

    /* ── Security ── */
    case 'SECURITY_DECISION': {
      const pending = action.decision === 'REQUIRES_CONFIRMATION' && action.requestId
        ? {
            action: action.action ?? 'Unknown action',
            reason: action.reason,
            requestId: action.requestId,
            timestamp: new Date().toISOString(),
          }
        : state.securityState.pendingConfirmation;

      next = {
        ...state,
        securityState: {
          ...state.securityState,
          policyDecision: action.decision,
          pendingConfirmation: pending,
        },
        transcript: boundedAppend(
          state.transcript,
          makeTranscriptEntry('SECURITY', `[${action.decision}] ${action.reason}`),
          MAX_TRANSCRIPT_ENTRIES,
        ),
      };
      break;
    }

    case 'SECURITY_CONFIRMATION_RESPONSE':
      next = {
        ...state,
        securityState: {
          ...state.securityState,
          userConfirmed: action.confirmed,
          pendingConfirmation: null,
          policyDecision: action.confirmed ? 'ALLOW' : 'DENY',
        },
        transcript: boundedAppend(
          state.transcript,
          makeTranscriptEntry('SECURITY', action.confirmed ? 'User confirmed destructive action' : 'User denied destructive action'),
          MAX_TRANSCRIPT_ENTRIES,
        ),
      };
      break;

    case 'SECURITY_CAPABILITY_CHANGED':
      next = {
        ...state,
        securityState: {
          ...state.securityState,
          grantedCapabilities: {
            ...state.securityState.grantedCapabilities,
            [action.capability]: action.granted,
          },
        },
        transcript: boundedAppend(
          state.transcript,
          makeTranscriptEntry('SECURITY', `Capability '${action.capability}' ${action.granted ? 'GRANTED' : 'REVOKED'}`),
          MAX_TRANSCRIPT_ENTRIES,
        ),
      };
      break;

    /* ── System Metrics ── */
    case 'SYSTEM_METRICS_RECEIVED':
      next = { ...state, systemTelemetry: action.metrics };
      break;

    /* ── Transcript ── */
    case 'TRANSCRIPT_ADDED':
      next = {
        ...state,
        transcript: boundedAppend(
          state.transcript,
          makeTranscriptEntry(action.source, action.content),
          MAX_TRANSCRIPT_ENTRIES,
        ),
      };
      break;

    default:
      return state;
  }

  // Recompute avatar state after every action
  return recomputeAvatar(next);
}

/* ─── Hook ─── */

export function useUltronState() {
  return useReducer(ultronReducer, recomputeAvatar(INITIAL_STATE));
}

// Export for testing
export { ultronReducer };
