/* ULTRON Unified System State Types & Actions (Phase 4G.3)
 *
 * Central state contract representing the PRESENTATION PROJECTION of all
 * backend subsystems. The frontend never holds authoritative state.
 * All data originates from real backend endpoints or reflects real
 * hardware/provider status.
 *
 * SECURITY INVARIANT: This module defines display-only types.
 * No authorization, no tool execution, no policy decisions.
 */

import type { AvatarState } from '../avatar/types';
import type {
  ObjectDetection,
  ObjectTrack,
  PlannerStreamEvent,
  SceneSummary,
  SystemResourceInfo,
  VisionPlannerContext,
} from '../types';

/* ─── Connectivity ─── */

export type ConnectivityStatus = 'ONLINE' | 'OFFLINE' | 'CONNECTING' | 'RECONNECTING';
export type SubsystemStatus = 'ONLINE' | 'OFFLINE' | 'UNAVAILABLE' | 'ERROR';

/* ─── Audio Subsystem ─── */

export interface AudioSubsystemState {
  status: SubsystemStatus;
  micActive: boolean;
  vadActive: boolean;
  wakeWordTriggered: boolean;
  isSpeaking: boolean;
  currentTranscript: string;
  sttConfidence: number;
  ttsStatus: 'IDLE' | 'SPEAKING' | 'UNAVAILABLE';
  latencyMs: number;
}

/* ─── Vision Subsystem ─── */

export interface VisionSubsystemState {
  status: SubsystemStatus;
  cameraOnline: boolean;
  motionDetected: boolean;
  objectDetected: boolean;
  personDetected: boolean;
}

/* ─── Planner Subsystem ─── */

export interface PlannerSubsystemState {
  status: SubsystemStatus;
  currentEventType?: PlannerStreamEvent['event_type'];
  isExecuting: boolean;
}

/* ─── Security Subsystem ─── */

export interface SecuritySubsystemState {
  status: SubsystemStatus;
  policyDecision?: 'ALLOW' | 'DENY' | 'REQUIRES_CONFIRMATION';
  userConfirmed: boolean;
  pendingConfirmation: PendingConfirmation | null;
  grantedCapabilities: Record<string, boolean>;
}

export interface PendingConfirmation {
  action: string;
  reason: string;
  requestId: string;
  timestamp: string;
}

/* ─── Transcript & Event Log ─── */

export interface TranscriptEntry {
  id: string;
  timestamp: string;
  source: 'SYSTEM' | 'USER' | 'PLANNER' | 'AUDIO' | 'VISION' | 'SECURITY';
  content: string;
}

export interface PlannerEventEntry {
  id: string;
  eventType: PlannerStreamEvent['event_type'];
  payload: string;
  requestId: string;
  timestamp: string;
}

/* ─── Boot Health Check ─── */

export type BootPhase =
  | 'INITIALIZING'
  | 'CHECKING_BACKEND'
  | 'CHECKING_AUDIO'
  | 'CHECKING_VISION'
  | 'CHECKING_PLANNER'
  | 'CHECKING_POLICY'
  | 'READY'
  | 'BOOT_FAILED';

export interface BootCheckResult {
  subsystem: string;
  status: SubsystemStatus;
  detail: string;
}

/* ─── Unified System State ─── */

export interface UltronSystemState {
  connectivity: ConnectivityStatus;
  audioState: AudioSubsystemState;
  visionState: VisionSubsystemState;
  plannerState: PlannerSubsystemState;
  securityState: SecuritySubsystemState;
  avatarState: AvatarState;
  transcript: TranscriptEntry[];
  sceneSummary: SceneSummary | null;
  activeObjects: ObjectDetection[];
  activeTracks: ObjectTrack[];
  visionContext: VisionPlannerContext | null;
  plannerEvents: PlannerEventEntry[];
  systemTelemetry: SystemResourceInfo;
  bootPhase: BootPhase;
  bootResults: BootCheckResult[];
  backendVersion: string | null;
  backendKernelState: string | null;
}

/* ─── Actions ─── */

export type UltronAction =
  | { type: 'BOOT_PHASE_CHANGED'; phase: BootPhase }
  | { type: 'BOOT_CHECK_RESULT'; result: BootCheckResult }
  | { type: 'BACKEND_CONNECTED'; kernelState: string; version: string }
  | { type: 'BACKEND_DISCONNECTED' }
  | { type: 'BACKEND_RECONNECTING'; attempt: number }
  | { type: 'AUDIO_STATUS_RECEIVED'; status: SubsystemStatus }
  | { type: 'AUDIO_VAD_CHANGED'; vadActive: boolean }
  | { type: 'AUDIO_WAKE_WORD'; triggered: boolean }
  | { type: 'AUDIO_STT_RESULT'; transcript: string; confidence: number }
  | { type: 'AUDIO_TTS_START' }
  | { type: 'AUDIO_TTS_END' }
  | { type: 'AUDIO_MIC_TOGGLED'; active: boolean }
  | { type: 'VISION_STATUS_RECEIVED'; status: SubsystemStatus; cameraOnline: boolean }
  | { type: 'VISION_OBSERVATION_RECEIVED'; observation: {
      detections: ObjectDetection[];
      tracks: ObjectTrack[];
      sceneSummary: SceneSummary | null;
    } }
  | { type: 'VISION_CONTEXT_RECEIVED'; context: VisionPlannerContext }
  | { type: 'PLANNER_STATUS_RECEIVED'; status: SubsystemStatus }
  | { type: 'PLANNER_EVENT'; event: PlannerStreamEvent }
  | { type: 'PLANNER_EXECUTION_STARTED' }
  | { type: 'PLANNER_EXECUTION_ENDED' }
  | { type: 'SECURITY_DECISION'; decision: 'ALLOW' | 'DENY' | 'REQUIRES_CONFIRMATION'; reason: string; requestId?: string; action?: string }
  | { type: 'SECURITY_CONFIRMATION_RESPONSE'; confirmed: boolean; requestId: string }
  | { type: 'SECURITY_CAPABILITY_CHANGED'; capability: string; granted: boolean }
  | { type: 'SYSTEM_METRICS_RECEIVED'; metrics: SystemResourceInfo }
  | { type: 'TRANSCRIPT_ADDED'; source: TranscriptEntry['source']; content: string };

/* ─── Bounded Queue Constants ─── */

export const MAX_TRANSCRIPT_ENTRIES = 50;
export const MAX_PLANNER_EVENTS = 30;
