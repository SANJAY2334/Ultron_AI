/* ULTRON Avatar Domain Types & State Machine Contracts (Phase 4G.2) */

export type AvatarState =
  | 'IDLE'
  | 'LISTENING'
  | 'SPEECH_DETECTED'
  | 'THINKING'
  | 'PROCESSING'
  | 'SPEAKING'
  | 'TOOL_EXECUTION'
  | 'WARNING'
  | 'ERROR'
  | 'CONFIRMATION_REQUIRED'
  | 'OFFLINE';

export type AvatarViewMode = 'EMBEDDED' | 'FULLSCREEN' | 'COMPACT';

export interface AvatarAudioState {
  micActive: boolean;
  vadActive: boolean;
  wakeWordTriggered: boolean;
  isSpeaking: boolean;
}

export interface AvatarPlannerState {
  currentEventType?: 'THINKING' | 'TOOL_STARTED' | 'TOOL_COMPLETED' | 'PARTIAL_RESPONSE' | 'COMPLETED' | 'ERROR';
  isExecuting: boolean;
}

export interface AvatarSecurityState {
  policyDecision?: 'ALLOW' | 'DENY' | 'REQUIRES_CONFIRMATION';
  userConfirmed: boolean;
}

export interface AvatarVisionState {
  motionDetected: boolean;
  objectDetected: boolean;
  personDetected: boolean;
}

export interface AvatarTelemetry {
  state: AvatarState;
  viewMode: AvatarViewMode;
  fps: number;
  cpuBudgetLow: boolean;
  hasRenderError: boolean;
}
