/* ULTRON Avatar State Derivation Engine (Phase 4G.3)
 *
 * Deterministic priority-based state mapper. Derives a single AvatarState
 * from the intersection of audio, planner, security, and vision subsystems.
 *
 * Priority order (highest → lowest):
 *   OFFLINE → CONFIRMATION_REQUIRED → ERROR → TOOL_EXECUTION →
 *   THINKING → SPEECH_DETECTED → SPEAKING → LISTENING → IDLE
 *
 * SECURITY INVARIANT: Avatar state is a visual projection only.
 * It carries no authority and grants no permissions.
 */

import type {
  AvatarAudioState,
  AvatarPlannerState,
  AvatarSecurityState,
  AvatarState,
  AvatarVisionState,
} from './types';

export function deriveAvatarState(
  audio: AvatarAudioState,
  planner: AvatarPlannerState,
  security: AvatarSecurityState,
  _vision: AvatarVisionState,
  online: boolean = true
): AvatarState {
  // Priority 1: OFFLINE — system unreachable
  if (!online) {
    return 'OFFLINE';
  }

  // Priority 2: CONFIRMATION_REQUIRED — safety interlock waiting
  if (security.policyDecision === 'REQUIRES_CONFIRMATION') {
    return 'CONFIRMATION_REQUIRED';
  }

  // Priority 3: ERROR — planner error or security denial
  if (planner.currentEventType === 'ERROR') {
    return 'ERROR';
  }
  if (security.policyDecision === 'DENY') {
    return 'WARNING';
  }

  // Priority 4: TOOL_EXECUTION — planner executing a tool
  if (planner.currentEventType === 'TOOL_STARTED') {
    return 'TOOL_EXECUTION';
  }
  if (planner.currentEventType === 'TOOL_COMPLETED') {
    return 'PROCESSING';
  }

  // Priority 5: THINKING — planner reasoning
  if (planner.currentEventType === 'THINKING') {
    return 'THINKING';
  }

  // Priority 6: SPEECH_DETECTED — VAD active
  if (audio.vadActive) {
    return 'SPEECH_DETECTED';
  }

  // Priority 7: SPEAKING — TTS output or planner completed
  if (audio.isSpeaking || planner.currentEventType === 'COMPLETED') {
    return 'SPEAKING';
  }

  // Priority 8: LISTENING — wake word triggered
  if (audio.wakeWordTriggered) {
    return 'LISTENING';
  }

  // Priority 9: IDLE
  return 'IDLE';
}

export function getAvatarColor(state: AvatarState): {
  primary: string;
  glow: string;
  badge: string;
} {
  switch (state) {
    case 'LISTENING':
      return { primary: '#00ff66', glow: 'rgba(0, 255, 102, 0.6)', badge: 'text-emerald-400 border-emerald-500/40' };
    case 'SPEECH_DETECTED':
      return { primary: '#00ff66', glow: 'rgba(0, 255, 102, 0.8)', badge: 'text-emerald-300 border-emerald-400/50' };
    case 'THINKING':
    case 'PROCESSING':
    case 'TOOL_EXECUTION':
      return { primary: '#9d00ff', glow: 'rgba(157, 0, 255, 0.6)', badge: 'text-purple-400 border-purple-500/40' };
    case 'SPEAKING':
      return { primary: '#00f0ff', glow: 'rgba(0, 240, 255, 0.8)', badge: 'text-cyan-300 border-cyan-500/40' };
    case 'WARNING':
    case 'CONFIRMATION_REQUIRED':
      return { primary: '#ffaa00', glow: 'rgba(255, 170, 0, 0.8)', badge: 'text-amber-300 border-amber-500/40' };
    case 'ERROR':
      return { primary: '#ff0055', glow: 'rgba(255, 0, 85, 0.8)', badge: 'text-red-400 border-red-500/40' };
    case 'OFFLINE':
      return { primary: '#475569', glow: 'rgba(71, 85, 105, 0.3)', badge: 'text-slate-400 border-slate-700' };
    case 'IDLE':
    default:
      return { primary: '#00f0ff', glow: 'rgba(0, 240, 255, 0.4)', badge: 'text-cyan-400 border-cyan-500/30' };
  }
}
