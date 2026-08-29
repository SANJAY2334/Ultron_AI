/* ULTRON Live Multimodal Interface & Desktop HUD — Root Application (Phase 4G.4: Stage 7)
 *
 * Wires the unified state reducer to all subsystem components, supports dual-view rendering
 * (Command Center Dashboard vs Native Desktop HUD Overlay), enforces bounded exponential backoff
 * reconnect resilience, and coordinates native OS notifications.
 *
 * HARD SECURITY INVARIANTS:
 * - Frontend is a read-only presentation projection of backend state.
 * - Zero tool execution, zero policy decisions, zero direct authorization.
 * - Desktop Shell != ToolExecutor; Python backend remains sole authoritative kernel.
 * - Notifications and HUD are strictly presentation-only.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import './App.css';
import { Header } from './components/Header';
import { VisionFeed } from './components/VisionFeed';
import { AudioStreamFeed } from './components/AudioStreamFeed';
import { PlannerPanel } from './components/PlannerPanel';
import { SecurityPanel } from './components/SecurityPanel';
import { MemoryPanel } from './components/MemoryPanel';
import { BootSequence } from './components/BootSequence';
import { HUDOverlay } from './components/HUDOverlay';
import { AvatarContainer } from './avatar/AvatarContainer';
import { useUltronState } from './state/useUltronState';
import {
  checkBackendHealth,
  fetchSystemMetrics,
  fetchVisionObservation,
  fetchVisionContext,
  fetchAudioStatus,
} from './services/api';
import type { ObjectDetection, ObjectTrack, SceneSummary, VisionPlannerContext } from './types';

export function App() {
  const [state, dispatch] = useUltronState();
  const [booted, setBooted] = useState(false);
  const prevConnectivityRef = useRef<string>(state.connectivity);
  const prevPendingConfirmationRef = useRef<string | null>(null);
  const reconnectAttemptRef = useRef<number>(0);

  // Detect whether this renderer window is running in HUD Overlay mode
  const isHUDView =
    typeof window !== 'undefined' &&
    (window.location.search.includes('view=hud') ||
     window.location.hash.includes('hud') ||
     window.location.search.includes('hud'));

  const handleBootComplete = useCallback(() => {
    setBooted(true);
  }, []);

  /* ─── Global Shortcut Listener (Native Desktop Mode) ─── */
  useEffect(() => {
    if (window.ultronNative?.onGlobalShortcutTriggered) {
      const unsubscribe = window.ultronNative.onGlobalShortcutTriggered(() => {
        // Native shell brought window to focus
      });
      return () => unsubscribe();
    }
  }, []);

  /* ─── Native Notifications on Critical State Transitions ─── */
  useEffect(() => {
    if (!window.ultronNative?.sendNotification) return;

    // 1. Connectivity status transition notification
    if (prevConnectivityRef.current !== state.connectivity) {
      if (
        (state.connectivity === 'OFFLINE' || state.connectivity === 'RECONNECTING') &&
        prevConnectivityRef.current === 'ONLINE'
      ) {
        window.ultronNative.sendNotification({
          title: 'ULTRON Disconnected',
          body: 'Zero-Trust backend connection lost. Attempting automatic reconnection...',
          level: 'error',
        });
      } else if (
        state.connectivity === 'ONLINE' &&
        (prevConnectivityRef.current === 'OFFLINE' ||
         prevConnectivityRef.current === 'RECONNECTING' ||
         prevConnectivityRef.current === 'CONNECTING')
      ) {
        window.ultronNative.sendNotification({
          title: 'ULTRON Connected',
          body: 'Zero-Trust backend kernel is active and responding.',
          level: 'info',
        });
      }
      prevConnectivityRef.current = state.connectivity;
    }

    // 2. Safety Interlock Confirmation Requirement Notification
    const currentConfirmationReq = state.securityState.pendingConfirmation?.requestId ?? null;
    if (currentConfirmationReq && currentConfirmationReq !== prevPendingConfirmationRef.current) {
      const actionName = state.securityState.pendingConfirmation?.action ?? 'Operation';
      window.ultronNative.sendNotification({
        title: 'CONFIRMATION REQUIRED',
        body: `SafetyInterlock requires human authorization for: ${actionName}`,
        level: 'warning',
      });
      prevPendingConfirmationRef.current = currentConfirmationReq;
    } else if (!currentConfirmationReq) {
      prevPendingConfirmationRef.current = null;
    }
  }, [state.connectivity, state.securityState.pendingConfirmation]);

  /* ─── Periodic Backend Health Poll with Bounded Reconnect Backoff ─── */
  useEffect(() => {
    if (!booted && !isHUDView) return;

    let cancelled = false;
    let timerId: ReturnType<typeof setTimeout> | null = null;

    async function poll() {
      try {
        const health = await checkBackendHealth();
        if (cancelled) return;

        if (health.online) {
          reconnectAttemptRef.current = 0; // Reset retry backoff
          dispatch({
            type: 'BACKEND_CONNECTED',
            kernelState: health.kernelState ?? 'UNKNOWN',
            version: health.version ?? 'unknown',
          });
          // Normal polling interval when healthy: 10,000ms
          timerId = setTimeout(poll, 10000);
        } else {
          handleFailure();
        }
      } catch {
        if (!cancelled) {
          handleFailure();
        }
      }
    }

    function handleFailure() {
      reconnectAttemptRef.current += 1;
      const attempt = reconnectAttemptRef.current;

      if (attempt === 1) {
        dispatch({ type: 'BACKEND_DISCONNECTED' });
      } else {
        dispatch({ type: 'BACKEND_RECONNECTING', attempt });
      }

      // Bounded exponential backoff: 2000ms * (1.5 ^ attempt), capped at 15000ms
      const backoffMs = Math.min(15000, Math.round(2000 * Math.pow(1.5, Math.min(attempt - 1, 6))));
      timerId = setTimeout(poll, backoffMs);
    }

    poll();

    return () => {
      cancelled = true;
      if (timerId) clearTimeout(timerId);
    };
  }, [booted, isHUDView, dispatch]);

  /* ─── Periodic System Metrics Poll ─── */
  useEffect(() => {
    if ((!booted && !isHUDView) || state.connectivity !== 'ONLINE') return;

    let cancelled = false;

    async function poll() {
      const data = await fetchSystemMetrics();
      if (cancelled || !data) return;
      dispatch({
        type: 'SYSTEM_METRICS_RECEIVED',
        metrics: {
          cpu_percent: (data as Record<string, number>).cpu_usage_pct ?? 0,
          memory_used_gb: (data as Record<string, number>).memory_used_gb ?? 0,
          memory_total_gb: (data as Record<string, number>).memory_total_gb ?? 0,
          active_processes: (data as Record<string, number>).process_count ?? 0,
          uptime_seconds: (data as Record<string, number>).uptime_sec ?? 0,
        },
      });
    }

    poll();
    const interval = setInterval(poll, 5000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [booted, isHUDView, state.connectivity, dispatch]);

  /* ─── Periodic Vision Poll ─── */
  useEffect(() => {
    if ((!booted && !isHUDView) || state.visionState.status !== 'ONLINE') return;

    let cancelled = false;

    async function poll() {
      const [obsData, ctxData] = await Promise.all([
        fetchVisionObservation(),
        fetchVisionContext(),
      ]);
      if (cancelled) return;

      if (obsData) {
        dispatch({
          type: 'VISION_OBSERVATION_RECEIVED',
          observation: {
            detections: (obsData as { object_detections?: ObjectDetection[] }).object_detections ?? [],
            tracks: (obsData as { tracks?: ObjectTrack[] }).tracks ?? [],
            sceneSummary: (obsData as { scene_summary?: SceneSummary }).scene_summary ?? null,
          },
        });
      }

      if (ctxData) {
        dispatch({
          type: 'VISION_CONTEXT_RECEIVED',
          context: ctxData as unknown as VisionPlannerContext,
        });
      }
    }

    poll();
    const interval = setInterval(poll, 2000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [booted, isHUDView, state.visionState.status, dispatch]);

  /* ─── Periodic Audio Status Poll ─── */
  useEffect(() => {
    if ((!booted && !isHUDView) || state.audioState.status !== 'ONLINE') return;

    let cancelled = false;

    async function poll() {
      const data = await fetchAudioStatus();
      if (cancelled || !data) return;

      const audio = data as Record<string, unknown>;
      if (typeof audio.vad_active === 'boolean') {
        dispatch({ type: 'AUDIO_VAD_CHANGED', vadActive: audio.vad_active });
      }
      if (typeof audio.wake_word_triggered === 'boolean') {
        dispatch({ type: 'AUDIO_WAKE_WORD', triggered: audio.wake_word_triggered });
      }
      if (typeof audio.is_speaking === 'boolean') {
        if (audio.is_speaking) dispatch({ type: 'AUDIO_TTS_START' });
        else dispatch({ type: 'AUDIO_TTS_END' });
      }
      if (typeof audio.transcript === 'string' && audio.transcript) {
        dispatch({
          type: 'AUDIO_STT_RESULT',
          transcript: audio.transcript as string,
          confidence: (audio.stt_confidence as number) ?? 0,
        });
      }
    }

    poll();
    const interval = setInterval(poll, 1000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [booted, isHUDView, state.audioState.status, dispatch]);

  /* ─── Render View: Native Desktop HUD Overlay ─── */
  if (isHUDView) {
    return <HUDOverlay state={state} />;
  }

  /* ─── Render View: Boot Sequence for Command Center ─── */
  if (!booted) {
    return <BootSequence dispatch={dispatch} onComplete={handleBootComplete} />;
  }

  /* ─── Render View: Full Command Center Dashboard ─── */
  return (
    <div className="min-h-screen bg-cyber-grid p-4 md:p-6 max-w-[1800px] mx-auto flex flex-col">
      {/* Header */}
      <Header
        connectivity={state.connectivity}
        systemTelemetry={state.systemTelemetry}
        avatarState={state.avatarState}
        backendVersion={state.backendVersion}
        backendKernelState={state.backendKernelState}
      />

      {/* Main Grid */}
      <main className="grid grid-cols-1 lg:grid-cols-12 gap-6 flex-1">
        {/* Robotic Avatar (4 cols) */}
        <div className="lg:col-span-4">
          <AvatarContainer state={state.avatarState} />
        </div>

        {/* Vision Feed (8 cols) */}
        <div className="lg:col-span-8">
          <VisionFeed
            visionState={state.visionState}
            activeObjects={state.activeObjects}
            activeTracks={state.activeTracks}
            sceneSummary={state.sceneSummary}
          />
        </div>

        {/* Audio Stream (5 cols) */}
        <div className="lg:col-span-5">
          <AudioStreamFeed
            audioState={state.audioState}
            transcript={state.transcript}
            dispatch={dispatch}
          />
        </div>

        {/* Planner (7 cols) */}
        <div className="lg:col-span-7">
          <PlannerPanel
            plannerState={state.plannerState}
            plannerEvents={state.plannerEvents}
            visionContext={state.visionContext}
            dispatch={dispatch}
          />
        </div>

        {/* Security (6 cols) */}
        <div className="lg:col-span-6">
          <SecurityPanel
            securityState={state.securityState}
            dispatch={dispatch}
          />
        </div>

        {/* Memory & Hardware (6 cols) */}
        <div className="lg:col-span-6">
          <MemoryPanel
            systemTelemetry={state.systemTelemetry}
            connectivity={state.connectivity}
          />
        </div>
      </main>

      {/* Footer */}
      <footer className="mt-6 text-center text-xs font-code text-slate-500 border-t border-slate-900 pt-4 flex flex-col sm:flex-row items-center justify-between gap-2">
        <div>ULTRON MULTIMODAL INTERFACE — PHASE 4G.4 NATIVE</div>
        <div className="flex items-center gap-2">
          <span
            className={`w-2 h-2 rounded-full ${
              state.connectivity === 'ONLINE' ? 'bg-emerald-400' :
              state.connectivity === 'RECONNECTING' ? 'bg-amber-500 animate-bounce' :
              'bg-red-400'
            }`}
          />
          <span>FROZEN BACKEND SECURITY BOUNDARY</span>
        </div>
      </footer>
    </div>
  );
}

export default App;
