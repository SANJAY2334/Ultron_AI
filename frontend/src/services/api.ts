/* ULTRON API Service — Real Backend Integration Only (Phase 4G.3)
 *
 * All functions attempt real backend endpoints. When a subsystem endpoint
 * does not exist or the backend is unreachable, callers receive null or
 * an explicit error — never synthetic data.
 *
 * SECURITY INVARIANT: No authorization tokens are stored or rendered.
 * No chain-of-thought is exposed. No raw frames are transmitted.
 */

const BACKEND_BASE_URL = 'http://localhost:8000';
const API_PREFIX = '/api/v1';

/* ─── Types ─── */

export interface BackendHealthResult {
  online: boolean;
  kernelState: string | null;
  version: string | null;
  database: boolean;
  redis: boolean;
}

export interface SystemStatus {
  name: string;
  version: string;
  environment: string;
  debug: boolean;
  kernel_state: string;
  api_prefix: string;
}

export interface PlannerSSEEvent {
  eventType: 'start' | 'tool_status' | 'token' | 'completion' | 'error';
  data: Record<string, unknown>;
}

export interface PlannerResponse {
  response: string;
  planner_status: string;
  session_id: string;
  correlation_id: string;
  planner_id: string;
  tool_outputs: Array<{
    tool_name: string;
    status: string;
    execution_time_ms: number;
  }>;
}

/* ─── Backend Health ─── */

export async function checkBackendHealth(): Promise<BackendHealthResult> {
  try {
    const [healthRes, statusRes] = await Promise.all([
      fetch(`${BACKEND_BASE_URL}${API_PREFIX}/system/health`, {
        signal: AbortSignal.timeout(3000),
      }),
      fetch(`${BACKEND_BASE_URL}${API_PREFIX}/system/status`, {
        signal: AbortSignal.timeout(3000),
      }),
    ]);

    let kernelState: string | null = null;
    let database = false;
    let redis = false;
    let version: string | null = null;

    if (healthRes.ok) {
      const health = await healthRes.json();
      kernelState = health.kernel_state ?? null;
      database = health.components?.database === 'connected';
      redis = health.components?.redis === 'connected';
    }

    if (statusRes.ok) {
      const status: SystemStatus = await statusRes.json();
      version = status.version ?? null;
      if (!kernelState) kernelState = status.kernel_state ?? null;
    }

    return {
      online: healthRes.ok,
      kernelState,
      version,
      database,
      redis,
    };
  } catch {
    return { online: false, kernelState: null, version: null, database: false, redis: false };
  }
}

/* ─── Subsystem Status Probes ─── */

export type SubsystemProbeResult = 'ONLINE' | 'OFFLINE' | 'UNAVAILABLE';

export async function checkVisionSubsystem(): Promise<SubsystemProbeResult> {
  try {
    const res = await fetch(`${BACKEND_BASE_URL}${API_PREFIX}/vision/status`, {
      signal: AbortSignal.timeout(2000),
    });
    return res.ok ? 'ONLINE' : 'UNAVAILABLE';
  } catch {
    return 'UNAVAILABLE';
  }
}

export async function checkAudioSubsystem(): Promise<SubsystemProbeResult> {
  try {
    const res = await fetch(`${BACKEND_BASE_URL}${API_PREFIX}/audio/status`, {
      signal: AbortSignal.timeout(2000),
    });
    return res.ok ? 'ONLINE' : 'UNAVAILABLE';
  } catch {
    return 'UNAVAILABLE';
  }
}

export async function checkPlannerSubsystem(backendOnline: boolean): Promise<SubsystemProbeResult> {
  // Planner availability is derived from backend health — the /ai/chat endpoint
  // exists whenever the backend is running.
  return backendOnline ? 'ONLINE' : 'OFFLINE';
}

/* ─── Vision Data ─── */

export async function fetchVisionObservation(): Promise<Record<string, unknown> | null> {
  try {
    const res = await fetch(`${BACKEND_BASE_URL}${API_PREFIX}/vision/observation`, {
      signal: AbortSignal.timeout(2000),
    });
    if (res.ok) return await res.json();
  } catch { /* endpoint may not exist */ }
  return null;
}

export async function fetchVisionContext(): Promise<Record<string, unknown> | null> {
  try {
    const res = await fetch(`${BACKEND_BASE_URL}${API_PREFIX}/vision/context`, {
      signal: AbortSignal.timeout(2000),
    });
    if (res.ok) return await res.json();
  } catch { /* endpoint may not exist */ }
  return null;
}

/* ─── Audio Data ─── */

export async function fetchAudioStatus(): Promise<Record<string, unknown> | null> {
  try {
    const res = await fetch(`${BACKEND_BASE_URL}${API_PREFIX}/audio/status`, {
      signal: AbortSignal.timeout(2000),
    });
    if (res.ok) return await res.json();
  } catch { /* endpoint may not exist */ }
  return null;
}

/* ─── System Metrics ─── */

export async function fetchSystemMetrics(): Promise<Record<string, unknown> | null> {
  try {
    const res = await fetch(`${BACKEND_BASE_URL}${API_PREFIX}/system/info`, {
      signal: AbortSignal.timeout(2000),
    });
    if (res.ok) return await res.json();
  } catch { /* endpoint may not exist */ }
  return null;
}

/* ─── Planner SSE Streaming ─── */

export function connectPlannerStream(
  message: string,
  onEvent: (event: PlannerSSEEvent) => void,
  onError: (error: string) => void,
  onComplete: () => void,
): AbortController {
  const controller = new AbortController();

  (async () => {
    try {
      const res = await fetch(`${BACKEND_BASE_URL}${API_PREFIX}/ai/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message }),
        signal: controller.signal,
      });

      if (!res.ok || !res.body) {
        onError(`Backend returned status ${res.status}`);
        onComplete();
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let currentEventType = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';

        for (const line of lines) {
          const trimmed = line.trim();
          if (trimmed.startsWith('event: ')) {
            currentEventType = trimmed.slice(7).trim();
          } else if (trimmed.startsWith('data: ')) {
            try {
              const data = JSON.parse(trimmed.slice(6));
              onEvent({
                eventType: currentEventType as PlannerSSEEvent['eventType'],
                data,
              });
            } catch {
              // Malformed JSON line — skip
            }
          }
        }
      }

      onComplete();
    } catch (err) {
      if (!controller.signal.aborted) {
        onError(err instanceof Error ? err.message : String(err));
        onComplete();
      }
    }
  })();

  return controller;
}

/* ─── Planner Synchronous Fallback ─── */

export async function executePlannerCommand(message: string): Promise<PlannerResponse | null> {
  try {
    const res = await fetch(`${BACKEND_BASE_URL}${API_PREFIX}/ai/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message }),
      signal: AbortSignal.timeout(30000),
    });
    if (res.ok) return await res.json();
  } catch { /* network error */ }
  return null;
}
