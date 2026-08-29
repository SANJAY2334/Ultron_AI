/* ULTRON Domain Types & TypeScript Contracts (Phase 4G.1) */

export type AppMode = 'LIVE' | 'DEMO';

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
  confidence?: number;
}

export interface ObjectDetection {
  detection_id: string;
  label: string;
  confidence: number;
  bounding_box: BoundingBox;
  timestamp: string;
}

export interface ObjectTrack {
  track_id: string;
  class_name: string;
  confidence: number;
  bounding_box: BoundingBox;
  state: 'NEW' | 'ACTIVE' | 'LOST' | 'REACQUIRED' | 'ENDED';
  speed: number;
  direction: string;
  is_active: boolean;
}

export interface SceneSummary {
  scene_id: string;
  scene_state: string;
  motion_level: string;
  dominant_activity: string;
  object_count: number;
  class_counts: Record<string, number>;
  dominant_objects: string[];
  moving_track_count: number;
}

export interface VisionObservation {
  observation_id: string;
  timestamp: string;
  object_detections: ObjectDetection[];
  tracks: ObjectTrack[];
  scene_summary?: SceneSummary;
}

export interface VisionPlannerContext {
  observation_id: string;
  timestamp: string;
  scene_state: string;
  motion_level: string;
  dominant_activity: string;
  object_count: number;
  class_counts: Record<string, number>;
  dominant_objects: string[];
  moving_object_count: number;
  active_track_count: number;
  recent_events: string[];
}

export interface AudioTelemetry {
  subsystem: string;
  sample_rate: number;
  vad_active: boolean;
  wake_word_triggered: boolean;
  buffer_ms: number;
}

export interface PlannerStreamEvent {
  event_type: 'THINKING' | 'TOOL_STARTED' | 'TOOL_COMPLETED' | 'PARTIAL_RESPONSE' | 'COMPLETED' | 'ERROR';
  payload: string;
  request_id: string;
  timestamp: string;
}

export interface PolicyDecision {
  decision: 'ALLOW' | 'DENY' | 'REQUIRES_CONFIRMATION';
  reason: string;
  missing_capabilities: string[];
}

export interface ExecutionContext {
  session_id: string;
  user_id: string;
  granted_capabilities: string[];
  environment: Record<string, any>;
}

export interface SystemResourceInfo {
  cpu_percent: number;
  memory_used_gb: number;
  memory_total_gb: number;
  active_processes: number;
  uptime_seconds: number;
}
