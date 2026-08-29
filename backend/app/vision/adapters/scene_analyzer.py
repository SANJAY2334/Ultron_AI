"""Vision Scene Analyzer Adapter Implementation (Phase 4F.6).

Concrete implementation of ISceneAnalyzer executing deterministic rule-based scene understanding,
object statistics calculation, motion level classification, temporal scene comparison,
dominant activity derivation, and telemetry metrics (ZERO RAW IMAGE DEPENDENCIES).
"""

import asyncio
import time
from collections import Counter, deque
from datetime import UTC, datetime
from typing import Any

from app.vision.base import ISceneAnalyzer
from app.vision.exceptions import (
    SceneAnalysisProcessingError,
    SceneAnalysisTimeoutError,
    SceneAnalysisValidationError,
)
from app.vision.models import VisionObservation
from app.vision.scene import (
    DominantActivity,
    MotionLevel,
    SceneAnalysisConfig,
    SceneAnalysisTelemetry,
    SceneChange,
    SceneChangeType,
    SceneState,
    SceneSummary,
    create_scene_analysis_config,
)


class SceneAnalyzerAdapter(ISceneAnalyzer):
    """Deterministic Visual Scene Understanding Engine implementing ISceneAnalyzer."""

    def __init__(
        self,
        config: SceneAnalysisConfig | None = None,
        simulated_delay_sec: float = 0.0,
    ) -> None:
        """Initializes SceneAnalyzerAdapter.

        Args:
            config: Optional SceneAnalysisConfig configuration instance.
            simulated_delay_sec: Optional processing delay in seconds for timeout testing.
        """
        self.config = config or create_scene_analysis_config()
        self._simulated_delay_sec = simulated_delay_sec

        self._previous_observations: deque[VisionObservation] = deque(
            maxlen=max(1, min(self.config.max_previous_observations, 10))
        )
        self._scene_counter = 0
        self._last_telemetry: SceneAnalysisTelemetry | None = None

    def reset(self) -> None:
        """Resets all bounded observation history state."""
        self._previous_observations.clear()
        self._scene_counter = 0
        self._last_telemetry = None

    async def analyze(self, observation: VisionObservation) -> SceneSummary:
        """Analyzes structured VisionObservation and produces SceneSummary with timeout protection."""
        if observation is None or not hasattr(observation, "object_detections"):
            raise SceneAnalysisValidationError(
                "Invalid VisionObservation supplied for scene analysis."
            )

        start_time = time.perf_counter()
        timeout_sec = self.config.timeout_ms / 1000.0

        try:
            return await asyncio.wait_for(
                self._analyze_internal(observation),
                timeout=timeout_sec,
            )
        except TimeoutError as exc:
            self._record_telemetry(
                "err_scene",
                0,
                "UNKNOWN",
                "NONE",
                0.0,
                0.0,
                0.0,
                (time.perf_counter() - start_time) * 1000.0,
                False,
                "TIMEOUT",
            )
            raise SceneAnalysisTimeoutError(
                f"Scene analysis operation timed out after {self.config.timeout_ms}ms."
            ) from exc
        except Exception as exc:
            self._record_telemetry(
                "err_scene",
                0,
                "UNKNOWN",
                "NONE",
                0.0,
                0.0,
                0.0,
                (time.perf_counter() - start_time) * 1000.0,
                False,
                "SCENE_ERROR",
            )
            if isinstance(exc, (SceneAnalysisTimeoutError, SceneAnalysisValidationError)):
                raise
            raise SceneAnalysisProcessingError(f"Scene analysis processing failure: {exc}") from exc

    async def _analyze_internal(self, observation: VisionObservation) -> SceneSummary:
        """Internal deterministic scene understanding algorithm."""
        if self._simulated_delay_sec > 0:
            await asyncio.sleep(self._simulated_delay_sec)

        t0 = time.perf_counter()
        now = observation.timestamp or datetime.now(UTC)

        # Step 1: Object Statistics
        detections = observation.object_detections or []
        object_count = len(detections)

        class_counts_map: Counter[str] = Counter([d.label for d in detections])
        class_counts = dict(class_counts_map)
        object_classes = sorted(class_counts.keys())

        tracks = observation.tracks or []
        active_tracks = [t for t in tracks if getattr(t, "is_active", True)]
        active_track_count = len(active_tracks)

        moving_tracks = [
            t
            for t in active_tracks
            if getattr(t, "speed", 0.0) >= 5.0
            or str(getattr(t, "direction", "STATIONARY")).upper() != "STATIONARY"
        ]
        moving_track_count = len(moving_tracks)

        t_stats = (time.perf_counter() - t0) * 1000.0

        # Step 2: Temporal Scene Comparison
        t1 = time.perf_counter()
        entered_objects: list[str] = []
        exited_objects: list[str] = []
        scene_changes: list[SceneChange] = []

        prev_obs = self._previous_observations[-1] if self._previous_observations else None

        if prev_obs is not None:
            prev_tracks = prev_obs.tracks or []
            prev_active_ids = {t.track_id for t in prev_tracks if getattr(t, "is_active", True)}
            curr_active_ids = {t.track_id for t in active_tracks if hasattr(t, "track_id")}

            # Entered & exited tracks
            entered_ids = sorted(curr_active_ids - prev_active_ids)
            exited_ids = sorted(prev_active_ids - curr_active_ids)
            entered_objects = entered_ids
            exited_objects = exited_ids

            for trk_id in entered_ids:
                cls_lbl = next(
                    (t.class_name for t in active_tracks if t.track_id == trk_id),
                    "unknown",
                )
                scene_changes.append(
                    SceneChange(
                        event_id=f"sc_ent_{trk_id}_{now.timestamp()}",
                        event_type=SceneChangeType.OBJECT_ENTERED,
                        timestamp=now,
                        track_id=trk_id,
                        class_name=cls_lbl,
                    )
                )

            for trk_id in exited_ids:
                cls_lbl = next(
                    (t.class_name for t in prev_tracks if t.track_id == trk_id),
                    "unknown",
                )
                scene_changes.append(
                    SceneChange(
                        event_id=f"sc_ext_{trk_id}_{now.timestamp()}",
                        event_type=SceneChangeType.OBJECT_EXITED,
                        timestamp=now,
                        track_id=trk_id,
                        class_name=cls_lbl,
                    )
                )

            prev_obj_cnt = len(prev_obs.object_detections or [])
            if object_count != prev_obj_cnt:
                scene_changes.append(
                    SceneChange(
                        event_id=f"sc_cnt_{now.timestamp()}",
                        event_type=SceneChangeType.OBJECT_COUNT_CHANGED,
                        timestamp=now,
                        previous_state=str(prev_obj_cnt),
                        current_state=str(object_count),
                    )
                )

            prev_classes = sorted({d.label for d in (prev_obs.object_detections or [])})
            if object_classes != prev_classes:
                scene_changes.append(
                    SceneChange(
                        event_id=f"sc_cls_{now.timestamp()}",
                        event_type=SceneChangeType.OBJECT_CLASS_CHANGED,
                        timestamp=now,
                        previous_state=",".join(prev_classes),
                        current_state=",".join(object_classes),
                    )
                )

            prev_moving_cnt = len(
                [
                    t
                    for t in (prev_obs.tracks or [])
                    if getattr(t, "is_active", True)
                    and (
                        getattr(t, "speed", 0.0) >= 5.0
                        or str(getattr(t, "direction", "STATIONARY")).upper() != "STATIONARY"
                    )
                ]
            )

            if moving_track_count > 0 and prev_moving_cnt == 0:
                scene_changes.append(
                    SceneChange(
                        event_id=f"sc_mov_start_{now.timestamp()}",
                        event_type=SceneChangeType.MOTION_STARTED,
                        timestamp=now,
                    )
                )
            elif moving_track_count == 0 and prev_moving_cnt > 0:
                scene_changes.append(
                    SceneChange(
                        event_id=f"sc_mov_stop_{now.timestamp()}",
                        event_type=SceneChangeType.MOTION_STOPPED,
                        timestamp=now,
                    )
                )

        t_temp = (time.perf_counter() - t1) * 1000.0

        # Step 3: Motion Level Classification
        t2 = time.perf_counter()
        total_denom = max(1, active_track_count)
        motion_ratio = moving_track_count / float(total_denom)

        if moving_track_count == 0:
            motion_level = MotionLevel.NONE
        elif motion_ratio < self.config.low_motion_threshold:
            motion_level = MotionLevel.LOW
        elif motion_ratio < self.config.high_motion_threshold:
            motion_level = MotionLevel.MEDIUM
        else:
            motion_level = MotionLevel.HIGH

        # Step 4: Scene State Classification
        if object_count == 0:
            scene_state = SceneState.EMPTY
        elif object_count >= self.config.crowded_threshold:
            scene_state = SceneState.CROWDED
        elif moving_track_count >= 1 or motion_level in (MotionLevel.MEDIUM, MotionLevel.HIGH):
            scene_state = SceneState.ACTIVE
        elif moving_track_count == 0 or motion_level == MotionLevel.NONE:
            scene_state = SceneState.STABLE
        elif len(scene_changes) >= 3:
            scene_state = SceneState.CHANGING
        else:
            scene_state = SceneState.UNKNOWN

        # Step 5: Dominant Objects & Activity
        sorted_classes = sorted(
            class_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
        dominant_objects = [cls for cls, _cnt in sorted_classes[:2]]

        if object_count == 0:
            dominant_activity = DominantActivity.NONE
        elif scene_state == SceneState.CHANGING:
            dominant_activity = DominantActivity.SCENE_CHANGE
        elif moving_track_count == 0:
            dominant_activity = DominantActivity.STATIONARY
        elif moving_track_count == 1:
            dominant_activity = DominantActivity.MOVEMENT
        else:
            dominant_activity = DominantActivity.MULTIPLE_MOVEMENT

        t_class = (time.perf_counter() - t2) * 1000.0

        # Bounded history storage
        self._previous_observations.append(observation)
        self._scene_counter += 1
        scene_id = f"scene_{self._scene_counter:04d}"

        total_latency = (time.perf_counter() - t0) * 1000.0
        self._record_telemetry(
            scene_id=scene_id,
            object_count=object_count,
            scene_state=scene_state.value,
            motion_level=motion_level.value,
            t_stats=t_stats,
            t_temp=t_temp,
            t_class=t_class,
            total_latency=total_latency,
            success=True,
            error_code=None,
        )

        return SceneSummary(
            scene_id=scene_id,
            timestamp=now,
            object_count=object_count,
            object_classes=object_classes,
            class_counts=class_counts,
            active_track_count=active_track_count,
            moving_track_count=moving_track_count,
            entered_objects=entered_objects,
            exited_objects=exited_objects,
            scene_state=scene_state,
            motion_level=motion_level,
            dominant_objects=dominant_objects,
            dominant_activity=dominant_activity,
            scene_changes=scene_changes,
        )

    def _record_telemetry(
        self,
        scene_id: str,
        object_count: int,
        scene_state: str,
        motion_level: str,
        t_stats: float,
        t_temp: float,
        t_class: float,
        total_latency: float,
        success: bool,
        error_code: str | None,
    ) -> None:
        """Records telemetry metrics without storing raw pixels or video paths."""
        self._last_telemetry = SceneAnalysisTelemetry(
            scene_id=scene_id,
            object_count=object_count,
            scene_state=scene_state,
            motion_level=motion_level,
            statistics_latency_ms=round(t_stats, 2),
            temporal_comparison_latency_ms=round(t_temp, 2),
            classification_latency_ms=round(t_class, 2),
            total_latency_ms=round(total_latency, 2),
            success=success,
            error_code=error_code,
        )

    async def health(self) -> dict[str, Any]:
        """Probes operational health and metrics of scene analyzer adapter."""
        return {
            "subsystem": "scene_analyzer",
            "status": "READY",
            "scenes_analyzed": self._scene_counter,
            "bounded_history_depth": len(self._previous_observations),
            "last_scene_state": self._last_telemetry.scene_state
            if self._last_telemetry
            else "UNKNOWN",
            "last_latency_ms": self._last_telemetry.total_latency_ms
            if self._last_telemetry
            else 0.0,
        }
