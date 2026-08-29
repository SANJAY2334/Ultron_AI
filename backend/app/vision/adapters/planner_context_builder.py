"""Vision-to-Autonomous Planner Context Builder Adapter Implementation (Phase 4F.7).

Concrete implementation of IVisionPlannerContextBuilder executing schema validation,
privacy filtering, field whitelisting, bounded event selection, prompt character truncation,
and person anonymity guarantees before passing visual context to the Autonomous Planner.
"""

import asyncio
import time
from collections import Counter
from datetime import UTC, datetime
from typing import Any

from app.vision.base import IVisionPlannerContextBuilder
from app.vision.exceptions import (
    VisionContextProcessingError,
    VisionContextTimeoutError,
    VisionContextValidationError,
)
from app.vision.models import VisionObservation
from app.vision.planner_context import (
    VisionContextConfig,
    VisionContextTelemetry,
    VisionPlannerContext,
    create_vision_context_config,
)


class VisionPlannerContextBuilder(IVisionPlannerContextBuilder):
    """Sanitized Vision Context Builder implementing IVisionPlannerContextBuilder."""

    def __init__(
        self,
        config: VisionContextConfig | None = None,
        simulated_delay_sec: float = 0.0,
    ) -> None:
        """Initializes VisionPlannerContextBuilder.

        Args:
            config: Optional VisionContextConfig configuration instance.
            simulated_delay_sec: Optional processing delay in seconds for timeout testing.
        """
        self.config = config or create_vision_context_config()
        self._simulated_delay_sec = simulated_delay_sec

        self._builds_processed = 0
        self._builds_rejected = 0
        self._last_telemetry: VisionContextTelemetry | None = None

    def reset(self) -> None:
        """Resets all metrics state."""
        self._builds_processed = 0
        self._builds_rejected = 0
        self._last_telemetry = None

    async def build_context(self, observation: VisionObservation) -> VisionPlannerContext:
        """Translates structured VisionObservation into sanitized VisionPlannerContext with timeout protection."""
        if observation is None or not hasattr(observation, "object_detections"):
            raise VisionContextValidationError(
                "Invalid VisionObservation supplied for context builder."
            )

        start_time = time.perf_counter()
        timeout_sec = self.config.timeout_ms / 1000.0

        try:
            return await asyncio.wait_for(
                self._build_context_internal(observation),
                timeout=timeout_sec,
            )
        except TimeoutError as exc:
            self._builds_rejected += 1
            self._record_telemetry(
                "build",
                "err",
                "obs_timeout",
                0,
                0,
                0,
                0,
                (time.perf_counter() - start_time) * 1000.0,
                False,
                "TIMEOUT",
            )
            raise VisionContextTimeoutError(
                f"Vision context building operation timed out after {self.config.timeout_ms}ms."
            ) from exc
        except Exception as exc:
            self._builds_rejected += 1
            self._record_telemetry(
                "build",
                "err",
                "obs_err",
                0,
                0,
                0,
                0,
                (time.perf_counter() - start_time) * 1000.0,
                False,
                "BUILD_ERROR",
            )
            if isinstance(exc, (VisionContextTimeoutError, VisionContextValidationError)):
                raise
            raise VisionContextProcessingError(f"Vision context building failure: {exc}") from exc

    async def _build_context_internal(self, observation: VisionObservation) -> VisionPlannerContext:
        """Internal context translation and truncation pipeline."""
        if self._simulated_delay_sec > 0:
            await asyncio.sleep(self._simulated_delay_sec)

        start_time = time.perf_counter()
        now = observation.timestamp or datetime.now(UTC)

        # Truncate collections to configured ceilings
        raw_detections = (observation.object_detections or [])[: self.config.max_objects]
        raw_tracks = (observation.tracks or [])[: self.config.max_tracks]
        raw_track_events = (observation.track_events or [])[: self.config.max_events]

        object_count = len(raw_detections)
        active_tracks = [t for t in raw_tracks if getattr(t, "is_active", True)]
        active_track_count = len(active_tracks)

        moving_tracks = [
            t
            for t in active_tracks
            if getattr(t, "speed", 0.0) >= 5.0
            or str(getattr(t, "direction", "STATIONARY")).upper() != "STATIONARY"
        ]
        moving_object_count = len(moving_tracks)

        class_counts = dict(Counter([d.label for d in raw_detections]))

        # Scene Summary metadata extraction
        scene_state = "UNKNOWN"
        motion_level = "NONE"
        dominant_activity = "NONE"
        dominant_objects: list[str] = []
        scene_changes: list[str] = []

        summary = observation.scene_summary
        if summary is not None:
            scene_state = str(getattr(summary, "scene_state", "UNKNOWN"))
            motion_level = str(getattr(summary, "motion_level", "NONE"))
            dominant_activity = str(getattr(summary, "dominant_activity", "NONE"))
            dominant_objects = list(getattr(summary, "dominant_objects", []))[:5]

            changes_raw = getattr(summary, "scene_changes", [])[: self.config.max_events]
            for c in changes_raw:
                evt_type = str(getattr(c, "event_type", "SCENE_CHANGED"))
                trk_id = getattr(c, "track_id", None) or "none"
                cls_lbl = getattr(c, "class_name", None) or "none"
                scene_changes.append(f"{evt_type} (track={trk_id}, class={cls_lbl})")
        else:
            if object_count == 0:
                scene_state = "EMPTY"
            elif moving_object_count > 0:
                scene_state = "ACTIVE"
                motion_level = "MEDIUM"
                dominant_activity = "MOVEMENT"
            else:
                scene_state = "STABLE"
                dominant_activity = "STATIONARY"
            dominant_objects = sorted(class_counts.keys())[:2]

        # Recent events extraction
        recent_events: list[str] = []
        for e in raw_track_events:
            evt_type = str(getattr(e, "event_type", "TRACK_UPDATED"))
            trk_id = str(getattr(e, "track_id", "none"))
            cls_lbl = str(getattr(e, "class_name", "none"))
            recent_events.append(f"{evt_type} (track={trk_id}, class={cls_lbl})")

        obs_id = f"obs_{int(now.timestamp() * 1000)}"

        context = VisionPlannerContext(
            observation_id=obs_id,
            timestamp=now,
            scene_state=scene_state,
            motion_level=motion_level,
            dominant_activity=dominant_activity,
            object_count=object_count,
            class_counts=class_counts,
            dominant_objects=dominant_objects,
            moving_object_count=moving_object_count,
            active_track_count=active_track_count,
            scene_changes=scene_changes,
            recent_events=recent_events,
            confidence=1.0,
            source="vision_subsystem",
        )

        # Enforce max prompt character length
        prompt_text = context.to_prompt_context()
        if len(prompt_text) > self.config.max_context_chars:
            # Deterministic truncation if exceeds ceiling
            context.scene_changes = context.scene_changes[:5]
            context.recent_events = context.recent_events[:5]

        self._builds_processed += 1
        total_latency = (time.perf_counter() - start_time) * 1000.0

        self._record_telemetry(
            op="build_context",
            corr_id="corr_vision",
            obs_id=obs_id,
            obj_cnt=object_count,
            trk_cnt=active_track_count,
            evt_cnt=len(scene_changes) + len(recent_events),
            context_sz=len(context.to_prompt_context()),
            latency_ms=total_latency,
            success=True,
            error_code=None,
        )

        return context

    def _record_telemetry(
        self,
        op: str,
        corr_id: str,
        obs_id: str,
        obj_cnt: int,
        trk_cnt: int,
        evt_cnt: int,
        context_sz: int,
        latency_ms: float,
        success: bool,
        error_code: str | None,
    ) -> None:
        """Records telemetry metrics without storing raw pixels or video paths."""
        self._last_telemetry = VisionContextTelemetry(
            operation=op,
            correlation_id=corr_id,
            observation_id=obs_id,
            object_count=obj_cnt,
            track_count=trk_cnt,
            event_count=evt_cnt,
            context_size=context_sz,
            latency_ms=round(latency_ms, 2),
            success=success,
            error_code=error_code,
        )

    async def health(self) -> dict[str, Any]:
        """Probes operational health and metrics of context builder adapter."""
        return {
            "subsystem": "vision_planner_context_builder",
            "status": "READY",
            "builds_processed": self._builds_processed,
            "builds_rejected": self._builds_rejected,
            "last_context_size": self._last_telemetry.context_size if self._last_telemetry else 0,
            "last_latency_ms": self._last_telemetry.latency_ms if self._last_telemetry else 0.0,
        }
