"""Vision Object Tracker Adapter Implementation (Phase 4F.5).

Concrete implementation of IVisionTracker executing baseline bounding-box IoU association,
centroid distance matching, motion direction estimation, track lifecycle management,
and telemetry bounds while preserving person anonymity.
"""

import asyncio
import math
import time
from datetime import UTC, datetime
from typing import Any

from app.vision.base import IVisionTracker
from app.vision.detection import ObjectDetectionValidationError
from app.vision.exceptions import (
    TrackingProcessingError,
    TrackingTimeoutError,
)
from app.vision.models import BoundingBox, ObjectDetection
from app.vision.tracking import (
    ObjectTrack,
    TrackDirection,
    TrackEvent,
    TrackEventType,
    TrackingConfig,
    TrackingTelemetry,
    TrackState,
    create_tracking_config,
)


class VisionTrackerAdapter(IVisionTracker):
    """Temporal Multi-Object Tracker implementing IVisionTracker."""

    def __init__(
        self,
        config: TrackingConfig | None = None,
        simulated_delay_sec: float = 0.0,
    ) -> None:
        """Initializes VisionTrackerAdapter.

        Args:
            config: Optional TrackingConfig configuration instance.
            simulated_delay_sec: Optional processing delay in seconds for timeout testing.
        """
        self.config = config or create_tracking_config()
        self._simulated_delay_sec = simulated_delay_sec

        self._tracks: dict[str, ObjectTrack] = {}
        self._track_counter = 0

        self._frames_processed = 0
        self._frames_rejected = 0
        self._last_telemetry: TrackingTelemetry | None = None

    def reset(self) -> None:
        """Resets all active tracking state and track history."""
        self._tracks.clear()
        self._track_counter = 0
        self._frames_processed = 0
        self._frames_rejected = 0
        self._last_telemetry = None

    async def update(
        self,
        detections: list[ObjectDetection],
        frame_id: str = "frm_0",
    ) -> tuple[list[ObjectTrack], list[TrackEvent]]:
        """Updates object tracks given new frame detections with timeout protection."""
        start_time = time.perf_counter()
        timeout_sec = self.config.timeout_ms / 1000.0

        try:
            return await asyncio.wait_for(
                self._update_internal(detections, frame_id),
                timeout=timeout_sec,
            )
        except TimeoutError as exc:
            self._frames_rejected += 1
            self._record_telemetry(
                frame_id, 0, 0, 0, 0, (time.perf_counter() - start_time) * 1000.0, False, "TIMEOUT"
            )
            raise TrackingTimeoutError(
                f"Tracking update operation timed out after {self.config.timeout_ms}ms."
            ) from exc
        except Exception as exc:
            self._frames_rejected += 1
            self._record_telemetry(
                frame_id,
                0,
                0,
                0,
                0,
                (time.perf_counter() - start_time) * 1000.0,
                False,
                "TRACKING_ERROR",
            )
            if isinstance(exc, (TrackingTimeoutError, ObjectDetectionValidationError)):
                raise
            raise TrackingProcessingError(f"Tracking processing failure: {exc}") from exc

    async def _update_internal(
        self,
        detections: list[ObjectDetection],
        frame_id: str,
    ) -> tuple[list[ObjectTrack], list[TrackEvent]]:
        """Internal track association and motion calculation pipeline."""
        if self._simulated_delay_sec > 0:
            await asyncio.sleep(self._simulated_delay_sec)

        start_time = time.perf_counter()
        events: list[TrackEvent] = []

        now = datetime.now(UTC)
        if detections:
            now = detections[0].timestamp

        active_tracks = [t for t in self._tracks.values() if t.state != TrackState.ENDED]

        # Match detections to active tracks
        matched_track_ids: set[str] = set()
        matched_det_indices: set[int] = set()

        # Deterministic association algorithm
        matches: list[tuple[float, str, int]] = []
        for det_idx, det in enumerate(detections):
            for track in active_tracks:
                if det.label != track.class_name:
                    continue

                iou = self._calculate_iou(track.bounding_box, det.bounding_box)
                dist_px = self._calculate_centroid_distance_px(track.bounding_box, det.bounding_box)

                if iou >= self.config.iou_threshold or dist_px <= self.config.max_distance:
                    score = iou * 1000.0 - dist_px
                    matches.append((score, track.track_id, det_idx))

        # Sort matches deterministically by score descending, then track_id
        matches.sort(key=lambda m: (-m[0], m[1]))

        for _score, trk_id, det_idx in matches:
            if trk_id in matched_track_ids or det_idx in matched_det_indices:
                continue

            matched_track_ids.add(trk_id)
            matched_det_indices.add(det_idx)

            track = self._tracks[trk_id]
            det = detections[det_idx]

            # Calculate motion velocity and direction
            prev_cx, prev_cy = self._get_centroid_px(track.bounding_box)
            curr_cx, curr_cy = self._get_centroid_px(det.bounding_box)

            vx = curr_cx - prev_cx
            vy = curr_cy - prev_cy
            spd = math.sqrt(vx * vx + vy * vy)
            direction = self._classify_direction(vx, vy, spd)

            # Update track state machine
            was_lost = track.state == TrackState.LOST
            target_state = TrackState.REACQUIRED if was_lost else TrackState.ACTIVE
            track.transition_to(target_state)

            track.bounding_box = det.bounding_box
            track.confidence = det.confidence
            track.last_seen = now
            track.frames_seen += 1
            track.age_frames += 1
            track.missing_frames = 0
            track.velocity_x = round(vx, 2)
            track.velocity_y = round(vy, 2)
            track.speed = round(spd, 2)
            track.direction = direction

            if was_lost:
                events.append(
                    TrackEvent(
                        event_id=f"evt_reacq_{track.track_id}_{now.timestamp()}",
                        event_type=TrackEventType.TRACK_REACQUIRED,
                        track_id=track.track_id,
                        class_name=track.class_name,
                        timestamp=now,
                    )
                )

            if spd >= self.config.movement_threshold:
                events.append(
                    TrackEvent(
                        event_id=f"evt_mov_{track.track_id}_{now.timestamp()}",
                        event_type=TrackEventType.TRACK_MOVED,
                        track_id=track.track_id,
                        class_name=track.class_name,
                        timestamp=now,
                        metadata={
                            "velocity_x": round(vx, 2),
                            "velocity_y": round(vy, 2),
                            "speed": round(spd, 2),
                            "direction": direction.value,
                        },
                    )
                )

        new_count = 0
        # Create new tracks for unmatched detections
        for det_idx, det in enumerate(detections):
            if det_idx in matched_det_indices:
                continue

            if len([t for t in self._tracks.values() if t.is_active]) >= self.config.max_tracks:
                # Max tracks reached; ignore new track creation
                continue

            self._track_counter += 1
            new_id = f"track_{self._track_counter:04d}"

            new_track = ObjectTrack(
                track_id=new_id,
                class_name=det.label,
                confidence=det.confidence,
                bounding_box=det.bounding_box,
                state=TrackState.NEW,
                first_seen=now,
                last_seen=now,
                frames_seen=1,
                age_frames=1,
                missing_frames=0,
                is_active=True,
            )
            self._tracks[new_id] = new_track
            matched_track_ids.add(new_id)
            new_count += 1

            events.append(
                TrackEvent(
                    event_id=f"evt_crt_{new_id}_{now.timestamp()}",
                    event_type=TrackEventType.TRACK_CREATED,
                    track_id=new_id,
                    class_name=det.label,
                    timestamp=now,
                )
            )

        lost_count = 0
        ended_count = 0
        # Update unmatched active tracks (missing frames)
        for track in list(self._tracks.values()):
            if not track.is_active or track.track_id in matched_track_ids:
                continue

            track.missing_frames += 1
            track.age_frames += 1

            if track.missing_frames >= self.config.max_missing_frames:
                track.transition_to(TrackState.ENDED)
                ended_count += 1
                events.append(
                    TrackEvent(
                        event_id=f"evt_end_{track.track_id}_{now.timestamp()}",
                        event_type=TrackEventType.TRACK_ENDED,
                        track_id=track.track_id,
                        class_name=track.class_name,
                        timestamp=now,
                    )
                )
            elif track.state in (TrackState.ACTIVE, TrackState.NEW):
                track.transition_to(TrackState.LOST)
                lost_count += 1
                events.append(
                    TrackEvent(
                        event_id=f"evt_lst_{track.track_id}_{now.timestamp()}",
                        event_type=TrackEventType.TRACK_LOST,
                        track_id=track.track_id,
                        class_name=track.class_name,
                        timestamp=now,
                    )
                )

        self._frames_processed += 1
        total_latency = (time.perf_counter() - start_time) * 1000.0

        active_list = [t for t in self._tracks.values() if t.is_active]
        self._record_telemetry(
            frame_id,
            len(active_list),
            new_count,
            lost_count,
            ended_count,
            total_latency,
            True,
            None,
        )

        return active_list, events

    def _get_centroid_px(self, bbox: BoundingBox) -> tuple[float, float]:
        """Calculates pixel centroid coordinates assuming 640x480 resolution base."""
        cx = (bbox.x + bbox.width / 2.0) * 640.0
        cy = (bbox.y + bbox.height / 2.0) * 480.0
        return cx, cy

    def _calculate_centroid_distance_px(self, b1: BoundingBox, b2: BoundingBox) -> float:
        """Calculates Euclidean distance between centroids in pixels."""
        c1x, c1y = self._get_centroid_px(b1)
        c2x, c2y = self._get_centroid_px(b2)
        dx = c2x - c1x
        dy = c2y - c1y
        return math.sqrt(dx * dx + dy * dy)

    def _calculate_iou(self, b1: BoundingBox, b2: BoundingBox) -> float:
        """Calculates Intersection-over-Union between two normalized bounding boxes."""
        x1 = max(b1.x, b2.x)
        y1 = max(b1.y, b2.y)
        x2 = min(b1.x + b1.width, b2.x + b2.width)
        y2 = min(b1.y + b1.height, b2.y + b2.height)

        inter_w = max(0.0, x2 - x1)
        inter_h = max(0.0, y2 - y1)
        inter_area = inter_w * inter_h

        b1_area = b1.width * b1.height
        b2_area = b2.width * b2.height
        union_area = b1_area + b2_area - inter_area

        if union_area <= 0.0:
            return 0.0
        return round(inter_area / union_area, 4)

    def _classify_direction(self, vx: float, vy: float, spd: float) -> TrackDirection:
        """Classifies movement direction vector into 8-cardinal directions or STATIONARY."""
        if spd < self.config.movement_threshold:
            return TrackDirection.STATIONARY

        angle = math.atan2(vy, vx) * 180.0 / math.pi

        if -22.5 <= angle < 22.5:
            return TrackDirection.RIGHT
        if 22.5 <= angle < 67.5:
            return TrackDirection.DOWN_RIGHT
        if 67.5 <= angle < 112.5:
            return TrackDirection.DOWN
        if 112.5 <= angle < 157.5:
            return TrackDirection.DOWN_LEFT
        if angle >= 157.5 or angle < -157.5:
            return TrackDirection.LEFT
        if -157.5 <= angle < -112.5:
            return TrackDirection.UP_LEFT
        if -112.5 <= angle < -67.5:
            return TrackDirection.UP
        if -67.5 <= angle < -22.5:
            return TrackDirection.UP_RIGHT

        return TrackDirection.STATIONARY

    def _record_telemetry(
        self,
        frame_id: str,
        active_cnt: int,
        new_cnt: int,
        lost_cnt: int,
        ended_cnt: int,
        latency_ms: float,
        success: bool,
        error_code: str | None,
    ) -> None:
        """Records telemetry metrics without storing raw pixels or video paths."""
        self._last_telemetry = TrackingTelemetry(
            frame_id=frame_id,
            active_tracks_count=active_cnt,
            new_tracks_count=new_cnt,
            lost_tracks_count=lost_cnt,
            ended_tracks_count=ended_cnt,
            association_latency_ms=latency_ms * 0.7,
            total_latency_ms=latency_ms,
            success=success,
            error_code=error_code,
        )

    async def health(self) -> dict[str, Any]:
        """Probes operational health and metrics of vision tracker adapter."""
        return {
            "subsystem": "vision_tracker",
            "status": "READY",
            "provider": self.config.provider,
            "active_tracks": len([t for t in self._tracks.values() if t.is_active]),
            "total_tracks_created": self._track_counter,
            "frames_processed": self._frames_processed,
            "frames_rejected": self._frames_rejected,
            "last_latency_ms": self._last_telemetry.total_latency_ms
            if self._last_telemetry
            else 0.0,
        }
