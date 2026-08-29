"""Vision Processing Engine and Frame Pipeline (Phase 4F.3).

Concrete implementation of IVisionProcessor establishing the visual processing boundary
between raw VisionFrames and downstream perception observations.
Provides strict frame payload validation, format limits, processing timeout protection,
event loop safety, bounded previous-frame memory buffers, motion estimation signals,
and privacy-preserving telemetry while maintaining 100% decoupling from inference models.
"""

import asyncio
import logging
import time
from collections import deque
from typing import Any

from pydantic import BaseModel, Field

from app.vision.base import IVisionProcessor
from app.vision.config import VisionConfig, create_vision_config
from app.vision.exceptions import (
    VisionProcessorProcessingError,
    VisionProcessorTimeoutError,
    VisionProcessorValidationError,
)
from app.vision.models import (
    MotionEvent,
    PixelFormat,
    SceneEvent,
    SceneEventType,
    VisionFrame,
    VisionObservation,
)

logger = logging.getLogger(__name__)


class VisionProcessorTelemetry(BaseModel):
    """Telemetry record captured during frame processing (NEVER CONTAINS RAW PIXELS)."""

    frame_id: str = Field(min_length=1, description="Processed frame identifier")
    sequence_number: int = Field(ge=0, description="Frame sequence index")
    processing_latency_ms: float = Field(ge=0.0, description="Processing latency in ms")
    width: int = Field(gt=0, description="Frame width in pixels")
    height: int = Field(gt=0, description="Frame height in pixels")
    success: bool = Field(default=True, description="True if frame processing succeeded")
    error_code: str | None = Field(
        default=None, description="Sanitized error code string if failed"
    )


class VisionProcessor(IVisionProcessor):
    """Frame Processing Engine providing visual boundary validation and observation generation."""

    def __init__(
        self,
        config: VisionConfig | None = None,
        detector: Any | None = None,
        tracker: Any | None = None,
        scene_analyzer: Any | None = None,
        simulated_delay_sec: float = 0.0,
    ) -> None:
        """Initializes VisionProcessor.

        Args:
            config: Optional VisionConfig configuration instance.
            detector: Optional IObjectDetector engine instance.
            tracker: Optional IVisionTracker engine instance.
            scene_analyzer: Optional ISceneAnalyzer engine instance.
            simulated_delay_sec: Optional processing delay in seconds for timeout testing.
        """
        self.config = config or create_vision_config()
        self.detector = detector
        self.tracker = tracker
        self.scene_analyzer = scene_analyzer
        self._simulated_delay_sec = simulated_delay_sec

        # Bounded frame buffer for motion comparison (strictly max 1-5 frames)
        max_buf = max(0, min(self.config.max_previous_frames, 5))
        self._previous_frames: deque[VisionFrame] = deque(maxlen=max_buf)

        self._frames_processed = 0
        self._frames_rejected = 0
        self._last_telemetry: VisionProcessorTelemetry | None = None
        self._last_sequence = -1

    def _calculate_expected_bytes(self, width: int, height: int, pixel_format: PixelFormat) -> int:
        """Calculates expected frame payload byte length based on format."""
        if pixel_format in (PixelFormat.RGB24, PixelFormat.BGR24):
            return width * height * 3
        if pixel_format == PixelFormat.RGBA:
            return width * height * 4
        if pixel_format == PixelFormat.GRAY8:
            return width * height
        if pixel_format in (PixelFormat.NV12, PixelFormat.YUV420P):
            return int(width * height * 1.5)
        # Compressed formats JPEG/PNG vary; default to actual payload length if positive
        return 0

    def _validate_frame(self, frame: VisionFrame) -> None:
        """Validates incoming VisionFrame headers, dimensions, payload size, and timestamps."""
        if not frame.frame_id or not frame.frame_id.strip():
            raise VisionProcessorValidationError("VisionFrame contains empty frame_id.")

        if frame.sequence_number < 0:
            raise VisionProcessorValidationError(
                f"Invalid negative sequence_number ({frame.sequence_number})."
            )

        if frame.timestamp.tzinfo is None:
            raise VisionProcessorValidationError("VisionFrame timestamp must be timezone-aware.")

        if frame.width <= 0 or frame.height <= 0:
            raise VisionProcessorValidationError(
                f"Invalid frame dimensions ({frame.width}x{frame.height})."
            )

        if frame.width > self.config.max_width:
            raise VisionProcessorValidationError(
                f"Frame width ({frame.width}px) exceeds configured max_width ({self.config.max_width}px)."
            )

        if frame.height > self.config.max_height:
            raise VisionProcessorValidationError(
                f"Frame height ({frame.height}px) exceeds configured max_height ({self.config.max_height}px)."
            )

        payload_bytes = len(frame.payload)
        if payload_bytes == 0:
            raise VisionProcessorValidationError("VisionFrame contains empty payload bytes.")

        if payload_bytes > self.config.max_payload_bytes:
            raise VisionProcessorValidationError(
                f"Frame payload size ({payload_bytes} bytes) exceeds maximum limit ({self.config.max_payload_bytes} bytes)."
            )

        expected_bytes = self._calculate_expected_bytes(
            frame.width, frame.height, frame.format.pixel_format
        )
        if expected_bytes > 0 and payload_bytes != expected_bytes:
            raise VisionProcessorValidationError(
                f"Mismatched frame payload size ({payload_bytes} bytes) for format {frame.format.pixel_format.value}; expected {expected_bytes} bytes."
            )

    async def process(self, frame: VisionFrame) -> VisionObservation:
        """Validates and processes VisionFrame into a sanitized VisionObservation with timeout protection."""
        start_time = time.perf_counter()
        timeout_sec = self.config.processing_timeout_ms / 1000.0

        try:
            return await asyncio.wait_for(self._process_internal(frame), timeout=timeout_sec)
        except TimeoutError as exc:
            self._frames_rejected += 1
            self._record_telemetry(
                frame=frame,
                latency_ms=timeout_sec * 1000.0,
                success=False,
                error_code="VISION_PROCESSING_TIMEOUT",
            )
            raise VisionProcessorTimeoutError(
                f"Vision frame processing timed out after {self.config.processing_timeout_ms}ms."
            ) from exc
        except VisionProcessorValidationError:
            self._frames_rejected += 1
            self._record_telemetry(
                frame=frame,
                latency_ms=(time.perf_counter() - start_time) * 1000.0,
                success=False,
                error_code="VISION_VALIDATION_ERROR",
            )
            raise
        except Exception as exc:
            self._frames_rejected += 1
            self._record_telemetry(
                frame=frame,
                latency_ms=(time.perf_counter() - start_time) * 1000.0,
                success=False,
                error_code="VISION_PROCESSING_ERROR",
            )
            raise VisionProcessorProcessingError(
                f"Unexpected vision processing error: {exc}"
            ) from exc

    async def _process_internal(self, frame: VisionFrame) -> VisionObservation:
        """Internal frame processing execution."""
        self._validate_frame(frame)

        if self._simulated_delay_sec > 0:
            await asyncio.sleep(self._simulated_delay_sec)

        start_time = time.perf_counter()

        motion_events: list[MotionEvent] = []
        scene_events: list[SceneEvent] = []

        # Simple motion calculation if motion is enabled and previous frame exists
        if self.config.enable_motion and self._previous_frames:
            prev_frame = self._previous_frames[-1]
            motion_score = self._compute_motion_score(prev_frame, frame)
            if motion_score > 0.05:
                motion_events.append(
                    MotionEvent(
                        event_id=f"mot_{frame.frame_id}",
                        motion_score=motion_score,
                        timestamp=frame.timestamp,
                        confidence=0.9,
                    )
                )
                scene_events.append(
                    SceneEvent(
                        event_id=f"scn_mot_{frame.frame_id}",
                        event_type=SceneEventType.MOTION_STARTED,
                        confidence=0.9,
                        timestamp=frame.timestamp,
                        metadata={"motion_score": motion_score},
                    )
                )

        # Object Detection integration
        object_detections: list[Any] = []
        if self.detector is not None:
            try:
                object_detections = await self.detector.detect(frame)
            except Exception as exc:
                logger.warning(f"Object detector error during frame processing: {exc}")

        # Visual Object Tracking integration
        tracks: list[Any] = []
        track_events: list[Any] = []
        if self.tracker is not None and object_detections:
            try:
                tracks, track_events = await self.tracker.update(object_detections, frame.frame_id)
            except Exception as exc:
                logger.warning(f"Vision tracker error during frame processing: {exc}")

        # Store frame in bounded buffer
        if self.config.max_previous_frames > 0:
            self._previous_frames.append(frame)

        self._frames_processed += 1
        self._last_sequence = frame.sequence_number
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        self._record_telemetry(
            frame=frame,
            latency_ms=latency_ms,
            success=True,
            error_code=None,
        )

        obs = VisionObservation(
            timestamp=frame.timestamp,
            scene_events=scene_events,
            object_detections=object_detections,
            face_detections=[],
            motion_events=motion_events,
            tracks=tracks,
            track_events=track_events,
        )

        if self.scene_analyzer is not None:
            try:
                obs.scene_summary = await self.scene_analyzer.analyze(obs)
            except Exception as exc:
                logger.warning(f"Scene analyzer error during frame processing: {exc}")

        return obs

    def _compute_motion_score(self, frame1: VisionFrame, frame2: VisionFrame) -> float:
        """Computes simple bounded difference ratio between consecutive frames."""
        if frame1.width != frame2.width or frame1.height != frame2.height:
            return 0.0

        p1 = frame1.payload
        p2 = frame2.payload
        if len(p1) != len(p2) or not p1:
            return 0.0

        # Subsample diff calculation to remain non-blocking
        step = max(1, len(p1) // 1000)
        diff_count = 0
        total_samples = 0

        for i in range(0, len(p1), step):
            total_samples += 1
            if abs(p1[i] - p2[i]) > 30:
                diff_count += 1

        return round(diff_count / max(1, total_samples), 4)

    def _record_telemetry(
        self,
        frame: VisionFrame,
        latency_ms: float,
        success: bool,
        error_code: str | None,
    ) -> None:
        """Records telemetry metadata without storing raw image bytes."""
        self._last_telemetry = VisionProcessorTelemetry(
            frame_id=frame.frame_id,
            sequence_number=frame.sequence_number,
            processing_latency_ms=latency_ms,
            width=frame.width,
            height=frame.height,
            success=success,
            error_code=error_code,
        )

    async def health(self) -> dict[str, Any]:
        """Probes operational health and metrics of vision processor engine."""
        return {
            "subsystem": "vision_processor",
            "status": "RUNNING",
            "frames_processed": self._frames_processed,
            "frames_rejected": self._frames_rejected,
            "last_sequence": self._last_sequence,
            "buffered_previous_frames": len(self._previous_frames),
            "last_latency_ms": self._last_telemetry.processing_latency_ms
            if self._last_telemetry
            else 0.0,
        }
