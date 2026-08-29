"""Vision Intelligence Domain Models, State Machine, and Privacy Semantics (Phase 4F.1).

Defines strongly typed Pydantic models for FrameFormat, VisionFrame, BoundingBox, ObjectDetection,
FaceDetection (detection-only, zero biometric identity), MotionEvent, SceneEvent, VisionTelemetry,
VisionPrivacy, VisionSessionState state machine, and VisionObservation multimodal boundary.
"""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class PixelFormat(StrEnum):
    """Supported raw image and video frame pixel formats."""

    RGB24 = "rgb24"
    BGR24 = "bgr24"
    RGBA = "rgba"
    GRAY8 = "gray8"
    NV12 = "nv12"
    YUV420P = "yuv420p"
    JPEG = "jpeg"
    PNG = "png"


class FrameFormat(BaseModel):
    """Format specification for a vision frame."""

    width: int = Field(gt=0, description="Frame width in pixels")
    height: int = Field(gt=0, description="Frame height in pixels")
    channels: int = Field(default=3, gt=0, le=4, description="Number of color channels")
    pixel_format: PixelFormat = Field(
        default=PixelFormat.RGB24, description="Pixel color space format"
    )
    frame_rate: float = Field(
        default=30.0, gt=0.0, description="Target frame rate in frames per second"
    )


class VisionFrame(BaseModel):
    """Single camera/video frame payload container."""

    frame_id: str = Field(min_length=1, description="Unique frame tracking identifier")
    sequence_number: int = Field(ge=0, description="Monotonically increasing sequence index")
    timestamp: datetime = Field(description="Timezone-aware frame capture timestamp")
    width: int = Field(gt=0, description="Frame width in pixels")
    height: int = Field(gt=0, description="Frame height in pixels")
    format: FrameFormat = Field(description="Frame format details")
    payload: bytes = Field(min_length=1, description="Raw image/video binary frame bytes")

    @field_validator("timestamp")
    @classmethod
    def validate_timezone(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware.")
        return v

    def __repr__(self) -> str:
        """Safe representation suppressing raw image binary bytes."""
        return (
            f"VisionFrame(frame_id='{self.frame_id}', seq={self.sequence_number}, "
            f"size={self.width}x{self.height}, format={self.format.pixel_format.value}, "
            f"payload_bytes={len(self.payload)})"
        )

    def __str__(self) -> str:
        return repr(self)


class BoundingBox(BaseModel):
    """Normalized bounding box coordinates relative to frame dimensions [0.0, 1.0]."""

    x: float = Field(ge=0.0, le=1.0, description="Normalized top-left X coordinate [0.0, 1.0]")
    y: float = Field(ge=0.0, le=1.0, description="Normalized top-left Y coordinate [0.0, 1.0]")
    width: float = Field(gt=0.0, le=1.0, description="Normalized box width [0.0, 1.0]")
    height: float = Field(gt=0.0, le=1.0, description="Normalized box height [0.0, 1.0]")
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Bounding box localization confidence"
    )

    @model_validator(mode="after")
    def validate_box_boundaries(self) -> "BoundingBox":
        if self.x + self.width > 1.05:
            raise ValueError("Bounding box X + width exceeds normalized frame width 1.0.")
        if self.y + self.height > 1.05:
            raise ValueError("Bounding box Y + height exceeds normalized frame height 1.0.")
        return self


class ObjectDetection(BaseModel):
    """Non-identifying object detection record."""

    model_config = ConfigDict(extra="forbid")

    detection_id: str = Field(min_length=1, description="Unique detection tracking ID")
    label: str = Field(
        min_length=1, description="Detected object class label (e.g. 'chair', 'cup', 'person')"
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Object detection confidence score")
    bounding_box: BoundingBox = Field(description="Normalized bounding box")
    tracking_id: str | None = Field(default=None, description="Optional object tracking identifier")
    timestamp: datetime = Field(description="Timezone-aware detection timestamp")

    @field_validator("timestamp")
    @classmethod
    def validate_timezone(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware.")
        return v


class FaceLandmark(BaseModel):
    """Normalized non-identifying facial landmark landmark point."""

    landmark_type: str = Field(
        min_length=1, description="Landmark type (e.g. 'left_eye', 'nose_tip')"
    )
    x: float = Field(ge=0.0, le=1.0, description="Normalized X coordinate")
    y: float = Field(ge=0.0, le=1.0, description="Normalized Y coordinate")


class FaceDetection(BaseModel):
    """Detection-only face localization record (STRICTLY PROHIBITS BIOMETRIC IDENTIFICATION)."""

    face_id: str = Field(min_length=1, description="Unique non-identifying face region tracking ID")
    bounding_box: BoundingBox = Field(description="Normalized facial bounding box")
    confidence: float = Field(ge=0.0, le=1.0, description="Face detection confidence score")
    landmarks: list[FaceLandmark] = Field(
        default_factory=list, description="Non-identifying landmark points"
    )
    timestamp: datetime = Field(description="Timezone-aware detection timestamp")

    @field_validator("timestamp")
    @classmethod
    def validate_timezone(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware.")
        return v

    @model_validator(mode="after")
    def prohibit_biometric_fields(self) -> "FaceDetection":
        """Strictly enforces zero biometric identification or identity persistence."""
        forbidden_attrs = [
            "name",
            "identity",
            "person_id",
            "biometric_embedding",
            "face_recognition",
        ]
        for attr in forbidden_attrs:
            if hasattr(self, attr):
                raise ValueError(
                    f"Biometric identification field '{attr}' is strictly prohibited in ULTRON FaceDetection."
                )
        return self


class MotionEvent(BaseModel):
    """Motion detection event record."""

    event_id: str = Field(min_length=1, description="Unique motion event ID")
    motion_score: float = Field(
        ge=0.0, le=1.0, description="Normalized motion magnitude score [0.0, 1.0]"
    )
    region: BoundingBox | None = Field(
        default=None, description="Optional bounding box of motion activity"
    )
    timestamp: datetime = Field(description="Timezone-aware event timestamp")
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Motion detection confidence"
    )

    @field_validator("timestamp")
    @classmethod
    def validate_timezone(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware.")
        return v


class SceneEventType(StrEnum):
    """Enumeration of recognized visual scene events."""

    SCENE_CHANGED = "SCENE_CHANGED"
    OBJECT_ENTERED = "OBJECT_ENTERED"
    OBJECT_LEFT = "OBJECT_LEFT"
    MOTION_STARTED = "MOTION_STARTED"
    MOTION_STOPPED = "MOTION_STOPPED"
    PERSON_DETECTED = "PERSON_DETECTED"
    FACE_DETECTED = "FACE_DETECTED"


class SceneEvent(BaseModel):
    """Higher-level visual scene change event payload."""

    event_id: str = Field(min_length=1, description="Unique scene event ID")
    event_type: SceneEventType = Field(description="Event type discriminator")
    confidence: float = Field(ge=0.0, le=1.0, description="Event classification confidence score")
    timestamp: datetime = Field(description="Timezone-aware timestamp")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Sanitized event metadata attributes"
    )

    @field_validator("timestamp")
    @classmethod
    def validate_timezone(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware.")
        return v


class VisionTelemetry(BaseModel):
    """Privacy-preserving telemetry record for vision processing (NEVER STORES RAW PIXELS)."""

    frame_id: str = Field(min_length=1, description="Processed frame ID")
    sequence_number: int = Field(ge=0, description="Frame sequence number")
    processing_latency_ms: float = Field(ge=0.0, description="Frame processing latency in ms")
    objects_detected: int = Field(ge=0, description="Count of objects detected")
    faces_detected: int = Field(ge=0, description="Count of non-identifying face regions detected")
    motion_score: float = Field(ge=0.0, le=1.0, description="Motion intensity score")
    success: bool = Field(default=True, description="True if frame processing succeeded")
    error_code: str | None = Field(
        default=None, description="Sanitized error code string if failed"
    )


class VisionPrivacy(StrEnum):
    """Vision privacy classification semantics."""

    EPHEMERAL = "EPHEMERAL"  # Processed in-memory; raw frame discarded immediately
    SESSION = "SESSION"  # Retained in memory during active vision session
    PERSISTENT = "PERSISTENT"  # Persistent (Requires explicit user consent)


class VisionSessionState(StrEnum):
    """States of vision processing session state machine."""

    IDLE = "IDLE"
    INITIALIZING = "INITIALIZING"
    CAPTURING = "CAPTURING"
    PROCESSING = "PROCESSING"
    DETECTING = "DETECTING"
    TRACKING = "TRACKING"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


VALID_VISION_STATE_TRANSITIONS: dict[VisionSessionState, set[VisionSessionState]] = {
    VisionSessionState.IDLE: {VisionSessionState.INITIALIZING, VisionSessionState.ERROR},
    VisionSessionState.INITIALIZING: {
        VisionSessionState.CAPTURING,
        VisionSessionState.STOPPING,
        VisionSessionState.ERROR,
    },
    VisionSessionState.CAPTURING: {
        VisionSessionState.PROCESSING,
        VisionSessionState.STOPPING,
        VisionSessionState.ERROR,
    },
    VisionSessionState.PROCESSING: {
        VisionSessionState.DETECTING,
        VisionSessionState.TRACKING,
        VisionSessionState.CAPTURING,
        VisionSessionState.STOPPING,
        VisionSessionState.ERROR,
    },
    VisionSessionState.DETECTING: {
        VisionSessionState.TRACKING,
        VisionSessionState.CAPTURING,
        VisionSessionState.STOPPING,
        VisionSessionState.ERROR,
    },
    VisionSessionState.TRACKING: {
        VisionSessionState.CAPTURING,
        VisionSessionState.STOPPING,
        VisionSessionState.ERROR,
    },
    VisionSessionState.STOPPING: {VisionSessionState.STOPPED, VisionSessionState.ERROR},
    VisionSessionState.STOPPED: {VisionSessionState.IDLE, VisionSessionState.INITIALIZING},
    VisionSessionState.ERROR: {VisionSessionState.IDLE, VisionSessionState.STOPPED},
}


class InvalidVisionStateTransitionError(Exception):
    """Exception raised when an illegal vision state transition is requested."""

    def __init__(self, current: VisionSessionState, target: VisionSessionState) -> None:
        self.current = current
        self.target = target
        super().__init__(
            f"Invalid Vision session state transition: cannot transition from {current.value} to {target.value}."
        )


class VisionSessionStateMachine:
    """Deterministic state machine for managing Vision Session lifecycle."""

    def __init__(self, initial_state: VisionSessionState = VisionSessionState.IDLE) -> None:
        self._state = initial_state

    @property
    def state(self) -> VisionSessionState:
        return self._state

    def transition_to(self, target_state: VisionSessionState) -> VisionSessionState:
        """Transitions to target_state if valid under transition rules."""
        valid_targets = VALID_VISION_STATE_TRANSITIONS.get(self._state, set())
        if target_state not in valid_targets:
            raise InvalidVisionStateTransitionError(self._state, target_state)
        self._state = target_state
        return self._state

    def reset(self) -> None:
        self._state = VisionSessionState.IDLE


class VisionObservation(BaseModel):
    """Sanitized visual perception observation for multimodal planner integration (NO RAW FRAMES)."""

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timezone-aware observation timestamp",
    )
    scene_events: list[SceneEvent] = Field(default_factory=list, description="Visual scene events")
    object_detections: list[ObjectDetection] = Field(
        default_factory=list, description="Non-identifying object detections"
    )
    face_detections: list[FaceDetection] = Field(
        default_factory=list, description="Non-identifying face location detections"
    )
    motion_events: list[MotionEvent] = Field(
        default_factory=list, description="Motion detection events"
    )
    tracks: list[Any] = Field(default_factory=list, description="Active anonymous object tracks")
    track_events: list[Any] = Field(default_factory=list, description="Temporal tracking events")
    scene_summary: Any | None = Field(
        default=None, description="Deterministic visual scene summary"
    )

    @field_validator("timestamp")
    @classmethod
    def validate_timezone(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware.")
        return v
