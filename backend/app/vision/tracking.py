"""Vision Object Tracking Domain Models, Configuration, and Exception Taxonomy (Phase 4F.5).

Defines framework-agnostic temporal object tracking models, track lifecycle state machine,
motion direction calculation, track events, configuration, telemetry, and exception hierarchy.
Enforces strict person anonymity rules (zero biometric/person identity fields permitted).
"""

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.config import Settings, get_settings
from app.vision.exceptions import (
    TrackingConfigurationError,
    TrackingValidationError,
)
from app.vision.models import BoundingBox


# Track Enums
class TrackState(StrEnum):
    """Lifecycle states of an object track."""

    NEW = "NEW"
    ACTIVE = "ACTIVE"
    LOST = "LOST"
    REACQUIRED = "REACQUIRED"
    ENDED = "ENDED"


class TrackDirection(StrEnum):
    """Spatial movement direction relative to camera frame."""

    STATIONARY = "STATIONARY"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    UP = "UP"
    DOWN = "DOWN"
    UP_LEFT = "UP_LEFT"
    UP_RIGHT = "UP_RIGHT"
    DOWN_LEFT = "DOWN_LEFT"
    DOWN_RIGHT = "DOWN_RIGHT"


class TrackEventType(StrEnum):
    """Temporal track lifecycle event type."""

    TRACK_CREATED = "TRACK_CREATED"
    TRACK_UPDATED = "TRACK_UPDATED"
    TRACK_MOVED = "TRACK_MOVED"
    TRACK_LOST = "TRACK_LOST"
    TRACK_REACQUIRED = "TRACK_REACQUIRED"
    TRACK_ENDED = "TRACK_ENDED"


VALID_TRACK_STATE_TRANSITIONS: dict[TrackState, set[TrackState]] = {
    TrackState.NEW: {TrackState.ACTIVE, TrackState.LOST, TrackState.ENDED},
    TrackState.ACTIVE: {TrackState.ACTIVE, TrackState.LOST, TrackState.ENDED},
    TrackState.LOST: {TrackState.REACQUIRED, TrackState.ENDED},
    TrackState.REACQUIRED: {TrackState.ACTIVE, TrackState.LOST, TrackState.ENDED},
    TrackState.ENDED: set(),
}


class ObjectTrack(BaseModel):
    """Anonymous temporal object track record (STRICTLY PROHIBITS BIOMETRIC/PERSON IDENTITIES)."""

    model_config = ConfigDict(extra="forbid")

    track_id: str = Field(min_length=1, description="Anonymous tracking ID (e.g. 'track_001')")
    class_name: str = Field(
        min_length=1, description="Detected object class (e.g. 'person', 'chair')"
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Detection confidence score")
    bounding_box: BoundingBox = Field(description="Current normalized bounding box")
    state: TrackState = Field(default=TrackState.NEW, description="Current track lifecycle state")
    first_seen: datetime = Field(description="Timezone-aware timestamp when track was created")
    last_seen: datetime = Field(description="Timezone-aware timestamp of last track update")
    frames_seen: int = Field(default=1, ge=1, description="Total count of frames matched")
    age_frames: int = Field(default=1, ge=1, description="Total frame lifetime of track")
    missing_frames: int = Field(default=0, ge=0, description="Consecutive frames missing")
    velocity_x: float = Field(
        default=0.0, description="Horizontal velocity component in pixels/frame"
    )
    velocity_y: float = Field(
        default=0.0, description="Vertical velocity component in pixels/frame"
    )
    speed: float = Field(default=0.0, ge=0.0, description="Scalar speed magnitude in pixels/frame")
    direction: TrackDirection = Field(
        default=TrackDirection.STATIONARY, description="Movement direction classification"
    )
    is_active: bool = Field(default=True, description="True if track is active or lost")

    @field_validator("first_seen", "last_seen")
    @classmethod
    def validate_timezone(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware.")
        return v

    def transition_to(self, target: TrackState) -> None:
        """Executes validated track state transition."""
        allowed = VALID_TRACK_STATE_TRANSITIONS.get(self.state, set())
        if target not in allowed and target != self.state:
            raise TrackingValidationError(
                f"Invalid track state transition: cannot transition from {self.state.value} to {target.value}."
            )
        self.state = target
        if target == TrackState.ENDED:
            self.is_active = False


class TrackEvent(BaseModel):
    """Sanitized temporal track event record."""

    event_id: str = Field(min_length=1, description="Unique track event ID")
    event_type: TrackEventType = Field(description="Track event type discriminator")
    track_id: str = Field(min_length=1, description="Associated anonymous track ID")
    class_name: str = Field(min_length=1, description="Associated object class label")
    timestamp: datetime = Field(description="Timezone-aware event timestamp")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Sanitized event metadata attributes"
    )

    @field_validator("timestamp")
    @classmethod
    def validate_timezone(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware.")
        return v


class TrackingConfig(BaseModel):
    """Configuration parameters for visual object tracking engine."""

    enabled: bool = Field(default=True, description="True if visual tracking is enabled")
    provider: str = Field(default="baseline", description="Tracking engine provider identifier")
    iou_threshold: float = Field(
        default=0.30, ge=0.0, le=1.0, description="Minimum IoU threshold for detection association"
    )
    max_distance: float = Field(
        default=100.0, gt=0.0, description="Maximum centroid distance in pixels for association"
    )
    max_missing_frames: int = Field(
        default=10, ge=1, le=100, description="Maximum missing frames before track termination"
    )
    max_tracks: int = Field(
        default=50, gt=0, le=200, description="Maximum allowed concurrent active tracks"
    )
    timeout_ms: float = Field(
        default=50.0, gt=0.0, description="Maximum tracking update processing timeout in ms"
    )
    movement_threshold: float = Field(
        default=5.0,
        ge=0.0,
        description="Minimum pixel movement threshold for direction classification",
    )

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        valid_providers = {"baseline", "sort", "bytetrack", "mock"}
        if v.lower() not in valid_providers:
            raise TrackingConfigurationError(
                f"Invalid tracking provider '{v}'. Must be one of {valid_providers}."
            )
        return v.lower()


class TrackingTelemetry(BaseModel):
    """Telemetry record for tracking operations (ZERO RAW IMAGE BYTES)."""

    frame_id: str = Field(min_length=1, description="Target frame identifier")
    active_tracks_count: int = Field(ge=0, description="Current count of active tracks")
    new_tracks_count: int = Field(ge=0, description="Count of tracks created in frame")
    lost_tracks_count: int = Field(ge=0, description="Count of tracks lost in frame")
    ended_tracks_count: int = Field(ge=0, description="Count of tracks ended in frame")
    association_latency_ms: float = Field(
        ge=0.0, description="Association algorithm duration in ms"
    )
    total_latency_ms: float = Field(ge=0.0, description="Total tracking update duration in ms")
    success: bool = Field(default=True, description="True if tracking update succeeded")
    error_code: str | None = Field(
        default=None, description="Sanitized error code string if failed"
    )


def create_tracking_config(settings: Settings | None = None) -> TrackingConfig:
    """Constructs TrackingConfig derived from application Settings."""
    cfg = settings or get_settings()
    return TrackingConfig(
        enabled=getattr(cfg, "VISION_TRACKING_ENABLED", True),
        provider=getattr(cfg, "VISION_TRACKING_PROVIDER", "baseline"),
        iou_threshold=getattr(cfg, "VISION_TRACKING_IOU_THRESHOLD", 0.30),
        max_distance=getattr(cfg, "VISION_TRACKING_MAX_DISTANCE", 100.0),
        max_missing_frames=getattr(cfg, "VISION_TRACKING_MAX_MISSING_FRAMES", 10),
        max_tracks=getattr(cfg, "VISION_TRACKING_MAX_TRACKS", 50),
        timeout_ms=getattr(cfg, "VISION_TRACKING_TIMEOUT_MS", 50.0),
        movement_threshold=getattr(cfg, "VISION_TRACKING_MOVEMENT_THRESHOLD", 5.0),
    )
