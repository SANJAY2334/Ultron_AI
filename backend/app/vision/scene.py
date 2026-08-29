"""Vision Scene Understanding & Temporal Intelligence Domain Models (Phase 4F.6).

Defines framework-agnostic structured scene understanding models, scene states,
motion level classification, temporal scene change models, dominant activity rules,
configuration, telemetry, and exception hierarchy.
Operates EXCLUSIVELY on structured metadata (VisionObservation). ZERO raw image dependencies.
Enforces strict person anonymity rules (zero biometric/person identity fields permitted).
"""

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.config import Settings, get_settings


# Scene Enums
class SceneState(StrEnum):
    """Deterministic state classification of a visual scene."""

    EMPTY = "EMPTY"
    STABLE = "STABLE"
    ACTIVE = "ACTIVE"
    CHANGING = "CHANGING"
    CROWDED = "CROWDED"
    UNKNOWN = "UNKNOWN"


class MotionLevel(StrEnum):
    """Relative motion level classification within scene."""

    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class DominantActivity(StrEnum):
    """Deterministic classification of primary visual scene activity."""

    NONE = "NONE"
    STATIONARY = "STATIONARY"
    MOVEMENT = "MOVEMENT"
    MULTIPLE_MOVEMENT = "MULTIPLE_MOVEMENT"
    SCENE_CHANGE = "SCENE_CHANGE"


class SceneChangeType(StrEnum):
    """Temporal scene change event discriminator."""

    OBJECT_ENTERED = "OBJECT_ENTERED"
    OBJECT_EXITED = "OBJECT_EXITED"
    OBJECT_COUNT_CHANGED = "OBJECT_COUNT_CHANGED"
    OBJECT_CLASS_CHANGED = "OBJECT_CLASS_CHANGED"
    MOTION_STARTED = "MOTION_STARTED"
    MOTION_STOPPED = "MOTION_STOPPED"
    SCENE_BECAME_ACTIVE = "SCENE_BECAME_ACTIVE"
    SCENE_BECAME_STABLE = "SCENE_BECAME_STABLE"
    SCENE_CHANGED = "SCENE_CHANGED"


class SceneChange(BaseModel):
    """Sanitized temporal scene change record (NO IDENTITY FIELDS PERMITTED)."""

    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(min_length=1, description="Unique scene change event ID")
    event_type: SceneChangeType = Field(description="Scene change event type discriminator")
    timestamp: datetime = Field(description="Timezone-aware timestamp")
    track_id: str | None = Field(
        default=None, description="Associated anonymous track ID if available"
    )
    class_name: str | None = Field(
        default=None, description="Associated object class label if available"
    )
    previous_state: str | None = Field(default=None, description="Previous scene attribute state")
    current_state: str | None = Field(default=None, description="Current scene attribute state")
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Change classification confidence"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Sanitized event metadata attributes"
    )

    @field_validator("timestamp")
    @classmethod
    def validate_timezone(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware.")
        return v


class SceneSummary(BaseModel):
    """Structured privacy-preserving summary of visual scene perception (NO RAW IMAGES)."""

    model_config = ConfigDict(extra="forbid")

    scene_id: str = Field(min_length=1, description="Unique scene summary ID")
    timestamp: datetime = Field(description="Timezone-aware observation timestamp")
    object_count: int = Field(ge=0, description="Total count of objects present in scene")
    object_classes: list[str] = Field(
        default_factory=list, description="Sorted list of unique object classes"
    )
    class_counts: dict[str, int] = Field(
        default_factory=dict, description="Histogram count of objects per class"
    )
    active_track_count: int = Field(ge=0, description="Count of active object tracks")
    moving_track_count: int = Field(ge=0, description="Count of actively moving object tracks")
    entered_objects: list[str] = Field(
        default_factory=list, description="Anonymous track IDs entered in frame"
    )
    exited_objects: list[str] = Field(
        default_factory=list, description="Anonymous track IDs exited in frame"
    )
    scene_state: SceneState = Field(
        default=SceneState.UNKNOWN, description="Deterministic scene state"
    )
    motion_level: MotionLevel = Field(
        default=MotionLevel.NONE, description="Calculated motion level"
    )
    dominant_objects: list[str] = Field(
        default_factory=list, description="Dominant object class labels"
    )
    dominant_activity: DominantActivity = Field(
        default=DominantActivity.NONE, description="Deterministic dominant activity"
    )
    scene_changes: list[SceneChange] = Field(
        default_factory=list, description="Temporal scene change events"
    )

    @field_validator("timestamp")
    @classmethod
    def validate_timezone(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware.")
        return v


class SceneAnalysisConfig(BaseModel):
    """Configuration parameters for deterministic scene understanding engine."""

    enabled: bool = Field(default=True, description="True if visual scene understanding is enabled")
    max_previous_observations: int = Field(
        default=2, ge=1, le=10, description="Maximum bounded observation history depth"
    )
    crowded_threshold: int = Field(
        default=10, ge=1, description="Object count threshold for crowded scene state"
    )
    low_motion_threshold: float = Field(
        default=0.10, ge=0.0, le=1.0, description="Low motion level threshold ratio"
    )
    high_motion_threshold: float = Field(
        default=0.50, ge=0.0, le=1.0, description="High motion level threshold ratio"
    )
    change_threshold: float = Field(
        default=0.30, ge=0.0, le=1.0, description="Scene change detection sensitivity ratio"
    )
    timeout_ms: float = Field(
        default=50.0, gt=0.0, description="Maximum scene analysis timeout in ms"
    )


class SceneAnalysisTelemetry(BaseModel):
    """Telemetry record for scene understanding operations (ZERO RAW IMAGES)."""

    scene_id: str = Field(min_length=1, description="Target scene summary identifier")
    object_count: int = Field(ge=0, description="Count of objects analyzed")
    scene_state: str = Field(description="Derived scene state string")
    motion_level: str = Field(description="Derived motion level string")
    statistics_latency_ms: float = Field(ge=0.0, description="Object statistics latency in ms")
    temporal_comparison_latency_ms: float = Field(
        ge=0.0, description="Temporal comparison latency in ms"
    )
    classification_latency_ms: float = Field(
        ge=0.0, description="Scene state classification latency in ms"
    )
    total_latency_ms: float = Field(ge=0.0, description="Total scene analysis duration in ms")
    success: bool = Field(default=True, description="True if scene analysis succeeded")
    error_code: str | None = Field(
        default=None, description="Sanitized error code string if failed"
    )


def create_scene_analysis_config(settings: Settings | None = None) -> SceneAnalysisConfig:
    """Constructs SceneAnalysisConfig derived from application Settings."""
    cfg = settings or get_settings()
    return SceneAnalysisConfig(
        enabled=getattr(cfg, "VISION_SCENE_ANALYSIS_ENABLED", True),
        max_previous_observations=getattr(cfg, "VISION_SCENE_MAX_PREVIOUS_OBSERVATIONS", 2),
        crowded_threshold=getattr(cfg, "VISION_SCENE_CROWDED_THRESHOLD", 10),
        low_motion_threshold=getattr(cfg, "VISION_SCENE_LOW_MOTION_THRESHOLD", 0.10),
        high_motion_threshold=getattr(cfg, "VISION_SCENE_HIGH_MOTION_THRESHOLD", 0.50),
        change_threshold=getattr(cfg, "VISION_SCENE_CHANGE_THRESHOLD", 0.30),
        timeout_ms=getattr(cfg, "VISION_SCENE_TIMEOUT_MS", 50.0),
    )
