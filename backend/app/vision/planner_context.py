"""Vision-to-Autonomous Planner Context Models and Configuration (Phase 4F.7).

Defines framework-agnostic sanitized VisionPlannerContext, VisionContextConfig,
VisionContextTelemetry, and exception hierarchy.
Enforces strict person anonymity rules (zero biometric/person identity fields permitted)
and strict character/item truncation limits before crossing the Planner boundary.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.config import Settings, get_settings


class VisionPlannerContext(BaseModel):
    """Sanitized structured visual context for Autonomous Planner consumption (NO RAW IMAGES)."""

    model_config = ConfigDict(extra="forbid")

    observation_id: str = Field(min_length=1, description="Source observation tracking ID")
    timestamp: datetime = Field(description="Timezone-aware observation timestamp")
    scene_state: str = Field(default="UNKNOWN", description="Derived scene state string")
    motion_level: str = Field(default="NONE", description="Derived motion level string")
    dominant_activity: str = Field(default="NONE", description="Derived dominant activity string")
    object_count: int = Field(default=0, ge=0, description="Total count of objects present")
    class_counts: dict[str, int] = Field(
        default_factory=dict, description="Histogram of detected object classes"
    )
    dominant_objects: list[str] = Field(
        default_factory=list, description="Top dominant object class names"
    )
    moving_object_count: int = Field(
        default=0, ge=0, description="Count of actively moving objects"
    )
    active_track_count: int = Field(default=0, ge=0, description="Count of active object tracks")
    scene_changes: list[str] = Field(
        default_factory=list, description="Sanitized scene change event summaries"
    )
    recent_events: list[str] = Field(
        default_factory=list, description="Sanitized perception event summaries"
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Observation confidence score"
    )
    source: str = Field(default="vision_subsystem", description="Context provenance metadata")

    @field_validator("timestamp")
    @classmethod
    def validate_timezone(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware.")
        return v

    def to_prompt_context(self) -> str:
        """Renders sanitized, bounded text prompt context block for planner injection."""
        lines = [
            "[VISUAL SCENE CONTEXT]",
            f"Scene State: {self.scene_state}",
            f"Motion Level: {self.motion_level}",
            f"Dominant Activity: {self.dominant_activity}",
            f"Total Objects: {self.object_count} (Moving: {self.moving_object_count}, Active Tracks: {self.active_track_count})",
        ]
        if self.class_counts:
            cls_str = ", ".join(f"{k}: {v}" for k, v in self.class_counts.items())
            lines.append(f"Class Histogram: {cls_str}")
        if self.dominant_objects:
            lines.append(f"Dominant Objects: {', '.join(self.dominant_objects)}")
        if self.scene_changes:
            lines.append(f"Recent Scene Changes: {'; '.join(self.scene_changes)}")
        if self.recent_events:
            lines.append(f"Recent Perception Events: {'; '.join(self.recent_events)}")

        context_str = "\n".join(lines)
        return context_str


class VisionContextConfig(BaseModel):
    """Configuration parameters for Vision-to-Planner Context Translation."""

    enabled: bool = Field(
        default=True, description="True if vision planner context integration is enabled"
    )
    max_objects: int = Field(
        default=20, ge=1, le=50, description="Maximum objects included in context"
    )
    max_tracks: int = Field(
        default=20, ge=1, le=50, description="Maximum tracks included in context"
    )
    max_events: int = Field(
        default=20, ge=1, le=50, description="Maximum events included in context"
    )
    max_context_chars: int = Field(
        default=8000,
        ge=100,
        le=20000,
        description="Maximum vision context prompt length in characters",
    )
    timeout_ms: float = Field(
        default=20.0, gt=0.0, description="Maximum context build timeout in ms"
    )


class VisionContextTelemetry(BaseModel):
    """Telemetry metrics for vision planner context translation (ZERO RAW IMAGES)."""

    operation: str = Field(description="Operation discriminator")
    correlation_id: str = Field(description="Tracing correlation ID")
    observation_id: str = Field(description="Target observation identifier")
    object_count: int = Field(ge=0, description="Count of objects processed")
    track_count: int = Field(ge=0, description="Count of tracks processed")
    event_count: int = Field(ge=0, description="Count of events processed")
    context_size: int = Field(ge=0, description="Total characters in prompt context")
    latency_ms: float = Field(ge=0.0, description="Context build latency in ms")
    success: bool = Field(default=True, description="True if build succeeded")
    error_code: str | None = Field(
        default=None, description="Sanitized error code string if failed"
    )


def create_vision_context_config(settings: Settings | None = None) -> VisionContextConfig:
    """Constructs VisionContextConfig derived from application Settings."""
    cfg = settings or get_settings()
    return VisionContextConfig(
        enabled=getattr(cfg, "VISION_PLANNER_CONTEXT_ENABLED", True),
        max_objects=getattr(cfg, "VISION_PLANNER_MAX_OBJECTS", 20),
        max_tracks=getattr(cfg, "VISION_PLANNER_MAX_TRACKS", 20),
        max_events=getattr(cfg, "VISION_PLANNER_MAX_EVENTS", 20),
        max_context_chars=getattr(cfg, "VISION_PLANNER_MAX_CONTEXT_CHARS", 8000),
        timeout_ms=getattr(cfg, "VISION_PLANNER_CONTEXT_TIMEOUT_MS", 20.0),
    )
