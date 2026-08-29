"""Vision Subsystem Configuration Model and Helpers (Phase 4F.1).

Defines bounded resource configuration constraints for vision processing, frame resolution limits,
queue depth limits, drop policy, processing timeouts, and privacy defaults.
"""

from pydantic import BaseModel, Field, field_validator

from app.core.config import Settings, get_settings
from app.vision.models import VisionPrivacy


class VisionConfig(BaseModel):
    """Configuration parameters for Vision Subsystem frame capture and processing."""

    enabled: bool = Field(default=False, description="True if vision subsystem is enabled")
    provider: str = Field(default="local", description="Vision provider engine identifier")
    camera_device: str = Field(
        default="default", description="Target video capture device identifier"
    )
    camera_backend: str = Field(
        default="opencv", description="Camera hardware acquisition backend identifier"
    )
    camera_pixel_format: str = Field(
        default="RGB24", description="Default frame acquisition pixel format"
    )
    max_fps: float = Field(
        default=15.0, gt=0.0, le=60.0, description="Maximum capture/processing frame rate"
    )
    max_width: int = Field(
        default=1280, gt=0, le=3840, description="Maximum allowed frame width in pixels"
    )
    max_height: int = Field(
        default=720, gt=0, le=2160, description="Maximum allowed frame height in pixels"
    )
    max_buffered_frames: int = Field(
        default=10, gt=0, le=100, description="Maximum frame queue buffer depth"
    )
    processing_timeout_ms: float = Field(
        default=100.0, gt=0.0, description="Maximum frame processing timeout in ms"
    )
    max_payload_bytes: int = Field(
        default=10485760,
        gt=0,
        le=52428800,
        description="Maximum allowed frame payload size in bytes",
    )
    enable_motion: bool = Field(default=False, description="True if motion estimation is enabled")
    max_previous_frames: int = Field(
        default=1, ge=0, le=5, description="Maximum bounded frame history buffer depth"
    )
    drop_policy: str = Field(default="DROP_OLDEST", description="Queue backpressure drop policy")
    privacy: VisionPrivacy = Field(
        default=VisionPrivacy.EPHEMERAL, description="Default vision privacy classification"
    )

    @field_validator("drop_policy")
    @classmethod
    def validate_drop_policy(cls, v: str) -> str:
        valid_policies = {"DROP_OLDEST", "DROP_NEWEST", "BLOCK"}
        if v.upper() not in valid_policies:
            raise ValueError(f"Invalid drop policy '{v}'. Must be one of {valid_policies}.")
        return v.upper()


def create_vision_config(settings: Settings | None = None) -> VisionConfig:
    """Constructs VisionConfig derived from application Settings."""
    cfg = settings or get_settings()
    privacy_val = getattr(cfg, "VISION_PRIVACY", "ephemeral").upper()
    try:
        privacy_enum = VisionPrivacy(privacy_val)
    except ValueError:
        privacy_enum = VisionPrivacy.EPHEMERAL

    return VisionConfig(
        enabled=getattr(cfg, "VISION_ENABLED", False),
        provider=getattr(cfg, "VISION_PROVIDER", "local"),
        camera_device=getattr(cfg, "VISION_CAMERA_DEVICE", "default"),
        max_fps=getattr(cfg, "VISION_MAX_FPS", 15),
        max_width=getattr(cfg, "VISION_MAX_WIDTH", 1280),
        max_height=getattr(cfg, "VISION_MAX_HEIGHT", 720),
        max_buffered_frames=getattr(cfg, "VISION_MAX_BUFFERED_FRAMES", 10),
        processing_timeout_ms=getattr(cfg, "VISION_PROCESSING_TIMEOUT_MS", 100.0),
        privacy=privacy_enum,
    )
