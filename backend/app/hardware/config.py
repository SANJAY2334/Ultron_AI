"""Hardware Subsystem Configuration Schema (Phase 4H.1).

Provides strongly-typed, validated hardware configuration derived from
application environment settings.
"""

from typing import Literal

from pydantic import BaseModel, Field

from app.core.config import Settings, get_settings


class HardwareConfig(BaseModel):
    """Hardware Subsystem Configuration parameters."""

    audio_capture_provider: Literal["mock", "sounddevice", "auto"] = Field(
        default="auto", description="Audio capture hardware driver provider"
    )
    vision_capture_provider: Literal["mock", "opencv", "directshow", "auto"] = Field(
        default="auto", description="Video capture hardware driver provider"
    )
    telemetry_interval_sec: float = Field(
        default=1.0, gt=0.1, le=60.0, description="Resource telemetry sampling interval in seconds"
    )
    mock_fallback_enabled: bool = Field(
        default=True, description="Allow falling back to mock device when no hardware is present"
    )
    probe_timeout_sec: float = Field(
        default=3.0, gt=0.1, le=30.0, description="Maximum timeout for hardware probing"
    )


def create_hardware_config(settings: Settings | None = None) -> HardwareConfig:
    """Constructs HardwareConfig derived from application Settings."""
    cfg = settings or get_settings()
    return HardwareConfig(
        audio_capture_provider=getattr(cfg, "AUDIO_CAPTURE_PROVIDER", "auto"),  # type: ignore[arg-type]
        vision_capture_provider=getattr(cfg, "VISION_CAPTURE_PROVIDER", "auto"),  # type: ignore[arg-type]
        telemetry_interval_sec=getattr(cfg, "HARDWARE_TELEMETRY_INTERVAL_SEC", 1.0),
        mock_fallback_enabled=getattr(cfg, "HARDWARE_MOCK_FALLBACK_ENABLED", True),
        probe_timeout_sec=getattr(cfg, "HARDWARE_PROBE_TIMEOUT_SEC", 3.0),
    )
