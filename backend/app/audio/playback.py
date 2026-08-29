"""Audio Playback Configuration, Telemetry, and Exceptions (Phase 4E.1).

Defines PlaybackConfig, PlaybackTelemetry, and sanitized Playback exception hierarchy.
"""

from datetime import datetime

from pydantic import BaseModel, Field, field_validator

from app.audio.models import PlaybackState


class PlaybackConfig(BaseModel):
    """Configuration parameters for Audio Playback providers."""

    max_buffered_chunks: int = Field(
        default=100, gt=0, description="Maximum bounded playback queue size"
    )
    backpressure_policy: str = Field(
        default="DROP_OLDEST", description="Queue overflow backpressure policy"
    )
    sample_rate: int = Field(default=16000, gt=0, description="Target playback sample rate in Hz")
    channels: int = Field(default=1, gt=0, description="Target playback channel count")
    timeout_ms: float = Field(
        default=1000.0, gt=0.0, description="Playback operation timeout limit in ms"
    )

    @field_validator("backpressure_policy")
    @classmethod
    def validate_policy(cls, v: str) -> str:
        valid_policies = {"DROP_OLDEST", "DROP_NEWEST", "BLOCK"}
        v_upper = v.upper()
        if v_upper not in valid_policies:
            raise ValueError(
                f"Invalid backpressure policy '{v}'. Supported: {sorted(valid_policies)}"
            )
        return v_upper


class PlaybackTelemetry(BaseModel):
    """Telemetry record captured during audio playback (excludes raw audio payload bytes)."""

    state: PlaybackState = Field(description="Current playback state")
    chunks_received: int = Field(ge=0, description="Total AudioChunks received")
    chunks_played: int = Field(ge=0, description="Total AudioChunks played")
    chunks_dropped: int = Field(ge=0, description="Total AudioChunks dropped due to backpressure")
    total_duration_ms: float = Field(ge=0.0, description="Total played audio duration in ms")
    latency_ms: float = Field(ge=0.0, description="Playback enqueue latency in ms")
    timestamp: datetime = Field(description="Timezone-aware telemetry timestamp")
    session_id: str | None = Field(default=None, description="Session ID")
    correlation_id: str | None = Field(default=None, description="Tracing correlation ID")

    @field_validator("timestamp")
    @classmethod
    def validate_timezone(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware.")
        return v


# Exception Taxonomy for Audio Playback Subsystem
class PlaybackError(Exception):
    """Base application exception for all Audio Playback failures."""


class PlaybackConfigurationError(PlaybackError):
    """Raised when PlaybackConfig parameters are invalid."""


class PlaybackValidationError(PlaybackError):
    """Raised when incoming AudioChunks violate sequence or format rules."""


class PlaybackStateError(PlaybackError):
    """Raised when illegal PlaybackState transitions are attempted."""


class PlaybackTimeoutError(PlaybackError):
    """Raised when playback operations exceed configured timeout_ms."""


class PlaybackDeviceNotFoundError(PlaybackError):
    """Raised when target speaker or audio output hardware device is unreachable."""
