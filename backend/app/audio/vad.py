"""Voice Activity Detection (VAD) Domain Models, Configuration, and Exceptions (Phase 4C.3).

Defines VADConfig, VADTelemetry, and the exception taxonomy for the VAD subsystem.
"""

from datetime import datetime

from pydantic import BaseModel, Field, field_validator

from app.audio.models import VoiceActivityState


class VADConfig(BaseModel):
    """Configuration parameters for Voice Activity Detection."""

    speech_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for classifying speech [0.0, 1.0]",
    )
    minimum_speech_duration_ms: float = Field(
        default=100.0,
        gt=0.0,
        description="Minimum continuous speech duration required to trigger SPEECH_START",
    )
    minimum_silence_duration_ms: float = Field(
        default=300.0,
        gt=0.0,
        description="Minimum continuous silence duration required to trigger SPEECH_END",
    )
    debounce_ms: float = Field(
        default=50.0,
        ge=0.0,
        description="Debounce window in milliseconds to prevent state oscillation",
    )
    processing_timeout_ms: float = Field(
        default=100.0,
        gt=0.0,
        description="Maximum allowed VAD processing timeout per chunk in milliseconds",
    )


class VADTelemetry(BaseModel):
    """Telemetry record captured during VAD processing (excludes raw audio payload)."""

    timestamp: datetime = Field(description="Timezone-aware timestamp")
    chunk_id: str = Field(description="Target AudioChunk ID")
    sequence_number: int = Field(ge=0, description="Chunk sequence number")
    duration_ms: float = Field(gt=0.0, description="Chunk duration in ms")
    detected_state: VoiceActivityState = Field(description="Detected VAD state")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Normalized energy/speech confidence score"
    )
    processing_latency_ms: float = Field(ge=0.0, description="Processing duration in milliseconds")
    error_code: str | None = Field(
        default=None, description="Error code string if processing failed"
    )

    @field_validator("timestamp")
    @classmethod
    def validate_timezone(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware.")
        return v


# VAD Subsystem Exception Taxonomy
class VADError(Exception):
    """Base application exception for all VAD subsystem failures."""


class VADConfigurationError(VADError):
    """Raised when VAD configuration parameters are invalid or unsupported."""


class VADProcessingError(VADError):
    """Raised when audio payload processing or RMS computation fails."""


class VADBackendError(VADError):
    """Raised when the underlying VAD backend engine encounters an internal failure."""


class VADTimeoutError(VADError):
    """Raised when VAD chunk processing exceeds the processing_timeout_ms threshold."""
