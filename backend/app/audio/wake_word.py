"""Wake Word Detection Subsystem Configuration, Exceptions, and Telemetry (Phase 4C.6).

Defines WakeWordConfig, WakeWordTelemetry, and sanitized WakeWord exception taxonomy.
"""

from datetime import datetime

from pydantic import BaseModel, Field, field_validator


class WakeWordConfig(BaseModel):
    """Configuration parameters for Wake Word Detection providers."""

    enabled: bool = Field(default=True, description="True if wake-word detection is enabled")
    wake_words: list[str] = Field(
        default_factory=lambda: ["hey ultron", "ultron"],
        description="List of target wake word trigger phrases",
    )
    confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for triggering [0.0, 1.0]",
    )
    cooldown_ms: float = Field(
        default=2000.0,
        ge=0.0,
        description="Cooldown window in ms after activation to suppress duplicate triggers",
    )
    timeout_ms: float = Field(
        default=1000.0, gt=0.0, description="Provider detection request timeout in ms"
    )
    retry_count: int = Field(
        default=2, ge=0, description="Maximum retry count for transient provider errors"
    )
    max_buffered_chunks: int = Field(
        default=100, gt=0, description="Maximum allowed AudioChunk buffer count"
    )
    max_detection_window_ms: float = Field(
        default=5000.0, gt=0.0, description="Maximum rolling audio window duration in ms"
    )
    provider: str = Field(default="mock_wake_word", description="Wake-word engine provider name")
    model: str = Field(default="openwakeword-v1", description="Target wake-word model identifier")

    @field_validator("wake_words")
    @classmethod
    def validate_wake_words_list(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("wake_words list cannot be empty.")
        cleaned = [w.strip().lower() for w in v if w and w.strip()]
        if not cleaned:
            raise ValueError("wake_words list must contain at least one valid trigger phrase.")
        return cleaned


class WakeWordTelemetry(BaseModel):
    """Telemetry record captured during wake-word evaluation (excludes raw audio bytes)."""

    detection_id: str = Field(description="Unique detection event identifier")
    wake_word: str = Field(description="Triggered wake word string")
    confidence: float = Field(ge=0.0, le=1.0, description="Normalized detection confidence")
    latency_ms: float = Field(ge=0.0, description="Evaluation latency in ms")
    provider: str = Field(description="Wake-word engine provider name")
    model: str = Field(description="Wake-word model identifier")
    sequence_number: int = Field(ge=0, description="AudioChunk sequence number")
    success: bool = Field(description="True if evaluation succeeded")
    error_code: str | None = Field(
        default=None, description="Sanitized error code string if failed"
    )
    correlation_id: str | None = Field(default=None, description="Tracing correlation identifier")

    @field_validator("timestamp", check_fields=False)
    @classmethod
    def validate_timestamp(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware.")
        return v


# Exception Taxonomy for Wake Word Subsystem
class WakeWordError(Exception):
    """Base application exception for all Wake Word subsystem failures."""


class WakeWordConfigurationError(WakeWordError):
    """Raised when Wake Word configuration parameters are invalid."""


class WakeWordValidationError(WakeWordError):
    """Raised when incoming audio chunks or sequence numbers violate validation rules."""


class WakeWordProviderError(WakeWordError):
    """Raised when a wake-word engine encounters an unrecoverable operational failure."""


class WakeWordTimeoutError(WakeWordError):
    """Raised when a wake-word evaluation request exceeds timeout_ms."""


class WakeWordUnavailableError(WakeWordError):
    """Raised when the wake-word engine service is unreachable or unavailable."""


class WakeWordProcessingError(WakeWordError):
    """Raised when audio frame decoding or rolling window evaluation fails."""
