"""Speech-to-Text (STT) Subsystem Configuration, Exceptions, and Telemetry (Phase 4C.4).

Defines STTConfig, STTTelemetry, and sanitized STT exception taxonomy.
"""

from pydantic import BaseModel, Field, field_validator


class STTConfig(BaseModel):
    """Configuration parameters for Speech-to-Text (STT) providers."""

    provider_name: str = Field(default="mock_stt", description="STT provider identifier")
    model_name: str = Field(default="whisper-1", description="Target STT transcription model")
    language: str = Field(default="en", description="Target language code (ISO 639-1)")
    timeout_ms: float = Field(default=5000.0, gt=0.0, description="Provider request timeout in ms")
    max_audio_duration_ms: float = Field(
        default=60000.0,
        gt=0.0,
        description="Maximum total audio duration allowed per request in ms",
    )
    streaming_enabled: bool = Field(
        default=True, description="True if real-time streaming partial results enabled"
    )
    partial_transcript_enabled: bool = Field(
        default=True, description="True if partial unfinalized transcripts should be emitted"
    )
    confidence_threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for accepting transcripts [0.0, 1.0]",
    )
    max_transcript_length: int = Field(
        default=10000, gt=0, description="Maximum allowed transcript text length in characters"
    )
    max_chunks: int = Field(
        default=1000, gt=0, description="Maximum allowed AudioChunk count per request"
    )
    max_payload_bytes: int = Field(
        default=10485760,
        gt=0,
        description="Maximum allowed cumulative raw audio bytes (default: 10MB)",
    )
    retry_count: int = Field(
        default=2, ge=0, description="Maximum retry count for transient provider errors"
    )

    @field_validator("language")
    @classmethod
    def validate_language_code(cls, v: str) -> str:
        v_clean = v.strip().lower()
        if not v_clean or len(v_clean) > 5:
            raise ValueError(f"Invalid language code '{v}'.")
        return v_clean


class STTTelemetry(BaseModel):
    """Telemetry record captured during STT transcription (excludes raw audio bytes and full text)."""

    transcript_id: str = Field(description="Unique transcript identifier")
    provider: str = Field(description="STT provider name")
    model: str = Field(description="STT model name")
    duration_ms: float = Field(ge=0.0, description="Total audio duration transcribed in ms")
    latency_ms: float = Field(ge=0.0, description="Processing latency in ms")
    language: str = Field(description="Detected or configured language code")
    confidence: float | None = Field(
        default=None, description="Normalized transcript confidence [0.0, 1.0]"
    )
    segment_count: int = Field(ge=0, description="Total transcript segments count")
    is_final: bool = Field(description="True if final transcript, False if partial")
    success: bool = Field(description="True if operation succeeded")
    error_code: str | None = Field(default=None, description="Sanitized error string if failed")
    correlation_id: str | None = Field(default=None, description="Tracing correlation identifier")


# Exception Taxonomy for STT Subsystem
class STTError(Exception):
    """Base application exception for all STT subsystem failures."""


class STTConfigurationError(STTError):
    """Raised when STT configuration parameters are invalid."""


class STTValidationError(STTError):
    """Raised when incoming audio chunks or parameters violate validation rules."""


class STTProviderError(STTError):
    """Raised when an STT provider encounters an unrecoverable operational failure."""


class STTTimeoutError(STTError):
    """Raised when an STT provider request exceeds timeout_ms."""


class STTUnavailableError(STTError):
    """Raised when the STT provider service is unreachable or credentials are missing."""


class STTProcessingError(STTError):
    """Raised when raw speech audio decoding or segment assembly fails."""
