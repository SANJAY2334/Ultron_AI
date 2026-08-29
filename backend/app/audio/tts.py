"""Text-to-Speech (TTS) Subsystem Configuration, Exceptions, and Telemetry (Phase 4C.5).

Defines TTSConfig, TTSTelemetry, and sanitized TTS exception taxonomy.
"""

from pydantic import BaseModel, Field, field_validator


class TTSConfig(BaseModel):
    """Configuration parameters for Text-to-Speech (TTS) providers."""

    provider_name: str = Field(default="mock_tts", description="TTS provider identifier")
    model_name: str = Field(default="tts-1", description="Target TTS synthesis model")
    voice: str = Field(default="en-US-Standard-A", description="Target voice name")
    language: str = Field(default="en", description="Target language code (ISO 639-1)")
    sample_rate: int = Field(default=16000, description="Target PCM sample rate in Hz")
    channels: int = Field(default=1, ge=1, description="Target audio channels count")
    encoding: str = Field(default="pcm_s16le", description="Target audio sample encoding")
    speed: float = Field(
        default=1.0, ge=0.25, le=4.0, description="Speech synthesis speed multiplier [0.25, 4.0]"
    )
    pitch: float = Field(
        default=0.0, ge=-20.0, le=20.0, description="Speech synthesis pitch shift [-20.0, 20.0]"
    )
    timeout_ms: float = Field(
        default=5000.0, gt=0.0, description="Provider synthesis request timeout in ms"
    )
    streaming_enabled: bool = Field(
        default=True, description="True if real-time audio chunk streaming is enabled"
    )
    max_text_length: int = Field(
        default=5000, gt=0, description="Maximum allowed text length in characters"
    )
    max_audio_duration_ms: float = Field(
        default=120000.0, gt=0.0, description="Maximum allowed audio synthesis duration in ms"
    )
    max_chunks: int = Field(
        default=1000, gt=0, description="Maximum allowed streaming audio chunk count"
    )
    max_chunk_size_bytes: int = Field(
        default=65536, gt=0, description="Maximum allowed size per audio chunk in bytes"
    )
    retry_count: int = Field(
        default=2, ge=0, description="Maximum retry count for transient provider errors"
    )

    @field_validator("sample_rate")
    @classmethod
    def validate_sample_rate(cls, v: int) -> int:
        if v not in {8000, 16000, 24000, 44100, 48000}:
            raise ValueError(f"Unsupported TTS sample rate {v}Hz.")
        return v


class TTSTelemetry(BaseModel):
    """Telemetry record captured during TTS synthesis (excludes raw audio payload and full text)."""

    synthesis_id: str = Field(description="Unique synthesis request identifier")
    provider: str = Field(description="TTS provider name")
    model: str = Field(description="TTS model name")
    voice: str = Field(description="Selected voice identifier")
    duration_ms: float = Field(ge=0.0, description="Synthesized audio duration in ms")
    latency_ms: float = Field(ge=0.0, description="Processing latency in ms")
    audio_chunk_count: int = Field(ge=0, description="Count of audio chunks generated")
    success: bool = Field(description="True if operation succeeded")
    error_code: str | None = Field(default=None, description="Sanitized error string if failed")
    correlation_id: str | None = Field(default=None, description="Tracing correlation identifier")


# Exception Taxonomy for TTS Subsystem
class TTSError(Exception):
    """Base application exception for all TTS subsystem failures."""


class TTSConfigurationError(TTSError):
    """Raised when TTS configuration parameters are invalid."""


class TTSValidationError(TTSError):
    """Raised when synthesis text or parameters violate validation rules."""


class TTSProviderError(TTSError):
    """Raised when a TTS provider encounters an unrecoverable operational failure."""


class TTSTimeoutError(TTSError):
    """Raised when a TTS provider request exceeds timeout_ms."""


class TTSUnavailableError(TTSError):
    """Raised when the TTS provider service is unreachable or credentials are missing."""


class TTSProcessingError(TTSError):
    """Raised when speech synthesis rendering or audio byte assembly fails."""
