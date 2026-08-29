"""Audio Session Domain Models, Configuration, and Exception Taxonomy (Phase 4C.7).

Defines AudioSessionConfig, AudioSessionSessionTelemetry, and sanitized session exceptions.
"""

from datetime import datetime

from pydantic import BaseModel, Field, field_validator

from app.audio.models import AudioSessionState


class AudioSessionConfig(BaseModel):
    """Configuration parameters for AudioSessionManager orchestration."""

    max_concurrent_sessions: int = Field(
        default=5, gt=0, description="Maximum allowed concurrent audio sessions"
    )
    max_queued_chunks: int = Field(
        default=100, gt=0, description="Maximum bounded capture chunk queue size"
    )
    max_speech_duration_ms: float = Field(
        default=30000.0, gt=0.0, description="Maximum single speech segment duration in ms"
    )
    max_pending_synthesis_chunks: int = Field(
        default=500, gt=0, description="Maximum bounded synthesis chunk queue size"
    )
    max_background_tasks: int = Field(
        default=20, gt=0, description="Maximum tracked background asyncio tasks"
    )
    shutdown_timeout_ms: float = Field(
        default=5000.0, gt=0.0, description="Session shutdown timeout limit in ms"
    )
    backpressure_policy: str = Field(
        default="DROP_OLDEST", description="Queue overflow backpressure policy"
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


class AudioSessionTelemetry(BaseModel):
    """Telemetry record captured for audio sessions (excludes raw audio bytes and transcript text)."""

    session_id: str = Field(description="Unique session string identifier")
    correlation_id: str = Field(description="Tracing correlation identifier")
    state: AudioSessionState = Field(description="Current session state")
    duration_ms: float = Field(ge=0.0, description="Session total active duration in ms")
    chunks_processed: int = Field(ge=0, description="Count of AudioChunks processed")
    chunks_dropped: int = Field(
        ge=0, description="Count of AudioChunks dropped due to backpressure"
    )
    speech_segments: int = Field(ge=0, description="Count of completed speech segments")
    transcripts_generated: int = Field(ge=0, description="Count of generated transcripts")
    tts_requests: int = Field(ge=0, description="Count of TTS synthesis requests executed")
    errors: int = Field(ge=0, description="Count of operational session errors encountered")
    latency_ms: float = Field(ge=0.0, description="Average pipeline processing latency in ms")
    timestamp: datetime = Field(description="Timezone-aware telemetry timestamp")

    @field_validator("timestamp")
    @classmethod
    def validate_timezone(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware.")
        return v


# Audio Session Subsystem Exception Taxonomy
class AudioSessionError(Exception):
    """Base application exception for all Audio Session Orchestrator failures."""


class AudioSessionConfigurationError(AudioSessionError):
    """Raised when AudioSessionConfig parameters are invalid."""


class AudioSessionLimitError(AudioSessionError):
    """Raised when session limits (e.g. max_concurrent_sessions) are exceeded."""


class AudioSessionStateError(AudioSessionError):
    """Raised when illegal session state transitions are attempted."""


class AudioSessionTimeoutError(AudioSessionError):
    """Raised when session startup, execution, or shutdown exceeds configured timeouts."""


class AudioSessionProcessingError(AudioSessionError):
    """Raised when errors occur during pipeline component coordination."""
