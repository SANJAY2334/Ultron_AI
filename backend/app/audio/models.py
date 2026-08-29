"""Audio Automation Domain Models & State Machine (Phase 4C.1).

Defines strongly typed, framework-agnostic Pydantic v2 domain models for AudioFormat,
AudioChunk, AudioStreamConfig, VoiceActivityEvent, TranscriptSegment, Transcript,
SpeechSynthesisRequest/Result, WakeWordDetection, AudioSessionState transitions,
AudioPrivacy, and AudioTelemetry metrics.
"""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator, model_validator


class AudioEncoding(StrEnum):
    """Supported PCM audio sample encoding representations."""

    PCM_S16LE = "pcm_s16le"
    PCM_S24LE = "pcm_s24le"
    PCM_S32LE = "pcm_s32le"
    PCM_F32LE = "pcm_f32le"


class AudioPrivacy(StrEnum):
    """Audio data retention and privacy classification."""

    EPHEMERAL = "EPHEMERAL"
    SESSION = "SESSION"
    PERSISTENT = "PERSISTENT"


class AudioFormat(BaseModel):
    """Characteristics of PCM raw audio data."""

    sample_rate: int = Field(default=16000, gt=0, description="Sampling rate in Hz (e.g. 16000)")
    channels: int = Field(
        default=1, ge=1, description="Number of audio channels (1=mono, 2=stereo)"
    )
    sample_width: int = Field(
        default=2, gt=0, description="Sample byte width (2 for s16le, 4 for f32le)"
    )
    encoding: AudioEncoding = Field(
        default=AudioEncoding.PCM_S16LE, description="PCM audio sample encoding"
    )

    @field_validator("sample_rate")
    @classmethod
    def validate_sample_rate(cls, v: int) -> int:
        valid_rates = {8000, 11025, 16000, 22050, 24000, 32000, 44100, 48000, 96000}
        if v not in valid_rates:
            raise ValueError(
                f"Unsupported sample_rate {v}Hz. Must be one of standard production rates: {sorted(valid_rates)}"
            )
        return v


class AudioChunk(BaseModel):
    """Single frame/slice of captured audio payload with timing metadata."""

    chunk_id: str = Field(min_length=1, description="Unique chunk identifier")
    sequence_number: int = Field(ge=0, description="Monotonically increasing sequence number")
    timestamp: datetime = Field(description="Timezone-aware capture timestamp")
    duration_ms: float = Field(gt=0, description="Chunk duration in milliseconds")
    audio_format: AudioFormat = Field(description="PCM format of payload")
    payload: bytes = Field(min_length=1, description="Raw binary audio bytes")
    correlation_id: str | None = Field(default=None, description="Tracing correlation ID")
    session_id: str | None = Field(default=None, description="Active session ID")
    privacy_level: AudioPrivacy = Field(
        default=AudioPrivacy.EPHEMERAL, description="Privacy retention policy"
    )
    retention_policy: str = Field(default="do_not_persist", description="Retention rule flag")

    @field_validator("timestamp")
    @classmethod
    def validate_timezone(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware.")
        return v

    def __repr__(self) -> str:
        """Custom repr preventing binary payload leak in logs."""
        return (
            f"AudioChunk(chunk_id='{self.chunk_id}', seq={self.sequence_number}, "
            f"duration={self.duration_ms}ms, bytes={len(self.payload)}, privacy={self.privacy_level})"
        )


class AudioStreamConfig(BaseModel):
    """Streaming configuration parameters for real-time speech processing."""

    sample_rate: int = Field(default=16000, gt=0, description="Sample rate in Hz")
    channels: int = Field(default=1, ge=1, description="Channel count")
    chunk_duration_ms: float = Field(
        default=30.0, gt=0.0, le=1000.0, description="Chunk duration in ms"
    )
    buffer_size: int = Field(default=4096, gt=0, description="Stream buffer size in bytes")
    max_latency_ms: float = Field(
        default=200.0, gt=0.0, description="Maximum allowed stream latency in ms"
    )

    @field_validator("sample_rate")
    @classmethod
    def validate_stream_sample_rate(cls, v: int) -> int:
        valid_rates = {8000, 16000, 24000, 44100, 48000}
        if v not in valid_rates:
            raise ValueError(f"Stream sample_rate {v} is invalid. Supported: {sorted(valid_rates)}")
        return v


class VoiceActivityState(StrEnum):
    """States of Voice Activity Detection (VAD)."""

    SILENCE = "SILENCE"
    SPEECH_START = "SPEECH_START"
    SPEAKING = "SPEAKING"
    SPEECH_END = "SPEECH_END"
    UNKNOWN = "UNKNOWN"


class VoiceActivityEvent(BaseModel):
    """VAD state transition event payload."""

    event_id: str = Field(min_length=1, description="Event identifier")
    state: VoiceActivityState = Field(description="VAD state")
    timestamp: datetime = Field(description="Timezone-aware event timestamp")
    confidence: float = Field(ge=0.0, le=1.0, description="Detection confidence score")
    chunk_id: str | None = Field(default=None, description="Triggering chunk ID")
    session_id: str | None = Field(default=None, description="Session ID")
    correlation_id: str | None = Field(default=None, description="Tracing correlation ID")

    @field_validator("timestamp")
    @classmethod
    def validate_timezone(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware.")
        return v


class TranscriptSegment(BaseModel):
    """Recognized speech segment payload."""

    segment_id: str = Field(min_length=1, description="Segment identifier")
    text: str = Field(description="Recognized text string")
    start_ms: float = Field(ge=0.0, description="Segment start offset in ms")
    end_ms: float = Field(ge=0.0, description="Segment end offset in ms")
    confidence: float = Field(ge=0.0, le=1.0, description="Recognition confidence")
    is_final: bool = Field(default=False, description="True if segment is final")
    language: str = Field(default="en", description="Language code string")

    @model_validator(mode="after")
    def validate_segment_times_and_text(self) -> "TranscriptSegment":
        if self.end_ms < self.start_ms:
            raise ValueError("end_ms must be greater than or equal to start_ms.")
        if self.is_final and not self.text.strip():
            raise ValueError("Completed final transcript segment cannot be empty text.")
        return self


class Transcript(BaseModel):
    """Complete speech-to-text transcript object."""

    transcript_id: str = Field(min_length=1, description="Transcript identifier")
    segments: list[TranscriptSegment] = Field(default_factory=list, description="Segments list")
    full_text: str = Field(description="Aggregated full transcript text")
    language: str = Field(default="en", description="Primary language code")
    confidence: float = Field(ge=0.0, le=1.0, description="Overall confidence score")
    is_final: bool = Field(default=False, description="True if transcript is final")
    session_id: str | None = Field(default=None, description="Session ID")
    correlation_id: str | None = Field(default=None, description="Correlation ID")


class SpeechSynthesisRequest(BaseModel):
    """Request payload for Text-to-Speech synthesis."""

    request_id: str = Field(min_length=1, description="Unique request ID")
    text: str = Field(min_length=1, max_length=5000, description="Text string to synthesize")
    voice: str = Field(min_length=1, description="Voice identifier string")
    language: str = Field(default="en-US", description="Target language code")
    speed: float = Field(default=1.0, ge=0.25, le=4.0, description="Speech rate multiplier")
    pitch: float = Field(
        default=0.0, ge=-20.0, le=20.0, description="Pitch modification in Hz/semitones"
    )
    format: AudioEncoding = Field(default=AudioEncoding.PCM_S16LE, description="Output PCM format")
    sample_rate: int = Field(default=24000, gt=0, description="Output sample rate")
    session_id: str | None = Field(default=None, description="Session ID")
    correlation_id: str | None = Field(default=None, description="Correlation ID")


class SpeechSynthesisResult(BaseModel):
    """Result of TTS speech synthesis."""

    request_id: str = Field(min_length=1, description="Matching request ID")
    audio_format: AudioFormat = Field(description="Synthesized audio format")
    duration_ms: float = Field(gt=0.0, description="Duration in milliseconds")
    audio_payload: bytes = Field(min_length=1, description="Synthesized binary audio bytes")
    session_id: str | None = Field(default=None, description="Session ID")
    correlation_id: str | None = Field(default=None, description="Correlation ID")

    def __repr__(self) -> str:
        return (
            f"SpeechSynthesisResult(request_id='{self.request_id}', "
            f"duration={self.duration_ms}ms, bytes={len(self.audio_payload)})"
        )


class WakeWordDetection(BaseModel):
    """Wake word detection event payload."""

    detection_id: str = Field(min_length=1, description="Detection identifier")
    wake_word: str = Field(min_length=1, description="Triggered wake word string")
    confidence: float = Field(ge=0.0, le=1.0, description="Detection confidence score")
    timestamp: datetime = Field(description="Timezone-aware detection timestamp")
    session_id: str | None = Field(default=None, description="Session ID")
    correlation_id: str | None = Field(default=None, description="Correlation ID")

    @field_validator("timestamp")
    @classmethod
    def validate_timezone(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware.")
        return v


class AudioSessionState(StrEnum):
    """States of full-duplex audio voice interaction session state machine."""

    IDLE = "IDLE"
    LISTENING = "LISTENING"
    SPEECH_DETECTED = "SPEECH_DETECTED"
    TRANSCRIBING = "TRANSCRIBING"
    THINKING = "THINKING"
    SYNTHESIZING = "SYNTHESIZING"
    PLAYING = "PLAYING"
    CANCELLING = "CANCELLING"
    ERROR = "ERROR"


class PlaybackState(StrEnum):
    """Lifecycle states of audio playback output devices."""

    IDLE = "IDLE"
    PLAYING = "PLAYING"
    PAUSED = "PAUSED"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


class InvalidAudioStateTransitionError(Exception):
    """Raised when an illegal AudioSessionState transition is attempted."""


# Valid state transitions lookup table
VALID_AUDIO_STATE_TRANSITIONS: dict[AudioSessionState, set[AudioSessionState]] = {
    AudioSessionState.IDLE: {
        AudioSessionState.LISTENING,
        AudioSessionState.CANCELLING,
        AudioSessionState.ERROR,
    },
    AudioSessionState.LISTENING: {
        AudioSessionState.SPEECH_DETECTED,
        AudioSessionState.IDLE,
        AudioSessionState.CANCELLING,
        AudioSessionState.ERROR,
    },
    AudioSessionState.SPEECH_DETECTED: {
        AudioSessionState.TRANSCRIBING,
        AudioSessionState.LISTENING,
        AudioSessionState.CANCELLING,
        AudioSessionState.ERROR,
    },
    AudioSessionState.TRANSCRIBING: {
        AudioSessionState.THINKING,
        AudioSessionState.LISTENING,
        AudioSessionState.CANCELLING,
        AudioSessionState.ERROR,
    },
    AudioSessionState.THINKING: {
        AudioSessionState.SYNTHESIZING,
        AudioSessionState.IDLE,
        AudioSessionState.CANCELLING,
        AudioSessionState.ERROR,
    },
    AudioSessionState.SYNTHESIZING: {
        AudioSessionState.PLAYING,
        AudioSessionState.IDLE,
        AudioSessionState.CANCELLING,
        AudioSessionState.ERROR,
    },
    AudioSessionState.PLAYING: {
        AudioSessionState.LISTENING,
        AudioSessionState.IDLE,
        AudioSessionState.CANCELLING,
        AudioSessionState.ERROR,
    },
    AudioSessionState.CANCELLING: {
        AudioSessionState.IDLE,
        AudioSessionState.ERROR,
    },
    AudioSessionState.ERROR: {
        AudioSessionState.IDLE,
    },
}


# Valid playback state transitions lookup table
VALID_PLAYBACK_STATE_TRANSITIONS: dict[PlaybackState, set[PlaybackState]] = {
    PlaybackState.IDLE: {PlaybackState.PLAYING, PlaybackState.ERROR},
    PlaybackState.PLAYING: {
        PlaybackState.PAUSED,
        PlaybackState.STOPPING,
        PlaybackState.STOPPED,
        PlaybackState.IDLE,
        PlaybackState.ERROR,
    },
    PlaybackState.PAUSED: {
        PlaybackState.PLAYING,
        PlaybackState.STOPPING,
        PlaybackState.STOPPED,
        PlaybackState.IDLE,
        PlaybackState.ERROR,
    },
    PlaybackState.STOPPING: {PlaybackState.STOPPED, PlaybackState.IDLE, PlaybackState.ERROR},
    PlaybackState.STOPPED: {PlaybackState.PLAYING, PlaybackState.IDLE, PlaybackState.ERROR},
    PlaybackState.ERROR: {PlaybackState.IDLE},
}


class AudioSessionStateMachine:
    """State machine governing valid full-duplex AudioSessionState transitions."""

    def __init__(
        self, initial_state: AudioSessionState = AudioSessionState.IDLE, session_id: str = "default"
    ) -> None:
        """Initializes AudioSessionStateMachine."""
        self._state = initial_state
        self.session_id = session_id

    @property
    def current_state(self) -> AudioSessionState:
        """Returns active AudioSessionState."""
        return self._state

    def transition(self, target_state: AudioSessionState) -> AudioSessionState:
        """Transitions state machine to target state if legal.

        Args:
            target_state: Desired AudioSessionState.

        Returns:
            AudioSessionState: New active state.

        Raises:
            InvalidAudioStateTransitionError: If transition is illegal.
        """
        allowed = VALID_AUDIO_STATE_TRANSITIONS.get(self._state, set())
        if target_state not in allowed:
            raise InvalidAudioStateTransitionError(
                f"Illegal Audio State Transition: Cannot transition from {self._state} to {target_state}."
            )
        self._state = target_state
        return self._state

    def cancel(self) -> AudioSessionState:
        """Explicitly cancels active session state back to CANCELLING or IDLE."""
        if self._state not in (AudioSessionState.IDLE, AudioSessionState.ERROR):
            self._state = AudioSessionState.CANCELLING
            self._state = AudioSessionState.IDLE
        return self._state


class AudioTelemetry(BaseModel):
    """Telemetry metrics tracking perceived end-to-end voice latency."""

    capture_latency_ms: float = Field(default=0.0, ge=0.0, description="Capture latency in ms")
    vad_latency_ms: float = Field(default=0.0, ge=0.0, description="VAD latency in ms")
    stt_latency_ms: float = Field(default=0.0, ge=0.0, description="STT latency in ms")
    planner_latency_ms: float = Field(default=0.0, ge=0.0, description="Planner latency in ms")
    tts_latency_ms: float = Field(default=0.0, ge=0.0, description="TTS latency in ms")
    total_latency_ms: float = Field(default=0.0, ge=0.0, description="Total pipeline latency in ms")
    timestamp: datetime = Field(description="Timezone-aware timestamp")
    session_id: str | None = Field(default=None, description="Session ID")
    correlation_id: str | None = Field(default=None, description="Correlation ID")

    @field_validator("timestamp")
    @classmethod
    def validate_timezone(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware.")
        return v
