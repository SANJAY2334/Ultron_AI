"""Abstract Provider-Agnostic Audio Domain Interfaces (Phase 4C.1).

Defines strongly typed abstract contracts for audio capture, VAD, STT, TTS, wake-word detection,
and audio session state management without depending on concrete SDKs or audio libraries.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterable
from typing import Any

from app.audio.models import (
    AudioChunk,
    AudioSessionState,
    PlaybackState,
    SpeechSynthesisRequest,
    SpeechSynthesisResult,
    Transcript,
    TranscriptSegment,
    VoiceActivityEvent,
    WakeWordDetection,
)


class IAudioCapture(ABC):
    """Abstract interface for audio capture hardware or stream sources."""

    @abstractmethod
    async def start(self) -> None:
        """Starts audio capture stream."""

    @abstractmethod
    async def stop(self) -> None:
        """Stops audio capture stream."""

    @abstractmethod
    async def read_chunk(self) -> AudioChunk:
        """Reads a single AudioChunk frame from the capture stream.

        Returns:
            AudioChunk: Captured audio chunk payload and metadata.
        """

    @abstractmethod
    async def health(self) -> dict[str, bool]:
        """Probes health and readiness status of capture device."""


class IVoiceActivityDetector(ABC):
    """Abstract interface for Voice Activity Detection (VAD) algorithms."""

    @abstractmethod
    async def process(self, chunk: AudioChunk) -> VoiceActivityEvent:
        """Processes an AudioChunk frame and detects voice activity state.

        Args:
            chunk: Input AudioChunk payload.

        Returns:
            VoiceActivityEvent: VAD state event object.
        """

    @abstractmethod
    async def reset(self) -> None:
        """Resets internal VAD state buffers."""

    @abstractmethod
    async def health(self) -> dict[str, bool]:
        """Probes health status of VAD detector engine."""


class ISpeechToTextProvider(ABC):
    """Abstract interface for Speech-to-Text (STT) transcription providers."""

    @abstractmethod
    async def transcribe(self, chunk_stream: AsyncIterable[AudioChunk]) -> Transcript:
        """Transcribes a stream of AudioChunk frames into a complete Transcript.

        Args:
            chunk_stream: Async iterable of AudioChunks.

        Returns:
            Transcript: Full transcript object.
        """

    @abstractmethod
    async def transcribe_chunk(self, chunk: AudioChunk) -> TranscriptSegment:
        """Transcribes a single AudioChunk for real-time partial streaming.

        Args:
            chunk: Single AudioChunk frame.

        Returns:
            TranscriptSegment: Partial or final transcript segment.
        """

    @abstractmethod
    async def health(self) -> dict[str, bool]:
        """Probes health status of STT engine."""

    @abstractmethod
    def capabilities(self) -> dict[str, Any]:
        """Returns STT provider capabilities metadata."""


class ITextToSpeechProvider(ABC):
    """Abstract interface for Text-to-Speech (TTS) synthesis providers."""

    @abstractmethod
    async def synthesize(self, request: SpeechSynthesisRequest) -> SpeechSynthesisResult:
        """Synthesizes text input into a complete audio speech payload.

        Args:
            request: SpeechSynthesisRequest payload.

        Returns:
            SpeechSynthesisResult: Complete synthesized audio result.
        """

    @abstractmethod
    def stream(self, request: SpeechSynthesisRequest) -> AsyncIterable[bytes]:
        """Streams synthesized PCM audio chunks asynchronously.

        Args:
            request: SpeechSynthesisRequest payload.

        Returns:
            AsyncIterable[bytes]: Async generator yielding binary audio chunks.
        """

    @abstractmethod
    async def health(self) -> dict[str, bool]:
        """Probes health status of TTS engine."""

    @abstractmethod
    def capabilities(self) -> dict[str, Any]:
        """Returns TTS provider capabilities metadata."""


class IWakeWordDetector(ABC):
    """Abstract interface for Wake Word Detection engines."""

    @abstractmethod
    async def detect(self, chunk: AudioChunk) -> WakeWordDetection | None:
        """Evaluates an AudioChunk for wake-word trigger activation.

        Args:
            chunk: Input AudioChunk.

        Returns:
            WakeWordDetection | None: Detection payload if triggered, else None.
        """

    @abstractmethod
    async def reset(self) -> None:
        """Resets internal wake word model state buffers."""

    @abstractmethod
    async def health(self) -> dict[str, bool]:
        """Probes health status of wake-word detector."""


class IAudioSessionManager(ABC):
    """Abstract interface for managing full-duplex audio voice interaction sessions."""

    @abstractmethod
    async def start_session(self, session_id: str) -> None:
        """Starts a new voice interaction session."""

    @abstractmethod
    async def transition(self, new_state: AudioSessionState) -> AudioSessionState:
        """Transitions active session state machine."""

    @abstractmethod
    async def cancel(self) -> None:
        """Cancels active voice session."""

    @abstractmethod
    def current_state(self) -> AudioSessionState:
        """Returns active AudioSessionState."""


class IAudioPlayback(ABC):
    """Abstract interface for audio playback output devices or stream sinks."""

    @abstractmethod
    async def play_chunk(self, chunk: AudioChunk) -> None:
        """Enqueues and plays an AudioChunk frame."""

    @abstractmethod
    async def stop(self) -> None:
        """Stops audio playback and clears pending playback buffer."""

    @abstractmethod
    async def pause(self) -> None:
        """Pauses active audio playback."""

    @abstractmethod
    async def resume(self) -> None:
        """Resumes paused audio playback."""

    @abstractmethod
    def get_state(self) -> PlaybackState:
        """Returns active PlaybackState."""

    @abstractmethod
    async def health(self) -> dict[str, Any]:
        """Probes health and readiness status of playback device."""
