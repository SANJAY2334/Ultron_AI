"""VAD Speech Segmenter and Audio Buffer Aggregator (Phase 4H.4).

Aggregates streaming 30ms AudioChunks from physical microphone capture into meaningful
speech segments using Voice Activity Detection (VAD) hysteresis for offline STT inference.
"""

import logging
from collections.abc import AsyncIterable

from app.audio.adapters.vad import VADAdapter
from app.audio.base import IVoiceActivityDetector
from app.audio.models import AudioChunk, VoiceActivityEvent, VoiceActivityState

logger = logging.getLogger(__name__)


class SpeechSegmenter:
    """Aggregates streaming AudioChunks into discrete speech utterance segments using VAD."""

    def __init__(
        self,
        vad_detector: IVoiceActivityDetector | None = None,
        silence_hysteresis_chunks: int = 15,  # ~450ms of silence @ 30ms/chunk
        max_segment_duration_ms: float = 15000.0,  # Max 15 seconds per utterance
        min_speech_chunks: int = 3,  # Min 90ms of speech to discard transient clicks
    ) -> None:
        """Initializes SpeechSegmenter.

        Args:
            vad_detector: IVoiceActivityDetector instance (defaults to VADAdapter).
            silence_hysteresis_chunks: Consecutive silence chunks required to finalize an utterance.
            max_segment_duration_ms: Maximum duration before forcing utterance segmentation.
            min_speech_chunks: Minimum speech chunks required to trigger a valid utterance.
        """
        self.vad = vad_detector or VADAdapter()
        self.silence_hysteresis_chunks = silence_hysteresis_chunks
        self.max_segment_duration_ms = max_segment_duration_ms
        self.min_speech_chunks = min_speech_chunks

        self._current_segment: list[AudioChunk] = []
        self._is_speaking = False
        self._consecutive_silence = 0
        self._speech_chunks_count = 0

    async def process_chunk(self, chunk: AudioChunk) -> list[AudioChunk] | None:
        """Processes a single AudioChunk and returns completed speech segment if finalized.

        Args:
            chunk: Input 30ms AudioChunk.

        Returns:
            list[AudioChunk] | None: Completed segment if speech utterance ended, else None.
        """
        vad_event: VoiceActivityEvent = await self.vad.process(chunk)
        state = vad_event.state

        if state in (VoiceActivityState.SPEAKING, VoiceActivityState.SPEECH_START):
            self._is_speaking = True
            self._consecutive_silence = 0
            self._speech_chunks_count += 1
            self._current_segment.append(chunk)

            # Check max duration boundary
            current_duration = sum(c.duration_ms for c in self._current_segment)
            if current_duration >= self.max_segment_duration_ms:
                logger.debug(f"Speech segment exceeded max duration ({current_duration:.0f}ms). Finalizing.")
                return self._finalize_segment()

        elif self._is_speaking:
            # Silence observed during an active speech utterance
            self._consecutive_silence += 1
            self._current_segment.append(chunk)

            if self._consecutive_silence >= self.silence_hysteresis_chunks or state == VoiceActivityState.SPEECH_END:
                logger.debug(
                    f"Utterance finalized: {len(self._current_segment)} chunks "
                    f"({self._speech_chunks_count} active speech)."
                )
                return self._finalize_segment()

        return None

    def _finalize_segment(self) -> list[AudioChunk] | None:
        """Finalizes and returns current speech segment if minimum threshold met."""
        if not self._current_segment:
            self._reset_state()
            return None

        if self._speech_chunks_count >= self.min_speech_chunks:
            completed_segment = list(self._current_segment)
            self._reset_state()
            return completed_segment

        # Discard transient noise/click
        self._reset_state()
        return None

    def _reset_state(self) -> None:
        """Clears in-memory speech segment buffers."""
        self._current_segment.clear()
        self._is_speaking = False
        self._consecutive_silence = 0
        self._speech_chunks_count = 0

    def reset(self) -> None:
        """Resets segmenter and clears internal state."""
        self._reset_state()

    async def segment_stream(self, chunk_stream: AsyncIterable[AudioChunk]) -> AsyncIterable[list[AudioChunk]]:
        """Asynchronously converts a stream of AudioChunks into completed speech segments."""
        async for chunk in chunk_stream:
            segment = await self.process_chunk(chunk)
            if segment:
                yield segment

        # Flush any remaining segment at end of stream
        final = self._finalize_segment()
        if final:
            yield final
