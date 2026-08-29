"""Mock Audio Playback Adapter (Phase 4E.1).

Deterministic, in-memory implementation of IAudioPlayback for testing audio output queue management,
sequence validation, backpressure drop policy, state machine transitions, barge-in cancellation, and telemetry.
"""

import asyncio
import logging
import time
from datetime import UTC, datetime
from typing import Any

from app.audio.base import IAudioPlayback
from app.audio.models import (
    VALID_PLAYBACK_STATE_TRANSITIONS,
    AudioChunk,
    PlaybackState,
)
from app.audio.playback import (
    PlaybackConfig,
    PlaybackError,
    PlaybackStateError,
    PlaybackTelemetry,
    PlaybackValidationError,
)

logger = logging.getLogger(__name__)


class MockPlaybackAdapter(IAudioPlayback):
    """Deterministic Mock implementation of IAudioPlayback."""

    def __init__(
        self,
        config: PlaybackConfig | None = None,
        simulated_delay_sec: float = 0.0,
        should_fail_on_play: bool = False,
    ) -> None:
        """Initializes MockPlaybackAdapter.

        Args:
            config: Optional PlaybackConfig configuration instance.
            simulated_delay_sec: Processing delay per chunk in seconds for testing.
            should_fail_on_play: True if play_chunk should simulate a failure.
        """
        self.config = config or PlaybackConfig()
        self._simulated_delay_sec = simulated_delay_sec
        self._should_fail_on_play = should_fail_on_play

        self._state = PlaybackState.IDLE
        self._queue: asyncio.Queue[AudioChunk] = asyncio.Queue(
            maxsize=self.config.max_buffered_chunks
        )
        self._last_sequence_number: int | None = None
        self._playback_task: asyncio.Task[Any] | None = None

        # Metrics tracking
        self._chunks_received = 0
        self._chunks_played = 0
        self._chunks_dropped = 0
        self._total_duration_ms = 0.0
        self._session_id: str | None = None
        self._correlation_id: str | None = None

    def get_state(self) -> PlaybackState:
        """Returns current active PlaybackState."""
        return self._state

    def _transition(self, target_state: PlaybackState) -> PlaybackState:
        """Validates and updates playback state machine."""
        allowed = VALID_PLAYBACK_STATE_TRANSITIONS.get(self._state, set())
        if target_state not in allowed:
            raise PlaybackStateError(
                f"Illegal PlaybackState transition: Cannot transition from {self._state} to {target_state}."
            )
        self._state = target_state
        return self._state

    async def health(self) -> dict[str, Any]:
        """Probes health and readiness status of playback adapter."""
        healthy = self._state != PlaybackState.ERROR
        return {
            "subsystem_playback": True,
            "healthy": healthy,
            "state": self._state.value,
            "queue_size": self._queue.qsize(),
            "chunks_played": self._chunks_played,
            "chunks_dropped": self._chunks_dropped,
        }

    def _validate_chunk(self, chunk: AudioChunk) -> None:
        """Validates AudioChunk payload, sequence numbers, and format rules."""
        if not chunk.payload:
            raise PlaybackValidationError("Cannot play empty AudioChunk payload.")

        if self._last_sequence_number is not None:
            if chunk.sequence_number == self._last_sequence_number:
                raise PlaybackValidationError(
                    f"Duplicate AudioChunk sequence number detected: seq={chunk.sequence_number}."
                )
            if chunk.sequence_number < self._last_sequence_number:
                raise PlaybackValidationError(
                    f"AudioChunk sequence regression detected: seq={chunk.sequence_number} < last_seq={self._last_sequence_number}."
                )

        self._last_sequence_number = chunk.sequence_number

    async def play_chunk(self, chunk: AudioChunk) -> None:
        """Enqueues and plays an AudioChunk frame with backpressure overflow handling."""
        if self._should_fail_on_play:
            self._transition(PlaybackState.ERROR)
            raise PlaybackError("Simulated playback device hardware failure.")

        self._validate_chunk(chunk)
        self._chunks_received += 1
        self._session_id = chunk.session_id
        self._correlation_id = chunk.correlation_id

        # Enforce bounded queue DROP_OLDEST backpressure policy
        if self._queue.full():
            try:
                self._queue.get_nowait()
                self._chunks_dropped += 1
            except asyncio.QueueEmpty:
                pass

        self._queue.put_nowait(chunk)

        if self._state in (PlaybackState.IDLE, PlaybackState.STOPPED):
            self._transition(PlaybackState.PLAYING)

        if self._playback_task is None or self._playback_task.done():
            self._playback_task = asyncio.create_task(self._playback_loop())

    async def pause(self) -> None:
        """Pauses active audio playback stream."""
        if self._state == PlaybackState.PLAYING:
            self._transition(PlaybackState.PAUSED)
            logger.info("MockPlaybackAdapter paused.")

    async def resume(self) -> None:
        """Resumes paused audio playback stream."""
        if self._state == PlaybackState.PAUSED:
            self._transition(PlaybackState.PLAYING)
            logger.info("MockPlaybackAdapter resumed.")

    async def stop(self) -> None:
        """Stops audio playback idempotently, clears pending queue, and releases tasks."""
        if self._state in (PlaybackState.STOPPED, PlaybackState.STOPPING, PlaybackState.IDLE):
            # Idempotent stop call
            return

        try:
            self._transition(PlaybackState.STOPPING)
        except PlaybackStateError:
            self._state = PlaybackState.STOPPING

        if self._playback_task and not self._playback_task.done():
            self._playback_task.cancel()
            try:
                await self._playback_task
            except asyncio.CancelledError:
                pass
            self._playback_task = None

        # Clear pending queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        self._last_sequence_number = None
        self._transition(PlaybackState.STOPPED)
        self._transition(PlaybackState.IDLE)
        logger.info("MockPlaybackAdapter stopped cleanly.")

    async def _playback_loop(self) -> None:
        """Background async loop pulling chunks from queue and executing playback."""
        try:
            while not self._queue.empty():
                if self._state == PlaybackState.PAUSED:
                    await asyncio.sleep(0.05)
                    continue

                if self._state != PlaybackState.PLAYING:
                    break

                chunk = await self._queue.get()
                start_time = time.perf_counter()

                if self._simulated_delay_sec > 0:
                    await asyncio.sleep(self._simulated_delay_sec)

                self._chunks_played += 1
                self._total_duration_ms += chunk.duration_ms
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0

                logger.debug(
                    f"MockPlaybackAdapter played chunk seq={chunk.sequence_number} ({chunk.duration_ms}ms) in {elapsed_ms:.1f}ms."
                )

            if self._state == PlaybackState.PLAYING and self._queue.empty():
                self._transition(PlaybackState.IDLE)

        except asyncio.CancelledError:
            logger.debug("MockPlaybackAdapter loop cancelled.")
            raise
        except Exception as exc:
            logger.error(f"MockPlaybackAdapter error in playback loop: {exc}")
            self._state = PlaybackState.ERROR

    def get_telemetry(self) -> PlaybackTelemetry:
        """Returns privacy-preserving PlaybackTelemetry without raw audio bytes."""
        return PlaybackTelemetry(
            state=self._state,
            chunks_received=self._chunks_received,
            chunks_played=self._chunks_played,
            chunks_dropped=self._chunks_dropped,
            total_duration_ms=self._total_duration_ms,
            latency_ms=1.0,
            timestamp=datetime.now(UTC),
            session_id=self._session_id,
            correlation_id=self._correlation_id,
        )
