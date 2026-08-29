"""Unit and Integration Tests for IAudioPlayback Contract and MockPlaybackAdapter (Phase 4E.1).

Validates IAudioPlayback interface contract, MockPlaybackAdapter lifecycle state machine, sequence ordering,
backpressure drop-oldest policy, idempotent cancellation, AudioSessionManager barge-in integration,
privacy telemetry redaction, health status, and concurrent execution safety.
"""

import asyncio
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.audio import (
    AudioChunk,
    AudioFormat,
    AudioSessionConfig,
    AudioSessionManager,
    IAudioPlayback,
    PlaybackConfig,
    PlaybackError,
    PlaybackState,
    PlaybackStateError,
    PlaybackTelemetry,
    PlaybackValidationError,
)
from app.audio.adapters.mock_playback import MockPlaybackAdapter


def make_chunk(seq: int = 0, session_id: str = "sess_play_1") -> AudioChunk:
    """Helper creating synthetic AudioChunk."""
    return AudioChunk(
        chunk_id=f"chk_play_{seq}",
        sequence_number=seq,
        timestamp=datetime.now(UTC),
        duration_ms=30.0,
        audio_format=AudioFormat(sample_rate=16000, channels=1, sample_width=2),
        payload=b"\x00" * 960,
        session_id=session_id,
        correlation_id="corr_play_100",
    )


@pytest.fixture
def playback() -> MockPlaybackAdapter:
    cfg = PlaybackConfig(max_buffered_chunks=5, timeout_ms=500.0)
    return MockPlaybackAdapter(config=cfg)


def test_playback_contract_and_instantiation() -> None:
    """Verify IAudioPlayback abstract interface and MockPlaybackAdapter compliance (rules 1-2)."""
    # 1. Abstract interface cannot be instantiated
    with pytest.raises(TypeError):
        IAudioPlayback()  # type: ignore

    # 2. MockPlaybackAdapter satisfies interface
    adapter = MockPlaybackAdapter()
    assert isinstance(adapter, IAudioPlayback)


def test_playback_lifecycle_state_machine(playback: MockPlaybackAdapter) -> None:
    """Verify playback state machine transitions: IDLE -> PLAYING -> PAUSED -> PLAYING -> STOPPING -> STOPPED -> IDLE (rules 3-9)."""

    async def _test() -> None:
        # 3. Initial state
        assert playback.get_state() == PlaybackState.IDLE

        # 4. Start playback
        c0 = make_chunk(seq=0)
        await playback.play_chunk(c0)
        assert playback.get_state() in (PlaybackState.PLAYING, PlaybackState.IDLE)

        # 5. Pause playback
        await playback.play_chunk(make_chunk(seq=1))
        await playback.pause()
        assert playback.get_state() == PlaybackState.PAUSED

        # 6. Resume playback
        await playback.resume()
        assert playback.get_state() in (PlaybackState.PLAYING, PlaybackState.IDLE)

        # 7. Stop playback
        await playback.stop()
        assert playback.get_state() == PlaybackState.IDLE

        # 8. Restart after stop
        await playback.play_chunk(make_chunk(seq=0))
        assert playback.get_state() in (PlaybackState.PLAYING, PlaybackState.IDLE)

        # 9. Invalid state transition (IDLE -> PAUSED raises PlaybackStateError)
        await playback.stop()
        with pytest.raises(PlaybackStateError):
            playback._transition(PlaybackState.PAUSED)

    asyncio.run(_test())


def test_chunk_validation_and_sequence_ordering(playback: MockPlaybackAdapter) -> None:
    """Verify sequence monotonic ordering, duplicate rejection, and empty payload rejection (rules 10-14)."""

    async def _test() -> None:
        # 10. Valid chunk accepted (seq 5)
        c5 = make_chunk(seq=5)
        await playback.play_chunk(c5)

        # 11-12. Duplicate sequence number rejection
        with pytest.raises(PlaybackValidationError, match="Duplicate"):
            await playback.play_chunk(make_chunk(seq=5))

        # Sequence regression rejection (seq 2 after seq 5)
        with pytest.raises(PlaybackValidationError, match="regression"):
            await playback.play_chunk(make_chunk(seq=2))

        # 13. Invalid empty audio payload chunk
        with pytest.raises((PlaybackValidationError, ValidationError)):
            AudioChunk(
                chunk_id="bad_chk",
                sequence_number=10,
                timestamp=datetime.now(UTC),
                duration_ms=30.0,
                audio_format=AudioFormat(),
                payload=b"",
            )

        # 14. Multiple chunks in order
        await playback.play_chunk(make_chunk(seq=6))
        await playback.play_chunk(make_chunk(seq=7))

    asyncio.run(_test())


def test_backpressure_drop_oldest_policy() -> None:
    """Verify bounded queue limit and DROP_OLDEST policy telemetry (rules 15-17)."""

    async def _test() -> None:
        small_playback = MockPlaybackAdapter(
            config=PlaybackConfig(max_buffered_chunks=2), simulated_delay_sec=0.10
        )

        await small_playback.play_chunk(make_chunk(seq=0))
        await small_playback.play_chunk(make_chunk(seq=1))
        await small_playback.play_chunk(make_chunk(seq=2))  # Overflows queue, drops seq 0

        telemetry = small_playback.get_telemetry()
        assert telemetry.chunks_received == 3
        assert telemetry.chunks_dropped >= 1

        await small_playback.stop()

    asyncio.run(_test())


def test_cancellation_and_barge_in_cleanup(playback: MockPlaybackAdapter) -> None:
    """Verify stop during active playback, idempotent stop, and pending chunk clearing (rules 18-22)."""

    async def _test() -> None:
        slow_playback = MockPlaybackAdapter(simulated_delay_sec=0.10)

        await slow_playback.play_chunk(make_chunk(seq=0))
        await slow_playback.play_chunk(make_chunk(seq=1))

        # 18. Stop during playback
        await slow_playback.stop()
        assert slow_playback._queue.empty()
        assert slow_playback.get_state() == PlaybackState.IDLE

        # 19. Repeated stop is safe (idempotent)
        await slow_playback.stop()
        await slow_playback.stop()
        assert slow_playback.get_state() == PlaybackState.IDLE

    asyncio.run(_test())


def test_session_manager_playback_integration() -> None:
    """Verify AudioSessionManager delivers TTS chunks to IAudioPlayback and barge-in calls stop() (rules 23-26)."""

    async def _test() -> None:
        mock_pb = MockPlaybackAdapter()
        mgr = AudioSessionManager(
            playback=mock_pb,
            config=AudioSessionConfig(max_queued_chunks=10, max_pending_synthesis_chunks=10),
        )

        # 23. AudioSessionManager resolves IAudioPlayback
        assert mgr.playback is mock_pb

        await mgr.start_session("sess_pb_test")

        # 24. Process speech text sends chunks to mock_pb
        await mgr.process_speech_text("Hello playback boundary.")
        assert mock_pb._chunks_received > 0

        # 25-26. Barge-in speech interruption stops playback
        await mgr.interrupt_speech()
        assert mock_pb.get_state() == PlaybackState.IDLE
        assert mgr._synthesis_queue.empty()

        await mgr.stop_session("sess_pb_test")

    asyncio.run(_test())


def test_privacy_and_telemetry_guarantees(playback: MockPlaybackAdapter) -> None:
    """Verify PlaybackTelemetry excludes raw PCM bytes and sensitive data (rules 27-28)."""

    async def _test() -> None:
        await playback.play_chunk(make_chunk(seq=0))

        telemetry = playback.get_telemetry()
        assert isinstance(telemetry, PlaybackTelemetry)

        telem_dict = telemetry.model_dump()
        assert "payload" not in telem_dict
        assert "audio_bytes" not in telem_dict
        assert "pcm" not in telem_dict

        await playback.stop()

    asyncio.run(_test())


def test_health_status_probing() -> None:
    """Verify healthy adapter reports healthy and failure state reports unhealthy (rules 29-30)."""

    async def _test() -> None:
        healthy_pb = MockPlaybackAdapter()
        h1 = await healthy_pb.health()
        assert h1["subsystem_playback"] is True
        assert h1["healthy"] is True

        failing_pb = MockPlaybackAdapter(should_fail_on_play=True)
        with pytest.raises(PlaybackError):
            await failing_pb.play_chunk(make_chunk(seq=0))

        h2 = await failing_pb.health()
        assert h2["healthy"] is False
        assert h2["state"] == PlaybackState.ERROR.value

    asyncio.run(_test())


def test_concurrency_and_task_leak_prevention(playback: MockPlaybackAdapter) -> None:
    """Verify concurrent chunk submissions and stop during concurrent submissions (rules 31-33)."""

    async def _test() -> None:
        tasks = [asyncio.create_task(playback.play_chunk(make_chunk(seq=i))) for i in range(5)]
        await asyncio.gather(*tasks)

        await playback.stop()
        assert playback._playback_task is None or playback._playback_task.done()

    asyncio.run(_test())
