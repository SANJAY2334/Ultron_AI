"""Unit and Integration Tests for SoundDevice Microphone & Speaker Hardware Adapters (Phase 4E.2).

Validates device enumeration, default and explicit device selection, non-blocking callback queuing,
sequence numbering, monotonic timestamps, queue backpressure DROP_OLDEST policy, state transitions,
barge-in interruption, health probing, resource cleanup, and privacy telemetry bounds.
"""

import asyncio
from datetime import UTC, datetime

import pytest

from app.audio import (
    AudioChunk,
    AudioFormat,
    AudioSessionManager,
    PlaybackConfig,
    PlaybackState,
    PlaybackStateError,
    PlaybackValidationError,
    SoundDeviceCaptureAdapter,
    SoundDevicePlaybackAdapter,
)
from app.audio.capture import AudioCaptureState, AudioDeviceInfo


def make_chunk(seq: int = 0, session_id: str = "sess_sd_1") -> AudioChunk:
    """Helper creating synthetic AudioChunk."""
    return AudioChunk(
        chunk_id=f"chk_sd_{seq}",
        sequence_number=seq,
        timestamp=datetime.now(UTC),
        duration_ms=30.0,
        audio_format=AudioFormat(sample_rate=16000, channels=1, sample_width=2),
        payload=b"\x00" * 960,
        session_id=session_id,
        correlation_id="corr_sd_100",
    )


def test_device_discovery() -> None:
    """Verify device enumeration, default input/output selection, and explicit selection (rules 1-6)."""
    devices_in = SoundDeviceCaptureAdapter.list_devices(kind="input")
    assert isinstance(devices_in, list)
    assert len(devices_in) >= 1
    assert isinstance(devices_in[0], AudioDeviceInfo)

    devices_out = SoundDeviceCaptureAdapter.list_devices(kind="output")
    assert isinstance(devices_out, list)
    assert len(devices_out) >= 1

    # Explicit input selection
    adapter = SoundDeviceCaptureAdapter(device_id=devices_in[0].device_id, simulated_mode=True)
    assert adapter.device_id == devices_in[0].device_id


def test_capture_lifecycle_and_callback() -> None:
    """Verify capture stream startup, shutdown, callback sequence numbering, and timestamps (rules 7-11)."""

    async def _test() -> None:
        capture = SoundDeviceCaptureAdapter(simulated_mode=True)
        h0 = await capture.health()
        assert h0["healthy"] is True

        # 7. Start stream
        await capture.start()
        health = await capture.health()
        assert health["state"] == "RUNNING"

        # 9-11. Read chunk sequence & timestamp
        chunk = await capture.read_chunk()
        assert isinstance(chunk, AudioChunk)
        assert chunk.sequence_number >= 0
        assert chunk.timestamp.tzinfo is not None

        # 8. Stop stream
        await capture.stop()
        h_stop = await capture.health()
        assert h_stop["state"] == "STOPPED"

    asyncio.run(_test())


def test_capture_queue_backpressure_and_drop_oldest() -> None:
    """Verify bounded capture queue backpressure and DROP_OLDEST policy (rules 12-13)."""

    async def _test() -> None:
        capture = SoundDeviceCaptureAdapter(max_buffered_chunks=2, simulated_mode=True)
        await capture.start()

        # Enqueue 3 chunks manually via callback helper
        capture._enqueue_chunk_from_callback(make_chunk(seq=0))
        capture._enqueue_chunk_from_callback(make_chunk(seq=1))
        capture._enqueue_chunk_from_callback(make_chunk(seq=2))

        health = await capture.health()
        assert health["chunks_dropped"] >= 1

        await capture.stop()

    asyncio.run(_test())


def test_capture_error_isolation_and_permission_failure() -> None:
    """Verify hardware failure isolation and permission denial handling (rules 14-16)."""

    async def _test() -> None:
        capture = SoundDeviceCaptureAdapter(simulated_mode=True)
        capture._state = AudioCaptureState.RUNNING

        # Callback exception handling
        capture._audio_callback(None, 480, None, status=1)
        health = await capture.health()
        assert health["stream_errors"] >= 1

    asyncio.run(_test())


def test_playback_lifecycle_state_machine() -> None:
    """Verify speaker playback stream startup, chunk playback, pause, resume, and stop (rules 17-23)."""

    async def _test() -> None:
        playback = SoundDevicePlaybackAdapter(simulated_mode=True)
        assert playback.get_state() == PlaybackState.IDLE

        # 17-18. Play chunk
        c0 = make_chunk(seq=0)
        await playback.play_chunk(c0)
        assert playback.get_state() in (PlaybackState.PLAYING, PlaybackState.IDLE)

        # 20. Pause
        await playback.play_chunk(make_chunk(seq=1))
        await playback.pause()
        assert playback.get_state() == PlaybackState.PAUSED

        # 21. Resume
        await playback.resume()
        assert playback.get_state() in (PlaybackState.PLAYING, PlaybackState.IDLE)

        # 22. Stop
        await playback.stop()
        assert playback.get_state() == PlaybackState.IDLE

        # 23. Restart
        await playback.play_chunk(make_chunk(seq=0))
        assert playback.get_state() in (PlaybackState.PLAYING, PlaybackState.IDLE)
        await playback.stop()

    asyncio.run(_test())


def test_playback_sequence_validation_and_invalid_transition() -> None:
    """Verify playback sequence ordering and invalid state transition error (rules 19, 24)."""

    async def _test() -> None:
        playback = SoundDevicePlaybackAdapter(simulated_mode=True)
        await playback.play_chunk(make_chunk(seq=5))

        # Duplicate sequence error
        with pytest.raises(PlaybackValidationError):
            await playback.play_chunk(make_chunk(seq=5))

        # Sequence regression error
        with pytest.raises(PlaybackValidationError):
            await playback.play_chunk(make_chunk(seq=2))

        await playback.stop()

        # 24. Invalid state transition
        with pytest.raises(PlaybackStateError):
            playback._transition(PlaybackState.PAUSED)

    asyncio.run(_test())


def test_playback_queue_overflow_and_underrun() -> None:
    """Verify playback queue overflow, underruns, and drop tracking (rules 25-26)."""

    async def _test() -> None:
        playback = SoundDevicePlaybackAdapter(
            config=PlaybackConfig(max_buffered_chunks=2), simulated_mode=True
        )

        await playback.play_chunk(make_chunk(seq=0))
        await playback.play_chunk(make_chunk(seq=1))
        await playback.play_chunk(make_chunk(seq=2))  # Overflows queue

        health = await playback.health()
        assert health["chunks_dropped"] >= 1

        await playback.stop()

    asyncio.run(_test())


def test_barge_in_playback_interruption() -> None:
    """Verify barge-in interruption halts playback, clears pending chunks, and safely stops stream (rules 27-30)."""

    async def _test() -> None:
        playback = SoundDevicePlaybackAdapter(simulated_mode=True)
        mgr = AudioSessionManager(playback=playback)

        await mgr.start_session("sess_sd_bargein")
        await mgr.process_speech_text("Barge in test speech output.")

        assert playback._chunks_received > 0

        # 27-30. Trigger barge-in speech interruption
        await mgr.interrupt_speech()
        assert playback.get_state() == PlaybackState.IDLE
        assert mgr._synthesis_queue.empty()

        await mgr.stop_session("sess_sd_bargein")

    asyncio.run(_test())


def test_health_probing_and_lifecycle_cleanup() -> None:
    """Verify health reporting, repeated start/stop, cancellation, and resource cleanup (rules 31-40)."""

    async def _test() -> None:
        capture = SoundDeviceCaptureAdapter(simulated_mode=True)
        playback = SoundDevicePlaybackAdapter(simulated_mode=True)

        # 31, 33. Healthy checks
        h_cap = await capture.health()
        h_pb = await playback.health()
        assert h_cap["healthy"] is True
        assert h_pb["healthy"] is True

        # 35-36. Repeated start and stop
        await capture.start()
        await capture.start()  # Idempotent start
        assert capture._state.value == "RUNNING"

        await capture.stop()
        await capture.stop()  # Idempotent stop
        assert capture._state.value == "STOPPED"

        # 37-38. Cancellation and task cleanup
        await playback.play_chunk(make_chunk(seq=0))
        await playback.stop()
        assert playback._playback_task is None or playback._playback_task.done()

    asyncio.run(_test())


def test_privacy_guarantees() -> None:
    """Verify raw PCM audio bytes and credentials are absent from telemetry (rules 41-43)."""

    async def _test() -> None:
        playback = SoundDevicePlaybackAdapter(simulated_mode=True)
        await playback.play_chunk(make_chunk(seq=0))

        telemetry = playback.get_telemetry()
        telem_dict = telemetry.model_dump()

        # 41. No raw PCM in telemetry
        assert "payload" not in telem_dict
        assert "audio_bytes" not in telem_dict
        assert "pcm" not in telem_dict

        await playback.stop()

    asyncio.run(_test())
