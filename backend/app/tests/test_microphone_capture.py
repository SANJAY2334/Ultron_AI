"""Unit and Integration Tests for Microphone Capture Adapter (Phase 4C.2).

Validates MicrophoneCaptureAdapter, device discovery, explicit device selection, invalid device rejection,
configuration validation, monotonic sequence numbering, DROP_OLDEST backpressure, cancellation cleanup,
privacy protection (no raw payload leakage), sanitized error taxonomy, and IAudioCapture contract compliance.
"""

import asyncio

import pytest

from app.audio import AudioChunk, AudioFormat, AudioStreamConfig, IAudioCapture
from app.audio.adapters.microphone import MicrophoneCaptureAdapter
from app.audio.capture import (
    AudioCaptureConfigurationError,
    AudioCaptureError,
    AudioCaptureState,
    AudioDeviceInfo,
    AudioDeviceNotFoundError,
)


@pytest.fixture
def adapter() -> MicrophoneCaptureAdapter:
    return MicrophoneCaptureAdapter(max_buffered_chunks=5, mock_mode=True)


def test_device_discovery(adapter: MicrophoneCaptureAdapter) -> None:
    """Verify list_input_devices and get_default_input_device without activating hardware."""
    devices = adapter.list_input_devices()
    assert len(devices) >= 1
    assert any(d.is_default for d in devices)

    default_dev = adapter.get_default_input_device()
    assert isinstance(default_dev, AudioDeviceInfo)
    assert default_dev.is_default is True
    assert default_dev.input_channels >= 1


def test_explicit_device_selection_and_not_found_rejection() -> None:
    """Verify requesting invalid device_id raises AudioDeviceNotFoundError."""

    async def _test() -> None:
        # Invalid device
        bad_adapter = MicrophoneCaptureAdapter(device_id="non_existent_mic_id")
        with pytest.raises(AudioDeviceNotFoundError, match="was not found"):
            await bad_adapter.start()

        # Valid explicit device
        valid_adapter = MicrophoneCaptureAdapter(device_id="dev_default_mic")
        await valid_adapter.start()
        assert valid_adapter.current_state == AudioCaptureState.RUNNING
        await valid_adapter.stop()

    asyncio.run(_test())


def test_invalid_configuration_rejection() -> None:
    """Verify invalid sample_rate in AudioStreamConfig raises AudioCaptureConfigurationError."""

    async def _test() -> None:
        bad_cfg = AudioStreamConfig(sample_rate=16000)
        bad_cfg.sample_rate = 99999  # Bypassing validator to test adapter safety

        bad_adapter = MicrophoneCaptureAdapter(config=bad_cfg)
        with pytest.raises(AudioCaptureConfigurationError):
            await bad_adapter.start()

    asyncio.run(_test())


def test_microphone_capture_lifecycle_and_monotonic_chunks(
    adapter: MicrophoneCaptureAdapter,
) -> None:
    """Verify start/read_chunk/stop lifecycle, monotonic sequence numbers, and payload integrity."""

    async def _test() -> None:
        assert adapter.current_state == AudioCaptureState.STOPPED

        await adapter.start()
        assert adapter.current_state == AudioCaptureState.RUNNING

        # Read 3 consecutive chunks
        chunk0 = await adapter.read_chunk()
        chunk1 = await adapter.read_chunk()
        chunk2 = await adapter.read_chunk()

        assert isinstance(chunk0, AudioChunk)
        assert chunk0.sequence_number == 0
        assert chunk1.sequence_number == 1
        assert chunk2.sequence_number == 2

        assert chunk0.duration_ms == 30.0
        assert len(chunk0.payload) == 960  # 16000 * 0.03 * 1 * 2 = 960 bytes
        assert chunk0.timestamp.tzinfo is not None

        await adapter.stop()
        assert adapter.current_state == AudioCaptureState.STOPPED

        # Repeated stop is safe and idempotent
        await adapter.stop()
        assert adapter.current_state == AudioCaptureState.STOPPED

    asyncio.run(_test())


def test_backpressure_drop_oldest_policy() -> None:
    """Verify bounded queue DROP_OLDEST backpressure policy when consumer is slow."""

    async def _test() -> None:
        # Buffer size = 2
        small_adapter = MicrophoneCaptureAdapter(max_buffered_chunks=2)
        await small_adapter.start()

        # Wait for 5 chunks to be produced in background (overflowing buffer size of 2)
        await asyncio.sleep(0.20)

        health = await small_adapter.health()
        assert health["chunks_captured"] >= 5
        assert health["chunks_dropped"] >= 2
        assert health["buffered_chunks"] == 2

        await small_adapter.stop()

    asyncio.run(_test())


def test_cancellation_and_cleanup(adapter: MicrophoneCaptureAdapter) -> None:
    """Verify read_chunk cancellation releases background tasks cleanly."""

    async def _test() -> None:
        await adapter.start()

        task = asyncio.create_task(adapter.read_chunk())
        await asyncio.sleep(0.01)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert adapter.current_state == AudioCaptureState.STOPPED

    asyncio.run(_test())


def test_privacy_boundaries_no_raw_payload_leaks(adapter: MicrophoneCaptureAdapter) -> None:
    """Verify health diagnostics and custom repr do not expose raw binary audio bytes."""

    async def _test() -> None:
        await adapter.start()
        chunk = await adapter.read_chunk()

        health = await adapter.health()
        assert "payload" not in health
        assert "audio_bytes" not in health

        chunk_repr = repr(chunk)
        assert r"\x00" not in chunk_repr
        assert "bytes=960" in chunk_repr

        await adapter.stop()

    asyncio.run(_test())


class FakeMicrophoneBackend(IAudioCapture):
    """Fake capture backend for dependency inversion integration testing."""

    def __init__(self) -> None:
        self._running = False
        self._seq = 0

    async def start(self) -> None:
        self._running = True

    async def stop(self) -> None:
        self._running = False

    async def read_chunk(self) -> AudioChunk:
        if not self._running:
            raise AudioCaptureError("Backend not running.")
        self._seq += 1
        return AudioChunk(
            chunk_id=f"fake_{self._seq}",
            sequence_number=self._seq,
            timestamp=pytest.importorskip("datetime").datetime.now(
                pytest.importorskip("datetime").UTC
            ),
            duration_ms=30.0,
            audio_format=AudioFormat(),
            payload=b"\x00" * 960,
        )

    async def health(self) -> dict[str, bool]:
        return {"fake_backend": True}


def test_dependency_inversion_fake_microphone_contract() -> None:
    """Verify IAudioCapture contract allows swapping FakeMicrophoneBackend seamlessly."""

    async def _test() -> None:
        cap: IAudioCapture = FakeMicrophoneBackend()
        await cap.start()
        chk = await cap.read_chunk()
        assert isinstance(chk, AudioChunk)
        assert chk.chunk_id == "fake_1"
        await cap.stop()

    asyncio.run(_test())
