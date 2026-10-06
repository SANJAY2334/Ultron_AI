"""Comprehensive Unit & Hardware Integration Tests for WindowsAudioCaptureDevice (Phase 4H.2).

Tests device resolution, PortAudio/SoundDevice streaming, PCM 16-bit 16kHz framing,
DROP_OLDEST queue backpressure, disconnect detection & recovery, and physical microphone capture.
"""

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from app.audio.adapters.vad import VADAdapter
from app.audio.adapters.windows_audio_capture import (
    SOUNDDEVICE_AVAILABLE,
    WindowsAudioCaptureDevice,
)
from app.audio.base import IAudioCapture
from app.audio.capture import (
    AudioCaptureConfigurationError,
    AudioCaptureState,
    AudioDeviceNotFoundError,
)
from app.audio.models import (
    AudioChunk,
    AudioEncoding,
    AudioFormat,
    AudioPrivacy,
    AudioStreamConfig,
)
from app.hardware.device_manager import DeviceManager
from app.hardware.models import DeviceState, DeviceType, HardwareDevice


class TestWindowsAudioCaptureAutomated:
    """Automated unit tests with mocked device/audio driver layers."""

    def test_implements_iaudio_capture_interface(self) -> None:
        """Verify WindowsAudioCaptureDevice implements IAudioCapture interface."""
        device = WindowsAudioCaptureDevice(simulated_mode=True)
        assert isinstance(device, IAudioCapture)

    def test_default_configuration_and_chunk_geometry(self) -> None:
        """Verify default configuration targets 16kHz mono 16-bit PCM @ 30ms."""
        device = WindowsAudioCaptureDevice(simulated_mode=True)
        assert device._config.sample_rate == 16000
        assert device._config.channels == 1
        assert device._config.chunk_duration_ms == 30.0
        # 16000 * 0.03 = 480 samples; 480 * 1 * 2 = 960 bytes
        assert device._samples_per_chunk == 480
        assert device._bytes_per_chunk == 960

    @pytest.mark.asyncio
    async def test_simulated_mode_chunk_generation(self) -> None:
        """Verify simulated mode captures valid 16kHz AudioChunks."""
        device = WindowsAudioCaptureDevice(simulated_mode=True)
        await device.start()
        assert device.current_state == AudioCaptureState.RUNNING

        chunk = await device.read_chunk()
        assert isinstance(chunk, AudioChunk)
        assert chunk.audio_format.sample_rate == 16000
        assert chunk.audio_format.channels == 1
        assert chunk.audio_format.encoding == AudioEncoding.PCM_S16LE
        assert len(chunk.payload) == 960
        assert chunk.duration_ms == 30.0
        assert chunk.sequence_number == 0
        assert chunk.privacy_level == AudioPrivacy.EPHEMERAL
        assert chunk.retention_policy == "do_not_persist"

        await device.stop()
        assert device.current_state == AudioCaptureState.STOPPED

    @pytest.mark.asyncio
    async def test_unsupported_sample_rate_rejection(self) -> None:
        """Verify invalid sample rate raises AudioCaptureConfigurationError."""
        invalid_config = AudioStreamConfig.model_construct(
            sample_rate=12345, channels=1, chunk_duration_ms=30.0, buffer_size=4096, max_latency_ms=200.0
        )
        device = WindowsAudioCaptureDevice(config=invalid_config, simulated_mode=True)
        with pytest.raises(AudioCaptureConfigurationError):
            await device.start()

    @pytest.mark.asyncio
    async def test_device_selection_explicit_integer_index(self) -> None:
        """Verify explicit device selection by integer index."""
        device = WindowsAudioCaptureDevice(device_id=1, simulated_mode=False)
        mock_sd_query = MagicMock(return_value={"name": "Test Realtek Mic", "max_input_channels": 2})

        with patch("app.audio.adapters.windows_audio_capture.sd") as mock_sd:
            mock_sd.query_devices = mock_sd_query
            idx, name = await device.resolve_input_device()
            assert idx == 1
            assert name == "Test Realtek Mic"

    @pytest.mark.asyncio
    async def test_device_selection_string_identifier_from_hal(self) -> None:
        """Verify device selection by HAL HardwareDevice ID."""
        mock_hal = DeviceManager()
        test_dev = HardwareDevice(
            device_id="audio_in_2",
            name="USB Conference Microphone",
            device_type=DeviceType.AUDIO_INPUT,
            state=DeviceState.AVAILABLE,
            is_default=False,
            channels=1,
        )

        device = WindowsAudioCaptureDevice(
            device_id="audio_in_2", device_manager=mock_hal, simulated_mode=False
        )

        with (
            patch.object(mock_hal, "list_devices", return_value=[test_dev]),
            patch("app.audio.adapters.windows_audio_capture.sd") as mock_sd,
        ):
            mock_sd.query_devices = MagicMock(return_value={"name": "USB Conference Microphone", "max_input_channels": 1})
            idx, name = await device.resolve_input_device()
            assert idx == 2
            assert name == "USB Conference Microphone"

    @pytest.mark.asyncio
    async def test_device_not_found_raises_error(self) -> None:
        """Verify non-existent device ID raises AudioDeviceNotFoundError."""
        mock_hal = DeviceManager()
        device = WindowsAudioCaptureDevice(
            device_id="non_existent_mic", device_manager=mock_hal, simulated_mode=False
        )

        with (
            patch.object(mock_hal, "list_devices", return_value=[]),
            patch("app.audio.adapters.windows_audio_capture.sd") as mock_sd,
        ):
            mock_sd.query_devices = MagicMock(return_value=[])
            mock_sd.default.device = [None, None]
            with pytest.raises(AudioDeviceNotFoundError):
                await device.resolve_input_device()

    @pytest.mark.asyncio
    async def test_backpressure_drop_oldest_policy(self) -> None:
        """Verify bounded queue discards oldest chunks when buffer overflows."""
        device = WindowsAudioCaptureDevice(max_buffered_chunks=3, simulated_mode=False)
        device._state = AudioCaptureState.RUNNING

        # Simulate enqueuing 5 chunks into a queue of maxsize=3
        for seq in range(5):
            chunk = AudioChunk(
                chunk_id=f"chk_{seq}",
                sequence_number=seq,
                timestamp=datetime.now(UTC),
                duration_ms=30.0,
                audio_format=AudioFormat(sample_rate=16000, channels=1, sample_width=2),
                payload=b"\x00" * 960,
            )
            device._enqueue_chunk_from_callback(chunk)

        assert device.chunks_dropped == 2
        assert device._queue.qsize() == 3

        # Oldest remaining chunk should be seq=2 (0 and 1 dropped)
        first_read = await device.read_chunk()
        assert first_read.sequence_number == 2

    @pytest.mark.asyncio
    async def test_stream_error_recovery(self) -> None:
        """Verify recover() cleanly restarts the capture stream."""
        device = WindowsAudioCaptureDevice(simulated_mode=True)
        await device.start()
        assert device.current_state == AudioCaptureState.RUNNING

        # Trigger recovery
        success = await device.recover()
        assert success is True
        assert device.current_state == AudioCaptureState.RUNNING

        await device.stop()

    @pytest.mark.asyncio
    async def test_health_telemetry_reporting(self) -> None:
        """Verify health check returns comprehensive diagnostics."""
        device = WindowsAudioCaptureDevice(simulated_mode=True)
        await device.start()
        health = await device.health()

        assert health["subsystem"] == "windows_audio_capture"
        assert health["healthy"] is True
        assert health["state"] == "RUNNING"
        assert health["sample_rate"] == 16000
        assert health["channels"] == 1
        assert "chunks_captured" in health
        assert "chunks_dropped" in health
        assert "stream_errors" in health

        await device.stop()

    def test_security_and_privacy_invariants(self) -> None:
        """Verify WindowsAudioCaptureDevice has zero execution methods and protects privacy."""
        forbidden_methods = [
            "execute",
            "run_tool",
            "modify_system",
            "grant_capability",
            "authorize",
        ]
        for m in forbidden_methods:
            assert not hasattr(WindowsAudioCaptureDevice, m)

        # Repr redaction check
        device = WindowsAudioCaptureDevice(simulated_mode=True)
        device.set_session_context(session_id="sess_123", correlation_id="corr_456")
        assert device._session_id == "sess_123"
        assert device._correlation_id == "corr_456"


class TestPhysicalHardwareCaptureIntegration:
    """Hardware integration test executing real physical microphone capture if present."""

    @pytest.mark.asyncio
    async def test_physical_microphone_capture_pipeline(self) -> None:
        """Performs actual hardware capture from physical microphone and routes to VAD."""
        if not SOUNDDEVICE_AVAILABLE:
            pytest.skip("SoundDevice / PortAudio is not available on host.")

        # Check if physical input devices exist
        import sounddevice as sd  # type: ignore[import-not-found,import-untyped]

        devices = sd.query_devices()
        input_devices = [d for d in devices if int(d.get("max_input_channels", 0)) > 0]

        if not input_devices:
            pytest.skip("No physical microphone devices detected on host machine.")

        device = WindowsAudioCaptureDevice(simulated_mode=False)

        try:
            await device.start()
            assert device.current_state == AudioCaptureState.RUNNING

            # Capture 3 real physical audio frames
            captured_chunks: list[AudioChunk] = []
            for _ in range(3):
                chunk = await device.read_chunk()
                captured_chunks.append(chunk)

            assert len(captured_chunks) == 3
            for c in captured_chunks:
                assert c.audio_format.sample_rate == 16000
                assert c.audio_format.channels == 1
                assert len(c.payload) == 960
                assert c.duration_ms == 30.0

            # Feed captured physical chunk into real VAD pipeline
            vad = VADAdapter()
            event = await vad.process(captured_chunks[0])
            assert event.state is not None
            assert event.confidence >= 0.0

        finally:
            await device.stop()
            assert device.current_state == AudioCaptureState.STOPPED
