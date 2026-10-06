"""Comprehensive Unit & Hardware Integration Tests for PiperTTS and WindowsAudioPlayback (Phase 4H.5).

Tests offline text-to-speech synthesis, Piper ONNX models, hardware output device resolution,
playback queue backpressure, barge-in cancellation, Zero-Trust security, and physical speaker playback.
"""

import asyncio
from datetime import UTC, datetime
from unittest.mock import patch

import pytest

from app.audio.adapters.piper_tts import PIPER_AVAILABLE, PiperTTSProvider
from app.audio.adapters.windows_audio_playback import (
    SOUNDDEVICE_AVAILABLE,
    WindowsAudioPlaybackDevice,
)
from app.audio.base import IAudioPlayback, ITextToSpeechProvider
from app.audio.models import (
    AudioChunk,
    AudioEncoding,
    AudioFormat,
    AudioPrivacy,
    PlaybackState,
    SpeechSynthesisRequest,
    SpeechSynthesisResult,
)
from app.audio.playback import (
    PlaybackConfig,
    PlaybackDeviceNotFoundError,
    PlaybackValidationError,
)
from app.audio.tts import (
    TTSConfig,
    TTSTimeoutError,
    TTSUnavailableError,
    TTSValidationError,
)


def _make_dummy_chunk(seq: int = 0, duration_ms: float = 30.0) -> AudioChunk:
    sample_count = int(16000 * (duration_ms / 1000.0))
    pcm_bytes = b"\x00\x00" * sample_count
    return AudioChunk(
        chunk_id=f"chk_play_{seq}",
        sequence_number=seq,
        timestamp=datetime.now(UTC),
        duration_ms=duration_ms,
        audio_format=AudioFormat(
            sample_rate=16000,
            channels=1,
            sample_width=2,
            encoding=AudioEncoding.PCM_S16LE,
        ),
        payload=pcm_bytes,
        privacy_level=AudioPrivacy.EPHEMERAL,
        retention_policy="do_not_persist",
    )


class TestPiperTTSAutomated:
    """Automated unit tests for PiperTTSProvider."""

    def test_implements_itts_provider_interface(self) -> None:
        """Verify PiperTTSProvider implements ITextToSpeechProvider."""
        provider = PiperTTSProvider(simulated_mode=True)
        assert isinstance(provider, ITextToSpeechProvider)

    def test_provider_construction_and_capabilities(self) -> None:
        """Verify capabilities metadata reporting."""
        provider = PiperTTSProvider(simulated_mode=True)
        caps = provider.capabilities()
        assert caps["provider_name"] == "piper"
        assert caps["offline_only"] is True
        assert caps["encoding"] == "pcm_s16le"

    def test_configuration_validation_rules(self) -> None:
        """Verify sample_rate validation in TTSConfig."""
        cfg = TTSConfig(sample_rate=16000)
        assert cfg.sample_rate == 16000

        with pytest.raises(ValueError, match="Unsupported TTS sample rate"):
            TTSConfig(sample_rate=12345)

    @pytest.mark.asyncio
    async def test_empty_and_whitespace_text_rejected(self) -> None:
        """Verify empty text raises TTSValidationError."""
        provider = PiperTTSProvider(simulated_mode=True)
        req = SpeechSynthesisRequest(
            request_id="req_empty",
            text="   ",
            voice="en_US-lessac-low",
        )
        with pytest.raises(TTSValidationError, match="cannot be empty"):
            await provider.synthesize(req)

    @pytest.mark.asyncio
    async def test_oversized_text_rejected(self) -> None:
        """Verify text exceeding max_text_length raises TTSValidationError."""
        cfg = TTSConfig(max_text_length=20)
        provider = PiperTTSProvider(config=cfg, simulated_mode=True)
        req = SpeechSynthesisRequest(
            request_id="req_long",
            text="This sentence is definitely longer than 20 characters.",
            voice="en_US-lessac-low",
        )
        with pytest.raises(TTSValidationError, match="exceeds limit"):
            await provider.synthesize(req)

    @pytest.mark.asyncio
    async def test_unsupported_characters_handled_cleanly(self) -> None:
        """Verify special unicode and punctuation characters are processed without crashing."""
        provider = PiperTTSProvider(simulated_mode=True)
        req = SpeechSynthesisRequest(
            request_id="req_unicode",
            text="System status: 100% OK! #ULTRON @2026 -> [test].",
            voice="en_US-lessac-low",
        )
        res = await provider.synthesize(req)
        assert isinstance(res, SpeechSynthesisResult)
        assert res.duration_ms > 0
        assert len(res.audio_payload) > 0

    @pytest.mark.asyncio
    async def test_model_load_failure_handling(self) -> None:
        """Verify missing model file raises TTSUnavailableError."""
        provider = PiperTTSProvider(model_path="non_existent_model.onnx", simulated_mode=False)
        req = SpeechSynthesisRequest(
            request_id="req_fail",
            text="Hello world",
            voice="en_US-lessac-low",
        )
        with pytest.raises(TTSUnavailableError):
            await provider.synthesize(req)

    @pytest.mark.asyncio
    async def test_synthesis_timeout_handling(self) -> None:
        """Verify synthesis exceeding timeout_ms raises TTSTimeoutError."""
        cfg = TTSConfig(timeout_ms=50.0)
        provider = PiperTTSProvider(config=cfg, simulated_mode=False)
        req = SpeechSynthesisRequest(
            request_id="req_to",
            text="Hello world",
            voice="en_US-lessac-low",
        )
        with patch("asyncio.wait_for", side_effect=TimeoutError()):
            with pytest.raises(TTSTimeoutError):
                await provider.synthesize(req)

    @pytest.mark.asyncio
    async def test_streaming_sentence_chunk_generation(self) -> None:
        """Verify stream() generates non-empty audio slices."""
        provider = PiperTTSProvider(simulated_mode=True)
        req = SpeechSynthesisRequest(
            request_id="req_stream",
            text="Streaming chunk test.",
            voice="en_US-lessac-low",
        )
        chunks: list[bytes] = []
        async for c in provider.stream(req):
            chunks.append(c)
        assert len(chunks) > 0
        assert all(len(c) > 0 for c in chunks)

    def test_security_invariant_model_output_is_not_authorization(self) -> None:
        """Verify TTS provider contains zero tool execution, shell, or capability granting APIs."""
        forbidden_methods = [
            "execute_tool",
            "run_command",
            "modify_file",
            "grant_capability",
            "authorize",
            "elevate",
            "execute_shell",
            "bypass_policy",
        ]
        for m in forbidden_methods:
            assert not hasattr(PiperTTSProvider, m)

    def test_privacy_invariants_no_audio_persistence(self) -> None:
        """Verify TTS provider contains no file or database persistence mechanisms."""
        forbidden_attrs = [
            "save_to_disk",
            "persist_audio",
            "db_session",
            "redis_client",
            "write_wav_file",
        ]
        for attr in forbidden_attrs:
            assert not hasattr(PiperTTSProvider, attr)


class TestWindowsAudioPlaybackAutomated:
    """Automated unit tests for WindowsAudioPlaybackDevice."""

    def test_implements_iaudio_playback_interface(self) -> None:
        """Verify WindowsAudioPlaybackDevice implements IAudioPlayback."""
        dev = WindowsAudioPlaybackDevice(simulated_mode=True)
        assert isinstance(dev, IAudioPlayback)

    @pytest.mark.asyncio
    async def test_speaker_device_selection_explicit_index(self) -> None:
        """Verify explicit integer output device selection."""
        with patch("app.audio.adapters.windows_audio_playback.sd") as mock_sd:
            mock_sd.query_devices.return_value = {"name": "Test Realtek Output", "max_output_channels": 2}
            dev = WindowsAudioPlaybackDevice(device_id=3, simulated_mode=False)
            idx, name = await dev.resolve_device()
            assert idx == 3
            assert "Realtek" in name

    @pytest.mark.asyncio
    async def test_speaker_device_selection_hal_identifier(self) -> None:
        """Verify HAL device_id format 'audio_out_3' resolution."""
        with patch("app.audio.adapters.windows_audio_playback.sd") as mock_sd:
            mock_sd.query_devices.return_value = {"name": "Test Speakers", "max_output_channels": 2}
            dev = WindowsAudioPlaybackDevice(device_id="audio_out_3", simulated_mode=False)
            idx, name = await dev.resolve_device()
            assert idx == 3
            assert name == "Test Speakers"

    @pytest.mark.asyncio
    async def test_invalid_speaker_device_raises_error(self) -> None:
        """Verify non-existent speaker raises PlaybackDeviceNotFoundError."""
        with patch("app.audio.adapters.windows_audio_playback.sd") as mock_sd:
            mock_sd.query_devices.return_value = []
            dev = WindowsAudioPlaybackDevice(device_id="non_existent_speaker_xyz", simulated_mode=False)
            with pytest.raises(PlaybackDeviceNotFoundError):
                await dev.resolve_device()

    @pytest.mark.asyncio
    async def test_bounded_queue_drop_oldest_backpressure(self) -> None:
        """Verify queue overflow drops oldest chunk and increments drop metric."""
        cfg = PlaybackConfig(max_buffered_chunks=2)
        dev = WindowsAudioPlaybackDevice(config=cfg, simulated_mode=True)

        c1 = _make_dummy_chunk(seq=1)
        c2 = _make_dummy_chunk(seq=2)
        c3 = _make_dummy_chunk(seq=3)

        await dev.play_chunk(c1)
        await dev.play_chunk(c2)
        await dev.play_chunk(c3)

        assert dev._chunks_dropped >= 1
        await dev.stop()

    @pytest.mark.asyncio
    async def test_duplicate_sequence_number_rejected(self) -> None:
        """Verify duplicate AudioChunk sequence number raises PlaybackValidationError."""
        dev = WindowsAudioPlaybackDevice(simulated_mode=True)
        c1 = _make_dummy_chunk(seq=1)
        c2 = _make_dummy_chunk(seq=1)

        await dev.play_chunk(c1)
        with pytest.raises(PlaybackValidationError, match="Duplicate sequence"):
            await dev.play_chunk(c2)
        await dev.stop()

    @pytest.mark.asyncio
    async def test_sequence_regression_rejected(self) -> None:
        """Verify regressing sequence number raises PlaybackValidationError."""
        dev = WindowsAudioPlaybackDevice(simulated_mode=True)
        c1 = _make_dummy_chunk(seq=5)
        c2 = _make_dummy_chunk(seq=2)

        await dev.play_chunk(c1)
        with pytest.raises(PlaybackValidationError, match="sequence regression"):
            await dev.play_chunk(c2)
        await dev.stop()

    @pytest.mark.asyncio
    async def test_playback_barge_in_interruption(self) -> None:
        """Verify stop() flushes queue and resets state to IDLE immediately."""
        dev = WindowsAudioPlaybackDevice(simulated_mode=True)
        c1 = _make_dummy_chunk(seq=1, duration_ms=500.0)
        c2 = _make_dummy_chunk(seq=2, duration_ms=500.0)

        await dev.play_chunk(c1)
        await dev.play_chunk(c2)
        assert dev.get_state() == PlaybackState.PLAYING

        await dev.stop()
        assert dev.get_state() == PlaybackState.IDLE
        assert dev._queue.empty()

    @pytest.mark.asyncio
    async def test_pause_and_resume_lifecycle(self) -> None:
        """Verify PAUSED and PLAYING transitions."""
        dev = WindowsAudioPlaybackDevice(simulated_mode=True)
        c1 = _make_dummy_chunk(seq=1, duration_ms=500.0)

        await dev.play_chunk(c1)
        await dev.pause()
        assert dev.get_state() == PlaybackState.PAUSED

        await dev.resume()
        assert dev.get_state() == PlaybackState.PLAYING
        await dev.stop()

    @pytest.mark.asyncio
    async def test_playback_recovery_from_error(self) -> None:
        """Verify recover() resets state machine and handles gracefully."""
        dev = WindowsAudioPlaybackDevice(simulated_mode=True)
        dev._state = PlaybackState.ERROR

        await dev.recover()
        assert dev.get_state() == PlaybackState.IDLE

    @pytest.mark.asyncio
    async def test_end_to_end_tts_to_playback_pipeline(self) -> None:
        """Verify response text -> PiperTTS -> WindowsAudioPlaybackDevice pipeline."""
        tts = PiperTTSProvider(simulated_mode=True)
        playback = WindowsAudioPlaybackDevice(simulated_mode=True)

        req = SpeechSynthesisRequest(
            request_id="req_pipeline",
            text="Pipeline integration test.",
            voice="en_US-lessac-low",
        )
        res = await tts.synthesize(req)
        assert len(res.audio_payload) > 0

        # Wrap in chunk and play
        chunk = AudioChunk(
            chunk_id="chk_e2e",
            sequence_number=1,
            timestamp=datetime.now(UTC),
            duration_ms=res.duration_ms,
            audio_format=res.audio_format,
            payload=res.audio_payload,
        )
        await playback.play_chunk(chunk)
        assert playback.is_speaking() is True
        await playback.stop()
        assert playback.is_speaking() is False


class TestPhysicalHardwarePlaybackIntegration:
    """Hardware integration test executing real Piper ONNX synthesis -> physical speaker output."""

    @pytest.mark.asyncio
    async def test_physical_piper_synthesis_and_speaker_playback(self) -> None:
        """Performs actual Piper voice synthesis and plays audio through physical Windows speakers."""
        if not PIPER_AVAILABLE:
            pytest.skip("piper-tts is not available on host.")
        if not SOUNDDEVICE_AVAILABLE:
            pytest.skip("sounddevice is not available on host.")

        import sounddevice as sd  # type: ignore[import-untyped]

        # Verify physical output speakers exist
        devs = sd.query_devices()
        outputs = [d for d in devs if int(d.get("max_output_channels", 0)) > 0]
        if not outputs:
            pytest.skip("No physical audio output speakers/headphones found on host.")

        cfg = TTSConfig(model_name="en_US-lessac-low", timeout_ms=30000.0)
        tts = PiperTTSProvider(config=cfg, simulated_mode=False)
        playback = WindowsAudioPlaybackDevice(simulated_mode=False)

        try:
            # 1. Synthesize known phrase locally with Piper
            test_phrase = "ULTRON local speech test successful."
            req = SpeechSynthesisRequest(
                request_id="req_hw_tts_001",
                text=test_phrase,
                voice="en_US-lessac-low",
                speed=1.0,
            )
            res = await tts.synthesize(req)
            assert isinstance(res, SpeechSynthesisResult)
            assert res.duration_ms > 500.0
            assert len(res.audio_payload) > 1000

            # 2. Output to physical speaker via WindowsAudioPlaybackDevice
            chunk = AudioChunk(
                chunk_id="chk_hw_playback_001",
                sequence_number=0,
                timestamp=datetime.now(UTC),
                duration_ms=min(res.duration_ms, 500.0),  # Play initial slice for non-intrusive CI test
                audio_format=res.audio_format,
                payload=res.audio_payload[: int(16000 * 2 * 0.5)],
            )
            await playback.play_chunk(chunk)
            assert playback.get_state() == PlaybackState.PLAYING

            # Allow brief playback
            await asyncio.sleep(0.6)
            await playback.stop()
            assert playback.get_state() == PlaybackState.IDLE

        finally:
            await playback.stop()
            await tts.close()
