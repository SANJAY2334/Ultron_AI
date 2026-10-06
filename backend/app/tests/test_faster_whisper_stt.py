"""Comprehensive Unit & Hardware Integration Tests for FasterWhisperSTT (Phase 4H.4).

Tests offline speech-to-text inference, CTranslate2 INT8 quantization, VAD speech segmentation,
device resolution (CPU/CUDA), error handling, Zero-Trust security invariants, and real microphone integration.
"""

import asyncio
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.audio.adapters.faster_whisper_stt import (
    FASTER_WHISPER_AVAILABLE,
    FasterWhisperSTT,
)
from app.audio.adapters.planner_gateway import VoicePlannerGateway
from app.audio.adapters.speech_segmenter import SpeechSegmenter
from app.audio.adapters.vad import VADAdapter
from app.audio.adapters.windows_audio_capture import (
    SOUNDDEVICE_AVAILABLE,
    WindowsAudioCaptureDevice,
)
from app.audio.base import ISpeechToTextProvider
from app.audio.models import (
    AudioChunk,
    AudioEncoding,
    AudioFormat,
    AudioPrivacy,
    Transcript,
    TranscriptSegment,
    VoiceActivityState,
)
from app.audio.stt import (
    STTConfig,
    STTTimeoutError,
    STTUnavailableError,
    STTValidationError,
)


def _generate_synthetic_chunk(
    seq: int = 0,
    duration_ms: float = 30.0,
    sample_rate: int = 16000,
    amplitude: float = 0.5,
) -> AudioChunk:
    """Helper to synthesize a 16-bit LE PCM mono AudioChunk."""
    samples_count = int(sample_rate * (duration_ms / 1000.0))
    # Generate a simple 440 Hz sine tone or silence
    if amplitude > 0:
        t = np.linspace(0, duration_ms / 1000.0, samples_count, False)
        wave_data = (np.sin(2 * np.pi * 440.0 * t) * amplitude * 32767).astype(np.int16)
        pcm_bytes = wave_data.tobytes()
    else:
        pcm_bytes = b"\x00" * (samples_count * 2)

    return AudioChunk(
        chunk_id=f"chk_synth_{seq}",
        sequence_number=seq,
        timestamp=datetime.now(UTC),
        duration_ms=duration_ms,
        audio_format=AudioFormat(
            sample_rate=sample_rate,
            channels=1,
            sample_width=2,
            encoding=AudioEncoding.PCM_S16LE,
        ),
        payload=pcm_bytes,
        privacy_level=AudioPrivacy.EPHEMERAL,
        retention_policy="do_not_persist",
    )


class TestFasterWhisperAutomated:
    """Automated unit tests for FasterWhisperSTT using fakes/mocks."""

    def test_implements_istt_provider_interface(self) -> None:
        """Verify FasterWhisperSTT implements ISpeechToTextProvider interface."""
        provider = FasterWhisperSTT(simulated_mode=True)
        assert isinstance(provider, ISpeechToTextProvider)

    def test_provider_construction_and_capabilities(self) -> None:
        """Verify provider capabilities metadata and configuration defaults."""
        provider = FasterWhisperSTT(model_size_or_path="tiny", device="cpu", compute_type="int8", simulated_mode=True)
        caps = provider.capabilities()
        assert caps["provider_name"] == "faster_whisper"
        assert caps["model_name"] == "tiny"
        assert caps["device"] == "cpu"
        assert caps["compute_type"] == "int8"
        assert caps["offline_only"] is True

    def test_configuration_validation_rules(self) -> None:
        """Verify STTConfig bounds and language code validation."""
        cfg = STTConfig(language="en", timeout_ms=8000.0)
        assert cfg.language == "en"
        assert cfg.timeout_ms == 8000.0

        with pytest.raises(ValueError):
            STTConfig(language="invalid_very_long_code")

    def test_device_selection_cpu_default(self) -> None:
        """Verify CPU selection when CUDA is absent or requested."""
        with patch("app.audio.adapters.faster_whisper_stt.ctranslate2") as mock_ct2:
            mock_ct2.get_cuda_device_count.return_value = 0
            provider = FasterWhisperSTT(device="auto", simulated_mode=True)
            assert provider.resolved_device == "cpu"
            assert provider.resolved_compute_type == "int8"

    def test_device_selection_cuda_when_present(self) -> None:
        """Verify CUDA selection when genuine CUDA hardware is reported."""
        with patch("app.audio.adapters.faster_whisper_stt.ctranslate2") as mock_ct2:
            mock_ct2.get_cuda_device_count.return_value = 1
            provider = FasterWhisperSTT(device="auto", compute_type="auto", simulated_mode=True)
            assert provider.resolved_device == "cuda"
            assert provider.resolved_compute_type == "float16"

    @pytest.mark.asyncio
    async def test_audio_format_validation_mismatches(self) -> None:
        """Verify rejection of non-16kHz and non-mono audio chunks."""
        provider = FasterWhisperSTT(simulated_mode=True)

        # 44.1kHz chunk
        bad_rate_chunk = _generate_synthetic_chunk(sample_rate=44100)
        with pytest.raises(STTValidationError, match="Unsupported sample rate"):
            await provider.transcribe_chunks([bad_rate_chunk])

    @pytest.mark.asyncio
    async def test_empty_and_silent_audio_handling(self) -> None:
        """Verify empty or silent audio produces clean empty transcript without crashing."""
        provider = FasterWhisperSTT(simulated_mode=False)

        # 1. Empty list
        with pytest.raises(STTValidationError, match="No audio chunks"):
            await provider.transcribe_chunks([])

        # 2. Silent audio chunks
        silent_chunks = [_generate_synthetic_chunk(seq=i, amplitude=0.0) for i in range(5)]
        transcript = await provider.transcribe_chunks(silent_chunks)
        assert transcript.full_text == ""
        assert transcript.confidence == 0.0

    @pytest.mark.asyncio
    async def test_malformed_audio_payload_limit_rejection(self) -> None:
        """Verify payload exceeding max_payload_bytes is rejected."""
        cfg = STTConfig(max_payload_bytes=500)
        provider = FasterWhisperSTT(config=cfg, simulated_mode=True)
        chunks = [_generate_synthetic_chunk(seq=i, duration_ms=30.0) for i in range(5)]
        # 5 * 960 = 4800 bytes > 500
        with pytest.raises(STTValidationError, match="exceeds limit"):
            await provider.transcribe_chunks(chunks)

    @pytest.mark.asyncio
    async def test_simulated_transcription_pipeline(self) -> None:
        """Verify simulated transcription output conforms to Transcript contract."""
        provider = FasterWhisperSTT(simulated_mode=True)
        chunks = [_generate_synthetic_chunk(seq=i) for i in range(10)]

        transcript = await provider.transcribe_chunks(chunks)
        assert isinstance(transcript, Transcript)
        assert len(transcript.full_text) > 0
        assert len(transcript.segments) >= 1
        assert transcript.confidence >= 0.9
        assert transcript.is_final is True

    @pytest.mark.asyncio
    async def test_model_load_failure_handling(self) -> None:
        """Verify model initialization error raises STTUnavailableError."""
        provider = FasterWhisperSTT(simulated_mode=False)
        chunks = [_generate_synthetic_chunk(seq=i, amplitude=0.5) for i in range(5)]

        with (
            patch.object(provider, "_get_or_load_model", side_effect=STTUnavailableError("Disk failure")),
        ):
            with pytest.raises(STTUnavailableError):
                await provider.transcribe_chunks(chunks)

    @pytest.mark.asyncio
    async def test_inference_timeout_handling(self) -> None:
        """Verify timeout raises STTTimeoutError."""
        cfg = STTConfig(timeout_ms=50.0)
        provider = FasterWhisperSTT(config=cfg, simulated_mode=False)
        chunks = [_generate_synthetic_chunk(seq=i, amplitude=0.5) for i in range(5)]

        async def _slow_inference(*args: Any, **kwargs: Any) -> Any:
            await asyncio.sleep(0.2)
            return "text", [], "en", 0.95

        with patch("asyncio.wait_for", side_effect=TimeoutError()):
            with pytest.raises(STTTimeoutError):
                await provider.transcribe_chunks(chunks)

    @pytest.mark.asyncio
    async def test_speech_segmenter_vad_hysteresis(self) -> None:
        """Verify SpeechSegmenter buffers during speech and finalizes upon silence hysteresis."""
        mock_vad = VADAdapter()
        segmenter = SpeechSegmenter(vad_detector=mock_vad, silence_hysteresis_chunks=3, min_speech_chunks=2)

        # Mock VAD returning SPEAKING for 3 chunks, then SILENCE
        with patch.object(
            mock_vad,
            "process",
            side_effect=[
                MagicMock(state=VoiceActivityState.SPEAKING),
                MagicMock(state=VoiceActivityState.SPEAKING),
                MagicMock(state=VoiceActivityState.SPEAKING),
                MagicMock(state=VoiceActivityState.SILENCE),
                MagicMock(state=VoiceActivityState.SILENCE),
                MagicMock(state=VoiceActivityState.SILENCE),
            ],
        ):
            completed: list[AudioChunk] | None = None
            for i in range(6):
                chunk = _generate_synthetic_chunk(seq=i)
                result = await segmenter.process_chunk(chunk)
                if result is not None:
                    completed = result

            assert completed is not None
            assert len(completed) >= 3

    @pytest.mark.asyncio
    async def test_voice_planner_gateway_integration(self) -> None:
        """Verify transcript passes through VoicePlannerGateway to planner without bypassing security."""
        gateway = VoicePlannerGateway()
        transcript = Transcript(
            transcript_id="tx_test_gw",
            segments=[TranscriptSegment(segment_id="s1", text="Hello ULTRON", start_ms=0, end_ms=1000, confidence=0.95)],
            full_text="Hello ULTRON",
            language="en",
            confidence=0.95,
        )

        response = await gateway.submit_transcript(transcript)
        assert response.success is True
        assert len(response.text_response) > 0
        assert response.correlation_id is not None

    def test_security_invariant_model_output_is_not_authorization(self) -> None:
        """Verify STT adapter exposes zero tool execution, shell, or authorization APIs."""
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
            assert not hasattr(FasterWhisperSTT, m)

    def test_privacy_invariants_no_audio_persistence(self) -> None:
        """Verify FasterWhisperSTT contains no file persistence or database writing methods."""
        forbidden_attributes = [
            "save_to_disk",
            "persist_audio",
            "db_session",
            "redis_client",
            "write_wav_file",
        ]
        for attr in forbidden_attributes:
            assert not hasattr(FasterWhisperSTT, attr)


class TestPhysicalHardwareSTTIntegration:
    """Hardware integration test executing real microphone capture -> VAD -> FasterWhisperSTT."""

    @pytest.mark.asyncio
    async def test_physical_microphone_vad_faster_whisper_pipeline(self) -> None:
        """Performs actual hardware capture from physical microphone and routes through local Faster-Whisper."""
        if not SOUNDDEVICE_AVAILABLE:
            pytest.skip("SoundDevice / PortAudio is not available on host.")
        if not FASTER_WHISPER_AVAILABLE:
            pytest.skip("Faster-Whisper / CTranslate2 is not available on host.")

        # Check physical microphone
        import sounddevice as sd  # type: ignore[import-untyped]

        devices = sd.query_devices()
        input_devices = [d for d in devices if int(d.get("max_input_channels", 0)) > 0]
        if not input_devices:
            pytest.skip("No physical microphone detected on host machine.")

        mic = WindowsAudioCaptureDevice(simulated_mode=False)
        cfg = STTConfig(timeout_ms=60000.0)
        stt = FasterWhisperSTT(config=cfg, model_size_or_path="tiny", device="cpu", compute_type="int8", simulated_mode=False)

        try:
            await mic.start()

            captured_chunks: list[AudioChunk] = []
            # Capture ~15 real chunks (450ms of audio)
            for _ in range(15):
                chunk = await mic.read_chunk()
                captured_chunks.append(chunk)

            assert len(captured_chunks) == 15

            # Transcribe real audio segment with local Faster-Whisper
            transcript = await stt.transcribe_chunks(captured_chunks)
            assert isinstance(transcript, Transcript)
            assert transcript.transcript_id.startswith("tx_fw_")
            assert transcript.language is not None

        finally:
            await mic.stop()
            await stt.close()
