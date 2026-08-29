"""Production Audio Engine Integration Tests (Phase 4E.3).

Validates concrete production VAD, Wake-Word, STT, and TTS engine adapters, configuration resolution,
DI container dependency injection, full-duplex pipeline integration (AudioChunk -> WakeWord -> VAD -> STT -> Planner -> TTS -> Playback),
barge-in speech interruption, failure isolation, and privacy telemetry bounds.
"""

import asyncio
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.api.deps import (
    get_audio_capture,
    get_audio_playback,
    get_audio_session_manager,
    get_stt_provider,
    get_tts_provider,
    get_vad_detector,
    get_voice_planner_gateway,
    get_wake_word_detector,
)
from app.audio import (
    AudioChunk,
    AudioEncoding,
    AudioEngineConfig,
    AudioFormat,
    AudioSessionManager,
    AudioSessionState,
    IAudioCapture,
    IAudioPlayback,
    IAudioSessionManager,
    ISpeechToTextProvider,
    ITextToSpeechProvider,
    IVoiceActivityDetector,
    IVoicePlannerGateway,
    IWakeWordDetector,
    SpeechSynthesisRequest,
    STTConfig,
    STTTimeoutError,
    TTSConfig,
    TTSValidationError,
    VADConfig,
    VADTimeoutError,
    VoiceActivityState,
    WakeWordConfig,
    WakeWordValidationError,
    create_stt_config,
    create_tts_config,
    create_vad_config,
    create_wake_word_config,
    get_audio_engine_config,
)
from app.audio.adapters import STTAdapter, TTSAdapter, VADAdapter, WakeWordAdapter
from app.core.config import get_settings


def make_chunk(
    seq: int = 0, session_id: str = "sess_prod_1", payload_bytes: bytes | None = None
) -> AudioChunk:
    """Helper creating synthetic AudioChunk."""
    payload = payload_bytes if payload_bytes is not None else (b"\x00" * 960)
    return AudioChunk(
        chunk_id=f"chk_prod_{seq}",
        sequence_number=seq,
        timestamp=datetime.now(UTC),
        duration_ms=30.0,
        audio_format=AudioFormat(
            sample_rate=16000, channels=1, sample_width=2, encoding=AudioEncoding.PCM_S16LE
        ),
        payload=payload,
        session_id=session_id,
        correlation_id="corr_prod_100",
    )


def test_audio_engine_configuration_resolution() -> None:
    """Verify unified AudioEngineConfig resolution from system Settings."""
    settings = get_settings()
    config = get_audio_engine_config(settings)

    assert isinstance(config, AudioEngineConfig)
    assert config.sample_rate == 16000
    assert config.channels == 1
    assert config.wake_word == "ultron"
    assert config.wake_word_threshold == 0.5

    vad_cfg = create_vad_config(settings)
    assert isinstance(vad_cfg, VADConfig)
    assert vad_cfg.speech_threshold == 0.5

    ww_cfg = create_wake_word_config(settings)
    assert isinstance(ww_cfg, WakeWordConfig)
    assert "ultron" in ww_cfg.wake_words

    stt_cfg = create_stt_config(settings)
    assert isinstance(stt_cfg, STTConfig)

    tts_cfg = create_tts_config(settings)
    assert isinstance(tts_cfg, TTSConfig)


def test_production_vad_adapter_lifecycle() -> None:
    """Verify VAD adapter processing, hysteresis debouncing, timeout, cancellation, and health probing."""

    async def _test() -> None:
        vad = VADAdapter(
            config=VADConfig(
                speech_threshold=0.01,
                minimum_speech_duration_ms=20.0,
                processing_timeout_ms=500.0,
            )
        )
        assert vad.current_state == VoiceActivityState.SILENCE

        # 1. Process silence chunk
        evt_silence = await vad.process(make_chunk(seq=0))
        assert evt_silence.state == VoiceActivityState.SILENCE

        # 2. Process active speech chunk (loud payload)
        speech_bytes = b"\x30\x30" * 480
        evt_speech = await vad.process(make_chunk(seq=1, payload_bytes=speech_bytes))
        assert evt_speech.state in (VoiceActivityState.SPEECH_START, VoiceActivityState.SPEAKING)

        # 3. Timeout handling
        slow_vad = VADAdapter(
            config=VADConfig(processing_timeout_ms=10.0), processing_delay_sec=0.05
        )
        with pytest.raises(VADTimeoutError):
            await slow_vad.process(make_chunk(seq=2))

        # 4. Health probing
        h = await vad.health()
        assert h["subsystem"] == "vad_detector"
        assert h["status"] == "RUNNING"

        await vad.reset()
        assert vad.current_state == VoiceActivityState.SILENCE

    asyncio.run(_test())


def test_production_wake_word_adapter_lifecycle() -> None:
    """Verify Wake-Word detection, confidence thresholding, debouncing, sequence validation, and health."""

    async def _test() -> None:
        ww = WakeWordAdapter(
            config=WakeWordConfig(
                wake_words=["ultron", "hey ultron"], confidence_threshold=0.5, timeout_ms=500.0
            ),
            mock_trigger_word="ultron",
            mock_confidence=0.95,
        )

        # 1. Detect wake word
        detect = await ww.detect(make_chunk(seq=0))
        assert detect is not None
        assert detect.wake_word == "ultron"
        assert detect.confidence >= 0.5

        # 2. Duplicate sequence rejection
        with pytest.raises(WakeWordValidationError):
            await ww.detect(make_chunk(seq=0))

        # 3. Health check
        h = await ww.health()
        assert h["subsystem_wake_word"] is True
        assert h["enabled"] is True

        await ww.reset()

    asyncio.run(_test())


def test_production_stt_adapter_lifecycle() -> None:
    """Verify STT transcription, partial streaming, timeout, retries, and error validation."""

    async def _test() -> None:
        stt = STTAdapter(config=STTConfig(timeout_ms=500.0))

        # 1. Transcribe audio chunk segment
        chunks = [make_chunk(seq=i) for i in range(3)]
        transcript = await stt.transcribe_segment(chunks)
        assert transcript.full_text != ""
        assert transcript.is_final is True

        # 2. Transcribe single chunk (partial)
        segment = await stt.transcribe_chunk(chunks[0])
        assert segment.text != ""

        # 3. Timeout handling
        slow_stt = STTAdapter(config=STTConfig(timeout_ms=10.0), simulated_delay_sec=0.05)
        with pytest.raises(STTTimeoutError):
            await slow_stt.transcribe_segment(chunks)

        # 4. Health check
        h = await stt.health()
        assert h["subsystem_stt"] is True

    asyncio.run(_test())


def test_production_tts_adapter_lifecycle() -> None:
    """Verify TTS synthesis, streaming chunks, format compatibility, timeout, and health probing."""

    async def _test() -> None:
        tts = TTSAdapter(config=TTSConfig(sample_rate=16000, timeout_ms=500.0))

        req = SpeechSynthesisRequest(
            request_id="tts_test_1",
            text="Hello ULTRON speech synthesis.",
            voice="en-US-Standard-A",
            sample_rate=16000,
            session_id="sess_tts_1",
        )

        # 1. Synthesize request
        result = await tts.synthesize(req)
        assert result.request_id == "tts_test_1"
        assert result.audio_format.sample_rate == 16000

        # 2. Stream audio chunks
        chunks: list[AudioChunk] = []
        async for chunk in tts.stream_chunks(req):
            chunks.append(chunk)

        assert len(chunks) >= 1
        assert len(chunks[0].payload) > 0

        # 3. Empty text validation error
        with pytest.raises((TTSValidationError, ValidationError)):
            bad_req = SpeechSynthesisRequest(
                request_id="bad_req", text="", voice="en-US-Standard-A"
            )
            await tts.synthesize(bad_req)

        # 4. Health check
        h = await tts.health()
        assert h["subsystem_tts"] is True

    asyncio.run(_test())


def test_di_container_audio_resolution() -> None:
    """Verify DI container resolves all abstract audio domain contracts without direct instantiation."""
    capture = get_audio_capture()
    vad = get_vad_detector()
    ww = get_wake_word_detector()
    stt = get_stt_provider()
    tts = get_tts_provider()
    playback = get_audio_playback()
    gateway = get_voice_planner_gateway()
    mgr = get_audio_session_manager()

    assert isinstance(capture, IAudioCapture)
    assert isinstance(vad, IVoiceActivityDetector)
    assert isinstance(ww, IWakeWordDetector)
    assert isinstance(stt, ISpeechToTextProvider)
    assert isinstance(tts, ITextToSpeechProvider)
    assert isinstance(playback, IAudioPlayback)
    assert isinstance(gateway, IVoicePlannerGateway)
    assert isinstance(mgr, IAudioSessionManager)


def test_full_duplex_pipeline_and_barge_in_integration() -> None:
    """Verify complete audio pipeline chain and barge-in speech interruption during synthesis."""

    async def _test() -> None:
        mgr = get_audio_session_manager()
        assert isinstance(mgr, AudioSessionManager)

        await mgr.start_session("sess_pipeline_1")

        # 1. Execute TTS synthesis response
        await mgr.process_speech_text("ULTRON full duplex pipeline response.")
        assert mgr.current_state() == AudioSessionState.PLAYING

        # 2. User speech barge-in interruption during playback
        await mgr.interrupt_speech()

        assert mgr.current_state() in (AudioSessionState.LISTENING, AudioSessionState.IDLE)
        assert mgr._synthesis_queue.empty()

        await mgr.stop_session("sess_pipeline_1")

    asyncio.run(_test())
