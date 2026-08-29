"""Unit Tests for Text-to-Speech (TTS) Subsystem and Adapter (Phase 4C.5).

Validates TTSConfig parameters, input text validation, voice/speed/pitch checking, synthesis execution,
streaming AudioChunk generation, monotonic sequence numbering, transient retries, timeout protection,
cancellation cleanup, privacy protection (no raw payload leakage), and ITextToSpeechProvider compliance.
"""

import asyncio

import pytest
from pydantic import ValidationError

from app.audio import (
    AudioChunk,
    ITextToSpeechProvider,
    SpeechSynthesisRequest,
    SpeechSynthesisResult,
)
from app.audio.adapters.tts import TTSAdapter
from app.audio.tts import (
    TTSConfig,
    TTSProviderError,
    TTSTimeoutError,
    TTSValidationError,
)


def make_request(
    text: str = "Hello ULTRON speech test.",
    voice: str = "en-US-Standard-A",
    speed: float = 1.0,
    pitch: float = 0.0,
) -> SpeechSynthesisRequest:
    """Helper creating synthetic SpeechSynthesisRequest."""
    return SpeechSynthesisRequest(
        request_id="tts_req_101",
        text=text,
        voice=voice,
        speed=speed,
        pitch=pitch,
        session_id="sess_tts_1",
        correlation_id="corr_tts_100",
    )


@pytest.fixture
def tts() -> TTSAdapter:
    return TTSAdapter(config=TTSConfig(timeout_ms=500.0, retry_count=1))


def test_tts_config_validation() -> None:
    """Verify TTSConfig parameters validation rules 1-6."""
    cfg = TTSConfig(provider_name="test_tts", speed=1.5, pitch=-5.0)
    assert cfg.provider_name == "test_tts"
    assert cfg.speed == 1.5

    # Invalid speed < 0.25
    with pytest.raises(ValidationError):
        TTSConfig(speed=0.1)

    # Invalid pitch > 20.0
    with pytest.raises(ValidationError):
        TTSConfig(pitch=25.0)

    # Invalid timeout <= 0
    with pytest.raises(ValidationError):
        TTSConfig(timeout_ms=0.0)

    # Invalid retry_count < 0
    with pytest.raises(ValidationError):
        TTSConfig(retry_count=-1)

    # Invalid sample rate
    with pytest.raises(ValidationError):
        TTSConfig(sample_rate=12345)


def test_text_and_parameter_validation(tts: TTSAdapter) -> None:
    """Verify input validation rules 7-11 (empty text, speed/pitch bounds, unsupported voice)."""

    async def _test() -> None:
        # 7. Empty text model validation
        with pytest.raises((TTSValidationError, ValidationError)):
            make_request(text="")

        # 8. Whitespace only text adapter validation
        with pytest.raises(TTSValidationError, match="empty or whitespace-only"):
            await tts.synthesize(make_request(text="   \n\t "))

        # 9. Excessive text length
        short_cfg_tts = TTSAdapter(config=TTSConfig(max_text_length=10))
        with pytest.raises(TTSValidationError, match="exceeds maximum limit"):
            await short_cfg_tts.synthesize(make_request(text="This text is way too long!"))

        # 10. Unsupported voice
        with pytest.raises(TTSValidationError, match="Unsupported TTS voice"):
            await tts.synthesize(make_request(voice="unsupported_voice_name"))

        # 27. Invalid speed rejection (> 4.0)
        with pytest.raises((TTSValidationError, ValidationError)):
            make_request(speed=5.0)

        # 28. Invalid pitch rejection (< -20.0)
        with pytest.raises((TTSValidationError, ValidationError)):
            make_request(pitch=-30.0)

    asyncio.run(_test())


def test_successful_synthesis_and_metadata(tts: TTSAdapter) -> None:
    """Verify synthesis execution and metadata mapping rules 12-16 & 24-26."""

    async def _test() -> None:
        req = make_request(speed=1.0, pitch=0.0)
        res = await tts.synthesize(req)

        assert isinstance(res, SpeechSynthesisResult)
        assert res.request_id == "tts_req_101"
        assert res.duration_ms > 0.0
        assert len(res.audio_payload) > 0
        assert res.session_id == "sess_tts_1"
        assert res.correlation_id == "corr_tts_100"

        # 24-26. Boundary speed and pitch
        req_bounds = make_request(speed=0.25, pitch=-20.0)
        res_bounds = await tts.synthesize(req_bounds)
        assert res_bounds.duration_ms > 0.0

    asyncio.run(_test())


def test_streaming_and_monotonic_chunks(tts: TTSAdapter) -> None:
    """Verify streaming AudioChunk generation and monotonic sequence numbers (rules 17-23)."""

    async def _test() -> None:
        req = make_request()

        chunks: list[AudioChunk] = []
        async for chunk in tts.stream_chunks(req):
            chunks.append(chunk)

        assert len(chunks) >= 1
        seqs = [c.sequence_number for c in chunks]
        assert seqs == list(range(len(chunks)))  # Monotonic starting at 0: [0, 1, 2, ...]

        # Test stream(request) bytes generator
        byte_chunks: list[bytes] = []
        async for b in tts.stream(req):
            byte_chunks.append(b)

        assert len(byte_chunks) == len(chunks)

    asyncio.run(_test())


def test_transient_failure_retry_and_exhaustion() -> None:
    """Verify transient error retries up to retry_count (rules 34-35)."""

    async def _test() -> None:
        # Retry count = 1, transient failure = 1 -> succeeds on retry
        retry_tts = TTSAdapter(config=TTSConfig(retry_count=1), transient_failures_to_simulate=1)
        res = await retry_tts.synthesize(make_request())
        assert isinstance(res, SpeechSynthesisResult)

        # Retry count = 1, transient failure = 2 -> raises TTSProviderError
        fail_tts = TTSAdapter(config=TTSConfig(retry_count=1), transient_failures_to_simulate=2)
        with pytest.raises(TTSProviderError, match="transient TTS network failure"):
            await fail_tts.synthesize(make_request())

    asyncio.run(_test())


def test_timeout_protection() -> None:
    """Verify TTSTimeoutError is raised when synthesis exceeds timeout_ms (rule 33)."""

    async def _test() -> None:
        timeout_tts = TTSAdapter(config=TTSConfig(timeout_ms=10.0), simulated_delay_sec=0.10)
        with pytest.raises(TTSTimeoutError):
            await timeout_tts.synthesize(make_request())

    asyncio.run(_test())


def test_cancellation_handling() -> None:
    """Verify asyncio.CancelledError during TTS request is re-raised cleanly (rules 36-38)."""

    async def _test() -> None:
        slow_tts = TTSAdapter(simulated_delay_sec=0.10)
        task = asyncio.create_task(slow_tts.synthesize(make_request()))
        await asyncio.sleep(0.01)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(_test())


def test_privacy_guarantees(tts: TTSAdapter) -> None:
    """Verify telemetry and health exclude raw audio bytes and full spoken text (rules 39-41)."""

    async def _test() -> None:
        req = make_request()
        await tts.synthesize(req)

        health = await tts.health()
        assert health["subsystem_tts"] is True

        telemetry = tts._last_telemetry
        assert telemetry is not None
        assert "audio_payload" not in telemetry.model_dump()
        assert "text" not in telemetry.model_dump()

    asyncio.run(_test())


def test_interface_compliance_and_determinism(tts: TTSAdapter) -> None:
    """Verify ITextToSpeechProvider compliance and output determinism (rules 46-47)."""

    async def _test() -> None:
        assert isinstance(tts, ITextToSpeechProvider)

        req = make_request()
        r1 = await tts.synthesize(req)
        r2 = await tts.synthesize(req)

        assert r1.duration_ms == r2.duration_ms
        assert r1.audio_format.sample_rate == r2.audio_format.sample_rate

    asyncio.run(_test())
