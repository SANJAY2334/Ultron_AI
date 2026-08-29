"""Unit Tests for Speech-to-Text (STT) Subsystem and Adapter (Phase 4C.4).

Validates STTConfig parameters, input AudioChunk list validation, out-of-order chunk rejection,
transcription execution, streaming partial & final transcripts, transient retries, timeout protection,
cancellation cleanup, privacy protection (no raw payload leakage), and ISpeechToTextProvider compliance.
"""

import asyncio
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.audio import AudioChunk, AudioFormat, ISpeechToTextProvider, Transcript, TranscriptSegment
from app.audio.adapters.stt import STTAdapter
from app.audio.stt import (
    STTConfig,
    STTProviderError,
    STTTimeoutError,
    STTValidationError,
)


def make_chunk(seq: int = 0, duration_ms: float = 30.0, session_id: str = "sess_1") -> AudioChunk:
    """Helper creating synthetic AudioChunk."""
    return AudioChunk(
        chunk_id=f"chk_stt_{seq}",
        sequence_number=seq,
        timestamp=datetime.now(UTC),
        duration_ms=duration_ms,
        audio_format=AudioFormat(sample_rate=16000, channels=1, sample_width=2),
        payload=b"\x00" * 960,
        session_id=session_id,
        correlation_id="corr_100",
    )


@pytest.fixture
def stt() -> STTAdapter:
    return STTAdapter(config=STTConfig(timeout_ms=500.0, retry_count=1))


def test_stt_config_validation() -> None:
    """Verify STTConfig parameters validation rules 1-5."""
    cfg = STTConfig(provider_name="test_p", timeout_ms=1000.0, confidence_threshold=0.5)
    assert cfg.provider_name == "test_p"
    assert cfg.language == "en"

    # Invalid timeout <= 0
    with pytest.raises(ValidationError):
        STTConfig(timeout_ms=0.0)

    # Invalid confidence threshold > 1.0
    with pytest.raises(ValidationError):
        STTConfig(confidence_threshold=1.5)

    # Invalid max_transcript_length <= 0
    with pytest.raises(ValidationError):
        STTConfig(max_transcript_length=0)

    # Invalid retry_count < 0
    with pytest.raises(ValidationError):
        STTConfig(retry_count=-1)


def test_audio_input_validation(stt: STTAdapter) -> None:
    """Verify input validation rules 6-12 (empty list, out of order, max duration/bytes/chunks)."""

    async def _test() -> None:
        # 6. Empty audio list
        with pytest.raises(STTValidationError, match="empty list"):
            await stt.transcribe_segment([])

        # 9. Out-of-order sequence numbers (seq 1 before seq 0)
        c0 = make_chunk(seq=5)
        c1 = make_chunk(seq=2)
        with pytest.raises(STTValidationError, match="Out-of-order"):
            await stt.transcribe_segment([c0, c1])

        # 11. Excessive audio duration
        short_cfg_stt = STTAdapter(config=STTConfig(max_audio_duration_ms=50.0))
        c_long = make_chunk(seq=0, duration_ms=100.0)
        with pytest.raises(STTValidationError, match="exceeds maximum limit"):
            await short_cfg_stt.transcribe_segment([c_long])

        # 38. Maximum chunk count limit
        small_chunks_stt = STTAdapter(config=STTConfig(max_chunks=2))
        with pytest.raises(STTValidationError, match="AudioChunk count"):
            await small_chunks_stt.transcribe_segment([make_chunk(0), make_chunk(1), make_chunk(2)])

    asyncio.run(_test())


def test_successful_transcription_and_metadata(stt: STTAdapter) -> None:
    """Verify transcription execution and metadata mapping (rules 13-17)."""

    async def _test() -> None:
        c0 = make_chunk(seq=0, duration_ms=30.0)
        c1 = make_chunk(seq=1, duration_ms=30.0)

        res = await stt.transcribe_segment([c0, c1])
        assert isinstance(res, Transcript)
        assert res.is_final is True
        assert res.language == "en"
        assert res.confidence == 0.95
        assert res.session_id == "sess_1"
        assert res.correlation_id == "corr_100"
        assert len(res.segments) == 1
        assert res.segments[0].is_final is True

    asyncio.run(_test())


def test_streaming_and_partial_transcripts(stt: STTAdapter) -> None:
    """Verify streaming partial vs final transcript behavior (rules 18-22)."""

    async def _test() -> None:
        c0 = make_chunk(seq=0)

        # Single partial chunk transcription
        part_seg = await stt.transcribe_chunk(c0)
        assert isinstance(part_seg, TranscriptSegment)
        assert part_seg.is_final is False
        assert "Partial" in part_seg.text

    asyncio.run(_test())


def test_transient_failure_retry_and_exhaustion() -> None:
    """Verify transient error retries up to retry_count (rules 28-29)."""

    async def _test() -> None:
        # Simulate 1 transient failure with retry_count=1 -> succeeds on 2nd attempt
        retry_stt = STTAdapter(config=STTConfig(retry_count=1), transient_failures_to_simulate=1)
        res = await retry_stt.transcribe_segment([make_chunk(0)])
        assert isinstance(res, Transcript)

        # Simulate 2 transient failures with retry_count=1 -> raises STTProviderError
        fail_stt = STTAdapter(config=STTConfig(retry_count=1), transient_failures_to_simulate=2)
        with pytest.raises(STTProviderError, match="transient STT network failure"):
            await fail_stt.transcribe_segment([make_chunk(0)])

    asyncio.run(_test())


def test_timeout_protection() -> None:
    """Verify STTTimeoutError is raised when transcription exceeds timeout_ms (rule 27)."""

    async def _test() -> None:
        timeout_stt = STTAdapter(config=STTConfig(timeout_ms=10.0), simulated_delay_sec=0.10)
        with pytest.raises(STTTimeoutError):
            await timeout_stt.transcribe_segment([make_chunk(0)])

    asyncio.run(_test())


def test_cancellation_handling() -> None:
    """Verify asyncio.CancelledError during STT request is re-raised cleanly (rules 30-32)."""

    async def _test() -> None:
        slow_stt = STTAdapter(simulated_delay_sec=0.10)
        task = asyncio.create_task(slow_stt.transcribe_segment([make_chunk(0)]))
        await asyncio.sleep(0.01)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(_test())


def test_privacy_guarantees(stt: STTAdapter) -> None:
    """Verify telemetry and health exclude raw audio bytes and full text (rules 33-35)."""

    async def _test() -> None:
        c0 = make_chunk(seq=0)
        await stt.transcribe_segment([c0])

        health = await stt.health()
        assert health["subsystem_stt"] is True

        telemetry = stt._last_telemetry
        assert telemetry is not None
        assert "payload" not in telemetry.model_dump()
        assert "full_text" not in telemetry.model_dump()

    asyncio.run(_test())


def test_interface_compliance_and_determinism(stt: STTAdapter) -> None:
    """Verify ISpeechToTextProvider compliance and output determinism (rules 36-37)."""

    async def _test() -> None:
        assert isinstance(stt, ISpeechToTextProvider)

        c0 = make_chunk(seq=0)
        t1 = await stt.transcribe_segment([c0])
        t2 = await stt.transcribe_segment([c0])

        assert t1.confidence == t2.confidence
        assert t1.is_final == t2.is_final
        assert t1.language == t2.language

    asyncio.run(_test())
