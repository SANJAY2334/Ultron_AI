"""Unit Tests for Wake Word Detection Subsystem and Adapter (Phase 4C.6).

Validates WakeWordConfig parameters, monotonic sequence validation, rolling buffer window limits,
confidence thresholding, cooldown duplicate suppression, transient retries, timeout protection,
cancellation cleanup, privacy protection (no raw payload leakage), and IWakeWordDetector compliance.
"""

import asyncio
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.audio import AudioChunk, AudioFormat, IWakeWordDetector, WakeWordDetection
from app.audio.adapters.wake_word import WakeWordAdapter
from app.audio.wake_word import (
    WakeWordConfig,
    WakeWordProviderError,
    WakeWordTimeoutError,
    WakeWordValidationError,
)


def make_chunk(
    seq: int = 0, duration_ms: float = 30.0, session_id: str = "sess_ww_1"
) -> AudioChunk:
    """Helper creating synthetic AudioChunk."""
    return AudioChunk(
        chunk_id=f"chk_ww_{seq}",
        sequence_number=seq,
        timestamp=datetime.now(UTC),
        duration_ms=duration_ms,
        audio_format=AudioFormat(sample_rate=16000, channels=1, sample_width=2),
        payload=b"\x00" * 960,
        session_id=session_id,
        correlation_id="corr_ww_100",
    )


@pytest.fixture
def ww_adapter() -> WakeWordAdapter:
    cfg = WakeWordConfig(
        wake_words=["hey ultron", "ultron"],
        confidence_threshold=0.5,
        cooldown_ms=100.0,
        timeout_ms=500.0,
        retry_count=1,
    )
    return WakeWordAdapter(config=cfg, mock_trigger_word="hey ultron", mock_confidence=0.95)


def test_wake_word_config_validation() -> None:
    """Verify WakeWordConfig parameter validation rules 1-8."""
    cfg = WakeWordConfig(wake_words=["ultron"], confidence_threshold=0.6)
    assert cfg.wake_words == ["ultron"]
    assert cfg.confidence_threshold == 0.6

    # 2. Empty wake_words list
    with pytest.raises(ValidationError):
        WakeWordConfig(wake_words=[])

    # 3. Invalid confidence threshold > 1.0
    with pytest.raises(ValidationError):
        WakeWordConfig(confidence_threshold=1.5)

    # 4. Invalid cooldown < 0
    with pytest.raises(ValidationError):
        WakeWordConfig(cooldown_ms=-10.0)

    # 5. Invalid timeout <= 0
    with pytest.raises(ValidationError):
        WakeWordConfig(timeout_ms=0.0)

    # 6. Invalid retry_count < 0
    with pytest.raises(ValidationError):
        WakeWordConfig(retry_count=-1)

    # 7. Invalid max_buffered_chunks <= 0
    with pytest.raises(ValidationError):
        WakeWordConfig(max_buffered_chunks=0)

    # 8. Invalid max_detection_window_ms <= 0
    with pytest.raises(ValidationError):
        WakeWordConfig(max_detection_window_ms=0.0)


def test_detection_and_confidence_threshold(ww_adapter: WakeWordAdapter) -> None:
    """Verify successful wake-word detection, metadata mapping, and confidence thresholding (rules 9-14)."""

    async def _test() -> None:
        c0 = make_chunk(seq=0)
        res = await ww_adapter.detect(c0)

        assert isinstance(res, WakeWordDetection)
        assert res.wake_word == "hey ultron"
        assert res.confidence == 0.95
        assert res.session_id == "sess_ww_1"
        assert res.correlation_id == "corr_ww_100"

        # 11. Below-threshold detection rejection
        low_conf_adapter = WakeWordAdapter(
            config=WakeWordConfig(confidence_threshold=0.8),
            mock_trigger_word="hey ultron",
            mock_confidence=0.5,  # 0.5 < 0.8 -> returns None
        )
        res_low = await low_conf_adapter.detect(make_chunk(seq=0))
        assert res_low is None

        # 13. Unknown wake word rejection
        unmatched_adapter = WakeWordAdapter(
            config=WakeWordConfig(wake_words=["ultron"]),
            mock_trigger_word="unmatched_word",
        )
        res_unmatched = await unmatched_adapter.detect(make_chunk(seq=0))
        assert res_unmatched is None

    asyncio.run(_test())


def test_cooldown_duplicate_suppression(ww_adapter: WakeWordAdapter) -> None:
    """Verify cooldown window suppresses duplicate trigger activations (rules 15-17)."""

    async def _test() -> None:
        # First trigger succeeds
        res1 = await ww_adapter.detect(make_chunk(seq=0))
        assert isinstance(res1, WakeWordDetection)

        # Immediate second trigger suppressed by cooldown_ms=100.0
        res2 = await ww_adapter.detect(make_chunk(seq=1))
        assert res2 is None

        # Wait for cooldown to expire
        await asyncio.sleep(0.12)

        # Third trigger succeeds
        res3 = await ww_adapter.detect(make_chunk(seq=2))
        assert isinstance(res3, WakeWordDetection)

    asyncio.run(_test())


def test_audio_sequence_validation_and_buffer_limits() -> None:
    """Verify monotonic sequence numbers, regression rejection, and buffer window limits (rules 18-23)."""

    async def _test() -> None:
        adapter = WakeWordAdapter(config=WakeWordConfig(max_buffered_chunks=3))

        await adapter.detect(make_chunk(seq=0))
        await adapter.detect(make_chunk(seq=1))

        # 19. Duplicate sequence number rejection
        with pytest.raises(WakeWordValidationError, match="regression"):
            await adapter.detect(make_chunk(seq=1))

        # 20. Sequence regression rejection (seq 0 after seq 1)
        with pytest.raises(WakeWordValidationError, match="regression"):
            await adapter.detect(make_chunk(seq=0))

        # 21. Malformed audio (empty payload validation)
        with pytest.raises((WakeWordValidationError, ValidationError)):
            AudioChunk(
                chunk_id="bad_chunk",
                sequence_number=10,
                timestamp=datetime.now(UTC),
                duration_ms=30.0,
                audio_format=AudioFormat(),
                payload=b"",
            )

        # 23. Buffer limit enforcement
        await adapter.detect(make_chunk(seq=2))
        await adapter.detect(make_chunk(seq=3))
        health = await adapter.health()
        assert health["buffered_chunks"] == 3

    asyncio.run(_test())


def test_transient_failure_retry_and_exhaustion() -> None:
    """Verify transient error retries up to retry_count (rules 29-30)."""

    async def _test() -> None:
        # Retry count = 1, transient failure = 1 -> succeeds on retry
        retry_adapter = WakeWordAdapter(
            config=WakeWordConfig(retry_count=1),
            mock_trigger_word="hey ultron",
            transient_failures_to_simulate=1,
        )
        res = await retry_adapter.detect(make_chunk(seq=0))
        assert isinstance(res, WakeWordDetection)

        # Retry count = 1, transient failure = 2 -> raises WakeWordProviderError
        fail_adapter = WakeWordAdapter(
            config=WakeWordConfig(retry_count=1),
            mock_trigger_word="hey ultron",
            transient_failures_to_simulate=2,
        )
        with pytest.raises(WakeWordProviderError, match="transient wake-word network failure"):
            await fail_adapter.detect(make_chunk(seq=0))

    asyncio.run(_test())


def test_timeout_protection() -> None:
    """Verify WakeWordTimeoutError is raised when evaluation exceeds timeout_ms (rule 28)."""

    async def _test() -> None:
        timeout_adapter = WakeWordAdapter(
            config=WakeWordConfig(timeout_ms=10.0), simulated_delay_sec=0.10
        )
        with pytest.raises(WakeWordTimeoutError):
            await timeout_adapter.detect(make_chunk(seq=0))

    asyncio.run(_test())


def test_cancellation_handling() -> None:
    """Verify asyncio.CancelledError during evaluation is re-raised cleanly (rules 31-33)."""

    async def _test() -> None:
        slow_adapter = WakeWordAdapter(simulated_delay_sec=0.10)
        task = asyncio.create_task(slow_adapter.detect(make_chunk(seq=0)))
        await asyncio.sleep(0.01)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(_test())


def test_privacy_guarantees(ww_adapter: WakeWordAdapter) -> None:
    """Verify telemetry and health exclude raw audio bytes (rules 34-36)."""

    async def _test() -> None:
        c0 = make_chunk(seq=0)
        await ww_adapter.detect(c0)

        health = await ww_adapter.health()
        assert health["subsystem_wake_word"] is True

        telemetry = ww_adapter._last_telemetry
        assert telemetry is not None
        assert "payload" not in telemetry.model_dump()
        assert "audio_bytes" not in telemetry.model_dump()

    asyncio.run(_test())


def test_interface_compliance_and_determinism(ww_adapter: WakeWordAdapter) -> None:
    """Verify IWakeWordDetector compliance and output determinism (rules 37-38)."""

    async def _test() -> None:
        assert isinstance(ww_adapter, IWakeWordDetector)

        c0 = make_chunk(seq=0)
        res1 = await ww_adapter.detect(c0)

        await ww_adapter.reset()
        res2 = await ww_adapter.detect(c0)

        assert res1 is not None and res2 is not None
        assert res1.wake_word == res2.wake_word
        assert res1.confidence == res2.confidence

    asyncio.run(_test())
