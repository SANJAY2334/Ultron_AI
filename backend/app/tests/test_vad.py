"""Unit Tests for Voice Activity Detection (VAD) Adapter (Phase 4C.3).

Validates VADAdapter state transitions (SILENCE -> SPEECH_START -> SPEAKING -> SPEECH_END -> SILENCE),
threshold confidence classification, noise burst filtering, hysteresis debouncing, timeout handling,
cancellation cleanup, privacy protection (no raw payload leakage), and IVoiceActivityDetector contract.
"""

import asyncio
import struct
from datetime import UTC, datetime

import pytest

from app.audio import AudioChunk, AudioFormat, IVoiceActivityDetector, VoiceActivityState
from app.audio.adapters.vad import VADAdapter
from app.audio.vad import VADConfig, VADProcessingError, VADTimeoutError


def make_pcm_chunk(amplitude: int, duration_ms: float = 30.0, seq: int = 0) -> AudioChunk:
    """Helper creating synthetic 16-bit PCM audio chunk with specified amplitude."""
    num_samples = int(16000 * (duration_ms / 1000.0))
    fmt = f"<{num_samples}h"
    samples = [amplitude] * num_samples
    payload = struct.pack(fmt, *samples)

    return AudioChunk(
        chunk_id=f"chk_vad_{seq}",
        sequence_number=seq,
        timestamp=datetime.now(UTC),
        duration_ms=duration_ms,
        audio_format=AudioFormat(sample_rate=16000, channels=1, sample_width=2),
        payload=payload,
    )


@pytest.fixture
def vad() -> VADAdapter:
    cfg = VADConfig(
        speech_threshold=0.5,
        minimum_speech_duration_ms=60.0,  # 2 chunks of 30ms
        minimum_silence_duration_ms=90.0,  # 3 chunks of 30ms
        debounce_ms=50.0,
        processing_timeout_ms=100.0,
    )
    return VADAdapter(config=cfg)


def test_basic_classification_and_state_transitions(vad: VADAdapter) -> None:
    """Verify full state transition pipeline: SILENCE -> SPEECH_START -> SPEAKING -> SPEECH_END -> SILENCE."""

    async def _test() -> None:
        # 1. Silence -> SILENCE
        c_sil1 = make_pcm_chunk(amplitude=100, seq=0)
        evt1 = await vad.process(c_sil1)
        assert evt1.state == VoiceActivityState.SILENCE

        # 2. Speech chunk 1 (30ms accumulated, below 60ms threshold -> stays SILENCE)
        c_sp1 = make_pcm_chunk(amplitude=20000, seq=1)
        evt2 = await vad.process(c_sp1)
        assert evt2.state == VoiceActivityState.SILENCE

        # 3. Speech chunk 2 (60ms accumulated -> triggers SPEECH_START)
        c_sp2 = make_pcm_chunk(amplitude=20000, seq=2)
        evt3 = await vad.process(c_sp2)
        assert evt3.state == VoiceActivityState.SPEECH_START

        # 4. Speech chunk 3 -> SPEAKING
        c_sp3 = make_pcm_chunk(amplitude=20000, seq=3)
        evt4 = await vad.process(c_sp3)
        assert evt4.state == VoiceActivityState.SPEAKING

        # 5. Silence chunk 1 (30ms silence accumulated < 90ms -> stays SPEAKING hysteresis)
        c_sil2 = make_pcm_chunk(amplitude=100, seq=4)
        evt5 = await vad.process(c_sil2)
        assert evt5.state == VoiceActivityState.SPEAKING

        # 6. Silence chunk 2 (60ms silence) -> stays SPEAKING
        c_sil3 = make_pcm_chunk(amplitude=100, seq=5)
        evt6 = await vad.process(c_sil3)
        assert evt6.state == VoiceActivityState.SPEAKING

        # 7. Silence chunk 3 (90ms silence -> triggers SPEECH_END)
        c_sil4 = make_pcm_chunk(amplitude=100, seq=6)
        evt7 = await vad.process(c_sil4)
        assert evt7.state == VoiceActivityState.SPEECH_END

        # 8. Silence chunk 4 -> SILENCE
        c_sil5 = make_pcm_chunk(amplitude=100, seq=7)
        evt8 = await vad.process(c_sil5)
        assert evt8.state == VoiceActivityState.SILENCE

    asyncio.run(_test())


def test_confidence_threshold_boundary(vad: VADAdapter) -> None:
    """Verify confidence classification below, at, and above speech_threshold."""

    async def _test() -> None:
        # High amplitude -> high confidence
        c_high = make_pcm_chunk(amplitude=24000)
        e_high = await vad.process(c_high)
        assert e_high.confidence >= 0.5

        await vad.reset()

        # Low amplitude -> low confidence
        c_low = make_pcm_chunk(amplitude=500)
        e_low = await vad.process(c_low)
        assert e_low.confidence < 0.5

    asyncio.run(_test())


def test_noise_burst_filtering(vad: VADAdapter) -> None:
    """Verify short noise burst below minimum_speech_duration_ms does not trigger SPEECH_START."""

    async def _test() -> None:
        # Single 30ms speech chunk (minimum_speech_duration_ms is 60ms)
        c_noise = make_pcm_chunk(amplitude=25000, duration_ms=30.0, seq=0)
        evt1 = await vad.process(c_noise)
        assert evt1.state == VoiceActivityState.SILENCE

        # Immediately followed by silence
        c_sil = make_pcm_chunk(amplitude=100, duration_ms=30.0, seq=1)
        evt2 = await vad.process(c_sil)
        assert evt2.state == VoiceActivityState.SILENCE

    asyncio.run(_test())


def test_silence_hysteresis_and_debouncing(vad: VADAdapter) -> None:
    """Verify brief inter-word quiet pauses do not prematurely trigger SPEECH_END."""

    async def _test() -> None:
        # Establish SPEAKING state
        await vad.process(make_pcm_chunk(25000, seq=0))
        await vad.process(make_pcm_chunk(25000, seq=1))
        assert vad.current_state == VoiceActivityState.SPEECH_START

        await vad.process(make_pcm_chunk(25000, seq=2))
        assert vad.current_state == VoiceActivityState.SPEAKING

        # Brief 30ms silence (less than minimum_silence_duration_ms=90ms)
        evt_pause = await vad.process(make_pcm_chunk(100, seq=3))
        assert evt_pause.state == VoiceActivityState.SPEAKING

        # Resume speech immediately
        evt_resume = await vad.process(make_pcm_chunk(25000, seq=4))
        assert evt_resume.state == VoiceActivityState.SPEAKING

    asyncio.run(_test())


def test_malformed_chunk_and_error_handling(vad: VADAdapter) -> None:
    """Verify malformed audio payload raises VADProcessingError."""

    async def _test() -> None:
        # Misaligned 3-byte payload for 16-bit PCM (sample_width=2 requires even byte count)
        bad_chunk = AudioChunk(
            chunk_id="bad_chk",
            sequence_number=99,
            timestamp=datetime.now(UTC),
            duration_ms=30.0,
            audio_format=AudioFormat(sample_rate=16000, channels=1, sample_width=2),
            payload=b"\x00\x01\x02",
        )
        with pytest.raises(VADProcessingError):
            await vad.process(bad_chunk)

    asyncio.run(_test())


def test_processing_timeout_handling() -> None:
    """Verify VADTimeoutError is raised when processing exceeds timeout_ms."""

    async def _test() -> None:
        # 10ms timeout configuration with artificial 100ms processing delay
        short_vad = VADAdapter(
            config=VADConfig(processing_timeout_ms=10.0), processing_delay_sec=0.1
        )
        chunk = make_pcm_chunk(1000)

        with pytest.raises(VADTimeoutError):
            await short_vad.process(chunk)

    asyncio.run(_test())


def test_cancellation_handling(vad: VADAdapter) -> None:
    """Verify asyncio.CancelledError during VAD processing is handled cleanly."""

    async def _test() -> None:
        slow_vad = VADAdapter(processing_delay_sec=0.1)
        chunk = make_pcm_chunk(1000)
        task = asyncio.create_task(slow_vad.process(chunk))
        await asyncio.sleep(0.01)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(_test())


def test_privacy_guarantees_no_raw_payload_in_telemetry(vad: VADAdapter) -> None:
    """Verify telemetry and health reports exclude raw binary PCM bytes."""

    async def _test() -> None:
        chunk = make_pcm_chunk(15000)
        await vad.process(chunk)

        health = await vad.health()
        assert "payload" not in health
        assert "pcm_bytes" not in health
        assert health["last_telemetry"] is not None
        assert "payload" not in health["last_telemetry"]

    asyncio.run(_test())


def test_determinism(vad: VADAdapter) -> None:
    """Verify identical audio input produces exact same confidence score and state."""

    async def _test() -> None:
        c1 = make_pcm_chunk(18000, seq=0)
        c2 = make_pcm_chunk(18000, seq=0)

        e1 = await vad.process(c1)
        await vad.reset()
        e2 = await vad.process(c2)

        assert e1.confidence == e2.confidence
        assert e1.state == e2.state

    asyncio.run(_test())


def test_interface_compliance_and_lifecycle(vad: VADAdapter) -> None:
    """Verify VADAdapter implements IVoiceActivityDetector interface and supports reset/health."""

    async def _test() -> None:
        assert isinstance(vad, IVoiceActivityDetector)

        await vad.process(make_pcm_chunk(20000, seq=0))
        await vad.process(make_pcm_chunk(20000, seq=1))
        assert vad.current_state == VoiceActivityState.SPEECH_START

        await vad.reset()
        assert vad.current_state == VoiceActivityState.SILENCE

        health = await vad.health()
        assert health["status"] == "RUNNING"
        assert health["current_state"] == "SILENCE"

    asyncio.run(_test())
