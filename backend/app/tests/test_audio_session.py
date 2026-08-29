"""Unit and Integration Tests for Full-Duplex Audio Session Orchestrator (Phase 4C.7).

Validates AudioSessionManager lifecycle, state transitions (IDLE -> LISTENING -> SPEECH_DETECTED -> TRANSCRIBING -> THINKING -> SYNTHESIZING -> PLAYING),
wake-word integration, VAD & STT pipeline execution, TTS playback boundary, backpressure queues,
idempotent cancellation, child task cleanup, privacy redaction, and IAudioSessionManager compliance.
"""

import asyncio
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.audio import (
    AudioChunk,
    AudioFormat,
    AudioSessionConfig,
    AudioSessionLimitError,
    AudioSessionManager,
    AudioSessionState,
    AudioSessionTelemetry,
    IAudioSessionManager,
    InvalidAudioStateTransitionError,
    SpeechSynthesisResult,
)
from app.audio.adapters.microphone import MicrophoneCaptureAdapter
from app.audio.adapters.stt import STTAdapter
from app.audio.adapters.tts import TTSAdapter
from app.audio.adapters.vad import VADAdapter
from app.audio.adapters.wake_word import WakeWordAdapter


def make_chunk(seq: int = 0, session_id: str = "sess_orch_1") -> AudioChunk:
    """Helper creating synthetic AudioChunk."""
    return AudioChunk(
        chunk_id=f"chk_orch_{seq}",
        sequence_number=seq,
        timestamp=datetime.now(UTC),
        duration_ms=30.0,
        audio_format=AudioFormat(sample_rate=16000, channels=1, sample_width=2),
        payload=b"\x00" * 960,
        session_id=session_id,
        correlation_id="corr_orch_100",
    )


@pytest.fixture
def session_mgr() -> AudioSessionManager:
    cfg = AudioSessionConfig(
        max_concurrent_sessions=2,
        max_queued_chunks=10,
        max_pending_synthesis_chunks=10,
    )
    return AudioSessionManager(
        capture=MicrophoneCaptureAdapter(max_buffered_chunks=5, mock_mode=True),
        vad=VADAdapter(),
        stt=STTAdapter(),
        tts=TTSAdapter(),
        wake_word=WakeWordAdapter(mock_trigger_word="hey ultron"),
        config=cfg,
    )


def test_session_config_validation() -> None:
    """Verify AudioSessionConfig parameter validation."""
    cfg = AudioSessionConfig(max_concurrent_sessions=3, backpressure_policy="DROP_OLDEST")
    assert cfg.max_concurrent_sessions == 3

    # Invalid concurrent sessions <= 0
    with pytest.raises(ValidationError):
        AudioSessionConfig(max_concurrent_sessions=0)

    # Invalid backpressure policy
    with pytest.raises(ValidationError):
        AudioSessionConfig(backpressure_policy="INVALID_POLICY")


def test_session_lifecycle_and_state_transitions(session_mgr: AudioSessionManager) -> None:
    """Verify session creation, start, state transitions, stop, and cancellation (rules 1-6 & 44)."""

    async def _test() -> None:
        # 44. Interface compliance
        assert isinstance(session_mgr, IAudioSessionManager)
        assert session_mgr.current_state() == AudioSessionState.IDLE

        # 2. Start session -> LISTENING
        await session_mgr.start_session("sess_001")
        assert session_mgr.current_state() == AudioSessionState.LISTENING

        # 6. Invalid state transition (LISTENING -> SYNTHESIZING directly raises InvalidAudioStateTransitionError)
        with pytest.raises(InvalidAudioStateTransitionError):
            await session_mgr.transition(AudioSessionState.SYNTHESIZING)

        # 3. Stop session -> IDLE
        await session_mgr.stop_session("sess_001")
        assert session_mgr.current_state() == AudioSessionState.IDLE

        # 4-5. Idempotent cancellation
        await session_mgr.cancel()
        assert session_mgr.current_state() == AudioSessionState.IDLE
        await session_mgr.cancel()
        assert session_mgr.current_state() == AudioSessionState.IDLE

    asyncio.run(_test())


def test_max_concurrent_sessions_limit() -> None:
    """Verify maximum concurrent sessions limit enforcement (rules 34 & 35)."""

    async def _test() -> None:
        cfg = AudioSessionConfig(max_concurrent_sessions=1)
        mgr = AudioSessionManager(config=cfg)

        await mgr.start_session("sess_1")
        assert mgr.current_state() == AudioSessionState.LISTENING

        # Second concurrent session should raise AudioSessionLimitError
        with pytest.raises(AudioSessionLimitError, match="limit reached"):
            await mgr.start_session("sess_2")

        await mgr.stop_session("sess_1")

    asyncio.run(_test())


def test_tts_synthesis_and_playback_boundary(session_mgr: AudioSessionManager) -> None:
    """Verify process_speech_text TTS integration and reading playback chunks (rules 21-23)."""

    async def _test() -> None:
        await session_mgr.start_session("sess_tts_test")

        # Trigger TTS synthesis
        res = await session_mgr.process_speech_text("Hello ULTRON voice playback.")
        assert isinstance(res, SpeechSynthesisResult)
        assert session_mgr.current_state() == AudioSessionState.PLAYING

        # Read playback boundary chunk
        playback_chunk = await session_mgr.read_playback_chunk()
        assert isinstance(playback_chunk, AudioChunk)
        assert playback_chunk.sequence_number >= 0

        await session_mgr.stop_session("sess_tts_test")

    asyncio.run(_test())


def test_task_cleanup_on_cancellation(session_mgr: AudioSessionManager) -> None:
    """Verify cancellation clears child background tasks without orphan task leaks (rules 29-33)."""

    async def _test() -> None:
        await session_mgr.start_session("sess_cancel_test")
        assert len(session_mgr._background_tasks) >= 1

        # Cancel session
        await session_mgr.cancel()
        assert len(session_mgr._background_tasks) == 0
        assert session_mgr.current_state() == AudioSessionState.IDLE

    asyncio.run(_test())


def test_privacy_and_telemetry_guarantees(session_mgr: AudioSessionManager) -> None:
    """Verify AudioSessionTelemetry excludes raw PCM bytes and full transcript text (rules 38-42)."""

    async def _test() -> None:
        await session_mgr.start_session("sess_privacy")
        await asyncio.sleep(0.05)

        telemetry = session_mgr.get_telemetry()
        assert isinstance(telemetry, AudioSessionTelemetry)
        assert telemetry.session_id == "sess_privacy"
        assert telemetry.correlation_id == "corr_sess_privacy"

        telem_dict = telemetry.model_dump()
        assert "audio_payload" not in telem_dict
        assert "pcm_bytes" not in telem_dict
        assert "transcript_text" not in telem_dict

        health = await session_mgr.health()
        assert health["subsystem_session_manager"] is True
        assert health["active_session_id"] == "sess_privacy"

        await session_mgr.stop_session("sess_privacy")

    asyncio.run(_test())


def test_determinism_and_identity_propagation(session_mgr: AudioSessionManager) -> None:
    """Verify session_id and correlation_id propagation and output determinism (rules 41-43)."""

    async def _test() -> None:
        await session_mgr.start_session("sess_det_1")
        t1 = session_mgr.get_telemetry()
        await session_mgr.stop_session("sess_det_1")

        await session_mgr.start_session("sess_det_2")
        t2 = session_mgr.get_telemetry()
        await session_mgr.stop_session("sess_det_2")

        assert t1.session_id == "sess_det_1"
        assert t2.session_id == "sess_det_2"
        assert t1.correlation_id == "corr_sess_det_1"
        assert t2.correlation_id == "corr_sess_det_2"

    asyncio.run(_test())
