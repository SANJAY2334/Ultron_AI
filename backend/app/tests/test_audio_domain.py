"""Unit Tests for Audio Domain Models, State Machine, and Interfaces (Phase 4C.1).

Validates AudioFormat, AudioChunk, AudioStreamConfig, VoiceActivityEvent, TranscriptSegment,
Transcript, SpeechSynthesisRequest/Result, WakeWordDetection, AudioSessionStateMachine,
AudioPrivacy defaults, AudioTelemetry metrics, and abstract interface instantiation rules.
"""

from datetime import UTC, datetime

import pytest

from app.audio import (
    AudioChunk,
    AudioEncoding,
    AudioFormat,
    AudioPrivacy,
    AudioSessionState,
    AudioSessionStateMachine,
    AudioStreamConfig,
    AudioTelemetry,
    IAudioCapture,
    IAudioSessionManager,
    InvalidAudioStateTransitionError,
    ISpeechToTextProvider,
    ITextToSpeechProvider,
    IVoiceActivityDetector,
    IWakeWordDetector,
    SpeechSynthesisRequest,
    SpeechSynthesisResult,
    Transcript,
    TranscriptSegment,
    VoiceActivityEvent,
    VoiceActivityState,
    WakeWordDetection,
)


def test_audio_format_validation() -> None:
    """Verify AudioFormat accepts valid PCM parameters and rejects invalid values."""
    fmt = AudioFormat(
        sample_rate=16000, channels=1, sample_width=2, encoding=AudioEncoding.PCM_S16LE
    )
    assert fmt.sample_rate == 16000
    assert fmt.channels == 1
    assert fmt.encoding == "pcm_s16le"

    # Invalid sample rate
    with pytest.raises(ValueError, match="Unsupported sample_rate"):
        AudioFormat(sample_rate=12345)

    # Invalid channels
    with pytest.raises(ValueError):
        AudioFormat(channels=0)


def test_audio_chunk_validation_and_privacy_security() -> None:
    """Verify AudioChunk timing, timezone, non-empty payload, and repr secrets protection."""
    now = datetime.now(UTC)
    fmt = AudioFormat()

    chunk = AudioChunk(
        chunk_id="chk_101",
        sequence_number=0,
        timestamp=now,
        duration_ms=30.0,
        audio_format=fmt,
        payload=b"\x00\x01\x02\x03\x04" * 10,
        privacy_level=AudioPrivacy.EPHEMERAL,
    )
    assert chunk.chunk_id == "chk_101"
    assert chunk.privacy_level == AudioPrivacy.EPHEMERAL

    # Verify custom repr excludes raw binary bytes payload
    chunk_repr = repr(chunk)
    assert "bytes=50" in chunk_repr
    assert r"\x00\x01" not in chunk_repr

    # Naive timestamp rejection
    with pytest.raises(ValueError, match="timezone-aware"):
        AudioChunk(
            chunk_id="chk_err",
            sequence_number=1,
            timestamp=datetime.now(),  # Naive!
            duration_ms=30.0,
            audio_format=fmt,
            payload=b"123",
        )

    # Empty payload rejection
    with pytest.raises(ValueError):
        AudioChunk(
            chunk_id="chk_err2",
            sequence_number=1,
            timestamp=now,
            duration_ms=30.0,
            audio_format=fmt,
            payload=b"",
        )


def test_audio_stream_config() -> None:
    """Verify AudioStreamConfig defaults and validation."""
    cfg = AudioStreamConfig()
    assert cfg.sample_rate == 16000
    assert cfg.channels == 1
    assert cfg.chunk_duration_ms == 30.0
    assert cfg.max_latency_ms == 200.0

    with pytest.raises(ValueError):
        AudioStreamConfig(sample_rate=9999)


def test_voice_activity_event() -> None:
    """Verify VoiceActivityEvent state transitions and confidence limits."""
    now = datetime.now(UTC)
    event = VoiceActivityEvent(
        event_id="vad_001",
        state=VoiceActivityState.SPEECH_START,
        timestamp=now,
        confidence=0.95,
    )
    assert event.state == VoiceActivityState.SPEECH_START
    assert event.confidence == 0.95

    # Out-of-bounds confidence score rejection
    with pytest.raises(ValueError):
        VoiceActivityEvent(
            event_id="vad_err",
            state=VoiceActivityState.SPEAKING,
            timestamp=now,
            confidence=1.5,
        )


def test_transcript_and_segment_validation() -> None:
    """Verify Transcript and TranscriptSegment timing invariants and text requirements."""
    seg1 = TranscriptSegment(
        segment_id="seg_1",
        text="Hello ULTRON",
        start_ms=0.0,
        end_ms=500.0,
        confidence=0.92,
        is_final=True,
    )
    assert seg1.text == "Hello ULTRON"

    # end_ms < start_ms validation
    with pytest.raises(ValueError, match="end_ms must be greater than or equal to start_ms"):
        TranscriptSegment(
            segment_id="seg_err",
            text="Invalid time",
            start_ms=1000.0,
            end_ms=500.0,
            confidence=0.90,
        )

    # Empty text for final segment rejection
    with pytest.raises(ValueError, match="Completed final transcript segment cannot be empty text"):
        TranscriptSegment(
            segment_id="seg_empty",
            text="  ",
            start_ms=0.0,
            end_ms=100.0,
            confidence=0.8,
            is_final=True,
        )

    transcript = Transcript(
        transcript_id="tr_1",
        segments=[seg1],
        full_text="Hello ULTRON",
        confidence=0.92,
        is_final=True,
    )
    assert transcript.transcript_id == "tr_1"


def test_speech_synthesis_models() -> None:
    """Verify SpeechSynthesisRequest and SpeechSynthesisResult properties."""
    req = SpeechSynthesisRequest(
        request_id="tts_001",
        text="Greeting human.",
        voice="en-US-Standard-A",
        speed=1.0,
        pitch=0.0,
    )
    assert req.text == "Greeting human."

    # Excessive text payload rejection (> 5000 chars)
    with pytest.raises(ValueError):
        SpeechSynthesisRequest(
            request_id="tts_huge",
            text="A" * 6000,
            voice="en-US-Standard-A",
        )

    # Out of bounds speed rejection
    with pytest.raises(ValueError):
        SpeechSynthesisRequest(
            request_id="tts_fast",
            text="Speed check",
            voice="en-US-Standard-A",
            speed=10.0,
        )

    fmt = AudioFormat()
    res = SpeechSynthesisResult(
        request_id="tts_001",
        audio_format=fmt,
        duration_ms=1200.0,
        audio_payload=b"\x00\x00" * 100,
    )
    assert res.duration_ms == 1200.0
    assert "bytes=200" in repr(res)


def test_wake_word_detection() -> None:
    """Verify WakeWordDetection model validation."""
    now = datetime.now(UTC)
    detect = WakeWordDetection(
        detection_id="ww_1",
        wake_word="hey ultron",
        confidence=0.98,
        timestamp=now,
    )
    assert detect.wake_word == "hey ultron"
    assert detect.confidence == 0.98


def test_audio_session_state_machine_legal_and_illegal_transitions() -> None:
    """Verify AudioSessionStateMachine allows legal state transitions and blocks illegal transitions."""
    sm = AudioSessionStateMachine(initial_state=AudioSessionState.IDLE)
    assert sm.current_state == AudioSessionState.IDLE

    # Legal Sequence: IDLE -> LISTENING -> SPEECH_DETECTED -> TRANSCRIBING -> THINKING -> SYNTHESIZING -> PLAYING -> LISTENING -> IDLE
    assert sm.transition(AudioSessionState.LISTENING) == AudioSessionState.LISTENING
    assert sm.transition(AudioSessionState.SPEECH_DETECTED) == AudioSessionState.SPEECH_DETECTED
    assert sm.transition(AudioSessionState.TRANSCRIBING) == AudioSessionState.TRANSCRIBING
    assert sm.transition(AudioSessionState.THINKING) == AudioSessionState.THINKING
    assert sm.transition(AudioSessionState.SYNTHESIZING) == AudioSessionState.SYNTHESIZING
    assert sm.transition(AudioSessionState.PLAYING) == AudioSessionState.PLAYING
    assert sm.transition(AudioSessionState.IDLE) == AudioSessionState.IDLE

    # Illegal Transition: IDLE -> SYNTHESIZING directly (must raise InvalidAudioStateTransitionError)
    with pytest.raises(InvalidAudioStateTransitionError, match="Illegal Audio State Transition"):
        sm.transition(AudioSessionState.SYNTHESIZING)

    # Illegal Transition: LISTENING -> THINKING directly
    sm.transition(AudioSessionState.LISTENING)
    with pytest.raises(InvalidAudioStateTransitionError):
        sm.transition(AudioSessionState.THINKING)

    # Cancel support
    assert sm.cancel() == AudioSessionState.IDLE


def test_audio_telemetry() -> None:
    """Verify AudioTelemetry metrics validation."""
    now = datetime.now(UTC)
    telem = AudioTelemetry(
        capture_latency_ms=10.0,
        vad_latency_ms=15.0,
        stt_latency_ms=120.0,
        planner_latency_ms=45.0,
        tts_latency_ms=50.0,
        total_latency_ms=240.0,
        timestamp=now,
    )
    assert telem.total_latency_ms == 240.0

    # Negative latency rejection
    with pytest.raises(ValueError):
        AudioTelemetry(capture_latency_ms=-5.0, timestamp=now)


def test_abstract_interfaces_cannot_be_instantiated() -> None:
    """Verify abstract interfaces raise TypeError when instantiated without subclassing."""
    with pytest.raises(TypeError):
        IAudioCapture()  # type: ignore[abstract]

    with pytest.raises(TypeError):
        IVoiceActivityDetector()  # type: ignore[abstract]

    with pytest.raises(TypeError):
        ISpeechToTextProvider()  # type: ignore[abstract]

    with pytest.raises(TypeError):
        ITextToSpeechProvider()  # type: ignore[abstract]

    with pytest.raises(TypeError):
        IWakeWordDetector()  # type: ignore[abstract]

    with pytest.raises(TypeError):
        IAudioSessionManager()  # type: ignore[abstract]
