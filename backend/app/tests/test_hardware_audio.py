"""Phase 4E.2-H Hardware Validation and Hardening Test Suite.

Provides automated mock-mode hardware assertions for normal CI/testing baseline and an opt-in
physical hardware smoke test enabled via ULTRON_HARDWARE_TEST=1.

Validates device enumeration, 16kHz S16LE format negotiation, physical/simulated audio capture,
playback tone output, stream resource lifecycle cleanup, device error isolation, barge-in interruption,
and privacy telemetry redaction.
"""

import asyncio
import math
import os
import struct
from datetime import UTC, datetime

import pytest

from app.audio import (
    AudioChunk,
    AudioEncoding,
    AudioFormat,
    AudioSessionManager,
    AudioSessionState,
    PlaybackState,
    SoundDeviceCaptureAdapter,
    SoundDevicePlaybackAdapter,
)
from app.audio.capture import AudioDeviceInfo

# Check if developer requested explicit physical hardware testing
RUN_HARDWARE_TEST = os.environ.get("ULTRON_HARDWARE_TEST", "0") == "1"


def generate_test_tone(
    duration_ms: float = 300.0, frequency_hz: float = 440.0, sample_rate: int = 16000
) -> bytes:
    """Generates a synthetic 16kHz mono 16-bit PCM LE sine wave test tone payload."""
    total_samples = int(sample_rate * (duration_ms / 1000.0))
    pcm_bytes = bytearray()
    for i in range(total_samples):
        t = i / sample_rate
        sample_val = int(32767.0 * 0.5 * math.sin(2.0 * math.pi * frequency_hz * t))
        pcm_bytes.extend(struct.pack("<h", sample_val))
    return bytes(pcm_bytes)


def make_chunk(seq: int = 0, session_id: str = "sess_hw_1") -> AudioChunk:
    """Helper creating synthetic AudioChunk."""
    return AudioChunk(
        chunk_id=f"chk_hw_{seq}",
        sequence_number=seq,
        timestamp=datetime.now(UTC),
        duration_ms=30.0,
        audio_format=AudioFormat(
            sample_rate=16000, channels=1, sample_width=2, encoding=AudioEncoding.PCM_S16LE
        ),
        payload=b"\x00" * 960,
        session_id=session_id,
        correlation_id="corr_hw_100",
    )


def test_device_enumeration_sanitization() -> None:
    """Validate device discovery and information sanitization (rule 1)."""
    inputs = SoundDeviceCaptureAdapter.list_devices(kind="input")
    outputs = SoundDeviceCaptureAdapter.list_devices(kind="output")

    assert isinstance(inputs, list)
    assert len(inputs) >= 1
    assert isinstance(outputs, list)
    assert len(outputs) >= 1

    for dev in inputs + outputs:
        assert isinstance(dev, AudioDeviceInfo)
        assert hasattr(dev, "device_id")
        assert hasattr(dev, "name")
        assert hasattr(dev, "input_channels")
        assert hasattr(dev, "default_sample_rate")
        # Ensure zero credentials or secret paths in device metadata
        dev_str = str(dev.model_dump())
        assert "password" not in dev_str.lower()
        assert "secret" not in dev_str.lower()


def test_format_negotiation_and_validation() -> None:
    """Validate microphone and speaker 16kHz S16LE format negotiation (rules 2-3)."""
    capture = SoundDeviceCaptureAdapter(
        sample_rate=16000, channels=1, simulated_mode=not RUN_HARDWARE_TEST
    )
    playback = SoundDevicePlaybackAdapter(simulated_mode=not RUN_HARDWARE_TEST)

    assert capture.validate_format() is True
    assert playback.validate_format(sample_rate=16000, channels=1) is True


def test_stream_lifecycle_resource_cleanup() -> None:
    """Verify repeated start -> stop lifecycle cleans up all tasks and streams (rule 5)."""

    async def _test() -> None:
        capture = SoundDeviceCaptureAdapter(simulated_mode=not RUN_HARDWARE_TEST)

        for _ in range(3):
            await capture.start()
            health = await capture.health()
            assert health["state"] == "RUNNING"

            await capture.stop()
            h_stop = await capture.health()
            assert h_stop["state"] == "STOPPED"

        assert capture._stream is None

    asyncio.run(_test())


def test_device_failure_isolation() -> None:
    """Verify runtime stream error does not crash process (rule 6)."""

    async def _test() -> None:
        capture = SoundDeviceCaptureAdapter(simulated_mode=True)
        await capture.start()

        # Simulate PortAudio stream callback error
        capture._audio_callback(None, 480, None, status=1)

        health = await capture.health()
        assert health["stream_errors"] >= 1
        assert health["healthy"] is True  # Recoverable

        await capture.stop()

    asyncio.run(_test())


def test_barge_in_hardware_interruption_chain() -> None:
    """Verify TTS -> playback -> user speech -> interrupt_speech() -> stop -> LISTENING (rule 7)."""

    async def _test() -> None:
        playback = SoundDevicePlaybackAdapter(simulated_mode=not RUN_HARDWARE_TEST)
        mgr = AudioSessionManager(playback=playback)

        await mgr.start_session("sess_hw_bargein")
        await mgr.process_speech_text("Hardware barge in test text.")

        assert playback._chunks_received > 0

        # Execute barge-in speech interruption
        await mgr.interrupt_speech()

        assert playback.get_state() == PlaybackState.IDLE
        assert mgr._synthesis_queue.empty()
        assert mgr.current_state() in (AudioSessionState.LISTENING, AudioSessionState.IDLE)

        await mgr.stop_session("sess_hw_bargein")

    asyncio.run(_test())


def test_privacy_and_telemetry_audit() -> None:
    """Verify telemetry contains metadata only and excludes raw PCM payloads (rule 9)."""

    async def _test() -> None:
        playback = SoundDevicePlaybackAdapter(simulated_mode=not RUN_HARDWARE_TEST)
        await playback.play_chunk(make_chunk(seq=0))

        telemetry = playback.get_telemetry()
        telem_dict = telemetry.model_dump()

        assert "payload" not in telem_dict
        assert "audio_bytes" not in telem_dict
        assert "pcm" not in telem_dict
        assert "jwt" not in telem_dict
        assert "secret" not in telem_dict

        await playback.stop()

    asyncio.run(_test())


@pytest.mark.skipif(
    not RUN_HARDWARE_TEST, reason="Opt-in physical hardware test (ULTRON_HARDWARE_TEST=1 required)"
)
def test_physical_hardware_smoke_test() -> None:
    """Opt-in physical hardware smoke test (rules 4, 11).

    Executes when ULTRON_HARDWARE_TEST=1. Opens physical sounddevice InputStream and OutputStream,
    captures 5 AudioChunk frames, verifies chunk properties, plays a 440Hz test tone, and verifies cleanup.
    """

    async def _test() -> None:
        print("\n--- RUNNING PHYSICAL SOUNDDEVICE HARDWARE SMOKE TEST ---")

        # 1. Physical Microphone Capture
        capture = SoundDeviceCaptureAdapter(sample_rate=16000, channels=1, simulated_mode=False)
        assert capture.validate_format() is True

        await capture.start()

        chunks: list[AudioChunk] = []
        for _ in range(5):
            chunk = await capture.read_chunk()
            chunks.append(chunk)

        await capture.stop()

        assert len(chunks) == 5
        for i, c in enumerate(chunks):
            assert c.sequence_number == i
            assert len(c.payload) == 960  # 30ms @ 16kHz mono S16LE = 960 bytes
            assert c.duration_ms == pytest.approx(30.0, rel=1e-2)
            assert c.audio_format.sample_rate == 16000
            assert c.audio_format.channels == 1
            assert c.audio_format.encoding == AudioEncoding.PCM_S16LE
            assert c.timestamp.tzinfo is not None

        # 2. Physical Speaker Playback
        playback = SoundDevicePlaybackAdapter(simulated_mode=False)
        assert playback.validate_format(sample_rate=16000, channels=1) is True

        tone_bytes = generate_test_tone(duration_ms=300.0, frequency_hz=440.0, sample_rate=16000)

        # Play 10 frames of test tone
        for seq in range(10):
            tone_chunk = AudioChunk(
                chunk_id=f"chk_tone_{seq}",
                sequence_number=seq,
                timestamp=datetime.now(UTC),
                duration_ms=30.0,
                audio_format=AudioFormat(
                    sample_rate=16000, channels=1, sample_width=2, encoding=AudioEncoding.PCM_S16LE
                ),
                payload=tone_bytes[:960],
            )
            await playback.play_chunk(tone_chunk)

        await asyncio.sleep(0.35)
        await playback.stop()

        assert playback._chunks_played > 0
        assert capture._stream is None
        assert playback._stream is None

        print("--- PHYSICAL SOUNDDEVICE HARDWARE SMOKE TEST SUCCESSFUL ---")

    asyncio.run(_test())
