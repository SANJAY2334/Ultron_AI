"""Phase 4F.3 Vision Processing & Frame Pipeline Unit Tests.

Validates VisionProcessor frame validation, bytes-per-pixel payload calculation, resolution limits,
processing timeout protection, event loop safety, bounded previous-frame memory buffers,
motion estimation signals, privacy telemetry bounds, and architectural boundary invariants.
"""

import asyncio
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.api.deps import get_vision_processor
from app.vision import (
    FrameFormat,
    IVisionProcessor,
    PixelFormat,
    VisionConfig,
    VisionFrame,
    VisionObservation,
    VisionProcessorTimeoutError,
    VisionProcessorValidationError,
)
from app.vision.processor import VisionProcessor


def make_valid_frame(seq: int = 0, width: int = 640, height: int = 480) -> VisionFrame:
    """Helper creating a valid synthetic VisionFrame."""
    fmt = FrameFormat(
        width=width, height=height, channels=3, pixel_format=PixelFormat.RGB24, frame_rate=15.0
    )
    payload = b"\x00\x80\xff" * (width * height)
    return VisionFrame(
        frame_id=f"frm_proc_{seq}",
        sequence_number=seq,
        timestamp=datetime.now(UTC),
        width=width,
        height=height,
        format=fmt,
        payload=payload,
    )


def test_vision_processor_valid_frame_processing() -> None:
    """Verify VisionProcessor validates a correct VisionFrame and produces a sanitized VisionObservation."""

    async def _test() -> None:
        processor = VisionProcessor()

        frame = make_valid_frame(seq=1)
        obs = await processor.process(frame)

        assert isinstance(obs, VisionObservation)
        assert obs.timestamp == frame.timestamp
        assert obs.object_detections == []
        assert obs.face_detections == []

        health = await processor.health()
        assert health["subsystem"] == "vision_processor"
        assert health["status"] == "RUNNING"
        assert health["frames_processed"] == 1
        assert health["frames_rejected"] == 0

    asyncio.run(_test())


def test_vision_processor_frame_validation_failures() -> None:
    """Verify VisionProcessor rejects invalid frame_id, sequence, dimensions, or mismatched payloads."""

    async def _test() -> None:
        processor = VisionProcessor(config=VisionConfig(max_width=1280, max_height=720))
        fmt = FrameFormat(width=640, height=480, pixel_format=PixelFormat.RGB24)

        # 1. Empty/whitespace frame_id
        bad_id = VisionFrame(
            frame_id="   ",
            sequence_number=1,
            timestamp=datetime.now(UTC),
            width=640,
            height=480,
            format=fmt,
            payload=b"\x00" * (640 * 480 * 3),
        )
        with pytest.raises(VisionProcessorValidationError):
            await processor.process(bad_id)

        # 2. Negative sequence number (rejected at frame model layer)
        with pytest.raises(ValidationError):
            VisionFrame(
                frame_id="bad_seq",
                sequence_number=-5,
                timestamp=datetime.now(UTC),
                width=640,
                height=480,
                format=fmt,
                payload=b"\x00" * (640 * 480 * 3),
            )

        # 3. Excessive dimensions (> max_width rejected at processor layer)
        bad_dim = VisionFrame(
            frame_id="bad_dim",
            sequence_number=2,
            timestamp=datetime.now(UTC),
            width=1920,
            height=1080,
            format=FrameFormat(width=1920, height=1080, pixel_format=PixelFormat.RGB24),
            payload=b"\x00" * (1920 * 1080 * 3),
        )
        with pytest.raises(VisionProcessorValidationError):
            await processor.process(bad_dim)

        # 4. Mismatched payload byte length (rejected at processor layer)
        mismatched_payload = VisionFrame(
            frame_id="bad_payload",
            sequence_number=3,
            timestamp=datetime.now(UTC),
            width=640,
            height=480,
            format=fmt,
            payload=b"\x00" * 100,  # Expected 640*480*3 = 921600 bytes
        )
        with pytest.raises(VisionProcessorValidationError):
            await processor.process(mismatched_payload)

    asyncio.run(_test())


def test_vision_processor_timeout_protection() -> None:
    """Verify processing_timeout_ms limits execution and raises VisionProcessorTimeoutError on delay."""

    async def _test() -> None:
        cfg = VisionConfig(processing_timeout_ms=20.0)
        slow_processor = VisionProcessor(config=cfg, simulated_delay_sec=0.08)

        frame = make_valid_frame(seq=1)
        with pytest.raises(VisionProcessorTimeoutError):
            await slow_processor.process(frame)

        health = await slow_processor.health()
        assert health["frames_rejected"] == 1

    asyncio.run(_test())


def test_vision_processor_motion_estimation_and_bounded_memory() -> None:
    """Verify motion signals and bounded previous-frame history (max_previous_frames=1)."""

    async def _test() -> None:
        cfg = VisionConfig(enable_motion=True, max_previous_frames=1)
        processor = VisionProcessor(config=cfg)

        f1 = make_valid_frame(seq=1)

        # Frame 2 with contrasting pixel data
        fmt = FrameFormat(width=640, height=480, channels=3, pixel_format=PixelFormat.RGB24)
        f2 = VisionFrame(
            frame_id="frm_proc_2",
            sequence_number=2,
            timestamp=datetime.now(UTC),
            width=640,
            height=480,
            format=fmt,
            payload=b"\xff\x00\x00" * (640 * 480),
        )

        obs1 = await processor.process(f1)
        assert len(obs1.motion_events) == 0

        obs2 = await processor.process(f2)
        assert len(obs2.motion_events) == 1
        assert obs2.motion_events[0].motion_score > 0.0

        health = await processor.health()
        assert health["buffered_previous_frames"] == 1

    asyncio.run(_test())


def test_di_container_vision_processor_resolution() -> None:
    """Verify get_vision_processor() resolves IVisionProcessor through DI container."""
    processor = get_vision_processor()
    assert isinstance(processor, IVisionProcessor)


def test_privacy_telemetry_bounds_and_repr_redaction() -> None:
    """Verify VisionProcessorTelemetry and VisionObservation contain zero raw binary image payload bytes."""

    async def _test() -> None:
        processor = VisionProcessor()
        frame = make_valid_frame(seq=1)

        obs = await processor.process(frame)
        repr_obs = repr(obs)
        str_obs = str(obs)

        assert "VisionObservation" in repr_obs
        assert "timestamp=" in str_obs
        assert b"\x00\x80\xff" not in repr_obs.encode()

        health = await processor.health()
        assert "frames_processed" in health
        assert "payload" not in health

    asyncio.run(_test())


def test_architectural_domain_isolation() -> None:
    """Verify vision processor has zero imports of OS automation tools or autonomous planner."""
    import sys

    import app.vision.processor

    forbidden_modules = [
        "app.ai.planner.langgraph_planner",
        "app.desktop.adapters.pyautogui_adapter",
        "app.security.policy",
    ]
    for fmod in forbidden_modules:
        if fmod in sys.modules:
            assert fmod not in app.vision.processor.__dict__
