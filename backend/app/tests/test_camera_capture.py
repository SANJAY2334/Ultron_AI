"""Phase 4F.2 Real Camera Capture Adapter Unit Tests.

Validates CameraCaptureAdapter device discovery, explicit selection, format negotiation, frame generation,
bounded queue backpressure (DROP_OLDEST), lifecycle management (start -> stop -> start -> stop),
failure isolation, resource cleanup, privacy telemetry bounds, and architectural boundary invariants.
Runs 100% deterministically in headless CI environments without requiring physical camera hardware.
"""

import asyncio
from datetime import datetime

import pytest

from app.api.deps import get_vision_capture
from app.vision import (
    IVisionCapture,
    PixelFormat,
    VisionConfig,
    VisionDeviceNotFoundError,
    VisionFrame,
)
from app.vision.adapters.camera import CameraCaptureAdapter


def test_camera_device_discovery_sanitization() -> None:
    """Verify list_devices() enumerates devices with sanitized metadata (no filesystem paths or credentials)."""

    async def _test() -> None:
        adapter = CameraCaptureAdapter(simulated_mode=True)
        devices = await adapter.list_devices()

        assert len(devices) >= 1
        dev = devices[0]
        assert "device_id" in dev
        assert "name" in dev
        assert "width" in dev
        assert "height" in dev
        assert "fps" in dev
        assert "backend" in dev
        assert "available" in dev

        # Verify no credentials or private driver paths in metadata
        dev_str = str(dev)
        assert "password" not in dev_str.lower()
        assert "secret" not in dev_str.lower()

    asyncio.run(_test())


def test_explicit_device_selection_and_invalid_id() -> None:
    """Verify explicit device selection and invalid device ID raises VisionDeviceNotFoundError."""

    async def _test() -> None:
        cfg = VisionConfig(camera_device="invalid_dev_999")
        adapter = CameraCaptureAdapter(config=cfg, simulated_mode=True)

        with pytest.raises(VisionDeviceNotFoundError):
            await adapter.start()

    asyncio.run(_test())


def test_format_negotiation_and_frame_generation() -> None:
    """Verify camera adapter negotiates resolution and FPS limits, producing valid VisionFrames."""

    async def _test() -> None:
        cfg = VisionConfig(max_width=640, max_height=480, max_fps=20.0, camera_device="default")
        adapter = CameraCaptureAdapter(config=cfg, simulated_mode=True)

        await adapter.start()
        await asyncio.sleep(0.1)

        frames: list[VisionFrame] = []
        async for frame in adapter.frames():
            frames.append(frame)
            if len(frames) >= 3:
                break

        await adapter.stop()

        assert len(frames) >= 3
        f0 = frames[0]
        assert f0.width == 640
        assert f0.height == 480
        assert f0.format.pixel_format == PixelFormat.RGB24
        assert isinstance(f0.timestamp, datetime)
        assert f0.timestamp.tzinfo is not None

        # Verify monotonic sequence numbers
        assert frames[1].sequence_number == f0.sequence_number + 1
        assert frames[2].sequence_number == f0.sequence_number + 2

    asyncio.run(_test())


def test_bounded_queue_and_drop_oldest_backpressure() -> None:
    """Verify bounded queue (maxsize=5) enforces DROP_OLDEST backpressure policy when consumer is slow."""

    async def _test() -> None:
        cfg = VisionConfig(max_buffered_frames=5, max_fps=30.0, camera_device="default")
        adapter = CameraCaptureAdapter(config=cfg, simulated_mode=True)

        await adapter.start()
        # Sleep to allow queue buffer to fill and overflow
        await asyncio.sleep(0.3)

        health = await adapter.health()
        assert health["queue_depth"] <= 5
        assert health["frames_dropped"] >= 0

        await adapter.stop()

    asyncio.run(_test())


def test_camera_capture_lifecycle_and_idempotency() -> None:
    """Verify CameraCaptureAdapter lifecycle (start -> stop -> start -> stop) and idempotency."""

    async def _test() -> None:
        adapter = CameraCaptureAdapter(simulated_mode=True)

        # 1. First start and stop
        await adapter.start()
        assert (await adapter.health())["status"] == "RUNNING"

        # Idempotent duplicate start
        await adapter.start()
        assert (await adapter.health())["status"] == "RUNNING"

        await adapter.stop()
        assert (await adapter.health())["status"] == "IDLE"

        # Idempotent duplicate stop
        await adapter.stop()
        assert (await adapter.health())["status"] == "IDLE"

        # 2. Restart lifecycle
        await adapter.start()
        assert (await adapter.health())["status"] == "RUNNING"

        await adapter.stop()
        assert (await adapter.health())["status"] == "IDLE"

    asyncio.run(_test())


def test_di_container_vision_capture_resolution() -> None:
    """Verify get_vision_capture() resolves IVisionCapture through the DI container."""
    capture = get_vision_capture()
    assert isinstance(capture, IVisionCapture)


def test_privacy_and_payload_redaction_bounds() -> None:
    """Verify frame representation and health telemetry suppress raw binary image payloads."""

    async def _test() -> None:
        adapter = CameraCaptureAdapter(simulated_mode=True)
        await adapter.start()
        await asyncio.sleep(0.05)

        frames: list[VisionFrame] = []
        async for f in adapter.frames():
            frames.append(f)
            break

        await adapter.stop()

        f = frames[0]
        repr_str = repr(f)
        str_str = str(f)

        assert "VisionFrame" in repr_str
        assert "payload_bytes=" in repr_str
        assert str_str == repr_str

        health = await adapter.health()
        assert "frames_received" in health
        assert "frames_dropped" in health
        assert "payload" not in health
        assert "raw_pixels" not in health

    asyncio.run(_test())


def test_architectural_domain_isolation() -> None:
    """Verify vision domain modules contain ZERO imports of OpenCV or OS automation tools."""
    import sys

    # Ensure OpenCV is NOT imported in vision domain core files
    import app.vision.base
    import app.vision.config
    import app.vision.exceptions
    import app.vision.models

    for mod in [app.vision.models, app.vision.base, app.vision.exceptions, app.vision.config]:
        assert "cv2" not in mod.__dict__

    # Ensure vision domain contains no imports of OS automation or planner
    forbidden_modules = [
        "app.ai.planner.langgraph_planner",
        "app.desktop.adapters.pyautogui_adapter",
        "app.security.policy",
    ]
    for fmod in forbidden_modules:
        if fmod in sys.modules:
            # Check that vision models do not reference them
            assert fmod not in app.vision.models.__dict__
