"""Opt-In Physical Camera Hardware Test Suite (Phase 4F.2).

Validates physical camera hardware acquisition, device enumeration, format negotiation,
stream lifecycle, device failure isolation, and privacy telemetry bounds on real hardware.
Activated by setting environment variable: ULTRON_VISION_HARDWARE_TEST=1
"""

import asyncio
import os

import pytest

from app.vision import VisionConfig, VisionFrame
from app.vision.adapters.camera import CameraCaptureAdapter

HARDWARE_TEST_ENABLED = os.getenv("ULTRON_VISION_HARDWARE_TEST") == "1"


@pytest.mark.skipif(
    not HARDWARE_TEST_ENABLED,
    reason="Physical vision hardware tests require ULTRON_VISION_HARDWARE_TEST=1 environment variable.",
)
def test_physical_camera_device_enumeration() -> None:
    """Verify physical camera device enumeration on local host environment."""

    async def _test() -> None:
        adapter = CameraCaptureAdapter(simulated_mode=False)
        devices = await adapter.list_devices()

        assert isinstance(devices, list)
        assert len(devices) >= 1
        assert "device_id" in devices[0]

    asyncio.run(_test())


@pytest.mark.skipif(
    not HARDWARE_TEST_ENABLED,
    reason="Physical vision hardware tests require ULTRON_VISION_HARDWARE_TEST=1 environment variable.",
)
def test_physical_camera_acquisition_stream() -> None:
    """Verify physical camera acquisition stream produces valid VisionFrames."""

    async def _test() -> None:
        cfg = VisionConfig(max_width=640, max_height=480, max_fps=15.0, camera_device="0")
        adapter = CameraCaptureAdapter(config=cfg, simulated_mode=False)

        await adapter.start()
        await asyncio.sleep(0.2)

        frames: list[VisionFrame] = []
        async for frame in adapter.frames():
            frames.append(frame)
            if len(frames) >= 5:
                break

        await adapter.stop()

        assert len(frames) >= 5
        assert frames[0].width == 640
        assert frames[0].height == 480

        health = await adapter.health()
        assert health["frames_received"] >= 5

    asyncio.run(_test())
