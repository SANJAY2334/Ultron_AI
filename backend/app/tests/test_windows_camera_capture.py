"""Comprehensive Unit & Hardware Integration Tests for WindowsCameraCaptureDevice (Phase 4H.3).

Tests camera enumeration, DirectShow/OpenCV frame acquisition, RGB24 formatting,
DROP_OLDEST queue backpressure, recovery lifecycle, Zero-Trust security, and physical webcam capture.
"""

import asyncio
from datetime import UTC, datetime
from unittest.mock import patch

import pytest

from app.hardware.device_manager import DeviceManager
from app.hardware.models import DeviceState, DeviceType, HardwareDevice
from app.vision.adapters.object_detector import ObjectDetectorAdapter
from app.vision.adapters.planner_context_builder import VisionPlannerContextBuilder
from app.vision.adapters.windows_camera_capture import (
    OPENCV_AVAILABLE,
    WindowsCameraCaptureDevice,
)
from app.vision.base import IVisionCapture
from app.vision.config import VisionConfig
from app.vision.exceptions import (
    VisionDeviceNotFoundError,
)
from app.vision.models import (
    FrameFormat,
    PixelFormat,
    VisionFrame,
    VisionObservation,
    VisionSessionState,
)


class TestWindowsCameraCaptureAutomated:
    """Automated unit tests for camera capture abstraction with mocked driver layer."""

    def test_implements_ivision_capture_interface(self) -> None:
        """Verify WindowsCameraCaptureDevice implements IVisionCapture interface."""
        device = WindowsCameraCaptureDevice(simulated_mode=True)
        assert isinstance(device, IVisionCapture)

    def test_default_configuration_and_fps(self) -> None:
        """Verify default configuration targets."""
        config = VisionConfig(max_width=640, max_height=480, max_fps=15.0, max_buffered_frames=5)
        device = WindowsCameraCaptureDevice(config=config, simulated_mode=True)
        assert device.configured_fps == 15.0
        assert device.measured_fps == 0.0
        assert device.config.max_width == 640
        assert device.config.max_height == 480

    @pytest.mark.asyncio
    async def test_camera_enumeration_simulated(self) -> None:
        """Verify camera enumeration in simulated environment."""
        device = WindowsCameraCaptureDevice(simulated_mode=True)
        devices = await device.list_devices()
        assert len(devices) >= 1
        assert devices[0]["available"] is True
        assert devices[0]["backend"] == "simulated"

    @pytest.mark.asyncio
    async def test_camera_selection_explicit_integer_index(self) -> None:
        """Verify explicit camera resolution by integer device index."""
        device = WindowsCameraCaptureDevice(device_id=0, simulated_mode=False)
        idx, name = await device.resolve_camera_device()
        assert idx == 0
        assert "0" in name

    @pytest.mark.asyncio
    async def test_camera_selection_hal_device_id(self) -> None:
        """Verify camera resolution via HAL DeviceManager identifier."""
        mock_hal = DeviceManager()
        test_cam = HardwareDevice(
            device_id="video_in_1",
            name="Logitech HD Webcam C920",
            device_type=DeviceType.VIDEO_INPUT,
            state=DeviceState.AVAILABLE,
            is_default=True,
            channels=3,
        )

        device = WindowsCameraCaptureDevice(
            device_id="video_in_1", device_manager=mock_hal, simulated_mode=False
        )

        with patch.object(mock_hal, "list_devices", return_value=[test_cam]):
            idx, name = await device.resolve_camera_device()
            assert idx == 1
            assert name == "Logitech HD Webcam C920"

    @pytest.mark.asyncio
    async def test_invalid_camera_raises_error(self) -> None:
        """Verify non-existent camera raises VisionDeviceNotFoundError."""
        mock_hal = DeviceManager()
        device = WindowsCameraCaptureDevice(
            device_id="non_existent_camera", device_manager=mock_hal, simulated_mode=False
        )

        with patch.object(mock_hal, "list_devices", return_value=[]):
            with pytest.raises(VisionDeviceNotFoundError):
                await device.resolve_camera_device()

    @pytest.mark.asyncio
    async def test_simulated_frame_capture_and_format(self) -> None:
        """Verify simulated capture yields valid VisionFrame objects."""
        config = VisionConfig(max_width=320, max_height=240, max_fps=20.0, max_buffered_frames=5)
        device = WindowsCameraCaptureDevice(config=config, simulated_mode=True)

        await device.start()
        assert device.current_state == VisionSessionState.CAPTURING

        captured_frames: list[VisionFrame] = []
        async for frame in device.frames():
            captured_frames.append(frame)
            if len(captured_frames) >= 2:
                break

        await device.stop()
        assert device.current_state == VisionSessionState.STOPPED

        assert len(captured_frames) == 2
        for f in captured_frames:
            assert f.width == 320
            assert f.height == 240
            assert f.format.pixel_format == PixelFormat.RGB24
            assert f.timestamp.tzinfo is not None
            assert len(f.payload) == 320 * 240 * 3

    @pytest.mark.asyncio
    async def test_backpressure_drop_oldest_policy(self) -> None:
        """Verify bounded queue drops oldest frames when downstream reader is slow."""
        config = VisionConfig(max_width=100, max_height=100, max_fps=30.0, max_buffered_frames=3)
        device = WindowsCameraCaptureDevice(config=config, simulated_mode=True)

        await device.start()
        # Allow background task to produce more frames than queue capacity (maxsize=3)
        await asyncio.sleep(0.2)
        await device.stop()

        assert device.frames_dropped > 0
        assert device._queue.qsize() <= 3

    @pytest.mark.asyncio
    async def test_camera_recovery_lifecycle(self) -> None:
        """Verify recover() restarts the capture stream cleanly."""
        device = WindowsCameraCaptureDevice(simulated_mode=True)
        await device.start()
        assert device.current_state == VisionSessionState.CAPTURING

        # Invoke recover()
        recovered = await device.recover()
        assert recovered is True
        assert device.current_state == VisionSessionState.CAPTURING

        await device.stop()
        assert device.current_state == VisionSessionState.STOPPED

    @pytest.mark.asyncio
    async def test_camera_health_telemetry_reporting(self) -> None:
        """Verify health check returns complete diagnostics."""
        device = WindowsCameraCaptureDevice(simulated_mode=True)
        await device.start()
        health = await device.health()

        assert health["subsystem"] == "windows_camera_capture"
        assert health["healthy"] is True
        assert health["state"] == "CAPTURING"
        assert "configured_fps" in health
        assert "measured_fps" in health
        assert "frame_latency_ms" in health
        assert "frames_captured" in health
        assert "frames_dropped" in health
        assert "capture_errors" in health
        assert "queue_depth" in health

        await device.stop()

    def test_security_and_privacy_invariants(self) -> None:
        """Verify camera adapter exposes zero tool execution methods and protects privacy."""
        forbidden_methods = [
            "execute",
            "run_tool",
            "modify_file",
            "grant_capability",
            "authorize",
            "elevate",
            "execute_shell",
        ]
        for m in forbidden_methods:
            assert not hasattr(WindowsCameraCaptureDevice, m)

        # Repr redaction check
        frame = VisionFrame(
            frame_id="cam_test_repr",
            sequence_number=1,
            timestamp=datetime.now(UTC),
            width=640,
            height=480,
            format=FrameFormat(
                width=640,
                height=480,
                channels=3,
                pixel_format=PixelFormat.RGB24,
                frame_rate=15.0,
            ),
            payload=b"\x00" * 1000,
        )
        assert "b'\\x00" not in repr(frame)
        assert "payload_bytes=1000" in repr(frame)


class TestPhysicalHardwareCameraIntegration:
    """Physical hardware integration test executing real camera acquisition if webcam exists."""

    @pytest.mark.asyncio
    async def test_physical_webcam_capture_pipeline(self) -> None:
        """Performs actual hardware capture from physical Windows webcam and routes to vision pipeline."""
        if not OPENCV_AVAILABLE:
            pytest.skip("OpenCV is not available on host.")

        # Probe physical camera 0
        import cv2

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            pytest.skip("No physical webcam detected on host machine.")
        cap.release()

        config = VisionConfig(max_width=640, max_height=480, max_fps=15.0, max_buffered_frames=5)
        device = WindowsCameraCaptureDevice(device_id=0, config=config, simulated_mode=False)

        try:
            await device.start()
            assert device.current_state == VisionSessionState.CAPTURING

            captured_frames: list[VisionFrame] = []
            async for frame in device.frames():
                captured_frames.append(frame)
                if len(captured_frames) >= 3:
                    break

            assert len(captured_frames) == 3
            for f in captured_frames:
                assert f.width == 640
                assert f.height == 480
                assert f.format.pixel_format == PixelFormat.RGB24
                assert len(f.payload) == 640 * 480 * 3
                assert f.sequence_number >= 1

            # Performance telemetry validation
            health = await device.health()
            assert health["healthy"] is True
            assert health["frames_captured"] >= 3
            assert health["frame_latency_ms"] >= 0.0

            # Feed physical frame into existing VisionProcessor & ObjectDetector
            detector = ObjectDetectorAdapter()
            detections = await detector.detect(captured_frames[0])
            assert isinstance(detections, list)

            # Feed into VisionPlannerContextBuilder
            observation = VisionObservation(
                object_detections=detections,
            )
            builder = VisionPlannerContextBuilder()
            planner_context = await builder.build_context(observation)
            assert planner_context is not None

        finally:
            await device.stop()
            assert device.current_state == VisionSessionState.STOPPED
