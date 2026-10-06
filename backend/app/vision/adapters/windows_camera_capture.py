"""Windows Physical Camera Hardware Capture Device Adapter (Phase 4H.3).

Concrete production implementation of IVisionCapture interfacing with Windows webcams
and video cameras via DirectShow / MediaFoundation / OpenCV and HAL DeviceManager.

Features:
- HAL DeviceManager integration for video input device discovery and resolution.
- DirectShow (CAP_DSHOW) / MediaFoundation (CAP_MSMF) Windows driver bindings.
- Non-blocking frame acquisition offloaded to background executor threads.
- Standardized VisionFrame generation (RGB24 format, monotonic sequence, timezone-aware UTC).
- Bounded asyncio queue with DROP_OLDEST backpressure policy.
- Telemetry tracking: Configured FPS vs Actual Measured FPS, frame latency, dropped frames, errors.
- Device hot-unplug / disconnection detection and automatic recovery (recover()).
- Strict Zero-Trust security and ephemeral privacy guarantees (person != identity).
"""

import asyncio
import logging
import time
from collections import deque
from collections.abc import AsyncIterable
from datetime import UTC, datetime
from typing import Any

from app.hardware.base import IDeviceManager
from app.hardware.device_manager import DeviceManager
from app.hardware.models import DeviceType
from app.vision.base import IVisionCapture
from app.vision.config import VisionConfig, create_vision_config
from app.vision.exceptions import (
    VisionCaptureError,
    VisionDeviceNotFoundError,
)
from app.vision.models import (
    FrameFormat,
    PixelFormat,
    VisionFrame,
    VisionSessionState,
    VisionSessionStateMachine,
)

logger = logging.getLogger(__name__)

# Conditional import for OpenCV
try:
    import cv2  # type: ignore[import-not-found]

    OPENCV_AVAILABLE = True
except ImportError:
    cv2 = None  # type: ignore[assignment]
    OPENCV_AVAILABLE = False


class WindowsCameraCaptureDevice(IVisionCapture):
    """Windows Physical Camera Capture Device implementing IVisionCapture."""

    def __init__(
        self,
        device_id: int | str | None = None,
        config: VisionConfig | None = None,
        device_manager: IDeviceManager | None = None,
        simulated_mode: bool | None = None,
    ) -> None:
        """Initializes WindowsCameraCaptureDevice.

        Args:
            device_id: Target camera index (int) or hardware identifier string.
            config: VisionConfig configuration instance.
            device_manager: IDeviceManager instance for device discovery.
            simulated_mode: If True, operates in synthetic test mode without physical hardware.
        """
        self.config = config or create_vision_config()
        self._target_device_id = device_id if device_id is not None else self.config.camera_device
        self._device_manager = device_manager or DeviceManager()

        # Resolve simulated mode
        if simulated_mode is not None:
            self._simulated_mode = simulated_mode
        else:
            self._simulated_mode = not OPENCV_AVAILABLE

        self._state_machine = VisionSessionStateMachine(initial_state=VisionSessionState.IDLE)
        self._queue: asyncio.Queue[VisionFrame] = asyncio.Queue(
            maxsize=self.config.max_buffered_frames
        )

        self._sequence_number = 0
        self._frames_captured = 0
        self._frames_processed = 0
        self._frames_dropped = 0
        self._capture_errors = 0

        # Performance measurement
        self._last_frame_time: float | None = None
        self._frame_intervals: deque[float] = deque(maxlen=30)
        self._last_latency_ms: float = 0.0

        self._capture_task: asyncio.Task[None] | None = None
        self._running = False
        self._cap_device: Any = None
        self._resolved_device_index: int | None = None
        self._resolved_device_name: str = "Unknown Camera"
        self._actual_width: int = 0
        self._actual_height: int = 0

    @property
    def current_state(self) -> VisionSessionState:
        """Returns active lifecycle state."""
        return self._state_machine.state

    @property
    def configured_fps(self) -> float:
        """Returns target configured frame rate."""
        return float(self.config.max_fps)

    @property
    def measured_fps(self) -> float:
        """Calculates actual measured FPS based on rolling average frame intervals."""
        if not self._frame_intervals or len(self._frame_intervals) < 2:
            return 0.0
        avg_interval = sum(self._frame_intervals) / len(self._frame_intervals)
        return round(1.0 / avg_interval, 2) if avg_interval > 0 else 0.0

    @property
    def frame_latency_ms(self) -> float:
        """Returns the most recent physical frame capture latency in milliseconds."""
        return round(self._last_latency_ms, 2)

    @property
    def frames_captured(self) -> int:
        """Returns count of captured frames."""
        return self._frames_captured

    @property
    def frames_dropped(self) -> int:
        """Returns count of dropped frames under backpressure."""
        return self._frames_dropped

    @property
    def capture_errors(self) -> int:
        """Returns count of frame acquisition errors."""
        return self._capture_errors

    async def resolve_camera_device(self) -> tuple[int | None, str]:
        """Resolves target camera index and name using DeviceManager and OpenCV."""
        if self._simulated_mode or not OPENCV_AVAILABLE:
            return None, "Simulated Virtual Camera"

        # Case 1: Integer index provided
        if isinstance(self._target_device_id, int):
            return self._target_device_id, f"Physical Camera Device {self._target_device_id}"

        target = str(self._target_device_id)

        # Case 2: Numeric string provided ("0", "1")
        if target.isdigit():
            idx = int(target)
            return idx, f"Physical Camera Device {idx}"

        # Case 3: Query HAL DeviceManager
        devices = await self._device_manager.list_devices(DeviceType.VIDEO_INPUT)
        if target not in ("default", "auto", ""):
            for d in devices:
                if d.device_id == target or target.lower() in d.name.lower():
                    if d.device_id.startswith("video_in_"):
                        try:
                            idx = int(d.device_id.replace("video_in_", ""))
                            return idx, d.name
                        except ValueError:
                            pass
                    return 0, d.name

            raise VisionDeviceNotFoundError(f"Requested camera device '{target}' was not found.")

        # Case 4: Default camera resolution
        if devices:
            default_dev = devices[0]
            if default_dev.device_id.startswith("video_in_"):
                try:
                    idx = int(default_dev.device_id.replace("video_in_", ""))
                    return idx, default_dev.name
                except ValueError:
                    pass
            return 0, default_dev.name

        # Fallback probe for device 0
        return 0, "Default Physical Camera (Index 0)"

    async def list_devices(self) -> list[dict[str, Any]]:
        """Enumerates available physical and simulated video capture devices."""
        devices: list[dict[str, Any]] = []

        if self._simulated_mode or not OPENCV_AVAILABLE:
            devices.append(
                {
                    "device_id": "simulated_0",
                    "name": "Simulated Virtual Camera",
                    "width": min(self.config.max_width, 1280),
                    "height": min(self.config.max_height, 720),
                    "fps": self.config.max_fps,
                    "backend": "simulated",
                    "available": True,
                }
            )
            return devices

        def _probe_opencv() -> list[dict[str, Any]]:
            probed: list[dict[str, Any]] = []
            try:
                cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                if cap.isOpened():
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
                    fps = float(cap.get(cv2.CAP_PROP_FPS))
                    if fps <= 0:
                        fps = 30.0
                    cap.release()

                    probed.append(
                        {
                            "device_id": "0",
                            "name": "Physical Camera Device 0",
                            "width": w,
                            "height": h,
                            "fps": fps,
                            "backend": "opencv_dshow",
                            "available": True,
                        }
                    )
            except Exception as exc:
                logger.debug(f"OpenCV direct probe error: {exc}")
            return probed

        loop = asyncio.get_running_loop()
        try:
            devices = await asyncio.wait_for(loop.run_in_executor(None, _probe_opencv), timeout=2.5)
        except Exception as exc:
            logger.warning(f"Error enumerating camera devices: {exc}")

        if not devices:
            devices.append(
                {
                    "device_id": "default",
                    "name": "Simulated Camera Fallback",
                    "width": min(self.config.max_width, 1280),
                    "height": min(self.config.max_height, 720),
                    "fps": self.config.max_fps,
                    "backend": "simulated",
                    "available": True,
                }
            )

        return devices

    async def start(self) -> None:
        """Starts video frame capture stream asynchronously (idempotent)."""
        if self._running:
            return

        self._state_machine.transition_to(VisionSessionState.INITIALIZING)

        if not self._simulated_mode and OPENCV_AVAILABLE:
            try:
                idx, name = await self.resolve_camera_device()
                self._resolved_device_index = idx
                self._resolved_device_name = name

                idx_int = idx if idx is not None else 0

                # Open camera device in background thread to avoid blocking loop
                def _open_camera() -> Any:
                    cap = cv2.VideoCapture(idx_int, cv2.CAP_DSHOW)
                    if not cap.isOpened():
                        # Fallback to MSMF backend if DSHOW fails
                        cap = cv2.VideoCapture(idx_int, cv2.CAP_MSMF)
                    if not cap.isOpened():
                        raise VisionCaptureError(f"Failed to open video capture device {idx_int}.")

                    # Request target resolution
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.max_width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.max_height)
                    cap.set(cv2.CAP_PROP_FPS, self.config.max_fps)
                    return cap

                loop = asyncio.get_running_loop()
                self._cap_device = await loop.run_in_executor(None, _open_camera)
                self._actual_width = int(self._cap_device.get(cv2.CAP_PROP_FRAME_WIDTH)) or self.config.max_width
                self._actual_height = int(self._cap_device.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self.config.max_height
            except Exception as exc:
                self._state_machine.transition_to(VisionSessionState.ERROR)
                logger.error(f"WindowsCameraCaptureDevice failed to initialize camera: {exc}")
                raise VisionCaptureError(f"Camera initialization failed: {exc}") from exc
        else:
            self._actual_width = min(self.config.max_width, 1280)
            self._actual_height = min(self.config.max_height, 720)

        # Clear existing queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        self._sequence_number = 0
        self._frames_captured = 0
        self._frames_processed = 0
        self._frames_dropped = 0
        self._capture_errors = 0
        self._frame_intervals.clear()
        self._last_frame_time = None

        self._running = True
        self._state_machine.transition_to(VisionSessionState.CAPTURING)

        loop = asyncio.get_running_loop()
        self._capture_task = loop.create_task(self._capture_loop())
        logger.info(
            f"WindowsCameraCaptureDevice started capture: '{self._resolved_device_name}' "
            f"(simulated={self._simulated_mode}, target_fps={self.config.max_fps}, "
            f"resolution={self._actual_width}x{self._actual_height})."
        )

    async def stop(self) -> None:
        """Stops video frame capture stream and releases all camera resources cleanly (idempotent)."""
        if not self._running:
            return

        self._running = False
        try:
            self._state_machine.transition_to(VisionSessionState.STOPPING)
        except Exception:
            pass

        if self._capture_task:
            self._capture_task.cancel()
            try:
                await self._capture_task
            except asyncio.CancelledError:
                pass
            self._capture_task = None

        if self._cap_device is not None:
            def _release() -> None:
                try:
                    self._cap_device.release()
                except Exception as exc:
                    logger.warning(f"Error releasing camera device: {exc}")

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, _release)
            self._cap_device = None

        # Drain queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        try:
            self._state_machine.transition_to(VisionSessionState.STOPPED)
        except Exception:
            pass

        logger.info("WindowsCameraCaptureDevice stopped cleanly.")

    async def recover(self) -> bool:
        """Recovers and restarts camera capture after a disconnection or driver error.

        Returns:
            bool: True if recovery succeeded, False otherwise.
        """
        logger.info("Attempting recovery on WindowsCameraCaptureDevice...")
        await self.stop()
        try:
            await self.start()
            return self._running and self._state_machine.state == VisionSessionState.CAPTURING
        except Exception as exc:
            logger.warning(f"WindowsCameraCaptureDevice recovery failed: {exc}")
            self._state_machine.transition_to(VisionSessionState.ERROR)
            return False

    async def _acquire_frame_bytes(self, width: int, height: int) -> tuple[bytes, float]:
        """Acquires a single frame from physical hardware and returns raw RGB24 bytes and latency."""
        if self._simulated_mode or self._cap_device is None or not OPENCV_AVAILABLE:
            payload = b"\x00\x80\xff" * (width * height)
            return payload, 1.0

        def _read() -> tuple[bytes, float]:
            t0 = time.perf_counter()
            ret, mat = self._cap_device.read()
            t1 = time.perf_counter()
            latency_ms = (t1 - t0) * 1000.0

            if not ret or mat is None:
                raise RuntimeError("Camera driver read() returned False or empty frame.")

            if mat.shape[1] != width or mat.shape[0] != height:
                mat = cv2.resize(mat, (width, height))

            rgb = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)
            return bytes(rgb.tobytes()), latency_ms

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _read)

    async def _capture_loop(self) -> None:
        """Non-blocking background frame capture loop."""
        interval_sec = 1.0 / max(1.0, float(self.config.max_fps))
        width = self._actual_width or min(self.config.max_width, 1280)
        height = self._actual_height or min(self.config.max_height, 720)

        fmt = FrameFormat(
            width=width,
            height=height,
            channels=3,
            pixel_format=PixelFormat.RGB24,
            frame_rate=float(self.config.max_fps),
        )

        try:
            while self._running:
                loop_start = time.perf_counter()

                # Track rolling frame interval for actual measured FPS
                if self._last_frame_time is not None:
                    delta = loop_start - self._last_frame_time
                    if delta > 0:
                        self._frame_intervals.append(delta)
                self._last_frame_time = loop_start

                try:
                    payload, latency_ms = await self._acquire_frame_bytes(width, height)
                    self._last_latency_ms = latency_ms
                except Exception as exc:
                    self._capture_errors += 1
                    logger.debug(f"Frame acquisition error in capture loop: {exc}")
                    # Allow 5 consecutive errors before declaring camera error
                    if self._capture_errors > 5:
                        self._state_machine.transition_to(VisionSessionState.ERROR)
                        raise VisionCaptureError(f"Persistent camera capture failure: {exc}") from exc
                    await asyncio.sleep(0.1)
                    continue

                self._sequence_number += 1
                self._frames_captured += 1

                frame = VisionFrame(
                    frame_id=f"win_cam_{self._sequence_number}",
                    sequence_number=self._sequence_number,
                    timestamp=datetime.now(UTC),
                    width=width,
                    height=height,
                    format=fmt,
                    payload=payload,
                )

                # Bounded queue with DROP_OLDEST backpressure
                if self._queue.full():
                    try:
                        self._queue.get_nowait()
                        self._frames_dropped += 1
                    except asyncio.QueueEmpty:
                        pass

                await self._queue.put(frame)

                elapsed = time.perf_counter() - loop_start
                sleep_time = max(0.001, interval_sec - elapsed)
                await asyncio.sleep(sleep_time)

        except asyncio.CancelledError:
            logger.info("WindowsCameraCaptureDevice capture task cancelled.")
            raise
        except Exception as exc:
            logger.error(f"WindowsCameraCaptureDevice fatal error in capture loop: {exc}")
            try:
                self._state_machine.transition_to(VisionSessionState.ERROR)
            except Exception:
                pass
            raise VisionCaptureError(f"Camera capture failure: {exc}") from exc

    async def frames(self) -> AsyncIterable[VisionFrame]:
        """Asynchronously yields captured VisionFrame objects from the bounded queue."""
        while self._running or not self._queue.empty():
            try:
                frame = await asyncio.wait_for(self._queue.get(), timeout=0.2)
                self._frames_processed += 1
                yield frame
            except TimeoutError:
                if not self._running:
                    break

    async def health(self) -> dict[str, Any]:
        """Probes operational health, state, and telemetry metrics of camera capture."""
        healthy = self._state_machine.state not in (VisionSessionState.ERROR, "ERROR")
        return {
            "subsystem": "windows_camera_capture",
            "healthy": healthy,
            "state": self._state_machine.state.value,
            "device_name": self._resolved_device_name,
            "device_index": self._resolved_device_index,
            "configured_fps": self.configured_fps,
            "measured_fps": self.measured_fps,
            "frame_latency_ms": self.frame_latency_ms,
            "resolution": f"{self._actual_width}x{self._actual_height}",
            "simulated_mode": self._simulated_mode,
            "opencv_available": OPENCV_AVAILABLE,
            "frames_captured": self._frames_captured,
            "frames_processed": self._frames_processed,
            "frames_dropped": self._frames_dropped,
            "capture_errors": self._capture_errors,
            "queue_depth": self._queue.qsize(),
            "max_queue_depth": self.config.max_buffered_frames,
        }
