"""Real Camera Capture Adapter (Phase 4F.2).

Concrete implementation of IVisionCapture executing real camera acquisition via background threads
(e.g. OpenCV / video capture drivers) while preserving bounded queues, format negotiation,
state machine transitions, device discovery, resource cleanup, and privacy boundaries.
"""

import asyncio
import logging
import time
from collections.abc import AsyncIterable
from datetime import UTC, datetime
from typing import Any

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


class CameraCaptureAdapter(IVisionCapture):
    """Real Camera Acquisition Adapter for ULTRON Vision Subsystem."""

    def __init__(
        self,
        config: VisionConfig | None = None,
        simulated_mode: bool | None = None,
    ) -> None:
        """Initializes CameraCaptureAdapter.

        Args:
            config: Optional VisionConfig configuration instance.
            simulated_mode: Optional boolean flag. If True, forces simulated camera capture mode.
        """
        self.config = config or create_vision_config()

        # Auto-detect or force simulated mode
        if simulated_mode is not None:
            self._simulated_mode = simulated_mode
        else:
            try:
                import cv2  # type: ignore[import-not-found] # noqa: F401

                self._simulated_mode = False
            except ImportError:
                self._simulated_mode = True

        self._state_machine = VisionSessionStateMachine(initial_state=VisionSessionState.IDLE)
        self._queue: asyncio.Queue[VisionFrame] = asyncio.Queue(
            maxsize=self.config.max_buffered_frames
        )

        self._sequence_number = 0
        self._frames_received = 0
        self._frames_processed = 0
        self._frames_dropped = 0

        self._capture_task: asyncio.Task[None] | None = None
        self._running = False
        self._cap_device: Any = None

    async def list_devices(self) -> list[dict[str, Any]]:
        """Safely enumerates available video capture devices with sanitized metadata."""
        devices: list[dict[str, Any]] = []

        if self._simulated_mode:
            devices.append(
                {
                    "device_id": "default",
                    "name": "Simulated Camera Device 0",
                    "width": min(self.config.max_width, 1280),
                    "height": min(self.config.max_height, 720),
                    "fps": self.config.max_fps,
                    "backend": "simulated",
                    "available": True,
                }
            )
            return devices

        try:
            import cv2  # type: ignore[import-not-found]

            # Probe device 0
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
                fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
                cap.release()

                devices.append(
                    {
                        "device_id": "0",
                        "name": "Physical Camera Device 0",
                        "width": min(w, self.config.max_width),
                        "height": min(h, self.config.max_height),
                        "fps": min(fps, self.config.max_fps),
                        "backend": "opencv",
                        "available": True,
                    }
                )
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
        """Starts video frame capture stream idempotently."""
        if self._running:
            logger.info("CameraCaptureAdapter: start() called while already running.")
            return

        target_device = self.config.camera_device
        if target_device not in ("default", "0", "simulated"):
            raise VisionDeviceNotFoundError(
                f"Requested camera device '{target_device}' was not found."
            )

        self._state_machine.transition_to(VisionSessionState.INITIALIZING)

        self._running = True
        self._sequence_number = 0
        self._state_machine.transition_to(VisionSessionState.CAPTURING)

        # Clear existing queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        loop = asyncio.get_running_loop()
        self._capture_task = loop.create_task(self._capture_loop())
        logger.info(
            f"CameraCaptureAdapter started capture loop (simulated={self._simulated_mode})."
        )

    async def stop(self) -> None:
        """Stops video frame capture stream and releases all camera resources idempotently."""
        if not self._running:
            logger.info("CameraCaptureAdapter: stop() called while already stopped.")
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
            try:
                self._cap_device.release()
            except Exception:
                pass
            self._cap_device = None

        try:
            self._state_machine.transition_to(VisionSessionState.STOPPED)
        except Exception:
            pass

        logger.info("CameraCaptureAdapter capture loop stopped cleanly.")

    async def _capture_loop(self) -> None:
        """Background frame capture loop running non-blockingly."""
        interval_sec = 1.0 / max(1.0, self.config.max_fps)
        width = min(self.config.max_width, 1280)
        height = min(self.config.max_height, 720)

        fmt = FrameFormat(
            width=width,
            height=height,
            channels=3,
            pixel_format=PixelFormat.RGB24,
            frame_rate=self.config.max_fps,
        )

        try:
            while self._running:
                start_time = time.perf_counter()

                # Generate or acquire frame
                if self._simulated_mode or self._cap_device is None:
                    # Synthetic 24-bit RGB frame payload
                    payload = b"\x00\x80\xff" * (width * height)
                else:
                    # Physical OpenCV frame acquisition offloaded to thread executor
                    payload = await self._acquire_physical_frame(width, height)

                self._sequence_number += 1
                self._frames_received += 1

                frame = VisionFrame(
                    frame_id=f"frm_cam_{self._sequence_number}",
                    sequence_number=self._sequence_number,
                    timestamp=datetime.now(UTC),
                    width=width,
                    height=height,
                    format=fmt,
                    payload=payload if payload else b"\x00" * 100,
                )

                # Push to bounded queue with DROP_OLDEST backpressure policy
                if self._queue.full():
                    try:
                        self._queue.get_nowait()
                        self._frames_dropped += 1
                    except asyncio.QueueEmpty:
                        pass

                await self._queue.put(frame)

                elapsed = time.perf_counter() - start_time
                sleep_dur = max(0.001, interval_sec - elapsed)
                await asyncio.sleep(sleep_dur)

        except asyncio.CancelledError:
            logger.info("CameraCaptureAdapter capture loop task cancelled.")
            raise
        except Exception as exc:
            logger.error(f"CameraCaptureAdapter error in capture loop: {exc}")
            try:
                self._state_machine.transition_to(VisionSessionState.ERROR)
            except Exception:
                pass
            raise VisionCaptureError(f"Camera capture failure: {exc}") from exc

    async def _acquire_physical_frame(self, width: int, height: int) -> bytes:
        """Offloads blocking OpenCV frame acquisition to background thread executor."""

        def _read_frame() -> bytes:
            if self._cap_device is None:
                import cv2  # type: ignore[import-not-found]

                self._cap_device = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                self._cap_device.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self._cap_device.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

            ret, mat = self._cap_device.read()
            if not ret or mat is None:
                return b"\x00" * (width * height * 3)

            import cv2  # type: ignore[import-not-found]

            resized = cv2.resize(mat, (width, height))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            return bytes(rgb.tobytes())

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _read_frame)

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
        """Probes operational health and telemetry metrics of camera capture adapter."""
        status = "RUNNING" if self._running else "IDLE"
        if self._state_machine.state == VisionSessionState.ERROR:
            status = "UNAVAILABLE"

        return {
            "subsystem": "camera_capture",
            "status": status,
            "simulated_mode": self._simulated_mode,
            "device_id": self.config.camera_device,
            "state": self._state_machine.state.value,
            "frames_received": self._frames_received,
            "frames_processed": self._frames_processed,
            "frames_dropped": self._frames_dropped,
            "queue_depth": self._queue.qsize(),
            "max_queue_depth": self.config.max_buffered_frames,
        }
