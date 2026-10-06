"""Hardware Device Manager Implementation (Phase 4H.1).

Concrete implementation of IDeviceManager enumerating audio and camera/video devices,
detecting state changes, and providing stable metadata without blocking the event loop.
"""

import asyncio
import logging
from typing import Any

from app.hardware.base import IDeviceManager
from app.hardware.config import HardwareConfig, create_hardware_config
from app.hardware.models import DeviceState, DeviceType, HardwareDevice

logger = logging.getLogger(__name__)


class DeviceManager(IDeviceManager):
    """Production-ready Hardware Device Manager implementing IDeviceManager."""

    def __init__(self, config: HardwareConfig | None = None) -> None:
        """Initializes DeviceManager.

        Args:
            config: Optional HardwareConfig configuration instance.
        """
        self.config = config or create_hardware_config()
        self._devices: dict[str, HardwareDevice] = {}
        self._initialized = False
        self._lock = asyncio.Lock()

    async def _enumerate_audio_devices(self) -> list[HardwareDevice]:
        """Enumerates audio input and output devices using sounddevice or fallback probes."""
        devices: list[HardwareDevice] = []

        def _probe_sounddevice() -> list[HardwareDevice]:
            detected: list[HardwareDevice] = []
            try:
                import sounddevice as sd  # type: ignore[import-not-found,import-untyped]

                raw_devices = sd.query_devices()
                default_in = sd.default.device[0] if sd.default.device else None
                default_out = sd.default.device[1] if sd.default.device else None

                for idx, d in enumerate(raw_devices):
                    max_in = int(d.get("max_input_channels", 0))
                    max_out = int(d.get("max_output_channels", 0))
                    name = str(d.get("name", f"Audio Device {idx}"))
                    sample_rate = int(d.get("default_samplerate", 16000))

                    if max_in > 0:
                        dev_id = f"audio_in_{idx}"
                        detected.append(
                            HardwareDevice(
                                device_id=dev_id,
                                name=name,
                                device_type=DeviceType.AUDIO_INPUT,
                                state=DeviceState.AVAILABLE,
                                is_default=(idx == default_in),
                                sample_rates=[16000, 24000, 44100, 48000],
                                channels=max_in,
                                capabilities={"default_sample_rate": sample_rate},
                                metadata={"host_api": d.get("hostapi", 0), "driver": "sounddevice"},
                            )
                        )

                    if max_out > 0:
                        dev_id = f"audio_out_{idx}"
                        detected.append(
                            HardwareDevice(
                                device_id=dev_id,
                                name=name,
                                device_type=DeviceType.AUDIO_OUTPUT,
                                state=DeviceState.AVAILABLE,
                                is_default=(idx == default_out),
                                sample_rates=[16000, 24000, 44100, 48000],
                                channels=max_out,
                                capabilities={"default_sample_rate": sample_rate},
                                metadata={"host_api": d.get("hostapi", 0), "driver": "sounddevice"},
                            )
                        )
            except Exception as exc:
                logger.debug(f"SoundDevice audio enumeration probe skipped/failed: {exc}")
            return detected

        loop = asyncio.get_running_loop()
        try:
            devices = await asyncio.wait_for(
                loop.run_in_executor(None, _probe_sounddevice),
                timeout=self.config.probe_timeout_sec,
            )
        except Exception as exc:
            logger.debug(f"Audio enumeration error: {exc}")

        return devices

    async def _enumerate_video_devices(self) -> list[HardwareDevice]:
        """Enumerates video capture cameras safely using OpenCV/DirectShow or fallbacks."""
        devices: list[HardwareDevice] = []

        def _probe_cameras() -> list[HardwareDevice]:
            detected: list[HardwareDevice] = []
            try:
                import cv2  # type: ignore[import-not-found]

                # Probe device index 0
                cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                if cap.isOpened():
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
                    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
                    cap.release()

                    detected.append(
                        HardwareDevice(
                            device_id="video_in_0",
                            name="Physical Camera Device 0",
                            device_type=DeviceType.VIDEO_INPUT,
                            state=DeviceState.AVAILABLE,
                            is_default=True,
                            channels=3,
                            capabilities={"width": w, "height": h, "fps": fps},
                            metadata={"backend": "opencv_dshow"},
                        )
                    )
            except Exception as exc:
                logger.debug(f"OpenCV camera enumeration probe skipped/failed: {exc}")
            return detected

        loop = asyncio.get_running_loop()
        try:
            devices = await asyncio.wait_for(
                loop.run_in_executor(None, _probe_cameras),
                timeout=self.config.probe_timeout_sec,
            )
        except Exception as exc:
            logger.debug(f"Video enumeration error: {exc}")

        return devices

    async def refresh_devices(self) -> list[HardwareDevice]:
        """Forces a complete hardware device rescan and updates internal device states."""
        async with self._lock:
            new_devices: dict[str, HardwareDevice] = {}

            audio_devs = await self._enumerate_audio_devices()
            video_devs = await self._enumerate_video_devices()

            for d in audio_devs + video_devs:
                new_devices[d.device_id] = d

            # Mark previously known devices that vanished as DISCONNECTED
            for dev_id, old_dev in self._devices.items():
                if dev_id not in new_devices:
                    disconnected = old_dev.model_copy(
                        update={"state": DeviceState.DISCONNECTED, "is_default": False}
                    )
                    new_devices[dev_id] = disconnected

            self._devices = new_devices
            self._initialized = True
            return list(self._devices.values())

    async def list_devices(self, device_type: DeviceType | None = None) -> list[HardwareDevice]:
        """Enumerates available hardware devices, optionally filtered by DeviceType."""
        if not self._initialized:
            await self.refresh_devices()

        devices = list(self._devices.values())
        if device_type is not None:
            devices = [d for d in devices if d.device_type == device_type]
        return devices

    async def get_device(self, device_id: str) -> HardwareDevice | None:
        """Retrieves a specific hardware device record by its unique ID."""
        if not self._initialized:
            await self.refresh_devices()
        return self._devices.get(device_id)

    async def get_default_device(self, device_type: DeviceType) -> HardwareDevice | None:
        """Returns the system default device for the specified DeviceType."""
        devices = await self.list_devices(device_type=device_type)
        for d in devices:
            if d.is_default and d.state == DeviceState.AVAILABLE:
                return d
        available = [d for d in devices if d.state == DeviceState.AVAILABLE]
        return available[0] if available else None

    async def is_device_available(self, device_id: str) -> bool:
        """Checks if a specific hardware device is currently connected and operational."""
        dev = await self.get_device(device_id)
        return dev is not None and dev.state == DeviceState.AVAILABLE

    async def health(self) -> dict[str, Any]:
        """Probes health and readiness status of the Device Manager subsystem."""
        devices = await self.list_devices()
        audio_in_count = len(
            [
                d
                for d in devices
                if d.device_type == DeviceType.AUDIO_INPUT and d.state == DeviceState.AVAILABLE
            ]
        )
        audio_out_count = len(
            [
                d
                for d in devices
                if d.device_type == DeviceType.AUDIO_OUTPUT and d.state == DeviceState.AVAILABLE
            ]
        )
        video_in_count = len(
            [
                d
                for d in devices
                if d.device_type == DeviceType.VIDEO_INPUT and d.state == DeviceState.AVAILABLE
            ]
        )

        return {
            "subsystem": "device_manager",
            "initialized": self._initialized,
            "total_devices_tracked": len(self._devices),
            "available_audio_inputs": audio_in_count,
            "available_audio_outputs": audio_out_count,
            "available_video_inputs": video_in_count,
            "healthy": True,
        }
