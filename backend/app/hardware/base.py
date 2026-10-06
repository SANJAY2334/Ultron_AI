"""Hardware Abstraction Layer (HAL) Abstract Interfaces (Phase 4H.1).

Defines provider-agnostic abstract contracts for DeviceManager,
HardwareProfileDetector, and ResourceTelemetryCollector.
"""

from abc import ABC, abstractmethod
from typing import Any

from app.hardware.models import (
    CpuProfile,
    DeviceType,
    GpuDeviceInfo,
    HardwareDevice,
    HardwareProfile,
    MemoryProfile,
    NpuProfile,
    ResourceTelemetry,
)


class IDeviceManager(ABC):
    """Abstract Interface for hardware device enumeration, inspection, and lifecycle monitoring."""

    @abstractmethod
    async def list_devices(self, device_type: DeviceType | None = None) -> list[HardwareDevice]:
        """Enumerates available hardware devices, optionally filtered by DeviceType.

        Args:
            device_type: Optional DeviceType filter (e.g. AUDIO_INPUT, VIDEO_INPUT).

        Returns:
            list[HardwareDevice]: Standardized hardware device metadata records.
        """

    @abstractmethod
    async def get_device(self, device_id: str) -> HardwareDevice | None:
        """Retrieves a specific hardware device record by its unique ID.

        Args:
            device_id: Unique hardware device identifier.

        Returns:
            HardwareDevice | None: Device record if found, else None.
        """

    @abstractmethod
    async def get_default_device(self, device_type: DeviceType) -> HardwareDevice | None:
        """Returns the system default device for the specified DeviceType.

        Args:
            device_type: Target DeviceType category.

        Returns:
            HardwareDevice | None: Default device record if available, else None.
        """

    @abstractmethod
    async def refresh_devices(self) -> list[HardwareDevice]:
        """Forces a hardware rescan to detect connected or disconnected devices.

        Returns:
            list[HardwareDevice]: Updated list of all enumerated devices.
        """

    @abstractmethod
    async def is_device_available(self, device_id: str) -> bool:
        """Checks if a specific hardware device is currently connected and operational.

        Args:
            device_id: Target device identifier.

        Returns:
            bool: True if device is available, False otherwise.
        """

    @abstractmethod
    async def health(self) -> dict[str, Any]:
        """Probes health and readiness status of the Device Manager subsystem."""


class IHardwareProfileDetector(ABC):
    """Abstract Interface for inspecting host compute resources and hardware acceleration."""

    @abstractmethod
    async def detect_profile(self) -> HardwareProfile:
        """Inspects and returns the comprehensive hardware capability profile of the host system."""

    @abstractmethod
    async def detect_cpu(self) -> CpuProfile:
        """Inspects CPU architecture, core counts, and instruction set extensions (AVX2, AVX-512)."""

    @abstractmethod
    async def detect_memory(self) -> MemoryProfile:
        """Inspects host physical RAM capacities."""

    @abstractmethod
    async def detect_gpus(self) -> list[GpuDeviceInfo]:
        """Inspects installed graphics processors, VRAM capacities, and compute capabilities."""

    @abstractmethod
    async def detect_npu(self) -> NpuProfile:
        """Probes presence of dedicated Neural Processing Units (NPU)."""


class IResourceTelemetryCollector(ABC):
    """Abstract Interface for real-time host resource monitoring and utilization telemetry."""

    @abstractmethod
    async def collect_telemetry(self) -> ResourceTelemetry:
        """Gathers and returns current snapshot of CPU, RAM, and GPU/VRAM utilization."""

    @abstractmethod
    async def health(self) -> dict[str, Any]:
        """Probes operational health of telemetry collector."""
