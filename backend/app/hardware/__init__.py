"""Hardware Abstraction Layer (HAL) Subsystem (Phase 4H.1).

Provides device management, hardware profiling, capability detection, and resource telemetry.
"""

from app.hardware.base import (
    IDeviceManager,
    IHardwareProfileDetector,
    IResourceTelemetryCollector,
)
from app.hardware.device_manager import DeviceManager
from app.hardware.models import (
    CpuArchitecture,
    CpuProfile,
    DeviceState,
    DeviceType,
    GpuDeviceInfo,
    GpuVendor,
    HardwareDevice,
    HardwareProfile,
    MemoryProfile,
    NpuProfile,
    ResourceTelemetry,
)
from app.hardware.profile import HardwareProfileDetector
from app.hardware.resource_manager import IResourceManager, ResourceManager
from app.hardware.resource_models import (
    ResourceAllocation,
    ResourceAllocationNotFoundError,
    ResourceBudget,
    ResourceCapacityExceededError,
    ResourceManagerError,
    ResourceRequest,
    ResourceRequestValidationError,
    ResourceSnapshot,
    ResourceTimeoutError,
    WorkloadClass,
    WorkloadPriority,
    WorkloadState,
)
from app.hardware.telemetry import ResourceTelemetryCollector

__all__ = [
    "IDeviceManager",
    "IHardwareProfileDetector",
    "IResourceTelemetryCollector",
    "IResourceManager",
    "DeviceManager",
    "HardwareProfileDetector",
    "ResourceTelemetryCollector",
    "ResourceManager",
    "DeviceType",
    "DeviceState",
    "HardwareDevice",
    "CpuArchitecture",
    "GpuVendor",
    "GpuDeviceInfo",
    "CpuProfile",
    "MemoryProfile",
    "NpuProfile",
    "HardwareProfile",
    "ResourceTelemetry",
    "WorkloadClass",
    "WorkloadPriority",
    "WorkloadState",
    "ResourceSnapshot",
    "ResourceBudget",
    "ResourceRequest",
    "ResourceAllocation",
    "ResourceManagerError",
    "ResourceCapacityExceededError",
    "ResourceRequestValidationError",
    "ResourceAllocationNotFoundError",
    "ResourceTimeoutError",
]
