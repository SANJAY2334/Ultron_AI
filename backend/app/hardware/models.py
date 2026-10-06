"""Hardware Domain Models and Telemetry Schemas (Phase 4H.1).

Defines strongly typed, framework-agnostic Pydantic v2 domain models for
HardwareDevice, CpuProfile, GpuDeviceInfo, MemoryProfile, NpuProfile,
HardwareProfile, and ResourceTelemetry.
"""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DeviceType(StrEnum):
    """Classification of hardware peripheral devices."""

    AUDIO_INPUT = "AUDIO_INPUT"
    AUDIO_OUTPUT = "AUDIO_OUTPUT"
    VIDEO_INPUT = "VIDEO_INPUT"


class DeviceState(StrEnum):
    """Operational lifecycle state of a hardware device."""

    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"
    ACTIVE = "ACTIVE"
    DISCONNECTED = "DISCONNECTED"
    ERROR = "ERROR"


class HardwareDevice(BaseModel):
    """Standardized metadata representation of an enumerated hardware device."""

    model_config = ConfigDict(extra="forbid")

    device_id: str = Field(min_length=1, description="Unique, stable hardware device identifier")
    name: str = Field(min_length=1, description="Human-readable device name")
    device_type: DeviceType = Field(description="Device type category discriminator")
    state: DeviceState = Field(default=DeviceState.AVAILABLE, description="Current device state")
    is_default: bool = Field(default=False, description="True if device is the system default")
    sample_rates: list[int] = Field(
        default_factory=list, description="Supported audio sampling rates in Hz (for audio devices)"
    )
    channels: int = Field(
        default=1, ge=0, description="Number of supported channels (audio) or video streams"
    )
    capabilities: dict[str, Any] = Field(
        default_factory=dict, description="Device-specific capability flags"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Sanitized hardware metadata (vendor, driver, backend)"
    )


class CpuArchitecture(StrEnum):
    """CPU instruction set architecture classifications."""

    X86_64 = "x86_64"
    ARM64 = "arm64"
    X86 = "x86"
    ARM = "arm"
    UNKNOWN = "unknown"


class GpuVendor(StrEnum):
    """GPU hardware vendor classifications."""

    NVIDIA = "NVIDIA"
    AMD = "AMD"
    INTEL = "INTEL"
    APPLE = "APPLE"
    UNKNOWN = "UNKNOWN"
    NONE = "NONE"


class GpuDeviceInfo(BaseModel):
    """Metadata and capability record for a detected graphics processing unit."""

    model_config = ConfigDict(extra="forbid")

    gpu_id: str = Field(min_length=1, description="Unique GPU device identifier index")
    name: str = Field(description="GPU hardware model description")
    vendor: GpuVendor = Field(default=GpuVendor.UNKNOWN, description="GPU hardware vendor")
    vram_total_mb: float = Field(ge=0.0, description="Total dedicated video memory in megabytes")
    vram_available_mb: float = Field(
        ge=0.0, description="Available dedicated video memory in megabytes"
    )
    cuda_available: bool = Field(
        default=False, description="True if NVIDIA CUDA runtime is available"
    )
    cuda_compute_capability: str | None = Field(
        default=None, description="CUDA compute capability major.minor string (e.g. '8.9')"
    )
    directml_available: bool = Field(
        default=False, description="True if Microsoft DirectML acceleration is supported"
    )
    driver_version: str | None = Field(default=None, description="Installed GPU driver version")


class CpuProfile(BaseModel):
    """Processor capabilities and architecture profile."""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(description="CPU model brand and specification string")
    architecture: CpuArchitecture = Field(
        default=CpuArchitecture.UNKNOWN, description="Instruction set architecture"
    )
    logical_cores: int = Field(gt=0, description="Count of logical processor threads")
    physical_cores: int = Field(gt=0, description="Count of physical processor cores")
    avx2_supported: bool = Field(
        default=False, description="True if Advanced Vector Extensions 2 (AVX2) is supported"
    )
    avx512_supported: bool = Field(
        default=False, description="True if AVX-512 vector extensions are supported"
    )


class MemoryProfile(BaseModel):
    """System host RAM memory characteristics."""

    model_config = ConfigDict(extra="forbid")

    total_ram_mb: float = Field(gt=0.0, description="Total physical system RAM in megabytes")
    available_ram_mb: float = Field(ge=0.0, description="Available physical system RAM in megabytes")


class NpuProfile(BaseModel):
    """Neural Processing Unit (NPU) capability profile."""

    model_config = ConfigDict(extra="forbid")

    available: bool = Field(
        default=False, description="True if a dedicated NPU accelerator is detected"
    )
    vendor: str = Field(default="NONE", description="NPU hardware vendor (e.g. Intel, Qualcomm)")
    name: str = Field(default="NONE", description="NPU accelerator model identifier")
    details: dict[str, Any] = Field(
        default_factory=dict, description="Sanitized NPU driver/driver metadata"
    )


class HardwareProfile(BaseModel):
    """Comprehensive host system hardware capability profile."""

    model_config = ConfigDict(extra="forbid")

    cpu: CpuProfile = Field(description="Host CPU capabilities profile")
    memory: MemoryProfile = Field(description="Host RAM memory profile")
    gpus: list[GpuDeviceInfo] = Field(
        default_factory=list, description="List of detected GPU devices"
    )
    npu: NpuProfile = Field(
        default_factory=lambda: NpuProfile(available=False, vendor="NONE", name="NONE"),
        description="NPU accelerator capability",
    )
    primary_accelerator: str = Field(
        default="CPU", description="Primary compute execution target (e.g. 'CUDA', 'DIRECTML', 'CPU')"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Detection timestamp"
    )

    @field_validator("timestamp")
    @classmethod
    def validate_timezone(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware.")
        return v


class ResourceTelemetry(BaseModel):
    """Instantaneous system resource utilization metrics."""

    model_config = ConfigDict(extra="forbid")

    cpu_percent: float = Field(ge=0.0, le=100.0, description="Current CPU utilization percentage")
    ram_used_mb: float = Field(ge=0.0, description="Used physical RAM in megabytes")
    ram_available_mb: float = Field(ge=0.0, description="Available physical RAM in megabytes")
    ram_total_mb: float = Field(gt=0.0, description="Total physical RAM in megabytes")
    ram_percent: float = Field(ge=0.0, le=100.0, description="RAM utilization percentage")
    gpu_percent: float | None = Field(
        default=None, ge=0.0, le=100.0, description="GPU compute utilization percentage if available"
    )
    vram_used_mb: float | None = Field(
        default=None, ge=0.0, description="Used VRAM in megabytes if available"
    )
    vram_available_mb: float | None = Field(
        default=None, ge=0.0, description="Available VRAM in megabytes if available"
    )
    vram_total_mb: float | None = Field(
        default=None, ge=0.0, description="Total VRAM in megabytes if available"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Telemetry sample timestamp"
    )

    @field_validator("timestamp")
    @classmethod
    def validate_timezone(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware.")
        return v
