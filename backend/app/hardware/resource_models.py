"""Hardware-Aware Resource Management Domain Models and Contracts (Phase 4H.7).

Defines strongly typed, framework-agnostic models and enums for WorkloadClass,
WorkloadPriority, WorkloadState, ResourceSnapshot, ResourceBudget, ResourceRequest,
and ResourceAllocation.
"""

from datetime import UTC, datetime
from enum import IntEnum, StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class WorkloadClass(StrEnum):
    """Classification of local computational and perceptual workloads."""

    STT = "STT"
    TTS = "TTS"
    VISION = "VISION"
    PLANNER = "PLANNER"
    SYSTEM = "SYSTEM"


class WorkloadPriority(IntEnum):
    """Priority levels for local workload scheduling and admission control."""

    SYSTEM = 50
    SAFETY_CRITICAL = 40
    INTERACTIVE_VOICE = 30
    INTERACTIVE_VISION = 20
    BACKGROUND_INFERENCE = 10


class WorkloadState(StrEnum):
    """Lifecycle state of a workload in the resource manager."""

    PENDING = "PENDING"
    ADMITTED = "ADMITTED"
    ACTIVE = "ACTIVE"
    REJECTED = "REJECTED"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    TIMED_OUT = "TIMED_OUT"
    FAILED = "FAILED"


class ResourceSnapshot(BaseModel):
    """Instantaneous snapshot of host compute, memory, and accelerator resources."""

    model_config = ConfigDict(extra="forbid")

    # CPU Information
    cpu_logical_cores: int = Field(gt=0, description="Logical CPU execution threads")
    cpu_physical_cores: int = Field(gt=0, description="Physical CPU processor cores")
    cpu_percent: float = Field(ge=0.0, le=100.0, description="Current CPU utilization percentage")
    cpu_architecture: str = Field(description="CPU instruction set architecture (e.g. 'x86_64')")
    cpu_model: str = Field(description="CPU model brand and specification string")
    avx2_supported: bool = Field(default=False, description="True if AVX2 vector extensions supported")

    # RAM Information
    ram_total_mb: float = Field(gt=0.0, description="Total physical host RAM in megabytes")
    ram_used_mb: float = Field(ge=0.0, description="Used physical host RAM in megabytes")
    ram_available_mb: float = Field(ge=0.0, description="Available physical host RAM in megabytes")
    ram_percent: float = Field(ge=0.0, le=100.0, description="RAM utilization percentage")

    # GPU Information
    gpu_available: bool = Field(default=False, description="True if usable GPU accelerator detected")
    gpu_name: str | None = Field(default=None, description="Model identifier of GPU if detected")
    vram_total_mb: float | None = Field(default=None, ge=0.0, description="Total VRAM in megabytes")
    vram_used_mb: float | None = Field(default=None, ge=0.0, description="Used VRAM in megabytes")
    vram_available_mb: float | None = Field(default=None, ge=0.0, description="Free VRAM in megabytes")
    gpu_percent: float | None = Field(default=None, ge=0.0, le=100.0, description="GPU load percentage")
    execution_provider: str = Field(
        default="CPUExecutionProvider", description="Active compute execution provider target"
    )

    # NPU Information
    npu_available: bool = Field(default=False, description="True if dedicated NPU accelerator present")
    npu_vendor: str = Field(default="NONE", description="NPU vendor identifier")
    npu_name: str = Field(default="NONE", description="NPU accelerator model name")

    # Workload Allocation Accounting
    active_workloads_count: int = Field(default=0, ge=0, description="Number of currently active workloads")
    allocated_threads: int = Field(default=0, ge=0, description="Total CPU threads currently allocated")
    allocated_ram_mb: float = Field(default=0.0, ge=0.0, description="Total RAM megabytes currently reserved")
    allocated_vram_mb: float = Field(default=0.0, ge=0.0, description="Total VRAM megabytes currently reserved")

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Snapshot sampling timestamp"
    )

    @field_validator("timestamp")
    @classmethod
    def validate_timezone(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware.")
        return v


class ResourceBudget(BaseModel):
    """Configurable thresholds governing admission control and over-allocation limits."""

    model_config = ConfigDict(extra="forbid")

    max_cpu_percent: float = Field(
        default=90.0, ge=10.0, le=100.0, description="Maximum total CPU utilization ceiling"
    )
    min_available_ram_mb: float = Field(
        default=256.0, ge=64.0, description="Minimum free RAM floor before rejecting workloads"
    )
    max_gpu_memory_percent: float = Field(
        default=90.0, ge=10.0, le=100.0, description="Maximum VRAM utilization ceiling"
    )
    max_concurrent_inference_jobs: int = Field(
        default=8, ge=1, le=32, description="Maximum concurrent inference workloads allowed"
    )
    max_threads_per_workload: int = Field(
        default=8, ge=1, le=32, description="Maximum CPU threads allowed per single workload request"
    )
    max_ram_per_workload_mb: float = Field(
        default=2048.0, ge=64.0, description="Maximum RAM allowed per single workload request"
    )
    default_workload_timeout_sec: float = Field(
        default=30.0, gt=0.0, le=300.0, description="Default lease timeout duration in seconds"
    )


class ResourceRequest(BaseModel):
    """Specification of resource requirements for admitting a local workload."""

    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1, description="Unique resource request identifier")
    workload_id: str = Field(min_length=1, description="Unique workload task identifier")
    workload_class: WorkloadClass = Field(description="Category of workload requesting resources")
    priority: WorkloadPriority = Field(
        default=WorkloadPriority.INTERACTIVE_VOICE, description="Scheduling priority level"
    )
    estimated_cpu_threads: int = Field(
        default=1, gt=0, le=16, description="Estimated CPU worker threads required"
    )
    estimated_ram_mb: float = Field(
        default=128.0, gt=0.0, description="Estimated RAM requirement in megabytes"
    )
    estimated_vram_mb: float = Field(
        default=0.0, ge=0.0, description="Estimated VRAM requirement in megabytes"
    )
    timeout_sec: float = Field(
        default=30.0, gt=0.0, le=300.0, description="Maximum duration lease will remain valid"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Creation timestamp"
    )

    @field_validator("timestamp")
    @classmethod
    def validate_timezone(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware.")
        return v


class ResourceAllocation(BaseModel):
    """Guaranteed resource allocation lease granted to an admitted workload."""

    model_config = ConfigDict(extra="forbid")

    allocation_id: str = Field(min_length=1, description="Unique allocation lease tracking ID")
    request_id: str = Field(min_length=1, description="Originating request ID")
    workload_id: str = Field(min_length=1, description="Associated workload ID")
    workload_class: WorkloadClass = Field(description="Workload classification")
    priority: WorkloadPriority = Field(description="Granted scheduling priority")
    granted_threads: int = Field(gt=0, description="Number of CPU worker threads granted")
    granted_ram_mb: float = Field(gt=0.0, description="Reserved RAM in megabytes")
    granted_vram_mb: float = Field(ge=0.0, description="Reserved VRAM in megabytes")
    acquired_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Lease acquisition timestamp"
    )
    expires_at: datetime = Field(description="Lease expiration timestamp")
    is_active: bool = Field(default=True, description="True while allocation is actively held")

    @field_validator("acquired_at", "expires_at")
    @classmethod
    def validate_timezone(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware.")
        return v


# Exception Taxonomy for Resource Management Subsystem
class ResourceManagerError(Exception):
    """Base application exception for all Resource Manager failures."""


class ResourceCapacityExceededError(ResourceManagerError):
    """Raised when host resources are insufficient to admit the requested workload."""


class ResourceRequestValidationError(ResourceManagerError):
    """Raised when ResourceRequest parameters exceed allowed safety bounds."""


class ResourceAllocationNotFoundError(ResourceManagerError):
    """Raised when attempting to release an unknown or expired allocation lease."""


class ResourceTimeoutError(ResourceManagerError):
    """Raised when waiting for resources exceeds the configured timeout."""
