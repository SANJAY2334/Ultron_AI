"""Hardware-Aware Resource Manager and Workload Coordinator (Phase 4H.7).

Implements IResourceManager providing centralized admission control, resource budgeting,
workload prioritization, lease lifecycle tracking, and graceful degradation across
concurrent local inference workloads (STT, TTS, Vision, Planner).

Architectural Invariants:
- RESOURCE MANAGEMENT != AUTHORIZATION. The Resource Manager only governs hardware capacity;
  it has zero tool execution, shell invocation, capability granting, or security policy authority.
- Ephemeral telemetry: zero persistence of raw audio, video frames, or sensitive user inputs.
- Safe lifecycle: bounded queues, lease expiration recovery, and guaranteed exception cleanup.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from app.hardware.base import IHardwareProfileDetector, IResourceTelemetryCollector
from app.hardware.profile import HardwareProfileDetector
from app.hardware.resource_models import (
    ResourceAllocation,
    ResourceBudget,
    ResourceCapacityExceededError,
    ResourceRequest,
    ResourceRequestValidationError,
    ResourceSnapshot,
    WorkloadClass,
    WorkloadPriority,
)
from app.hardware.telemetry import ResourceTelemetryCollector

logger = logging.getLogger(__name__)


class IResourceManager(ABC):
    """Abstract interface for hardware resource management and workload coordination."""

    @abstractmethod
    async def get_snapshot(self) -> ResourceSnapshot:
        """Gathers an instantaneous snapshot of host hardware resources and active allocations."""

    @abstractmethod
    async def request_resources(self, request: ResourceRequest) -> ResourceAllocation:
        """Requests and reserves hardware resources for a workload."""

    @abstractmethod
    async def release_resources(self, allocation_id: str) -> None:
        """Releases previously allocated hardware resources."""

    @abstractmethod
    def allocate(self, request: ResourceRequest) -> Any:
        """Async context manager for safe resource acquisition and guaranteed cleanup."""

    @abstractmethod
    async def should_throttle(self, workload_class: WorkloadClass) -> bool:
        """Determines whether a lower-priority workload should be throttled or deferred."""

    @abstractmethod
    async def health(self) -> dict[str, Any]:
        """Probes operational health and active workload allocation metrics."""


class ResourceManager(IResourceManager):
    """Production implementation of Hardware-Aware Resource Manager."""

    def __init__(
        self,
        budget: ResourceBudget | None = None,
        profile_detector: IHardwareProfileDetector | None = None,
        telemetry_collector: IResourceTelemetryCollector | None = None,
    ) -> None:
        """Initializes ResourceManager.

        Args:
            budget: Optional ResourceBudget configuring resource ceilings and safety limits.
            profile_detector: Optional hardware capability detector.
            telemetry_collector: Optional real-time resource telemetry collector.
        """
        self.budget = budget or ResourceBudget()
        self.profile_detector = profile_detector or HardwareProfileDetector()
        self.telemetry_collector = telemetry_collector or ResourceTelemetryCollector()

        self._lock = asyncio.Lock()
        self._active_allocations: dict[str, ResourceAllocation] = {}
        self._allocation_history: deque[ResourceAllocation] = deque(maxlen=100)

        self._total_requests = 0
        self._total_admitted = 0
        self._total_rejected = 0
        self._total_released = 0
        self._total_reaped = 0

        logger.info(
            f"ResourceManager initialized: max_cpu={self.budget.max_cpu_percent}%, "
            f"min_ram={self.budget.min_available_ram_mb}MB, "
            f"max_jobs={self.budget.max_concurrent_inference_jobs}."
        )

    def _reap_expired_allocations(self, now: datetime) -> int:
        """Identifies and reclaims abandoned or timed-out allocation leases."""
        expired_ids: list[str] = []
        for alloc_id, alloc in self._active_allocations.items():
            if alloc.expires_at <= now:
                expired_ids.append(alloc_id)

        for alloc_id in expired_ids:
            reaped = self._active_allocations.pop(alloc_id, None)
            if reaped:
                reaped.is_active = False
                self._total_reaped += 1
                logger.warning(
                    f"Reaped expired resource allocation lease '{alloc_id}' "
                    f"for workload '{reaped.workload_id}' ({reaped.workload_class})."
                )

        return len(expired_ids)

    async def get_snapshot(self) -> ResourceSnapshot:
        """Gathers real-time host hardware status combined with active allocation accounting."""
        hw_profile = await self.profile_detector.detect_profile()
        telemetry = await self.telemetry_collector.collect_telemetry()

        now = datetime.now(UTC)
        async with self._lock:
            self._reap_expired_allocations(now)

            active_count = len(self._active_allocations)
            alloc_threads = sum(a.granted_threads for a in self._active_allocations.values())
            alloc_ram = sum(a.granted_ram_mb for a in self._active_allocations.values())
            alloc_vram = sum(a.granted_vram_mb for a in self._active_allocations.values())

        # GPU information resolution
        gpu_avail = len(hw_profile.gpus) > 0 and hw_profile.gpus[0].vram_total_mb > 0
        gpu_name = hw_profile.gpus[0].name if hw_profile.gpus else None
        vram_tot = telemetry.vram_total_mb if gpu_avail else None
        vram_used = telemetry.vram_used_mb if gpu_avail else None
        vram_avail = telemetry.vram_available_mb if gpu_avail else None
        gpu_pct = telemetry.gpu_percent if gpu_avail else None

        exec_provider = "CPUExecutionProvider"
        if gpu_avail and hw_profile.gpus[0].cuda_available:
            exec_provider = "CUDAExecutionProvider"
        elif gpu_avail and hw_profile.gpus[0].directml_available:
            exec_provider = "DmlExecutionProvider"

        return ResourceSnapshot(
            cpu_logical_cores=hw_profile.cpu.logical_cores,
            cpu_physical_cores=hw_profile.cpu.physical_cores,
            cpu_percent=telemetry.cpu_percent,
            cpu_architecture=hw_profile.cpu.architecture.value,
            cpu_model=hw_profile.cpu.model,
            avx2_supported=hw_profile.cpu.avx2_supported,
            ram_total_mb=telemetry.ram_total_mb,
            ram_used_mb=telemetry.ram_used_mb,
            ram_available_mb=telemetry.ram_available_mb,
            ram_percent=telemetry.ram_percent,
            gpu_available=gpu_avail,
            gpu_name=gpu_name,
            vram_total_mb=vram_tot,
            vram_used_mb=vram_used,
            vram_available_mb=vram_avail,
            gpu_percent=gpu_pct,
            execution_provider=exec_provider,
            npu_available=hw_profile.npu.available,
            npu_vendor=hw_profile.npu.vendor,
            npu_name=hw_profile.npu.name,
            active_workloads_count=active_count,
            allocated_threads=alloc_threads,
            allocated_ram_mb=alloc_ram,
            allocated_vram_mb=alloc_vram,
            timestamp=now,
        )

    def _validate_request(self, request: ResourceRequest) -> None:
        """Validates incoming ResourceRequest against budget safety ceilings."""
        if request.estimated_cpu_threads > self.budget.max_threads_per_workload:
            raise ResourceRequestValidationError(
                f"Requested CPU threads ({request.estimated_cpu_threads}) exceeds "
                f"maximum limit ({self.budget.max_threads_per_workload})."
            )

        if request.estimated_ram_mb > self.budget.max_ram_per_workload_mb:
            raise ResourceRequestValidationError(
                f"Requested RAM ({request.estimated_ram_mb} MB) exceeds "
                f"maximum limit ({self.budget.max_ram_per_workload_mb} MB)."
            )

    async def request_resources(self, request: ResourceRequest) -> ResourceAllocation:
        """Evaluates admission capacity and grants a time-bounded resource allocation lease."""
        self._validate_request(request)
        self._total_requests += 1
        now = datetime.now(UTC)

        async with self._lock:
            self._reap_expired_allocations(now)

            # 1. Concurrency Check
            current_jobs = len(self._active_allocations)
            if current_jobs >= self.budget.max_concurrent_inference_jobs:
                # If incoming workload is higher priority than lowest active workload, permit eviction / priority admission
                min_priority_active = min(
                    (a.priority for a in self._active_allocations.values()),
                    default=WorkloadPriority.BACKGROUND_INFERENCE,
                )
                if request.priority <= min_priority_active:
                    self._total_rejected += 1
                    raise ResourceCapacityExceededError(
                        f"Max concurrent inference workloads ({self.budget.max_concurrent_inference_jobs}) reached. "
                        f"Workload priority {request.priority.name} insufficient to preempt active workloads."
                    )

            # 2. Host Telemetry & Resource Budget Validation
            telemetry = await self.telemetry_collector.collect_telemetry()
            hw_profile = await self.profile_detector.detect_profile()

            # Check RAM floor
            projected_available_ram = telemetry.ram_available_mb - request.estimated_ram_mb

            if projected_available_ram < self.budget.min_available_ram_mb:
                self._total_rejected += 1
                raise ResourceCapacityExceededError(
                    f"Insufficient RAM capacity: Host available={telemetry.ram_available_mb:.1f}MB, "
                    f"requested={request.estimated_ram_mb:.1f}MB, "
                    f"budget floor={self.budget.min_available_ram_mb:.1f}MB."
                )

            # Check CPU thread capacity
            current_reserved_threads = sum(a.granted_threads for a in self._active_allocations.values())
            max_permitted_threads = int(hw_profile.cpu.logical_cores * (self.budget.max_cpu_percent / 100.0))

            if current_reserved_threads + request.estimated_cpu_threads > max(1, max_permitted_threads):
                if request.priority < WorkloadPriority.INTERACTIVE_VOICE:
                    self._total_rejected += 1
                    raise ResourceCapacityExceededError(
                        f"CPU thread allocation ceiling reached: reserved={current_reserved_threads}, "
                        f"requested={request.estimated_cpu_threads}, max_permitted={max_permitted_threads}."
                    )

            # 3. Grant Allocation Lease
            timeout_sec = request.timeout_sec or self.budget.default_workload_timeout_sec
            expires_at = now + timedelta(seconds=timeout_sec)

            allocation = ResourceAllocation(
                allocation_id=f"alloc_{uuid4().hex[:12]}",
                request_id=request.request_id,
                workload_id=request.workload_id,
                workload_class=request.workload_class,
                priority=request.priority,
                granted_threads=request.estimated_cpu_threads,
                granted_ram_mb=request.estimated_ram_mb,
                granted_vram_mb=request.estimated_vram_mb,
                acquired_at=now,
                expires_at=expires_at,
                is_active=True,
            )

            self._active_allocations[allocation.allocation_id] = allocation
            self._allocation_history.append(allocation)
            self._total_admitted += 1

            logger.info(
                f"Resource lease granted: alloc='{allocation.allocation_id}', "
                f"workload='{request.workload_id}' ({request.workload_class}), "
                f"threads={allocation.granted_threads}, ram={allocation.granted_ram_mb}MB."
            )

            return allocation

    async def release_resources(self, allocation_id: str) -> None:
        """Releases an active resource allocation lease cleanly and idempotently."""
        async with self._lock:
            alloc = self._active_allocations.pop(allocation_id, None)
            if alloc is not None:
                alloc.is_active = False
                self._total_released += 1
                logger.info(
                    f"Resource lease released cleanly: alloc='{allocation_id}', "
                    f"workload='{alloc.workload_id}'."
                )

    @asynccontextmanager
    async def allocate(self, request: ResourceRequest) -> AsyncIterator[ResourceAllocation]:
        """Asynchronous context manager guaranteeing safe resource acquisition and release."""
        allocation = await self.request_resources(request)
        try:
            yield allocation
        finally:
            await self.release_resources(allocation.allocation_id)

    async def should_throttle(self, workload_class: WorkloadClass) -> bool:
        """Determines if a workload should be throttled to prioritize interactive voice."""
        now = datetime.now(UTC)
        async with self._lock:
            self._reap_expired_allocations(now)

            # Check if interactive voice (STT or TTS) is currently active
            voice_active = any(
                a.workload_class in (WorkloadClass.STT, WorkloadClass.TTS)
                for a in self._active_allocations.values()
            )

        # If interactive voice is active, throttle background vision inference if CPU or thread pressure is high
        if voice_active and workload_class == WorkloadClass.VISION:
            telemetry = await self.telemetry_collector.collect_telemetry()
            if telemetry.cpu_percent > 65.0 or telemetry.ram_percent > 85.0:
                logger.debug(
                    f"Throttling {workload_class} workload to preserve Interactive Voice responsiveness "
                    f"(CPU: {telemetry.cpu_percent}%, RAM: {telemetry.ram_percent}%)."
                )
                return True

        return False

    async def health(self) -> dict[str, Any]:
        """Probes operational health and active allocation telemetry."""
        now = datetime.now(UTC)
        async with self._lock:
            self._reap_expired_allocations(now)
            active_count = len(self._active_allocations)
            alloc_threads = sum(a.granted_threads for a in self._active_allocations.values())
            alloc_ram = sum(a.granted_ram_mb for a in self._active_allocations.values())

        return {
            "subsystem": "resource_manager",
            "healthy": True,
            "active_allocations_count": active_count,
            "allocated_threads": alloc_threads,
            "allocated_ram_mb": alloc_ram,
            "total_requests": self._total_requests,
            "total_admitted": self._total_admitted,
            "total_rejected": self._total_rejected,
            "total_released": self._total_released,
            "total_reaped": self._total_reaped,
            "budget": self.budget.model_dump(),
        }
