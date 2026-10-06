"""Comprehensive Unit & Hardware Integration Tests for ResourceManager (Phase 4H.7).

Validates hardware snapshot gathering, CPU/RAM/GPU capacity accounting, workload admission control,
priority scheduling, interactive voice prioritization, lease expiration reaping, memory safety,
and strict Zero-Trust boundaries (RESOURCE MANAGEMENT != AUTHORIZATION).
"""

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from app.hardware.base import IHardwareProfileDetector, IResourceTelemetryCollector
from app.hardware.models import (
    CpuArchitecture,
    CpuProfile,
    GpuDeviceInfo,
    GpuVendor,
    HardwareProfile,
    MemoryProfile,
    NpuProfile,
    ResourceTelemetry,
)
from app.hardware.resource_manager import IResourceManager, ResourceManager
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


def _make_mock_profile(
    logical_cores: int = 8,
    physical_cores: int = 4,
    total_ram_mb: float = 16384.0,
    has_gpu: bool = False,
) -> HardwareProfile:
    """Helper constructing mock HardwareProfile."""
    gpus: list[GpuDeviceInfo] = []
    if has_gpu:
        gpus.append(
            GpuDeviceInfo(
                gpu_id="gpu_0",
                name="NVIDIA GeForce RTX 4060",
                vendor=GpuVendor.NVIDIA,
                vram_total_mb=8192.0,
                vram_available_mb=6144.0,
                cuda_available=True,
                cuda_compute_capability="8.9",
            )
        )

    return HardwareProfile(
        cpu=CpuProfile(
            model="AMD Ryzen 5 7535HS",
            architecture=CpuArchitecture.X86_64,
            logical_cores=logical_cores,
            physical_cores=physical_cores,
            avx2_supported=True,
            avx512_supported=False,
        ),
        memory=MemoryProfile(
            total_ram_mb=total_ram_mb,
            available_ram_mb=total_ram_mb * 0.6,
        ),
        gpus=gpus,
        npu=NpuProfile(available=False, vendor="NONE", name="NONE"),
        primary_accelerator="CUDA" if has_gpu else "CPU",
    )


def _make_mock_telemetry(
    cpu_pct: float = 25.0,
    ram_used_mb: float = 6000.0,
    ram_avail_mb: float = 10000.0,
    ram_tot_mb: float = 16000.0,
    has_gpu: bool = False,
) -> ResourceTelemetry:
    """Helper constructing mock ResourceTelemetry."""
    return ResourceTelemetry(
        cpu_percent=cpu_pct,
        ram_used_mb=ram_used_mb,
        ram_available_mb=ram_avail_mb,
        ram_total_mb=ram_tot_mb,
        ram_percent=(ram_used_mb / ram_tot_mb) * 100.0,
        gpu_percent=30.0 if has_gpu else None,
        vram_used_mb=2048.0 if has_gpu else None,
        vram_available_mb=6144.0 if has_gpu else None,
        vram_total_mb=8192.0 if has_gpu else None,
        timestamp=datetime.now(UTC),
    )


class TestResourceManagerAutomated:
    """Automated unit tests for ResourceManager."""

    def test_implements_iresource_manager_interface(self) -> None:
        """Verify ResourceManager implements IResourceManager."""
        mgr = ResourceManager()
        assert isinstance(mgr, IResourceManager)

    @pytest.mark.asyncio
    async def test_snapshot_gathering_cpu_ram(self) -> None:
        """Verify snapshot correctly maps CPU and RAM hardware telemetry."""
        mock_prof = AsyncMock(spec=IHardwareProfileDetector)
        mock_prof.detect_profile.return_value = _make_mock_profile()
        mock_telem = AsyncMock(spec=IResourceTelemetryCollector)
        mock_telem.collect_telemetry.return_value = _make_mock_telemetry()

        mgr = ResourceManager(profile_detector=mock_prof, telemetry_collector=mock_telem)
        snap = await mgr.get_snapshot()

        assert isinstance(snap, ResourceSnapshot)
        assert snap.cpu_logical_cores == 8
        assert snap.cpu_physical_cores == 4
        assert snap.ram_total_mb == 16000.0
        assert snap.ram_available_mb == 10000.0
        assert snap.gpu_available is False
        assert snap.execution_provider == "CPUExecutionProvider"
        assert snap.npu_available is False

    @pytest.mark.asyncio
    async def test_snapshot_gpu_handling_when_available(self) -> None:
        """Verify snapshot reports GPU details when accelerator is present."""
        mock_prof = AsyncMock(spec=IHardwareProfileDetector)
        mock_prof.detect_profile.return_value = _make_mock_profile(has_gpu=True)
        mock_telem = AsyncMock(spec=IResourceTelemetryCollector)
        mock_telem.collect_telemetry.return_value = _make_mock_telemetry(has_gpu=True)

        mgr = ResourceManager(profile_detector=mock_prof, telemetry_collector=mock_telem)
        snap = await mgr.get_snapshot()

        assert snap.gpu_available is True
        assert snap.gpu_name == "NVIDIA GeForce RTX 4060"
        assert snap.vram_total_mb == 8192.0
        assert snap.execution_provider == "CUDAExecutionProvider"

    def test_resource_budget_validation(self) -> None:
        """Verify ResourceBudget bounds validation."""
        b = ResourceBudget(max_cpu_percent=85.0, min_available_ram_mb=512.0)
        assert b.max_cpu_percent == 85.0

        with pytest.raises(ValueError):
            ResourceBudget(max_cpu_percent=150.0)

        with pytest.raises(ValueError):
            ResourceBudget(min_available_ram_mb=10.0)

    @pytest.mark.asyncio
    async def test_resource_request_validation(self) -> None:
        """Verify request exceeding budget bounds raises ResourceRequestValidationError."""
        mgr = ResourceManager(budget=ResourceBudget(max_threads_per_workload=4, max_ram_per_workload_mb=1024.0))

        bad_req_threads = ResourceRequest(
            request_id="req_threads",
            workload_id="task_1",
            workload_class=WorkloadClass.STT,
            estimated_cpu_threads=8,  # exceeds 4
            estimated_ram_mb=256.0,
        )
        with pytest.raises(ResourceRequestValidationError, match="CPU threads"):
            await mgr.request_resources(bad_req_threads)

        bad_req_ram = ResourceRequest(
            request_id="req_ram",
            workload_id="task_2",
            workload_class=WorkloadClass.VISION,
            estimated_cpu_threads=2,
            estimated_ram_mb=4096.0,  # exceeds 1024
        )
        with pytest.raises(ResourceRequestValidationError, match="RAM"):
            await mgr.request_resources(bad_req_ram)

    @pytest.mark.asyncio
    async def test_resource_allocation_grant_and_accounting(self) -> None:
        """Verify valid request receives active allocation and updates internal accounting."""
        mock_prof = AsyncMock(spec=IHardwareProfileDetector)
        mock_prof.detect_profile.return_value = _make_mock_profile(logical_cores=8)
        mock_telem = AsyncMock(spec=IResourceTelemetryCollector)
        mock_telem.collect_telemetry.return_value = _make_mock_telemetry(ram_avail_mb=8000.0)

        mgr = ResourceManager(profile_detector=mock_prof, telemetry_collector=mock_telem)

        req = ResourceRequest(
            request_id="req_stt_01",
            workload_id="whisper_utterance_1",
            workload_class=WorkloadClass.STT,
            priority=WorkloadPriority.INTERACTIVE_VOICE,
            estimated_cpu_threads=2,
            estimated_ram_mb=512.0,
            timeout_sec=10.0,
        )

        alloc = await mgr.request_resources(req)
        assert isinstance(alloc, ResourceAllocation)
        assert alloc.is_active is True
        assert alloc.granted_threads == 2
        assert alloc.granted_ram_mb == 512.0
        assert alloc.allocation_id in mgr._active_allocations

        # Verify snapshot reflects accounting
        snap = await mgr.get_snapshot()
        assert snap.active_workloads_count == 1
        assert snap.allocated_threads == 2
        assert snap.allocated_ram_mb == 512.0

        # Release resources
        await mgr.release_resources(alloc.allocation_id)
        assert alloc.allocation_id not in mgr._active_allocations

        snap2 = await mgr.get_snapshot()
        assert snap2.active_workloads_count == 0
        assert snap2.allocated_threads == 0
        assert snap2.allocated_ram_mb == 0.0

    @pytest.mark.asyncio
    async def test_idempotent_duplicate_release_safety(self) -> None:
        """Verify releasing an already released allocation is safe and does not underflow counters."""
        mock_prof = AsyncMock(spec=IHardwareProfileDetector)
        mock_prof.detect_profile.return_value = _make_mock_profile()
        mock_telem = AsyncMock(spec=IResourceTelemetryCollector)
        mock_telem.collect_telemetry.return_value = _make_mock_telemetry()

        mgr = ResourceManager(profile_detector=mock_prof, telemetry_collector=mock_telem)
        req = ResourceRequest(
            request_id="req_dup",
            workload_id="workload_dup",
            workload_class=WorkloadClass.TTS,
            estimated_cpu_threads=1,
            estimated_ram_mb=128.0,
        )
        alloc = await mgr.request_resources(req)
        await mgr.release_resources(alloc.allocation_id)
        # Duplicate release
        await mgr.release_resources(alloc.allocation_id)
        await mgr.release_resources("non_existent_alloc_xyz")

        snap = await mgr.get_snapshot()
        assert snap.active_workloads_count == 0
        assert snap.allocated_threads == 0
        assert snap.allocated_ram_mb == 0.0

    @pytest.mark.asyncio
    async def test_insufficient_ram_rejection(self) -> None:
        """Verify request rejected when projected available RAM breaches floor."""
        mock_prof = AsyncMock(spec=IHardwareProfileDetector)
        mock_prof.detect_profile.return_value = _make_mock_profile()
        mock_telem = AsyncMock(spec=IResourceTelemetryCollector)
        # Only 300MB available RAM, floor is 256MB
        mock_telem.collect_telemetry.return_value = _make_mock_telemetry(ram_avail_mb=300.0)

        mgr = ResourceManager(
            budget=ResourceBudget(min_available_ram_mb=256.0),
            profile_detector=mock_prof,
            telemetry_collector=mock_telem,
        )

        req = ResourceRequest(
            request_id="req_big_ram",
            workload_id="heavy_task",
            workload_class=WorkloadClass.PLANNER,
            estimated_ram_mb=100.0,  # 300 - 100 = 200MB < 256MB floor!
        )
        with pytest.raises(ResourceCapacityExceededError, match="Insufficient RAM"):
            await mgr.request_resources(req)

    @pytest.mark.asyncio
    async def test_concurrency_ceiling_enforced(self) -> None:
        """Verify max_concurrent_inference_jobs ceiling rejects excess workloads."""
        mock_prof = AsyncMock(spec=IHardwareProfileDetector)
        mock_prof.detect_profile.return_value = _make_mock_profile()
        mock_telem = AsyncMock(spec=IResourceTelemetryCollector)
        mock_telem.collect_telemetry.return_value = _make_mock_telemetry(ram_avail_mb=8000.0)

        mgr = ResourceManager(
            budget=ResourceBudget(max_concurrent_inference_jobs=2),
            profile_detector=mock_prof,
            telemetry_collector=mock_telem,
        )

        r1 = ResourceRequest(
            request_id="r1", workload_id="w1", workload_class=WorkloadClass.STT,
            priority=WorkloadPriority.INTERACTIVE_VOICE,
        )
        r2 = ResourceRequest(
            request_id="r2", workload_id="w2", workload_class=WorkloadClass.TTS,
            priority=WorkloadPriority.INTERACTIVE_VOICE,
        )
        r3 = ResourceRequest(
            request_id="r3", workload_id="w3", workload_class=WorkloadClass.VISION,
            priority=WorkloadPriority.BACKGROUND_INFERENCE,
        )

        await mgr.request_resources(r1)
        await mgr.request_resources(r2)

        # 3rd request should be rejected due to capacity
        with pytest.raises(ResourceCapacityExceededError, match="Max concurrent"):
            await mgr.request_resources(r3)

    @pytest.mark.asyncio
    async def test_lease_expiration_reaping(self) -> None:
        """Verify expired leases are reaped and resources reclaimed."""
        mock_prof = AsyncMock(spec=IHardwareProfileDetector)
        mock_prof.detect_profile.return_value = _make_mock_profile()
        mock_telem = AsyncMock(spec=IResourceTelemetryCollector)
        mock_telem.collect_telemetry.return_value = _make_mock_telemetry()

        mgr = ResourceManager(profile_detector=mock_prof, telemetry_collector=mock_telem)

        req = ResourceRequest(
            request_id="req_fast_exp",
            workload_id="abandoned_task",
            workload_class=WorkloadClass.VISION,
            estimated_cpu_threads=2,
            estimated_ram_mb=256.0,
            timeout_sec=0.01,  # 10ms lease
        )

        alloc = await mgr.request_resources(req)
        assert len(mgr._active_allocations) == 1

        # Wait for lease to expire
        await asyncio.sleep(0.05)

        # Trigger check via get_snapshot
        snap = await mgr.get_snapshot()
        assert snap.active_workloads_count == 0
        assert snap.allocated_threads == 0
        assert alloc.allocation_id not in mgr._active_allocations
        assert mgr._total_reaped == 1

    @pytest.mark.asyncio
    async def test_context_manager_cleanup_on_success(self) -> None:
        """Verify allocate context manager acquires and releases cleanly."""
        mock_prof = AsyncMock(spec=IHardwareProfileDetector)
        mock_prof.detect_profile.return_value = _make_mock_profile()
        mock_telem = AsyncMock(spec=IResourceTelemetryCollector)
        mock_telem.collect_telemetry.return_value = _make_mock_telemetry()

        mgr = ResourceManager(profile_detector=mock_prof, telemetry_collector=mock_telem)
        req = ResourceRequest(
            request_id="req_cm_ok",
            workload_id="workload_cm",
            workload_class=WorkloadClass.STT,
        )

        async with mgr.allocate(req) as alloc:
            assert alloc.is_active is True
            assert len(mgr._active_allocations) == 1

        assert len(mgr._active_allocations) == 0

    @pytest.mark.asyncio
    async def test_context_manager_cleanup_on_exception(self) -> None:
        """Verify allocate context manager unconditionally releases when exception is raised."""
        mock_prof = AsyncMock(spec=IHardwareProfileDetector)
        mock_prof.detect_profile.return_value = _make_mock_profile()
        mock_telem = AsyncMock(spec=IResourceTelemetryCollector)
        mock_telem.collect_telemetry.return_value = _make_mock_telemetry()

        mgr = ResourceManager(profile_detector=mock_prof, telemetry_collector=mock_telem)
        req = ResourceRequest(
            request_id="req_cm_exc",
            workload_id="failing_workload",
            workload_class=WorkloadClass.VISION,
        )

        with pytest.raises(RuntimeError, match="Crash during inference"):
            async with mgr.allocate(req):
                assert len(mgr._active_allocations) == 1
                raise RuntimeError("Crash during inference")

        assert len(mgr._active_allocations) == 0

    @pytest.mark.asyncio
    async def test_context_manager_cleanup_on_cancellation(self) -> None:
        """Verify allocate context manager releases when cancelled."""
        mock_prof = AsyncMock(spec=IHardwareProfileDetector)
        mock_prof.detect_profile.return_value = _make_mock_profile()
        mock_telem = AsyncMock(spec=IResourceTelemetryCollector)
        mock_telem.collect_telemetry.return_value = _make_mock_telemetry()

        mgr = ResourceManager(profile_detector=mock_prof, telemetry_collector=mock_telem)
        req = ResourceRequest(
            request_id="req_cm_cancel",
            workload_id="cancelled_workload",
            workload_class=WorkloadClass.TTS,
        )

        async def _workload_task() -> None:
            async with mgr.allocate(req):
                assert len(mgr._active_allocations) == 1
                await asyncio.sleep(5.0)

        task = asyncio.create_task(_workload_task())
        await asyncio.sleep(0.02)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert len(mgr._active_allocations) == 0

    @pytest.mark.asyncio
    async def test_interactive_voice_prioritization_and_vision_throttling(self) -> None:
        """Verify should_throttle advises throttling background vision when interactive voice is active and load is high."""
        mock_prof = AsyncMock(spec=IHardwareProfileDetector)
        mock_prof.detect_profile.return_value = _make_mock_profile()
        mock_telem = AsyncMock(spec=IResourceTelemetryCollector)
        # Elevated CPU load (75%)
        mock_telem.collect_telemetry.return_value = _make_mock_telemetry(cpu_pct=75.0)

        mgr = ResourceManager(profile_detector=mock_prof, telemetry_collector=mock_telem)

        # Initially without voice active: should not throttle vision
        assert await mgr.should_throttle(WorkloadClass.VISION) is False

        # Now admit interactive voice workload (STT)
        voice_req = ResourceRequest(
            request_id="req_v1",
            workload_id="active_user_speech",
            workload_class=WorkloadClass.STT,
            priority=WorkloadPriority.INTERACTIVE_VOICE,
        )
        voice_alloc = await mgr.request_resources(voice_req)

        # Vision should now be advised to throttle
        assert await mgr.should_throttle(WorkloadClass.VISION) is True

        # When voice finishes:
        await mgr.release_resources(voice_alloc.allocation_id)
        assert await mgr.should_throttle(WorkloadClass.VISION) is False

    def test_security_invariant_resource_management_is_not_authorization(self) -> None:
        """Verify ResourceManager contains zero execution, capability, or shell APIs."""
        forbidden = [
            "execute_tool",
            "run_command",
            "grant_capability",
            "authorize",
            "bypass_policy",
            "bypass_safety",
            "execute_shell",
            "send_ipc",
        ]
        for f in forbidden:
            assert not hasattr(ResourceManager, f)

    def test_privacy_invariant_no_raw_payloads_in_snapshots(self) -> None:
        """Verify ResourceSnapshot contains no audio or video frame payloads."""
        forbidden = [
            "audio_payload",
            "pcm_bytes",
            "frame_payload",
            "pixel_bytes",
            "transcript",
            "credential",
        ]
        for f in forbidden:
            assert not hasattr(ResourceSnapshot, f)


class TestPhysicalHardwareResourceVerification:
    """Physical hardware integration test querying live Windows host metrics."""

    @pytest.mark.asyncio
    async def test_physical_hardware_resource_snapshot_and_coordination(self) -> None:
        """Collects real hardware snapshot on Windows host and executes concurrent workload leases."""
        mgr = ResourceManager()
        snap = await mgr.get_snapshot()

        assert isinstance(snap, ResourceSnapshot)
        assert snap.cpu_logical_cores > 0
        assert snap.ram_total_mb > 1024.0
        assert snap.ram_available_mb > 0.0
        assert snap.execution_provider in ("CPUExecutionProvider", "CUDAExecutionProvider", "DmlExecutionProvider")

        # Simulate concurrent STT + TTS + Vision workload admission on host
        req_stt = ResourceRequest(
            request_id="hw_req_stt",
            workload_id="realtek_mic_stt",
            workload_class=WorkloadClass.STT,
            priority=WorkloadPriority.INTERACTIVE_VOICE,
            estimated_cpu_threads=2,
            estimated_ram_mb=256.0,
        )
        req_vision = ResourceRequest(
            request_id="hw_req_vision",
            workload_id="webcam_yolo",
            workload_class=WorkloadClass.VISION,
            priority=WorkloadPriority.INTERACTIVE_VISION,
            estimated_cpu_threads=2,
            estimated_ram_mb=256.0,
        )
        req_tts = ResourceRequest(
            request_id="hw_req_tts",
            workload_id="piper_speech",
            workload_class=WorkloadClass.TTS,
            priority=WorkloadPriority.INTERACTIVE_VOICE,
            estimated_cpu_threads=2,
            estimated_ram_mb=128.0,
        )

        async with mgr.allocate(req_stt) as a_stt:
            assert a_stt.is_active is True
            async with mgr.allocate(req_vision) as a_vis:
                assert a_vis.is_active is True
                async with mgr.allocate(req_tts) as a_tts:
                    assert a_tts.is_active is True

                    h = await mgr.health()
                    assert h["active_allocations_count"] == 3
                    assert h["allocated_threads"] == 6
                    assert h["allocated_ram_mb"] == 640.0

        # All released
        h_end = await mgr.health()
        assert h_end["active_allocations_count"] == 0
        assert h_end["allocated_threads"] == 0
        assert h_end["allocated_ram_mb"] == 0.0
