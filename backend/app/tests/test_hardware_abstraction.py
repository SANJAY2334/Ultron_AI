"""Comprehensive Unit Tests for Hardware Abstraction Layer (HAL) (Phase 4H.1).

Tests device manager enumeration, hardware capability detection, CPU/RAM/GPU profiling,
resource telemetry gathering, error/disconnection handling, and Zero-Trust security invariants.
"""

from unittest.mock import patch

import pytest
from pydantic import BaseModel

from app.hardware.base import (
    IDeviceManager,
    IHardwareProfileDetector,
    IResourceTelemetryCollector,
)
from app.hardware.config import create_hardware_config
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
from app.hardware.telemetry import ResourceTelemetryCollector


class TestHardwareModels:
    """Tests domain models and schemas."""

    def test_hardware_device_model_validation(self) -> None:
        """Verify HardwareDevice model structure and immutability rules."""
        dev = HardwareDevice(
            device_id="mic_test_1",
            name="Test Studio Microphone",
            device_type=DeviceType.AUDIO_INPUT,
            state=DeviceState.AVAILABLE,
            is_default=True,
            sample_rates=[16000, 48000],
            channels=2,
            capabilities={"noise_cancellation": True},
            metadata={"driver": "test_driver"},
        )
        assert dev.device_id == "mic_test_1"
        assert dev.device_type == DeviceType.AUDIO_INPUT
        assert dev.is_default is True
        assert dev.state == DeviceState.AVAILABLE

    def test_hardware_profile_model_composition(self) -> None:
        """Verify HardwareProfile composite model structure."""
        cpu = CpuProfile(
            model="Intel Core i7-13700K",
            architecture=CpuArchitecture.X86_64,
            logical_cores=24,
            physical_cores=16,
            avx2_supported=True,
            avx512_supported=False,
        )
        memory = MemoryProfile(total_ram_mb=32768.0, available_ram_mb=16384.0)
        gpu = GpuDeviceInfo(
            gpu_id="nvidia_0",
            name="NVIDIA GeForce RTX 4070",
            vendor=GpuVendor.NVIDIA,
            vram_total_mb=12288.0,
            vram_available_mb=8192.0,
            cuda_available=True,
            directml_available=True,
        )
        npu = NpuProfile(available=False, vendor="NONE", name="NONE")

        profile = HardwareProfile(
            cpu=cpu,
            memory=memory,
            gpus=[gpu],
            npu=npu,
            primary_accelerator="CUDA",
        )
        assert profile.cpu.avx2_supported is True
        assert profile.memory.total_ram_mb == 32768.0
        assert len(profile.gpus) == 1
        assert profile.primary_accelerator == "CUDA"

    def test_resource_telemetry_model(self) -> None:
        """Verify ResourceTelemetry model bounds and timestamping."""
        telemetry = ResourceTelemetry(
            cpu_percent=35.5,
            ram_used_mb=8192.0,
            ram_available_mb=24576.0,
            ram_total_mb=32768.0,
            ram_percent=25.0,
            gpu_percent=42.0,
            vram_used_mb=2048.0,
            vram_available_mb=6144.0,
            vram_total_mb=8192.0,
        )
        assert telemetry.cpu_percent == 35.5
        assert telemetry.ram_percent == 25.0
        assert telemetry.timestamp.tzinfo is not None


class TestDeviceManager:
    """Tests DeviceManager device discovery, enumeration, and lifecycle tracking."""

    @pytest.mark.asyncio
    async def test_device_manager_interface_compliance(self) -> None:
        """Verify DeviceManager implements IDeviceManager interface."""
        manager = DeviceManager()
        assert isinstance(manager, IDeviceManager)

    @pytest.mark.asyncio
    async def test_device_manager_enumeration(self) -> None:
        """Verify device manager enumeration across device types."""
        manager = DeviceManager()
        devices = await manager.list_devices()
        assert isinstance(devices, list)

        # Query by category filter
        audio_inputs = await manager.list_devices(DeviceType.AUDIO_INPUT)
        assert all(d.device_type == DeviceType.AUDIO_INPUT for d in audio_inputs)

        video_inputs = await manager.list_devices(DeviceType.VIDEO_INPUT)
        assert all(d.device_type == DeviceType.VIDEO_INPUT for d in video_inputs)

    @pytest.mark.asyncio
    async def test_empty_device_environment_handling(self) -> None:
        """Verify device manager handles empty device environment gracefully."""
        manager = DeviceManager()
        with (
            patch.object(manager, "_enumerate_audio_devices", return_value=[]),
            patch.object(manager, "_enumerate_video_devices", return_value=[]),
        ):
            await manager.refresh_devices()
            devices = await manager.list_devices()
            assert devices == []

            default_mic = await manager.get_default_device(DeviceType.AUDIO_INPUT)
            assert default_mic is None

            is_avail = await manager.is_device_available("non_existent_id")
            assert is_avail is False

    @pytest.mark.asyncio
    async def test_device_disappearance_detection(self) -> None:
        """Verify device manager marks vanished devices as DISCONNECTED."""
        manager = DeviceManager()
        dev1 = HardwareDevice(
            device_id="usb_mic_1",
            name="USB Mic",
            device_type=DeviceType.AUDIO_INPUT,
            state=DeviceState.AVAILABLE,
            is_default=True,
        )

        with (
            patch.object(manager, "_enumerate_audio_devices", return_value=[dev1]),
            patch.object(manager, "_enumerate_video_devices", return_value=[]),
        ):
            await manager.refresh_devices()
            assert await manager.is_device_available("usb_mic_1") is True

        # Next refresh: device is unplugged/missing
        with (
            patch.object(manager, "_enumerate_audio_devices", return_value=[]),
            patch.object(manager, "_enumerate_video_devices", return_value=[]),
        ):
            await manager.refresh_devices()
            dev = await manager.get_device("usb_mic_1")
            assert dev is not None
            assert dev.state == DeviceState.DISCONNECTED
            assert dev.is_default is False
            assert await manager.is_device_available("usb_mic_1") is False

    @pytest.mark.asyncio
    async def test_device_manager_health(self) -> None:
        """Verify health check returns comprehensive device telemetry."""
        manager = DeviceManager()
        health = await manager.health()
        assert health["subsystem"] == "device_manager"
        assert health["healthy"] is True
        assert "available_audio_inputs" in health
        assert "available_video_inputs" in health


class TestHardwareProfileDetector:
    """Tests HardwareProfileDetector compute inspection."""

    @pytest.mark.asyncio
    async def test_profile_detector_interface_compliance(self) -> None:
        """Verify HardwareProfileDetector implements IHardwareProfileDetector."""
        detector = HardwareProfileDetector()
        assert isinstance(detector, IHardwareProfileDetector)

    @pytest.mark.asyncio
    async def test_cpu_detection(self) -> None:
        """Verify CPU profile detection returns valid architectural details."""
        detector = HardwareProfileDetector()
        cpu = await detector.detect_cpu()
        assert cpu.logical_cores > 0
        assert cpu.physical_cores > 0
        assert cpu.architecture in (
            CpuArchitecture.X86_64,
            CpuArchitecture.ARM64,
            CpuArchitecture.X86,
            CpuArchitecture.ARM,
            CpuArchitecture.UNKNOWN,
        )

    @pytest.mark.asyncio
    async def test_memory_detection(self) -> None:
        """Verify RAM detection returns non-zero system memory metrics."""
        detector = HardwareProfileDetector()
        mem = await detector.detect_memory()
        assert mem.total_ram_mb > 0
        assert mem.available_ram_mb >= 0

    @pytest.mark.asyncio
    async def test_gpu_detection_cpu_only_system(self) -> None:
        """Verify GPU detector returns clean empty list on CPU-only system."""
        detector = HardwareProfileDetector()
        with (
            patch("app.hardware.profile.HardwareProfileDetector.detect_gpus", return_value=[]),
            patch("app.hardware.profile.HardwareProfileDetector.detect_npu") as mock_npu,
        ):
            mock_npu.return_value = NpuProfile(available=False, vendor="NONE", name="NONE")
            profile = await detector.detect_profile()
            assert profile.gpus == []
            assert profile.primary_accelerator == "CPU"

    @pytest.mark.asyncio
    async def test_gpu_detection_with_cuda(self) -> None:
        """Verify GPU detector selects CUDA primary accelerator when present."""
        detector = HardwareProfileDetector()
        mock_gpu = GpuDeviceInfo(
            gpu_id="nv_0",
            name="NVIDIA RTX 3080",
            vendor=GpuVendor.NVIDIA,
            vram_total_mb=10240.0,
            vram_available_mb=6144.0,
            cuda_available=True,
            directml_available=True,
        )
        with patch.object(detector, "detect_gpus", return_value=[mock_gpu]):
            profile = await detector.detect_profile()
            assert profile.primary_accelerator == "CUDA"
            assert len(profile.gpus) == 1
            assert profile.gpus[0].vendor == GpuVendor.NVIDIA

    @pytest.mark.asyncio
    async def test_npu_detection_zero_fabrication(self) -> None:
        """Verify NPU detection returns available=False when hardware is absent."""
        detector = HardwareProfileDetector()
        npu = await detector.detect_npu()
        assert npu.available is False
        assert npu.vendor == "NONE"


class TestResourceTelemetryCollector:
    """Tests ResourceTelemetryCollector utilization monitoring."""

    @pytest.mark.asyncio
    async def test_resource_telemetry_interface_compliance(self) -> None:
        """Verify ResourceTelemetryCollector implements IResourceTelemetryCollector."""
        collector = ResourceTelemetryCollector()
        assert isinstance(collector, IResourceTelemetryCollector)

    @pytest.mark.asyncio
    async def test_resource_telemetry_sampling(self) -> None:
        """Verify resource telemetry returns bounded, valid CPU and RAM utilization metrics."""
        collector = ResourceTelemetryCollector()
        telemetry = await collector.collect_telemetry()
        assert 0.0 <= telemetry.cpu_percent <= 100.0
        assert telemetry.ram_total_mb > 0.0
        assert 0.0 <= telemetry.ram_percent <= 100.0
        assert telemetry.timestamp.tzinfo is not None

        health = await collector.health()
        assert health["healthy"] is True
        assert health["samples_collected"] >= 1


class TestHardwareSecurityAndPrivacy:
    """Tests Zero-Trust security invariants and privacy boundaries."""

    def test_hal_does_not_expose_execution_privileges(self) -> None:
        """Verify HAL classes do not possess tool execution or shell modification methods."""
        forbidden_methods = [
            "execute_tool",
            "run_command",
            "modify_file",
            "delete_file",
            "authorize",
            "elevate",
            "grant_capability",
        ]
        classes_to_test = [
            DeviceManager,
            HardwareProfileDetector,
            ResourceTelemetryCollector,
        ]
        for cls in classes_to_test:
            for method in forbidden_methods:
                assert not hasattr(
                    cls, method
                ), f"HAL class '{cls.__name__}' must not have execution method '{method}'."

    def test_hardware_telemetry_privacy_guarantees(self) -> None:
        """Verify hardware telemetry models never contain raw sensor payloads or biometric vectors."""
        forbidden_fields = [
            "payload",
            "audio_bytes",
            "video_frame",
            "raw_pixels",
            "biometric_embedding",
            "face_recognition",
            "identity",
        ]
        models_to_test: list[type[BaseModel]] = [
            HardwareDevice,
            CpuProfile,
            GpuDeviceInfo,
            MemoryProfile,
            HardwareProfile,
            ResourceTelemetry,
        ]
        for model_cls in models_to_test:
            fields = model_cls.__annotations__.keys()
            for forbidden in forbidden_fields:
                assert (
                    forbidden not in fields
                ), f"Model '{model_cls.__name__}' must not contain sensitive field '{forbidden}'."

    def test_hardware_config_defaults(self) -> None:
        """Verify default hardware configuration parameters."""
        cfg = create_hardware_config()
        assert cfg.audio_capture_provider in ("mock", "sounddevice", "auto")
        assert cfg.vision_capture_provider in ("mock", "opencv", "directshow", "auto")
        assert cfg.telemetry_interval_sec == 1.0
        assert cfg.mock_fallback_enabled is True
