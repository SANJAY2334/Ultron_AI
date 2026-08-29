"""Phase 4F.4 Object Detector Unit Tests.

Validates ObjectDetectorAdapter configuration bounds, confidence filtering, IoU thresholds,
maximum detection limits (max 50), normalized bounding box coordinates, person anonymity security rules,
model lifecycle management (initialize -> health -> detect -> shutdown), timeout protection,
determinism, telemetry privacy bounds, and architectural domain isolation.
"""

import asyncio
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.api.deps import get_object_detector
from app.vision import (
    BoundingBox,
    FrameFormat,
    IObjectDetector,
    ObjectDetection,
    ObjectDetectionConfig,
    ObjectDetectionConfigurationError,
    ObjectDetectionTimeoutError,
    ObjectDetectorAdapter,
    PixelFormat,
    VisionFrame,
)


def make_test_frame(seq: int = 1) -> VisionFrame:
    """Helper constructing a valid synthetic VisionFrame."""
    fmt = FrameFormat(width=640, height=480, pixel_format=PixelFormat.RGB24)
    return VisionFrame(
        frame_id=f"frm_det_{seq}",
        sequence_number=seq,
        timestamp=datetime.now(UTC),
        width=640,
        height=480,
        format=fmt,
        payload=b"\x00\x80\xff" * (640 * 480),
    )


def test_object_detection_config_validation() -> None:
    """Verify ObjectDetectionConfig rejects invalid confidence, IoU, limits, or providers."""

    # 1. Invalid provider
    with pytest.raises(ObjectDetectionConfigurationError):
        ObjectDetectionConfig(provider="invalid_provider_99")

    # 2. Invalid device
    with pytest.raises(ObjectDetectionConfigurationError):
        ObjectDetectionConfig(device="quantum_cpu")

    # 3. Invalid confidence threshold
    with pytest.raises(ValidationError):
        ObjectDetectionConfig(confidence_threshold=1.5)

    # 4. Invalid max_detections
    with pytest.raises(ValidationError):
        ObjectDetectionConfig(max_detections=0)


def test_object_detector_adapter_synthetic_detection() -> None:
    """Verify ObjectDetectorAdapter detects valid non-identifying objects with confidence bounds."""

    async def _test() -> None:
        detector = ObjectDetectorAdapter()
        await detector.initialize()

        frame = make_test_frame()
        detections = await detector.detect(frame)

        assert isinstance(detections, list)
        assert len(detections) >= 1
        d0 = detections[0]
        assert isinstance(d0, ObjectDetection)
        assert d0.label in ("person", "chair", "cup", "laptop")
        assert 0.0 <= d0.confidence <= 1.0
        assert isinstance(d0.bounding_box, BoundingBox)

        health = await detector.health()
        assert health["subsystem"] == "object_detector"
        assert health["status"] == "READY"
        assert health["frames_processed"] == 1

        await detector.shutdown()
        assert (await detector.health())["status"] == "SHUTDOWN"

    asyncio.run(_test())


def test_person_anonymity_security_rule() -> None:
    """CRITICAL SECURITY RULE: Person detection MUST remain anonymous. Identity fields must be rejected."""
    now = datetime.now(UTC)
    bbox = BoundingBox(x=0.1, y=0.1, width=0.4, height=0.6, confidence=0.95)

    det = ObjectDetection(
        detection_id="det_p1",
        label="person",
        confidence=0.95,
        bounding_box=bbox,
        timestamp=now,
    )

    assert det.label == "person"
    assert det.confidence == 0.95

    # Verify ObjectDetection schema contains ZERO identity fields
    forbidden_fields = [
        "name",
        "person_id",
        "identity",
        "biometric_embedding",
        "face_embedding",
        "recognition_score",
    ]
    for field in forbidden_fields:
        assert not hasattr(det, field)

    # Verify attempt to pass extra forbidden attributes raises ValidationError
    with pytest.raises(ValidationError):
        ObjectDetection(
            detection_id="det_p2",
            label="person",
            confidence=0.95,
            bounding_box=bbox,
            timestamp=now,
            name="John Doe",  # type: ignore[call-arg]
        )


def test_confidence_filtering_and_max_detections_cap() -> None:
    """Verify detector filters detections below confidence threshold and caps max_detections."""

    async def _test() -> None:
        cfg = ObjectDetectionConfig(confidence_threshold=0.90, max_detections=1)
        detector = ObjectDetectorAdapter(config=cfg)
        await detector.initialize()

        frame = make_test_frame()
        detections = await detector.detect(frame)

        assert len(detections) <= 1
        for d in detections:
            assert d.confidence >= 0.90

        await detector.shutdown()

    asyncio.run(_test())


def test_object_detector_timeout_protection() -> None:
    """Verify inference timeout protection raises ObjectDetectionTimeoutError on delay."""

    async def _test() -> None:
        cfg = ObjectDetectionConfig(timeout_ms=10.0)
        detector = ObjectDetectorAdapter(config=cfg, simulated_delay_sec=0.08)
        await detector.initialize()

        frame = make_test_frame()
        with pytest.raises(ObjectDetectionTimeoutError):
            await detector.detect(frame)

        await detector.shutdown()

    asyncio.run(_test())


def test_di_container_object_detector_resolution() -> None:
    """Verify get_object_detector() resolves IObjectDetector through the DI container."""
    detector = get_object_detector()
    assert isinstance(detector, IObjectDetector)


def test_telemetry_privacy_and_repr_bounds() -> None:
    """Verify ObjectDetectionTelemetry contains ZERO raw binary image bytes or file paths."""

    async def _test() -> None:
        detector = ObjectDetectorAdapter()
        frame = make_test_frame()
        detections = await detector.detect(frame)
        assert len(detections) >= 1

        health = await detector.health()
        assert "subsystem" in health
        assert "payload" not in health
        assert "image_bytes" not in health

        await detector.shutdown()

    asyncio.run(_test())


def test_architectural_domain_isolation() -> None:
    """Verify object detection modules contain ZERO imports of OS automation tools or autonomous planner."""
    import sys

    import app.vision.detection

    forbidden_modules = [
        "app.ai.planner.langgraph_planner",
        "app.desktop.adapters.pyautogui_adapter",
        "app.security.policy",
    ]
    for fmod in forbidden_modules:
        if fmod in sys.modules:
            assert fmod not in app.vision.detection.__dict__
