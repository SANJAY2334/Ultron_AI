"""Comprehensive Unit & Hardware Integration Tests for YOLOONNXDetector (Phase 4H.6).

Validates local offline YOLO object detection, ONNX Runtime execution, letterbox preprocessing,
vectorized NMS post-processing, privacy guarantees (PERSON != IDENTITY), Zero-Trust boundaries,
VisionProcessor integration, tracker/scene/planner context integration, and physical webcam inference.
"""

import os
from datetime import UTC, datetime
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from app.vision.adapters.planner_context_builder import VisionPlannerContextBuilder
from app.vision.adapters.scene_analyzer import SceneAnalyzerAdapter
from app.vision.adapters.tracker import VisionTrackerAdapter
from app.vision.adapters.windows_camera_capture import (
    OPENCV_AVAILABLE,
    WindowsCameraCaptureDevice,
)
from app.vision.adapters.yolo_onnx_detector import (
    ONNXRUNTIME_AVAILABLE,
    YOLOONNXDetector,
    _compute_iou,
    _nms,
)
from app.vision.base import IObjectDetector
from app.vision.detection import (
    ObjectDetectionConfig,
    ObjectDetectionConfigurationError,
    ObjectDetectionTimeoutError,
    ObjectDetectionUnavailableError,
)
from app.vision.exceptions import ObjectDetectionValidationError
from app.vision.models import (
    FrameFormat,
    PixelFormat,
    VisionFrame,
    VisionObservation,
)
from app.vision.planner_context import VisionPlannerContext
from app.vision.processor import VisionProcessor


def _make_dummy_frame(
    width: int = 640,
    height: int = 480,
    seq: int = 0,
    pixel_format: PixelFormat = PixelFormat.RGB24,
) -> VisionFrame:
    """Helper creating valid VisionFrame with synthetic image content."""
    raw_bytes = np.full((height, width, 3), 128, dtype=np.uint8).tobytes()
    return VisionFrame(
        frame_id=f"frm_test_{seq}",
        sequence_number=seq,
        timestamp=datetime.now(UTC),
        width=width,
        height=height,
        format=FrameFormat(width=width, height=height, pixel_format=pixel_format),
        payload=raw_bytes,
    )


class TestYOLOONNXDetectorAutomated:
    """Automated unit tests for YOLOONNXDetector."""

    def test_implements_iobject_detector_interface(self) -> None:
        """Verify YOLOONNXDetector implements IObjectDetector."""
        detector = YOLOONNXDetector(simulated_mode=True)
        assert isinstance(detector, IObjectDetector)

    def test_provider_construction_and_defaults(self) -> None:
        """Verify initialization defaults and capabilities reporting."""
        detector = YOLOONNXDetector(simulated_mode=True)
        assert detector.input_size == (640, 640)
        assert detector.requested_device == "cpu"

    def test_configuration_validation(self) -> None:
        """Verify confidence and iou thresholds bounds."""
        cfg = ObjectDetectionConfig(confidence_threshold=0.60, iou_threshold=0.40)
        assert cfg.confidence_threshold == 0.60
        assert cfg.iou_threshold == 0.40

        with pytest.raises(ValueError):
            ObjectDetectionConfig(confidence_threshold=1.5)

        with pytest.raises(ObjectDetectionConfigurationError):
            ObjectDetectionConfig(device="unsupported_device_xyz")

    @pytest.mark.asyncio
    async def test_missing_model_file_handling(self) -> None:
        """Verify non-existent model path raises ObjectDetectionUnavailableError."""
        detector = YOLOONNXDetector(model_path="non_existent_yolo.onnx", simulated_mode=False)
        with pytest.raises(ObjectDetectionUnavailableError):
            await detector.initialize()

    @pytest.mark.asyncio
    async def test_corrupt_onnx_model_handling(self, tmp_path: Any) -> None:
        """Verify corrupt/invalid ONNX file raises ObjectDetectionUnavailableError."""
        bad_file = tmp_path / "corrupt.onnx"
        bad_file.write_bytes(b"NOT_AN_ONNX_FILE_CORRUPT_BYTES")
        detector = YOLOONNXDetector(model_path=str(bad_file), simulated_mode=False)
        with pytest.raises(ObjectDetectionUnavailableError):
            await detector.initialize()

    def test_letterbox_preprocessing_geometry(self) -> None:
        """Verify aspect-ratio-preserving letterbox tensor construction."""
        detector = YOLOONNXDetector(simulated_mode=True)
        frame = _make_dummy_frame(width=1280, height=720)
        tensor, scale, (pad_w, pad_h) = detector._preprocess(frame)

        assert tensor.shape == (1, 3, 640, 640)
        assert tensor.dtype == np.float32
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0
        assert pad_w == 0.0  # Width is scaled to 640, height is padded
        assert pad_h > 0.0

    def test_unsupported_pixel_format_raises_error(self) -> None:
        """Verify unsupported pixel format in VisionFrame raises ObjectDetectionValidationError."""
        detector = YOLOONNXDetector(simulated_mode=True)
        bad_frame = VisionFrame(
            frame_id="frm_bad_fmt",
            sequence_number=1,
            timestamp=datetime.now(UTC),
            width=100,
            height=100,
            format=FrameFormat(width=100, height=100, pixel_format=PixelFormat.GRAY8),
            payload=b"\x00" * 10000,
        )
        with pytest.raises(ObjectDetectionValidationError):
            detector._preprocess(bad_frame)

    def test_iou_computation_utility(self) -> None:
        """Verify IoU calculation on known boxes."""
        b1 = (0.0, 0.0, 10.0, 10.0)
        b2 = (0.0, 0.0, 10.0, 10.0)
        assert _compute_iou(b1, b2) == pytest.approx(1.0)

        b3 = (20.0, 20.0, 30.0, 30.0)
        assert _compute_iou(b1, b3) == 0.0

    def test_nms_suppression_and_confidence_filtering(self) -> None:
        """Verify overlapping proposal suppression with NMS."""
        # Two identical person boxes with different scores
        boxes = [(10.0, 10.0, 100.0, 100.0), (12.0, 12.0, 98.0, 98.0)]
        scores = [0.95, 0.85]
        classes = [0, 0]  # both person

        selected = _nms(boxes, scores, classes, iou_threshold=0.45, max_detections=10)
        assert len(selected) == 1
        assert selected[0] == 0  # Highest score kept

    def test_max_detections_limit_enforced(self) -> None:
        """Verify NMS caps output at max_detections."""
        # 10 non-overlapping boxes
        boxes = [(i * 20.0, 0.0, (i + 1) * 20.0, 20.0) for i in range(10)]
        scores = [0.9] * 10
        classes = [0] * 10

        selected = _nms(boxes, scores, classes, iou_threshold=0.45, max_detections=3)
        assert len(selected) == 3

    @pytest.mark.asyncio
    async def test_synthesis_inference_timeout(self) -> None:
        """Verify inference exceeding timeout_ms raises ObjectDetectionTimeoutError."""
        cfg = ObjectDetectionConfig(timeout_ms=10.0)
        detector = YOLOONNXDetector(config=cfg, simulated_mode=False)
        detector._initialized = True
        detector._status = "READY"

        frame = _make_dummy_frame()
        with patch("asyncio.wait_for", side_effect=TimeoutError()):
            with pytest.raises(ObjectDetectionTimeoutError):
                await detector.detect(frame)

    def test_hardware_execution_provider_selection_cpu(self) -> None:
        """Verify CPU provider selected when CUDA is not present."""
        detector = YOLOONNXDetector(device="cpu", simulated_mode=False)
        providers = detector._select_execution_providers()
        assert providers == ["CPUExecutionProvider"]

    def test_hardware_execution_provider_cuda_fallback(self) -> None:
        """Verify CUDA request cleanly falls back to CPU if host lacks CUDA."""
        detector = YOLOONNXDetector(device="cuda", simulated_mode=False)
        providers = detector._select_execution_providers()
        # On this host, CUDA is not available in ORT
        assert "CPUExecutionProvider" in providers

    def test_privacy_invariant_person_is_not_identity(self) -> None:
        """Verify detected persons are labelled strictly as 'person' without biometric identities."""
        detector = YOLOONNXDetector(simulated_mode=True)
        # Synthetic raw output representing 1 proposal for person (class 0)
        raw_out = np.zeros((1, 84, 1), dtype=np.float32)
        raw_out[0, 0, 0] = 320.0  # cx
        raw_out[0, 1, 0] = 320.0  # cy
        raw_out[0, 2, 0] = 100.0  # w
        raw_out[0, 3, 0] = 200.0  # h
        raw_out[0, 4, 0] = 0.95   # score for class 0 (person)

        frame = _make_dummy_frame(width=640, height=640)
        dets = detector._postprocess(raw_out, frame, scale=1.0, pad=(0.0, 0.0))

        assert len(dets) == 1
        assert dets[0].label == "person"
        assert not hasattr(dets[0], "identity")
        assert not hasattr(dets[0], "name")
        assert not hasattr(dets[0], "face_id")
        assert not hasattr(dets[0], "biometric_vector")

    def test_security_invariant_model_output_is_not_authorization(self) -> None:
        """Verify YOLO detector exposes zero execution, capability, or shell APIs."""
        forbidden = [
            "execute_tool",
            "run_command",
            "grant_capability",
            "authorize",
            "bypass_policy",
            "execute_shell",
            "send_ipc",
        ]
        for f in forbidden:
            assert not hasattr(YOLOONNXDetector, f)

    def test_privacy_invariant_no_frame_persistence(self) -> None:
        """Verify detector does not contain persistence methods for raw frames."""
        forbidden = [
            "save_frame_to_disk",
            "persist_frame",
            "record_to_db",
            "redis_client",
        ]
        for f in forbidden:
            assert not hasattr(YOLOONNXDetector, f)

    @pytest.mark.asyncio
    async def test_clean_shutdown_and_health_probing(self) -> None:
        """Verify clean shutdown and health dictionary."""
        detector = YOLOONNXDetector(simulated_mode=True)
        await detector.initialize()
        h = await detector.health()
        assert h["subsystem"] == "yolo_onnx_detector"
        assert h["status"] == "READY"

        await detector.shutdown()
        assert detector._status == "SHUTDOWN"


class TestVisionPipelineIntegration:
    """Integration tests verifying YOLO detector inside the full vision stack."""

    @pytest.mark.asyncio
    async def test_yolo_to_vision_observation_pipeline(self) -> None:
        """Verify VisionProcessor + YOLOONNXDetector produces VisionObservation."""
        detector = YOLOONNXDetector(simulated_mode=True)
        processor = VisionProcessor(detector=detector)

        frame = _make_dummy_frame()
        obs = await processor.process(frame)

        assert isinstance(obs, VisionObservation)
        assert len(obs.object_detections) > 0
        assert any(d.label == "person" for d in obs.object_detections)

    @pytest.mark.asyncio
    async def test_yolo_to_tracker_integration(self) -> None:
        """Verify tracker updates tracks across frames with YOLO detections."""
        detector = YOLOONNXDetector(simulated_mode=True)
        tracker = VisionTrackerAdapter()
        processor = VisionProcessor(detector=detector, tracker=tracker)

        f1 = _make_dummy_frame(seq=1)
        f2 = _make_dummy_frame(seq=2)

        obs1 = await processor.process(f1)
        obs2 = await processor.process(f2)

        assert len(obs1.tracks) > 0
        assert len(obs2.tracks) > 0

    @pytest.mark.asyncio
    async def test_yolo_to_scene_analyzer_integration(self) -> None:
        """Verify scene analyzer processes observations generated with YOLO detections."""
        detector = YOLOONNXDetector(simulated_mode=True)
        analyzer = SceneAnalyzerAdapter()
        processor = VisionProcessor(detector=detector, scene_analyzer=analyzer)

        frame = _make_dummy_frame()
        obs = await processor.process(frame)

        summary = await analyzer.analyze(obs)
        assert summary.object_count >= 1

    @pytest.mark.asyncio
    async def test_yolo_to_vision_planner_context_integration(self) -> None:
        """Verify end-to-end translation from YOLO detections to sanitized VisionPlannerContext."""
        detector = YOLOONNXDetector(simulated_mode=True)
        processor = VisionProcessor(detector=detector)
        builder = VisionPlannerContextBuilder()

        frame = _make_dummy_frame()
        obs = await processor.process(frame)
        ctx = await builder.build_context(obs)

        assert isinstance(ctx, VisionPlannerContext)
        assert ctx.object_count > 0
        # Check that context representation does not contain raw pixel bytes
        ctx_repr = repr(ctx)
        assert "payload" not in ctx_repr


class TestPhysicalHardwareWebcamYOLOIntegration:
    """Hardware integration test executing real physical webcam capture -> local YOLO ONNX inference."""

    @pytest.mark.asyncio
    async def test_physical_webcam_to_local_yolo_inference(self) -> None:
        """Captures real frame from physical webcam and runs actual YOLOv8n ONNX inference."""
        if not OPENCV_AVAILABLE or not ONNXRUNTIME_AVAILABLE:
            pytest.skip("OpenCV or ONNX Runtime not available.")

        import cv2

        # Check if physical webcam is available
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            pytest.skip("Physical camera index 0 not available on host.")
        ret, test_img = cap.read()
        cap.release()

        if not ret or test_img is None:
            pytest.skip("Failed to read test frame from physical camera.")

        model_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "vision", "models", "yolo", "yolov8n.onnx"
        )
        if not os.path.exists(model_path):
            pytest.skip("Local yolov8n.onnx model file not found.")

        camera = WindowsCameraCaptureDevice(device_id=0, simulated_mode=False)
        cfg = ObjectDetectionConfig(
            confidence_threshold=0.25,
            iou_threshold=0.45,
            timeout_ms=5000.0,
            device="cpu",
        )
        detector = YOLOONNXDetector(config=cfg, model_path=model_path, simulated_mode=False)
        builder = VisionPlannerContextBuilder()

        try:
            await camera.start()
            await detector.initialize()

            captured_frame = None
            async for f in camera.frames():
                captured_frame = f
                break

            assert captured_frame is not None
            assert captured_frame.width > 0
            assert captured_frame.height > 0

            # Execute real local YOLO ONNX detection on the physical camera frame
            detections = await detector.detect(captured_frame)
            assert isinstance(detections, list)

            # Build observation and planner context
            obs = VisionObservation(
                timestamp=captured_frame.timestamp,
                object_detections=detections,
            )
            ctx = await builder.build_context(obs)
            assert isinstance(ctx, VisionPlannerContext)

            # Verify Zero-Trust and Person!=Identity invariants
            for d in detections:
                assert d.label != ""
                if d.label == "person":
                    assert not hasattr(d, "identity")

        finally:
            await camera.stop()
            await detector.shutdown()
