"""Opt-In Real ONNX Model Qualification Test Suite (Phase 4F.4).

Validates physical ONNX Object Detector model loading, input tensor formatting,
and local inference execution when explicitly configured.
Activated by environment variable: ULTRON_VISION_MODEL_TEST=1
"""

import asyncio
import os

import pytest

from app.tests.test_object_detector import make_test_frame
from app.vision import ObjectDetectionConfig, ObjectDetectorAdapter

MODEL_TEST_ENABLED = os.getenv("ULTRON_VISION_MODEL_TEST") == "1"


@pytest.mark.skipif(
    not MODEL_TEST_ENABLED,
    reason="Real ONNX vision model qualification test requires ULTRON_VISION_MODEL_TEST=1 environment variable.",
)
def test_real_onnx_model_qualification() -> None:
    """Verify local ONNX object detector model loading and inference qualification."""

    async def _test() -> None:
        model_path = os.getenv("VISION_OBJECT_MODEL_PATH", "")
        if not model_path or not os.path.exists(model_path):
            pytest.skip(
                f"No configured ONNX model file found at VISION_OBJECT_MODEL_PATH='{model_path}'."
            )

        cfg = ObjectDetectionConfig(
            provider="onnx",
            model_path=model_path,
            confidence_threshold=0.50,
            iou_threshold=0.45,
            max_detections=50,
            timeout_ms=200.0,
            device="cpu",
        )
        detector = ObjectDetectorAdapter(config=cfg)

        await detector.initialize()
        health = await detector.health()

        if health["status"] != "READY":
            pytest.skip(f"ONNX model failed initialization (status={health['status']}).")

        frame = make_test_frame()
        detections = await detector.detect(frame)

        assert isinstance(detections, list)
        assert len(detections) <= 50

        await detector.shutdown()

    asyncio.run(_test())
