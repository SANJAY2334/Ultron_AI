"""Phase 4F.4 Object Detection Pipeline Integration Tests.

Validates the full vision & object detection pipeline integration:
CameraCaptureAdapter -> IVisionCapture -> VisionFrame -> IVisionProcessor -> VisionProcessor -> IObjectDetector -> ObjectDetectorAdapter -> ObjectDetection[] -> VisionObservation.
Uses deterministic mock/synthetic detector for 100% reliable CI execution.
"""

import asyncio

from app.vision import (
    CameraCaptureAdapter,
    IVisionCapture,
    IVisionProcessor,
    ObjectDetection,
    ObjectDetectorAdapter,
    VisionConfig,
    VisionFrame,
    VisionObservation,
    VisionProcessor,
)


def test_full_object_detection_pipeline_integration() -> None:
    """Verify integration of CameraCaptureAdapter, VisionProcessor, ObjectDetectorAdapter, and VisionObservation."""

    async def _test() -> None:
        cfg = VisionConfig(
            max_width=640,
            max_height=480,
            max_fps=30.0,
            max_buffered_frames=5,
            camera_device="default",
        )

        capture: IVisionCapture = CameraCaptureAdapter(config=cfg, simulated_mode=True)
        detector = ObjectDetectorAdapter()
        processor: IVisionProcessor = VisionProcessor(config=cfg, detector=detector)

        await capture.start()
        await detector.initialize()
        await asyncio.sleep(0.1)

        observations: list[VisionObservation] = []
        frame_count = 0

        async for frame in capture.frames():
            assert isinstance(frame, VisionFrame)
            assert frame.sequence_number > 0

            obs = await processor.process(frame)
            assert isinstance(obs, VisionObservation)
            observations.append(obs)

            frame_count += 1
            if frame_count >= 3:
                break

        await capture.stop()
        await detector.shutdown()

        assert len(observations) >= 3
        obs0 = observations[0]
        assert obs0.timestamp is not None
        assert isinstance(obs0.object_detections, list)
        assert len(obs0.object_detections) >= 1

        d0 = obs0.object_detections[0]
        assert isinstance(d0, ObjectDetection)
        assert d0.label in ("person", "chair", "cup", "laptop")
        assert 0.0 <= d0.confidence <= 1.0

        # Verify person anonymity guarantee in observation outputs
        for det in obs0.object_detections:
            if det.label == "person":
                assert not hasattr(det, "name")
                assert not hasattr(det, "person_id")
                assert not hasattr(det, "identity")

    asyncio.run(_test())
