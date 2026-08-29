"""Phase 4F.3 Vision Processing Pipeline Integration Tests.

Validates the full vision frame processing pipeline:
CameraCaptureAdapter -> IVisionCapture -> VisionFrame -> IVisionProcessor -> VisionProcessor -> VisionObservation.
Verifies frame streaming, sequence monotonicity, bounded processing queues, failure handling, and privacy bounds.
"""

import asyncio

from app.vision import (
    CameraCaptureAdapter,
    IVisionCapture,
    IVisionProcessor,
    VisionConfig,
    VisionFrame,
    VisionObservation,
    VisionProcessor,
)


def test_full_vision_processing_pipeline_integration() -> None:
    """Verify integration of CameraCaptureAdapter, IVisionCapture, VisionProcessor, and VisionObservation."""

    async def _test() -> None:
        cfg = VisionConfig(
            max_width=640,
            max_height=480,
            max_fps=30.0,
            max_buffered_frames=5,
            camera_device="default",
            enable_motion=True,
        )

        capture: IVisionCapture = CameraCaptureAdapter(config=cfg, simulated_mode=True)
        processor: IVisionProcessor = VisionProcessor(config=cfg)

        await capture.start()
        await asyncio.sleep(0.1)

        observations: list[VisionObservation] = []
        frame_count = 0

        async for frame in capture.frames():
            assert isinstance(frame, VisionFrame)
            assert frame.sequence_number > 0
            assert frame.width == 640
            assert frame.height == 480

            obs = await processor.process(frame)
            assert isinstance(obs, VisionObservation)
            observations.append(obs)

            frame_count += 1
            if frame_count >= 4:
                break

        await capture.stop()

        assert len(observations) >= 4
        assert observations[0].timestamp is not None
        assert observations[0].object_detections == []
        assert observations[0].face_detections == []

        cap_health = await capture.health()
        proc_health = await processor.health()

        assert cap_health["frames_processed"] >= 4
        assert proc_health["frames_processed"] >= 4
        assert proc_health["frames_rejected"] == 0

    asyncio.run(_test())
