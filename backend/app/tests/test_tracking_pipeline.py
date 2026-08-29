"""Phase 4F.5 Vision Object Tracking Pipeline Integration Tests.

Validates end-to-end pipeline:
CameraCaptureAdapter -> IVisionCapture -> VisionFrame -> IVisionProcessor -> VisionProcessor -> IObjectDetector -> ObjectDetectorAdapter -> ObjectDetection[] -> IVisionTracker -> VisionTrackerAdapter -> ObjectTrack[] -> VisionObservation.
Uses deterministic synthetic acquisition and detection for 100% reliable CI execution.
"""

import asyncio

from app.vision import (
    CameraCaptureAdapter,
    IVisionCapture,
    IVisionProcessor,
    IVisionTracker,
    ObjectDetectorAdapter,
    ObjectTrack,
    TrackEvent,
    TrackingConfig,
    VisionConfig,
    VisionFrame,
    VisionObservation,
    VisionProcessor,
    VisionTrackerAdapter,
)


def test_full_vision_tracking_pipeline_integration() -> None:
    """Verify integration of capture, processor, object detector, tracker, and VisionObservation."""

    async def _test() -> None:
        v_cfg = VisionConfig(
            max_width=640,
            max_height=480,
            max_fps=30.0,
            max_buffered_frames=5,
            camera_device="default",
        )
        t_cfg = TrackingConfig(iou_threshold=0.30, max_tracks=10)

        capture: IVisionCapture = CameraCaptureAdapter(config=v_cfg, simulated_mode=True)
        detector = ObjectDetectorAdapter()
        tracker: IVisionTracker = VisionTrackerAdapter(config=t_cfg)
        processor: IVisionProcessor = VisionProcessor(
            config=v_cfg, detector=detector, tracker=tracker
        )

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
            if frame_count >= 4:
                break

        await capture.stop()
        await detector.shutdown()

        assert len(observations) >= 4
        obs_last = observations[-1]

        assert obs_last.timestamp is not None
        assert isinstance(obs_last.object_detections, list)
        assert len(obs_last.object_detections) >= 1
        assert isinstance(obs_last.tracks, list)
        assert len(obs_last.tracks) >= 1

        # Check track structure and person anonymity
        for trk in obs_last.tracks:
            assert isinstance(trk, ObjectTrack)
            assert trk.track_id.startswith("track_")
            if trk.class_name == "person":
                assert not hasattr(trk, "name")
                assert not hasattr(trk, "person_id")
                assert not hasattr(trk, "identity")

        # Check track events generated
        for evt in obs_last.track_events:
            assert isinstance(evt, TrackEvent)
            assert evt.track_id.startswith("track_")

    asyncio.run(_test())
