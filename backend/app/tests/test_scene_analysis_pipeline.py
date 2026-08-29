"""Phase 4F.6 Vision Scene Understanding Pipeline Integration Tests.

Validates end-to-end pipeline:
CameraCaptureAdapter -> VisionFrame -> VisionProcessor -> ObjectDetector -> ObjectDetection[] -> VisionTracker -> ObjectTrack[] -> VisionObservation -> SceneAnalyzer -> SceneSummary.
Uses deterministic synthetic acquisition and perception for 100% reliable CI execution.
"""

import asyncio

from app.vision import (
    CameraCaptureAdapter,
    ISceneAnalyzer,
    IVisionCapture,
    IVisionProcessor,
    IVisionTracker,
    ObjectDetectorAdapter,
    SceneAnalysisConfig,
    SceneAnalyzerAdapter,
    SceneState,
    SceneSummary,
    TrackingConfig,
    VisionConfig,
    VisionFrame,
    VisionObservation,
    VisionProcessor,
    VisionTrackerAdapter,
)


def test_full_vision_scene_analysis_pipeline_integration() -> None:
    """Verify integration of capture, processor, object detector, tracker, scene analyzer, and VisionObservation."""

    async def _test() -> None:
        v_cfg = VisionConfig(
            max_width=640,
            max_height=480,
            max_fps=30.0,
            max_buffered_frames=5,
            camera_device="default",
        )
        t_cfg = TrackingConfig(iou_threshold=0.30, max_tracks=10)
        s_cfg = SceneAnalysisConfig(crowded_threshold=10)

        capture: IVisionCapture = CameraCaptureAdapter(config=v_cfg, simulated_mode=True)
        detector: ObjectDetectorAdapter = ObjectDetectorAdapter()
        tracker: IVisionTracker = VisionTrackerAdapter(config=t_cfg)
        scene_analyzer: ISceneAnalyzer = SceneAnalyzerAdapter(config=s_cfg)

        processor: IVisionProcessor = VisionProcessor(
            config=v_cfg,
            detector=detector,
            tracker=tracker,
            scene_analyzer=scene_analyzer,
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

        # Check scene_summary populated on VisionObservation
        summary = obs_last.scene_summary
        assert summary is not None
        assert isinstance(summary, SceneSummary)
        assert summary.scene_id.startswith("scene_")
        assert summary.scene_state in (
            SceneState.STABLE,
            SceneState.ACTIVE,
            SceneState.EMPTY,
            SceneState.CHANGING,
        )
        assert summary.object_count >= 1

        # Check person anonymity
        for cls_lbl in summary.object_classes:
            assert isinstance(cls_lbl, str)
            assert cls_lbl not in ("Sanjay", "User")

        for trk_id in summary.entered_objects:
            assert trk_id.startswith("track_")

    asyncio.run(_test())
