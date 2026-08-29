"""Phase 4F.7 Vision-to-Autonomous Planner Pipeline Integration Tests.

Validates end-to-end pipeline:
Camera -> VisionFrame -> VisionProcessor -> ObjectDetector -> ObjectTracker -> SceneAnalyzer -> VisionObservation -> VisionPlannerContextBuilder -> VisionPlannerContext -> AgentState -> Response.
Uses deterministic synthetic acquisition and perception for 100% reliable CI execution.
"""

import asyncio

from app.ai.planner.base import AgentState
from app.ai.tools.base import ExecutionContext, ToolMetadata
from app.security.policy import PolicyEngine
from app.vision import (
    CameraCaptureAdapter,
    IVisionCapture,
    IVisionPlannerContextBuilder,
    IVisionProcessor,
    IVisionTracker,
    ObjectDetectorAdapter,
    SceneAnalysisConfig,
    SceneAnalyzerAdapter,
    TrackingConfig,
    VisionConfig,
    VisionContextConfig,
    VisionFrame,
    VisionObservation,
    VisionPlannerContext,
    VisionPlannerContextBuilder,
    VisionProcessor,
    VisionTrackerAdapter,
)


def test_full_vision_to_planner_pipeline_integration() -> None:
    """Verify integration of capture, processor, detector, tracker, scene analyzer, context builder, and AgentState."""

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
        c_cfg = VisionContextConfig(max_objects=10, max_tracks=10)

        capture: IVisionCapture = CameraCaptureAdapter(config=v_cfg, simulated_mode=True)
        detector: ObjectDetectorAdapter = ObjectDetectorAdapter()
        tracker: IVisionTracker = VisionTrackerAdapter(config=t_cfg)
        scene_analyzer = SceneAnalyzerAdapter(config=s_cfg)
        ctx_builder: IVisionPlannerContextBuilder = VisionPlannerContextBuilder(config=c_cfg)

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
            if frame_count >= 3:
                break

        await capture.stop()
        await detector.shutdown()

        assert len(observations) >= 3
        obs_last = observations[-1]

        # Build VisionPlannerContext
        vision_ctx = await ctx_builder.build_context(obs_last)
        assert isinstance(vision_ctx, VisionPlannerContext)
        assert vision_ctx.object_count >= 1

        # Create AgentState with vision context
        exec_ctx = ExecutionContext(session_id="pipeline_session", user_id="user_test")
        agent_state = AgentState(context=exec_ctx, vision_context=vision_ctx)

        assert agent_state.vision_context is not None
        prompt_text = agent_state.vision_context.to_prompt_context()
        assert "[VISUAL SCENE CONTEXT]" in prompt_text

        # Verify PolicyEngine boundary remains intact
        policy = PolicyEngine()
        tool_safe = ToolMetadata(
            name="list_dir", description="List directory", required_capabilities=[]
        )
        decision = policy.evaluate_tool_execution(tool_safe, agent_state.context)
        assert decision.decision == "ALLOW"

        tool_destructive = ToolMetadata(
            name="delete_file",
            description="Delete file",
            required_capabilities=["file:delete"],
            destructive=True,
            confirmation_required=True,
        )
        destructive_decision = policy.evaluate_tool_execution(tool_destructive, agent_state.context)
        assert destructive_decision.decision == "DENY"

    asyncio.run(_test())
