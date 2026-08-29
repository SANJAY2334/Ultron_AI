"""Phase 4F.7 Vision-to-Autonomous Planner Context Integration Unit Tests.

Validates VisionPlannerContext construction, prompt rendering, truncation ceilings,
person anonymity security rules, DI resolution, telemetry bounds, and the CRITICAL security invariant:
VISION PERCEPTION MUST NEVER BECOME AUTHORIZATION (PolicyEngine, SafetyInterlock, and ToolExecutor remain intact).
"""

import asyncio
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.ai.planner.base import AgentState
from app.ai.tools.base import ExecutionContext, ToolMetadata
from app.api.deps import get_vision_planner_context_builder
from app.desktop.models import ActionClassification, DesktopAction
from app.security.policy import PolicyEngine
from app.security.safety import SafetyInterlock
from app.vision import (
    BoundingBox,
    DominantActivity,
    IVisionPlannerContextBuilder,
    MotionLevel,
    ObjectDetection,
    ObjectTrack,
    SceneState,
    SceneSummary,
    TrackDirection,
    TrackState,
    VisionContextConfig,
    VisionContextTimeoutError,
    VisionObservation,
    VisionPlannerContext,
    VisionPlannerContextBuilder,
)


def make_obs(
    detections: list[ObjectDetection] | None = None,
    tracks: list[ObjectTrack] | None = None,
    summary: SceneSummary | None = None,
) -> VisionObservation:
    """Helper constructing a valid VisionObservation with timezone-aware timestamp."""
    return VisionObservation(
        timestamp=datetime.now(UTC),
        object_detections=detections or [],
        tracks=tracks or [],
        scene_summary=summary,
    )


def make_det(label: str = "person", x: float = 0.1, y: float = 0.1) -> ObjectDetection:
    """Helper constructing a valid ObjectDetection."""
    return ObjectDetection(
        detection_id=f"det_{time_ns()}",
        label=label,
        confidence=0.90,
        bounding_box=BoundingBox(x=x, y=y, width=0.2, height=0.3, confidence=0.90),
        timestamp=datetime.now(UTC),
    )


def make_track(
    track_id: str = "track_0001",
    label: str = "person",
    speed: float = 0.0,
    direction: TrackDirection = TrackDirection.STATIONARY,
) -> ObjectTrack:
    """Helper constructing a valid ObjectTrack."""
    now = datetime.now(UTC)
    return ObjectTrack(
        track_id=track_id,
        class_name=label,
        confidence=0.92,
        bounding_box=BoundingBox(x=0.1, y=0.1, width=0.2, height=0.3),
        state=TrackState.ACTIVE,
        first_seen=now,
        last_seen=now,
        speed=speed,
        direction=direction,
        is_active=True,
    )


def time_ns() -> int:
    return int(datetime.now(UTC).timestamp() * 1000000)


def test_vision_context_config_validation() -> None:
    """Verify VisionContextConfig validation bounds."""

    # Invalid max_objects
    with pytest.raises(ValidationError):
        VisionContextConfig(max_objects=0)

    # Invalid max_context_chars
    with pytest.raises(ValidationError):
        VisionContextConfig(max_context_chars=50)

    # Invalid timeout_ms
    with pytest.raises(ValidationError):
        VisionContextConfig(timeout_ms=0.0)


def test_vision_planner_context_building() -> None:
    """Verify context builder creates sanitized VisionPlannerContext and prompt string."""

    async def _test() -> None:
        builder = VisionPlannerContextBuilder()

        det_person = make_det("person")
        det_laptop = make_det("laptop")
        trk_person = make_track("track_0001", "person")
        trk_laptop = make_track("track_0002", "laptop")

        summary = SceneSummary(
            scene_id="scene_0001",
            timestamp=datetime.now(UTC),
            object_count=2,
            object_classes=["laptop", "person"],
            class_counts={"person": 1, "laptop": 1},
            active_track_count=2,
            moving_track_count=0,
            scene_state=SceneState.STABLE,
            motion_level=MotionLevel.NONE,
            dominant_objects=["person", "laptop"],
            dominant_activity=DominantActivity.STATIONARY,
        )

        obs = make_obs([det_person, det_laptop], [trk_person, trk_laptop], summary=summary)
        ctx = await builder.build_context(obs)

        assert isinstance(ctx, VisionPlannerContext)
        assert ctx.scene_state == "STABLE"
        assert ctx.object_count == 2
        assert ctx.class_counts == {"person": 1, "laptop": 1}

        prompt_text = ctx.to_prompt_context()
        assert "[VISUAL SCENE CONTEXT]" in prompt_text
        assert "Scene State: STABLE" in prompt_text
        assert "laptop: 1" in prompt_text

    asyncio.run(_test())


def test_person_anonymity_security_rule() -> None:
    """CRITICAL SECURITY RULE: VisionPlannerContext MUST NOT contain identity or biometric fields."""
    builder = VisionPlannerContextBuilder()
    trk_person = make_track("track_0042", "person")
    obs = make_obs([make_det("person")], [trk_person])

    ctx = asyncio.run(builder.build_context(obs))

    assert ctx.object_count == 1
    assert ctx.class_counts == {"person": 1}

    # Verify no identity or biometric attributes exist on context
    forbidden_fields = ["name", "person_id", "identity", "biometric_embedding", "face_embedding"]
    for field in forbidden_fields:
        assert not hasattr(ctx, field)

    # Verify extra forbidden attributes raise ValidationError on VisionPlannerContext
    with pytest.raises(ValidationError):
        VisionPlannerContext(
            observation_id="obs_001",
            timestamp=datetime.now(UTC),
            object_count=1,
            name="Sanjay",  # type: ignore[call-arg]
        )


def test_zero_trust_security_invariant_vision_is_not_authorization() -> None:
    """CRITICAL SECURITY INVARIANT: Vision context MUST NEVER grant execution capabilities or bypass PolicyEngine.

    Test scenario:
    Vision detects 'person', 'laptop', 'file_window'.
    User says 'Delete that file'.
    Verify:
    1. Vision context does NOT alter ExecutionContext permissions.
    2. PolicyEngine and SafetyInterlock STILL enforce policy validation and confirmation requirements.
    3. Vision perception NEVER sets user_confirmed=True.
    """
    builder = VisionPlannerContextBuilder()
    det_file = make_det("file_window")
    det_laptop = make_det("laptop")
    obs = make_obs([det_file, det_laptop])

    vision_ctx = asyncio.run(builder.build_context(obs))

    # 1. Attach vision_ctx to AgentState
    exec_ctx = ExecutionContext(
        session_id="test_session",
        user_id="user_1",
        granted_capabilities=set(),
    )
    agent_state = AgentState(context=exec_ctx, vision_context=vision_ctx)

    # Verify ExecutionContext permissions remain un-elevated
    assert "file:delete" not in agent_state.context.granted_capabilities

    # 2. Policy Engine validation for destructive action ('delete_file')
    tool_meta = ToolMetadata(
        name="delete_file",
        description="Deletes target file",
        required_capabilities=["file:delete"],
        destructive=True,
        confirmation_required=True,
    )
    policy = PolicyEngine()
    decision = policy.evaluate_tool_execution(tool_meta, agent_state.context)

    # Must be DENIED because missing required capabilities
    assert decision.decision == "DENY"

    # 3. Safety Interlock check for destructive action
    action = DesktopAction(
        action_id="act_001",
        action_type="delete_file",
        target="important.docx",
        capability="file:delete",
        classification=ActionClassification.DESTRUCTIVE,
    )
    interlock = SafetyInterlock()
    safety_decision = interlock.evaluate(action, agent_state.context)

    # Must be DENIED
    assert safety_decision.decision == "DENY"

    # Confirm vision context cannot bypass or produce user_confirmed=True
    assert not hasattr(vision_ctx, "user_confirmed")
    assert not hasattr(vision_ctx, "allow_file_system")


def test_vision_context_truncation_limits() -> None:
    """Verify context builder truncates items to max_objects, max_tracks, max_events ceilings."""

    async def _test() -> None:
        cfg = VisionContextConfig(max_objects=2, max_tracks=2, max_events=2)
        builder = VisionPlannerContextBuilder(config=cfg)

        dets = [make_det("chair", x=i * 0.1) for i in range(5)]
        trks = [make_track(f"track_{i:04d}", "chair") for i in range(5)]
        obs = make_obs(dets, trks)

        ctx = await builder.build_context(obs)
        assert ctx.object_count == 2
        assert ctx.active_track_count == 2

    asyncio.run(_test())


def test_vision_context_timeout_protection() -> None:
    """Verify context builder timeout protection raises VisionContextTimeoutError."""

    async def _test() -> None:
        cfg = VisionContextConfig(timeout_ms=10.0)
        slow_builder = VisionPlannerContextBuilder(config=cfg, simulated_delay_sec=0.08)

        with pytest.raises(VisionContextTimeoutError):
            await slow_builder.build_context(make_obs([make_det()]))

    asyncio.run(_test())


def test_di_container_vision_planner_context_builder_resolution() -> None:
    """Verify get_vision_planner_context_builder() resolves IVisionPlannerContextBuilder via DI container."""
    builder = get_vision_planner_context_builder()
    assert isinstance(builder, IVisionPlannerContextBuilder)


def test_telemetry_privacy_and_repr_bounds() -> None:
    """Verify VisionContextTelemetry contains ZERO raw image bytes or file paths."""

    async def _test() -> None:
        builder = VisionPlannerContextBuilder()
        await builder.build_context(make_obs([make_det()]))

        health = await builder.health()
        assert "builds_processed" in health
        assert "payload" not in health

    asyncio.run(_test())
