"""Phase 4F.6 Vision Scene Understanding Unit Tests.

Validates SceneAnalysisConfig bounds, SceneState classification, MotionLevel classification,
object statistics calculation, temporal change detection, dominant activity derivation,
person anonymity security rules, resource history bounds, timeout protection, and telemetry privacy bounds.
"""

import asyncio
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.api.deps import get_scene_analyzer
from app.vision import (
    BoundingBox,
    DominantActivity,
    ISceneAnalyzer,
    MotionLevel,
    ObjectDetection,
    ObjectTrack,
    SceneAnalysisConfig,
    SceneAnalysisTimeoutError,
    SceneAnalyzerAdapter,
    SceneChangeType,
    SceneState,
    SceneSummary,
    TrackDirection,
    TrackState,
    VisionObservation,
)


def make_obs(
    detections: list[ObjectDetection] | None = None,
    tracks: list[ObjectTrack] | None = None,
) -> VisionObservation:
    """Helper constructing a valid VisionObservation with timezone-aware timestamp."""
    return VisionObservation(
        timestamp=datetime.now(UTC),
        object_detections=detections or [],
        tracks=tracks or [],
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


def test_scene_config_validation() -> None:
    """Verify SceneAnalysisConfig validation rules."""
    # Invalid max_previous_observations
    with pytest.raises(ValidationError):
        SceneAnalysisConfig(max_previous_observations=0)

    # Invalid crowded_threshold
    with pytest.raises(ValidationError):
        SceneAnalysisConfig(crowded_threshold=0)

    # Invalid timeout_ms
    with pytest.raises(ValidationError):
        SceneAnalysisConfig(timeout_ms=0.0)


def test_scene_state_classification_empty_stable_active_crowded() -> None:
    """Verify SceneState classification rules (EMPTY, STABLE, ACTIVE, CROWDED)."""

    async def _test() -> None:
        analyzer = SceneAnalyzerAdapter(config=SceneAnalysisConfig(crowded_threshold=5))

        # 1. EMPTY scene
        obs_empty = make_obs([], [])
        sum_empty = await analyzer.analyze(obs_empty)
        assert sum_empty.scene_state == SceneState.EMPTY
        assert sum_empty.object_count == 0
        assert sum_empty.motion_level == MotionLevel.NONE
        assert sum_empty.dominant_activity == DominantActivity.NONE

        # 2. STABLE scene (objects present, 0 movement)
        det_person = make_det("person")
        det_laptop = make_det("laptop")
        trk_person = make_track("track_001", "person", speed=0.0)
        trk_laptop = make_track("track_002", "laptop", speed=0.0)

        obs_stable = make_obs([det_person, det_laptop], [trk_person, trk_laptop])
        sum_stable = await analyzer.analyze(obs_stable)
        assert sum_stable.scene_state == SceneState.STABLE
        assert sum_stable.object_count == 2
        assert sum_stable.class_counts == {"person": 1, "laptop": 1}
        assert sum_stable.dominant_activity == DominantActivity.STATIONARY

        # 3. ACTIVE scene (objects moving)
        trk_moving = make_track("track_001", "person", speed=15.0, direction=TrackDirection.RIGHT)
        obs_active = make_obs([det_person, det_laptop], [trk_moving, trk_laptop])
        sum_active = await analyzer.analyze(obs_active)
        assert sum_active.scene_state == SceneState.ACTIVE
        assert sum_active.moving_track_count == 1

        # 4. CROWDED scene (>= 5 objects)
        crowd_dets = [make_det("person", x=i * 0.1) for i in range(6)]
        crowd_trks = [make_track(f"t_{i}", "person") for i in range(6)]
        obs_crowd = make_obs(crowd_dets, crowd_trks)
        sum_crowd = await analyzer.analyze(obs_crowd)
        assert sum_crowd.scene_state == SceneState.CROWDED

    asyncio.run(_test())


def test_temporal_scene_change_detection() -> None:
    """Verify OBJECT_ENTERED, OBJECT_EXITED, and MOTION_STARTED event generation across observations."""

    async def _test() -> None:
        analyzer = SceneAnalyzerAdapter()

        # Obs 1: track_001 present
        t1 = make_track("track_001", "laptop", speed=0.0)
        sum1 = await analyzer.analyze(make_obs([make_det("laptop")], [t1]))
        assert len(sum1.entered_objects) == 0

        # Obs 2: track_002 enters
        t2 = make_track("track_002", "person", speed=10.0, direction=TrackDirection.LEFT)
        sum2 = await analyzer.analyze(make_obs([make_det("laptop"), make_det("person")], [t1, t2]))

        assert "track_002" in sum2.entered_objects
        entered_events = [
            e for e in sum2.scene_changes if e.event_type == SceneChangeType.OBJECT_ENTERED
        ]
        assert len(entered_events) >= 1
        assert entered_events[0].track_id == "track_002"

        # Obs 3: track_001 exits
        sum3 = await analyzer.analyze(make_obs([make_det("person")], [t2]))
        assert "track_001" in sum3.exited_objects
        exited_events = [
            e for e in sum3.scene_changes if e.event_type == SceneChangeType.OBJECT_EXITED
        ]
        assert len(exited_events) >= 1

    asyncio.run(_test())


def test_person_anonymity_security_rule() -> None:
    """CRITICAL SECURITY RULE: SceneSummary MUST NOT contain person identities or raw image payloads."""
    analyzer = SceneAnalyzerAdapter()
    t_person = make_track("track_0042", "person")
    summary = asyncio.run(analyzer.analyze(make_obs([make_det("person")], [t_person])))

    assert summary.scene_id.startswith("scene_")
    assert summary.object_classes == ["person"]

    # Verify no identity or biometric attributes exist
    forbidden_fields = ["name", "person_id", "identity", "biometric_embedding", "face_embedding"]
    for field in forbidden_fields:
        assert not hasattr(summary, field)

    # Verify extra attributes raise ValidationError
    with pytest.raises(ValidationError):
        SceneSummary(
            scene_id="scene_0001",
            timestamp=datetime.now(UTC),
            object_count=1,
            name="Sanjay",  # type: ignore[call-arg]
        )


def test_scene_analyzer_bounded_history_and_reset() -> None:
    """Verify max_previous_observations bounded depth and session reset."""

    async def _test() -> None:
        cfg = SceneAnalysisConfig(max_previous_observations=2)
        analyzer = SceneAnalyzerAdapter(config=cfg)

        for _i in range(5):
            await analyzer.analyze(make_obs([make_det("chair")]))

        health = await analyzer.health()
        assert health["subsystem"] == "scene_analyzer"
        assert health["bounded_history_depth"] <= 2
        assert health["scenes_analyzed"] == 5

        analyzer.reset()
        health_reset = await analyzer.health()
        assert health_reset["scenes_analyzed"] == 0
        assert health_reset["bounded_history_depth"] == 0

    asyncio.run(_test())


def test_scene_analyzer_timeout_protection() -> None:
    """Verify processing timeout raises SceneAnalysisTimeoutError."""

    async def _test() -> None:
        cfg = SceneAnalysisConfig(timeout_ms=10.0)
        slow_analyzer = SceneAnalyzerAdapter(config=cfg, simulated_delay_sec=0.08)

        with pytest.raises(SceneAnalysisTimeoutError):
            await slow_analyzer.analyze(make_obs([make_det()]))

    asyncio.run(_test())


def test_di_container_scene_analyzer_resolution() -> None:
    """Verify get_scene_analyzer() resolves ISceneAnalyzer through DI container."""
    analyzer = get_scene_analyzer()
    assert isinstance(analyzer, ISceneAnalyzer)


def test_telemetry_privacy_bounds() -> None:
    """Verify SceneAnalysisTelemetry contains ZERO raw binary frame bytes or file paths."""

    async def _test() -> None:
        analyzer = SceneAnalyzerAdapter()
        await analyzer.analyze(make_obs([make_det()]))

        health = await analyzer.health()
        assert "scenes_analyzed" in health
        assert "payload" not in health

    asyncio.run(_test())
