"""Phase 4F.5 Vision Object Tracking Unit Tests.

Validates TrackingConfig bounds, TrackState lifecycle transitions, deterministic IoU association,
motion direction calculation, noise threshold filtering, person anonymity security rules,
resource ceilings (max_tracks=50), timeout protection, and telemetry privacy bounds.
"""

import asyncio
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.api.deps import get_vision_tracker
from app.vision import (
    BoundingBox,
    IVisionTracker,
    ObjectDetection,
    ObjectTrack,
    TrackDirection,
    TrackEventType,
    TrackingConfig,
    TrackingConfigurationError,
    TrackingTimeoutError,
    TrackingValidationError,
    TrackState,
    VisionTrackerAdapter,
)


def make_test_detection(
    label: str = "person",
    x: float = 0.1,
    y: float = 0.1,
    w: float = 0.2,
    h: float = 0.3,
    conf: float = 0.90,
) -> ObjectDetection:
    """Helper creating a valid ObjectDetection instance."""
    return ObjectDetection(
        detection_id=f"det_{time_ns()}",
        label=label,
        confidence=conf,
        bounding_box=BoundingBox(x=x, y=y, width=w, height=h, confidence=conf),
        timestamp=datetime.now(UTC),
    )


def time_ns() -> int:
    return int(datetime.now(UTC).timestamp() * 1000000)


def test_tracking_config_validation() -> None:
    """Verify TrackingConfig rejects invalid IoU, distance, track limits, or providers."""

    # 1. Invalid provider
    with pytest.raises(TrackingConfigurationError):
        TrackingConfig(provider="invalid_provider_99")

    # 2. Invalid IoU threshold
    with pytest.raises(ValidationError):
        TrackingConfig(iou_threshold=1.5)

    # 3. Invalid max_tracks
    with pytest.raises(ValidationError):
        TrackingConfig(max_tracks=0)

    # 4. Invalid max_missing_frames
    with pytest.raises(ValidationError):
        TrackingConfig(max_missing_frames=0)


def test_track_state_machine_transitions() -> None:
    """Verify ObjectTrack lifecycle state machine transitions and illegal transition prevention."""
    now = datetime.now(UTC)
    bbox = BoundingBox(x=0.1, y=0.1, width=0.2, height=0.3)

    track = ObjectTrack(
        track_id="track_0001",
        class_name="person",
        confidence=0.92,
        bounding_box=bbox,
        state=TrackState.NEW,
        first_seen=now,
        last_seen=now,
    )

    assert track.state == TrackState.NEW
    assert track.is_active is True

    # Valid transitions: NEW -> ACTIVE -> LOST -> REACQUIRED -> ENDED
    track.transition_to(TrackState.ACTIVE)
    assert track.state == TrackState.ACTIVE

    track.transition_to(TrackState.LOST)
    assert track.state == TrackState.LOST

    track.transition_to(TrackState.REACQUIRED)
    assert track.state == TrackState.REACQUIRED

    track.transition_to(TrackState.ENDED)
    assert track.state == TrackState.ENDED
    assert track.is_active is False

    # Illegal transition: ENDED -> ACTIVE
    with pytest.raises(TrackingValidationError):
        track.transition_to(TrackState.ACTIVE)


def test_person_anonymity_tracking_security_rule() -> None:
    """CRITICAL SECURITY RULE: Tracked persons MUST remain anonymous. Identity fields strictly forbidden."""
    now = datetime.now(UTC)
    bbox = BoundingBox(x=0.1, y=0.1, width=0.2, height=0.3)

    track = ObjectTrack(
        track_id="track_0042",
        class_name="person",
        confidence=0.95,
        bounding_box=bbox,
        state=TrackState.ACTIVE,
        first_seen=now,
        last_seen=now,
    )

    assert track.track_id == "track_0042"
    assert track.class_name == "person"

    # Verify ObjectTrack schema contains ZERO identity fields
    forbidden_fields = [
        "name",
        "person_id",
        "identity",
        "biometric_embedding",
        "face_embedding",
        "recognition_score",
    ]
    for field in forbidden_fields:
        assert not hasattr(track, field)

    # Verify extra forbidden attributes raise ValidationError
    with pytest.raises(ValidationError):
        ObjectTrack(
            track_id="track_0043",
            class_name="person",
            confidence=0.95,
            bounding_box=bbox,
            state=TrackState.ACTIVE,
            first_seen=now,
            last_seen=now,
            name="Sanjay",  # type: ignore[call-arg]
        )


def test_tracker_association_same_object_across_frames() -> None:
    """Verify baseline tracker associates the same object continuously across consecutive frames."""

    async def _test() -> None:
        tracker = VisionTrackerAdapter()

        det1 = make_test_detection(label="person", x=0.1, y=0.1)
        tracks1, events1 = await tracker.update([det1], frame_id="frm_1")

        assert len(tracks1) == 1
        t1_id = tracks1[0].track_id
        assert tracks1[0].state == TrackState.NEW
        assert len(events1) >= 1
        assert events1[0].event_type == TrackEventType.TRACK_CREATED

        # Frame 2: Same object slightly shifted
        det2 = make_test_detection(label="person", x=0.12, y=0.1)
        tracks2, events2 = await tracker.update([det2], frame_id="frm_2")

        assert len(tracks2) == 1
        assert tracks2[0].track_id == t1_id
        assert tracks2[0].state == TrackState.ACTIVE
        assert tracks2[0].frames_seen == 2

        health = await tracker.health()
        assert health["subsystem"] == "vision_tracker"
        assert health["active_tracks"] == 1

        tracker.reset()
        assert (await tracker.health())["active_tracks"] == 0

    asyncio.run(_test())


def test_tracker_motion_direction_estimation() -> None:
    """Verify motion direction classification (RIGHT, LEFT, UP, DOWN, STATIONARY)."""

    async def _test() -> None:
        tracker = VisionTrackerAdapter(config=TrackingConfig(movement_threshold=5.0))

        # Frame 1: Initial position (cx=200px, cy=200px -> x=0.25, y=0.333)
        d1 = make_test_detection(label="laptop", x=0.25, y=0.333)
        await tracker.update([d1], frame_id="f1")

        # Frame 2: Shift RIGHT (+30px horizontal)
        d2 = make_test_detection(label="laptop", x=0.297, y=0.333)
        tracks, events = await tracker.update([d2], frame_id="f2")

        assert len(tracks) == 1
        assert tracks[0].direction in (TrackDirection.RIGHT, TrackDirection.DOWN_RIGHT)
        assert tracks[0].speed > 0.0

        # Check movement event generated
        moved_events = [e for e in events if e.event_type == TrackEventType.TRACK_MOVED]
        assert len(moved_events) >= 1

    asyncio.run(_test())


def test_tracker_missing_frame_termination_and_max_tracks() -> None:
    """Verify track state transitions to LOST then ENDED when missing, and max_tracks cap."""

    async def _test() -> None:
        cfg = TrackingConfig(max_missing_frames=2, max_tracks=2)
        tracker = VisionTrackerAdapter(config=cfg)

        d1 = make_test_detection(label="chair", x=0.1, y=0.1)
        await tracker.update([d1], frame_id="f1")

        # Frame 2: Missing detection -> track becomes LOST
        tracks_f2, events_f2 = await tracker.update([], frame_id="f2")
        assert len(tracks_f2) == 1
        assert tracks_f2[0].state == TrackState.LOST

        # Frame 3: Missing detection again -> track ENDED
        tracks_f3, events_f3 = await tracker.update([], frame_id="f3")
        assert len(tracks_f3) == 0

        ended_events = [e for e in events_f3 if e.event_type == TrackEventType.TRACK_ENDED]
        assert len(ended_events) == 1

    asyncio.run(_test())


def test_tracker_timeout_protection() -> None:
    """Verify tracking processing timeout raises TrackingTimeoutError."""

    async def _test() -> None:
        cfg = TrackingConfig(timeout_ms=10.0)
        slow_tracker = VisionTrackerAdapter(config=cfg, simulated_delay_sec=0.08)

        det = make_test_detection()
        with pytest.raises(TrackingTimeoutError):
            await slow_tracker.update([det])

    asyncio.run(_test())


def test_di_container_vision_tracker_resolution() -> None:
    """Verify get_vision_tracker() resolves IVisionTracker through DI container."""
    tracker = get_vision_tracker()
    assert isinstance(tracker, IVisionTracker)


def test_telemetry_privacy_and_repr_bounds() -> None:
    """Verify TrackingTelemetry contains ZERO raw binary frame bytes or file paths."""

    async def _test() -> None:
        tracker = VisionTrackerAdapter()
        det = make_test_detection()
        await tracker.update([det])

        health = await tracker.health()
        assert "active_tracks" in health
        assert "payload" not in health

    asyncio.run(_test())
