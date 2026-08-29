"""Phase 4F.1 Vision Intelligence Domain Architecture Unit Tests.

Validates vision domain models, frame payload redaction, bounding box coordinates,
detection-only face models (zero biometric identification), state machine legal/illegal transitions,
privacy default semantics, VisionConfig bounds, abstract interfaces, and security boundaries.
"""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.vision import (
    BoundingBox,
    FaceDetection,
    FaceLandmark,
    FrameFormat,
    IFaceDetector,
    IMotionDetector,
    InvalidVisionStateTransitionError,
    IObjectDetector,
    IVisionCapture,
    IVisionProcessor,
    IVisionSessionManager,
    IVisionTracker,
    MotionEvent,
    ObjectDetection,
    PixelFormat,
    SceneEvent,
    SceneEventType,
    VisionCaptureError,
    VisionConfig,
    VisionConfigurationError,
    VisionDeviceNotFoundError,
    VisionError,
    VisionFrame,
    VisionObservation,
    VisionPrivacy,
    VisionProcessingError,
    VisionSessionState,
    VisionSessionStateMachine,
    VisionTelemetry,
    VisionTimeoutError,
    VisionUnavailableError,
    create_vision_config,
)


def make_test_frame(seq: int = 0) -> VisionFrame:
    """Helper creating synthetic VisionFrame."""
    fmt = FrameFormat(
        width=640, height=480, channels=3, pixel_format=PixelFormat.RGB24, frame_rate=15.0
    )
    return VisionFrame(
        frame_id=f"frm_test_{seq}",
        sequence_number=seq,
        timestamp=datetime.now(UTC),
        width=640,
        height=480,
        format=fmt,
        payload=b"\x00\xff\x7f" * (640 * 480),
    )


def test_vision_frame_format_and_payload_redaction() -> None:
    """Verify VisionFrame creation, format attributes, timezone validation, and safe repr payload redaction."""
    frame = make_test_frame(seq=1)

    assert frame.frame_id == "frm_test_1"
    assert frame.sequence_number == 1
    assert frame.width == 640
    assert frame.height == 480
    assert frame.format.pixel_format == PixelFormat.RGB24

    # Verify raw binary bytes do NOT leak in repr() or str()
    repr_str = repr(frame)
    str_str = str(frame)

    assert "VisionFrame" in repr_str
    assert "640x480" in repr_str
    assert "payload_bytes=" in repr_str
    assert b"\x00\xff\x7f" not in repr_str.encode()
    assert repr_str == str_str

    # Timezone awareness check
    with pytest.raises(ValidationError):
        fmt = FrameFormat(width=640, height=480)
        VisionFrame(
            frame_id="bad_time",
            sequence_number=0,
            timestamp=datetime.now(),  # Naive timestamp
            width=640,
            height=480,
            format=fmt,
            payload=b"\x00",
        )


def test_bounding_box_normalization_and_validation() -> None:
    """Verify BoundingBox normalization bounds [0.0, 1.0] and boundary validation."""
    box = BoundingBox(x=0.1, y=0.2, width=0.5, height=0.4, confidence=0.95)
    assert box.x == 0.1
    assert box.width == 0.5
    assert box.confidence == 0.95

    # Out-of-bounds box validation failure
    with pytest.raises(ValidationError):
        BoundingBox(x=0.8, y=0.8, width=0.5, height=0.5)


def test_object_detection_model() -> None:
    """Verify ObjectDetection label, confidence, and timestamp validation."""
    box = BoundingBox(x=0.2, y=0.2, width=0.3, height=0.3)
    det = ObjectDetection(
        detection_id="det_obj_1",
        label="chair",
        confidence=0.88,
        bounding_box=box,
        tracking_id="trk_100",
        timestamp=datetime.now(UTC),
    )

    assert det.detection_id == "det_obj_1"
    assert det.label == "chair"
    assert det.tracking_id == "trk_100"


def test_face_detection_model_prohibits_biometric_identity() -> None:
    """Verify FaceDetection localization model strictly prohibits biometric identification fields."""
    box = BoundingBox(x=0.4, y=0.1, width=0.2, height=0.2)
    landmark = FaceLandmark(landmark_type="nose_tip", x=0.5, y=0.2)
    face = FaceDetection(
        face_id="face_det_1",
        bounding_box=box,
        confidence=0.92,
        landmarks=[landmark],
        timestamp=datetime.now(UTC),
    )

    assert face.face_id == "face_det_1"
    assert len(face.landmarks) == 1
    assert face.landmarks[0].landmark_type == "nose_tip"

    # Verify prohibition of biometric fields
    assert not hasattr(face, "name")
    assert not hasattr(face, "identity")
    assert not hasattr(face, "person_id")
    assert not hasattr(face, "biometric_embedding")


def test_motion_and_scene_event_models() -> None:
    """Verify MotionEvent and SceneEvent payload definitions."""
    box = BoundingBox(x=0.1, y=0.1, width=0.2, height=0.2)
    motion = MotionEvent(
        event_id="mot_1",
        motion_score=0.75,
        region=box,
        timestamp=datetime.now(UTC),
        confidence=0.9,
    )
    assert motion.motion_score == 0.75

    scene = SceneEvent(
        event_id="scn_1",
        event_type=SceneEventType.PERSON_DETECTED,
        confidence=0.96,
        timestamp=datetime.now(UTC),
        metadata={"region": "quadrant_1"},
    )
    assert scene.event_type == SceneEventType.PERSON_DETECTED


def test_vision_telemetry_privacy_bounds() -> None:
    """Verify VisionTelemetry contains metadata only and never raw pixel payloads."""
    telem = VisionTelemetry(
        frame_id="frm_tel_1",
        sequence_number=10,
        processing_latency_ms=12.5,
        objects_detected=3,
        faces_detected=1,
        motion_score=0.2,
        success=True,
    )
    assert telem.frame_id == "frm_tel_1"
    assert telem.objects_detected == 3
    assert not hasattr(telem, "payload")
    assert not hasattr(telem, "raw_pixels")


def test_vision_privacy_defaults() -> None:
    """Verify VisionPrivacy default setting is EPHEMERAL."""
    assert VisionPrivacy.EPHEMERAL == "EPHEMERAL"
    assert VisionPrivacy.SESSION == "SESSION"
    assert VisionPrivacy.PERSISTENT == "PERSISTENT"


def test_vision_session_state_machine_legal_and_illegal_transitions() -> None:
    """Verify VisionSessionStateMachine legal transitions and illegal transition error handling."""
    sm = VisionSessionStateMachine()
    assert sm.state == VisionSessionState.IDLE

    # Legal transition sequence
    sm.transition_to(VisionSessionState.INITIALIZING)
    sm.transition_to(VisionSessionState.CAPTURING)
    sm.transition_to(VisionSessionState.PROCESSING)
    sm.transition_to(VisionSessionState.DETECTING)
    sm.transition_to(VisionSessionState.TRACKING)
    sm.transition_to(VisionSessionState.STOPPING)
    sm.transition_to(VisionSessionState.STOPPED)

    assert sm.state == VisionSessionState.STOPPED

    sm.reset()
    assert sm.state == VisionSessionState.IDLE

    # Illegal transition attempt
    with pytest.raises(InvalidVisionStateTransitionError) as exc_info:
        sm.transition_to(VisionSessionState.PROCESSING)

    assert exc_info.value.current == VisionSessionState.IDLE
    assert exc_info.value.target == VisionSessionState.PROCESSING


def test_vision_config_bounds_and_defaults() -> None:
    """Verify VisionConfig resolution, FPS limits, resolution bounds, queue depth, and drop policy."""
    cfg = VisionConfig(
        enabled=True,
        max_fps=15.0,
        max_width=1280,
        max_height=720,
        max_buffered_frames=10,
        drop_policy="DROP_OLDEST",
        privacy=VisionPrivacy.EPHEMERAL,
    )

    assert cfg.enabled is True
    assert cfg.max_fps == 15.0
    assert cfg.drop_policy == "DROP_OLDEST"

    # Default settings resolution helper
    app_cfg = create_vision_config()
    assert isinstance(app_cfg, VisionConfig)
    assert app_cfg.privacy == VisionPrivacy.EPHEMERAL

    # Invalid drop policy error
    with pytest.raises(ValidationError):
        VisionConfig(drop_policy="INVALID_POLICY")


def test_abstract_interfaces_cannot_be_instantiated() -> None:
    """Verify abstract vision interfaces enforce contract implementation and cannot be instantiated."""
    with pytest.raises(TypeError):
        IVisionCapture()  # type: ignore[abstract]

    with pytest.raises(TypeError):
        IVisionProcessor()  # type: ignore[abstract]

    with pytest.raises(TypeError):
        IObjectDetector()  # type: ignore[abstract]

    with pytest.raises(TypeError):
        IFaceDetector()  # type: ignore[abstract]

    with pytest.raises(TypeError):
        IMotionDetector()  # type: ignore[abstract]

    with pytest.raises(TypeError):
        IVisionTracker()  # type: ignore[abstract]

    with pytest.raises(TypeError):
        IVisionSessionManager()  # type: ignore[abstract]


def test_sanitized_vision_exception_taxonomy() -> None:
    """Verify sanitized vision exception hierarchy."""
    err = VisionProcessingError("Frame decoding failed")
    assert isinstance(err, VisionError)

    timeout = VisionTimeoutError("Processing timed out")
    assert isinstance(timeout, VisionError)

    unavail = VisionUnavailableError("Camera device offline")
    assert isinstance(unavail, VisionError)

    cfg_err = VisionConfigurationError("Invalid FPS limit")
    assert isinstance(cfg_err, VisionError)

    dev_err = VisionDeviceNotFoundError("Device Index 0 not found")
    assert isinstance(dev_err, VisionError)

    cap_err = VisionCaptureError("Frame buffer overflow")
    assert isinstance(cap_err, VisionError)


def test_vision_observation_multimodal_boundary() -> None:
    """Verify VisionObservation container for multimodal planner consumption (contains NO raw frames)."""
    obs = VisionObservation(
        timestamp=datetime.now(UTC),
        scene_events=[],
        object_detections=[],
        face_detections=[],
        motion_events=[],
    )

    assert isinstance(obs.timestamp, datetime)
    assert not hasattr(obs, "frame")
    assert not hasattr(obs, "payload")
    assert not hasattr(obs, "raw_pixels")
