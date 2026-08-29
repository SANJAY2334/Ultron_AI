"""Provider-Agnostic Vision Intelligence Abstract Interfaces (Phase 4F.1).

Defines abstract base classes (ABC) for IVisionCapture, IVisionProcessor, IObjectDetector,
IFaceDetector, IMotionDetector, IVisionTracker, and IVisionSessionManager.
Ensures the vision domain remains 100% independent of OpenCV, YOLO, MediaPipe, ONNX, and PyTorch.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterable
from typing import Any

from app.vision.models import (
    FaceDetection,
    MotionEvent,
    ObjectDetection,
    VisionFrame,
    VisionObservation,
    VisionSessionState,
)


class IVisionCapture(ABC):
    """Abstract interface for video frame capture sources."""

    @abstractmethod
    async def list_devices(self) -> list[dict[str, Any]]:
        """Lists available video capture devices."""

    @abstractmethod
    async def start(self) -> None:
        """Starts video frame capture stream."""

    @abstractmethod
    async def stop(self) -> None:
        """Stops video frame capture stream."""

    @abstractmethod
    def frames(self) -> AsyncIterable[VisionFrame]:
        """Asynchronously yields captured VisionFrame objects."""

    @abstractmethod
    async def health(self) -> dict[str, Any]:
        """Probes operational health of capture device."""


class IVisionProcessor(ABC):
    """Abstract interface for high-level frame processing engine."""

    @abstractmethod
    async def process(self, frame: VisionFrame) -> VisionObservation:
        """Processes a single VisionFrame and returns sanitized perception observation."""

    @abstractmethod
    async def health(self) -> dict[str, Any]:
        """Probes operational health of vision processor engine."""


class IObjectDetector(ABC):
    """Abstract interface for non-identifying object detection engines."""

    @abstractmethod
    async def detect(self, frame: VisionFrame) -> list[ObjectDetection]:
        """Detects objects within a single VisionFrame."""

    @abstractmethod
    async def health(self) -> dict[str, Any]:
        """Probes operational health of object detector backend."""


class IFaceDetector(ABC):
    """Abstract interface for detection-only facial region localization engines."""

    @abstractmethod
    async def detect(self, frame: VisionFrame) -> list[FaceDetection]:
        """Detects facial bounding boxes within frame (NO BIOMETRIC IDENTIFICATION)."""

    @abstractmethod
    async def health(self) -> dict[str, Any]:
        """Probes operational health of face detector backend."""


class IMotionDetector(ABC):
    """Abstract interface for frame-to-frame motion estimation engines."""

    @abstractmethod
    async def detect(self, frame: VisionFrame) -> MotionEvent | None:
        """Detects motion intensity within frame."""

    @abstractmethod
    async def health(self) -> dict[str, Any]:
        """Probes operational health of motion detector backend."""


class IVisionTracker(ABC):
    """Abstract interface for visual multi-object tracking engines."""

    @abstractmethod
    async def update(self, detections: list[ObjectDetection], frame_id: str = "frm_0") -> Any:
        """Updates object tracks given new detections."""

    @abstractmethod
    def reset(self) -> None:
        """Resets active tracking state."""

    @abstractmethod
    async def health(self) -> dict[str, Any]:
        """Probes operational health of vision tracker backend."""


class ISceneAnalyzer(ABC):
    """Abstract interface for deterministic visual scene understanding engines."""

    @abstractmethod
    async def analyze(self, observation: VisionObservation) -> Any:
        """Analyzes structured VisionObservation and returns SceneSummary."""

    @abstractmethod
    def reset(self) -> None:
        """Resets bounded observation history state."""

    @abstractmethod
    async def health(self) -> dict[str, Any]:
        """Probes operational health of scene analyzer backend."""


class IVisionPlannerContextBuilder(ABC):
    """Abstract interface for constructing sanitized VisionPlannerContext for planner consumption."""

    @abstractmethod
    async def build_context(self, observation: VisionObservation) -> Any:
        """Translates structured VisionObservation into sanitized VisionPlannerContext."""

    @abstractmethod
    async def health(self) -> dict[str, Any]:
        """Probes operational health of vision planner context builder."""


class IVisionSessionManager(ABC):
    """Abstract interface for Vision Session Lifecycle Orchestrator."""

    @abstractmethod
    async def start_session(self, session_id: str) -> None:
        """Starts a vision perception session."""

    @abstractmethod
    async def stop_session(self, session_id: str) -> None:
        """Stops active vision perception session."""

    @abstractmethod
    async def process_frame(self, frame: VisionFrame) -> VisionObservation:
        """Processes VisionFrame and returns sanitized VisionObservation for planner consumption."""

    @abstractmethod
    def get_state(self) -> VisionSessionState:
        """Returns current state machine state."""

    @abstractmethod
    async def health(self) -> dict[str, Any]:
        """Probes health of vision session manager and underlying components."""
