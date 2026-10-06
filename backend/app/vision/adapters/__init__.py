"""Vision Subsystem Hardware and Intelligence Adapters for ULTRON.

Provides concrete adapters for camera acquisition, vision perception, object detection,
visual object tracking, scene understanding, and vision-to-planner context translation.
"""

from app.vision.adapters.camera import CameraCaptureAdapter
from app.vision.adapters.object_detector import ObjectDetectorAdapter
from app.vision.adapters.planner_context_builder import VisionPlannerContextBuilder
from app.vision.adapters.scene_analyzer import SceneAnalyzerAdapter
from app.vision.adapters.tracker import VisionTrackerAdapter
from app.vision.adapters.windows_camera_capture import WindowsCameraCaptureDevice
from app.vision.adapters.yolo_onnx_detector import YOLOONNXDetector

__all__ = [
    "CameraCaptureAdapter",
    "WindowsCameraCaptureDevice",
    "ObjectDetectorAdapter",
    "YOLOONNXDetector",
    "VisionTrackerAdapter",
    "SceneAnalyzerAdapter",
    "VisionPlannerContextBuilder",
]
