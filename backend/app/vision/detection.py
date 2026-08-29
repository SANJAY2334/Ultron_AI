"""Object Detection Domain Models, Configuration, and Exception Taxonomy (Phase 4F.4).

Defines provider-independent configuration models, telemetry records, and sanitized exceptions
for object detection operations without introducing vendor model dependencies.
"""

from pydantic import BaseModel, Field, field_validator

from app.core.config import Settings, get_settings
from app.vision.exceptions import VisionError


# Exception Taxonomy for Object Detection Subsystem
class ObjectDetectionError(VisionError):
    """Base application exception for all Object Detection failures."""


class ObjectDetectionConfigurationError(ObjectDetectionError):
    """Raised when invalid Object Detection configuration parameters are supplied."""


class ObjectDetectionValidationError(ObjectDetectionError):
    """Raised when object detection inputs or outputs fail validation bounds."""


class ObjectDetectionTimeoutError(ObjectDetectionError):
    """Raised when object detection inference exceeds the configured timeout."""


class ObjectDetectionProviderError(ObjectDetectionError):
    """Raised when the underlying inference backend or model encounters an execution error."""


class ObjectDetectionUnavailableError(ObjectDetectionError):
    """Raised when the object detection model file or provider engine is uninitialized/unavailable."""


class ObjectDetectionProcessingError(ObjectDetectionError):
    """Raised when post-processing or bounding box calculation fails."""


class ObjectDetectionConfig(BaseModel):
    """Configuration parameters for Object Detection inference engine."""

    provider: str = Field(
        default="mock", description="Inference backend provider identifier (mock, synthetic, onnx)"
    )
    model_path: str = Field(
        default="", description="Configured path to object detection model weights"
    )
    confidence_threshold: float = Field(
        default=0.50, ge=0.0, le=1.0, description="Minimum detection confidence score [0.0, 1.0]"
    )
    iou_threshold: float = Field(
        default=0.45, ge=0.0, le=1.0, description="Non-maximum suppression IoU threshold [0.0, 1.0]"
    )
    max_detections: int = Field(
        default=50, gt=0, le=200, description="Maximum allowed object detections per frame"
    )
    timeout_ms: float = Field(
        default=100.0, gt=0.0, description="Maximum inference timeout in milliseconds"
    )
    device: str = Field(default="cpu", description="Execution device hardware target (cpu, cuda)")

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        valid_providers = {"mock", "synthetic", "onnx", "local"}
        if v.lower() not in valid_providers:
            raise ObjectDetectionConfigurationError(
                f"Invalid object detector provider '{v}'. Must be one of {valid_providers}."
            )
        return v.lower()

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        valid_devices = {"cpu", "cuda", "directml", "tensorrt"}
        if v.lower() not in valid_devices:
            raise ObjectDetectionConfigurationError(
                f"Invalid execution device '{v}'. Must be one of {valid_devices}."
            )
        return v.lower()


class ObjectDetectionTelemetry(BaseModel):
    """Telemetry record for object detection inference operations (ZERO RAW IMAGE BYTES)."""

    frame_id: str = Field(min_length=1, description="Target frame identifier")
    detection_count: int = Field(ge=0, description="Number of valid objects detected")
    class_counts: dict[str, int] = Field(
        default_factory=dict, description="Summary counts per detected class label"
    )
    model_load_latency_ms: float = Field(
        default=0.0, ge=0.0, description="Model initialization duration in ms"
    )
    inference_latency_ms: float = Field(
        default=0.0, ge=0.0, description="Raw model inference duration in ms"
    )
    preprocessing_latency_ms: float = Field(
        default=0.0, ge=0.0, description="Pre-processing duration in ms"
    )
    postprocessing_latency_ms: float = Field(
        default=0.0, ge=0.0, description="Post-processing and NMS duration in ms"
    )
    total_latency_ms: float = Field(default=0.0, ge=0.0, description="Total pipeline latency in ms")
    success: bool = Field(default=True, description="True if detection pipeline succeeded")
    error_code: str | None = Field(
        default=None, description="Sanitized error code string if failed"
    )


def create_object_detection_config(settings: Settings | None = None) -> ObjectDetectionConfig:
    """Constructs ObjectDetectionConfig derived from application Settings."""
    cfg = settings or get_settings()
    return ObjectDetectionConfig(
        provider=getattr(cfg, "VISION_OBJECT_DETECTOR_PROVIDER", "mock"),
        model_path=getattr(cfg, "VISION_OBJECT_MODEL_PATH", ""),
        confidence_threshold=getattr(cfg, "VISION_OBJECT_CONFIDENCE_THRESHOLD", 0.50),
        iou_threshold=getattr(cfg, "VISION_OBJECT_IOU_THRESHOLD", 0.45),
        max_detections=getattr(cfg, "VISION_OBJECT_MAX_DETECTIONS", 50),
        timeout_ms=getattr(cfg, "VISION_OBJECT_TIMEOUT_MS", 100.0),
        device=getattr(cfg, "VISION_OBJECT_DEVICE", "cpu"),
    )
