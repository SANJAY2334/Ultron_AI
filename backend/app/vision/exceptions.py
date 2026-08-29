"""Sanitized Vision Subsystem Exception Taxonomy (Phase 4F.1 / Phase 4F.3 / Phase 4F.4 / Phase 4F.5).

Defines framework-agnostic sanitized exceptions for the Vision Intelligence subsystem.
Prevents camera driver tracebacks, raw frame bytes, or local file paths from leaking into logs or tracebacks.
"""


class VisionError(Exception):
    """Base application exception for all Vision Subsystem failures."""


class VisionConfigurationError(VisionError):
    """Raised when invalid Vision configuration parameters are supplied."""


class VisionDeviceNotFoundError(VisionError):
    """Raised when the specified camera or capture device cannot be located."""


class VisionCaptureError(VisionError):
    """Raised when frame capture fails or encounters stream disruption."""


class VisionProcessingError(VisionError):
    """Raised when frame processing or detection algorithm fails."""


class VisionTimeoutError(VisionError):
    """Raised when vision frame processing or capture request exceeds configured timeout."""


class VisionUnavailableError(VisionError):
    """Raised when vision subsystem or detection backend is unavailable."""


# Vision Processor Exception Taxonomy
class VisionProcessorError(VisionError):
    """Base exception for Vision Processor failures."""


class VisionProcessorConfigurationError(VisionProcessorError):
    """Raised when invalid VisionProcessor configuration parameters are supplied."""


class VisionProcessorValidationError(VisionProcessorError):
    """Raised when an incoming VisionFrame fails payload, resolution, or format validation."""


class VisionProcessorTimeoutError(VisionProcessorError):
    """Raised when frame processing exceeds the configured timeout."""


class VisionProcessorProcessingError(VisionProcessorError):
    """Raised when frame processing or motion calculation encounters an unexpected runtime error."""


# Object Detection Exception Taxonomy
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


# Vision Object Tracking Exception Taxonomy
class TrackingError(VisionError):
    """Base application exception for all Vision Object Tracking failures."""


class TrackingConfigurationError(TrackingError):
    """Raised when invalid Tracking configuration parameters are supplied."""


class TrackingValidationError(TrackingError):
    """Raised when track models or state transitions fail validation rules."""


class TrackingProcessingError(TrackingError):
    """Raised when track association or motion calculation encounters an execution error."""


class TrackingTimeoutError(TrackingError):
    """Raised when track update operations exceed the configured processing timeout."""


class TrackingProviderError(TrackingError):
    """Raised when underlying tracking provider engine encounters a failure."""


# Vision Scene Analysis Exception Taxonomy
class SceneAnalysisError(VisionError):
    """Base application exception for all Vision Scene Analysis failures."""


class SceneAnalysisConfigurationError(SceneAnalysisError):
    """Raised when invalid Scene Analysis configuration parameters are supplied."""


class SceneAnalysisValidationError(SceneAnalysisError):
    """Raised when scene summary inputs or validation fail."""


class SceneAnalysisProcessingError(SceneAnalysisError):
    """Raised when scene classification or temporal analysis encounters an error."""


class SceneAnalysisTimeoutError(SceneAnalysisError):
    """Raised when scene analysis processing exceeds the configured timeout."""


# Vision Planner Context Exception Taxonomy
class VisionContextError(VisionError):
    """Base application exception for all Vision Planner Context failures."""


class VisionContextConfigurationError(VisionContextError):
    """Raised when invalid VisionContext configuration parameters are supplied."""


class VisionContextValidationError(VisionContextError):
    """Raised when vision context payload validation fails bounds or whitelists."""


class VisionContextProcessingError(VisionContextError):
    """Raised when vision context construction or serialization fails."""


class VisionContextTimeoutError(VisionContextError):
    """Raised when vision context building exceeds the configured timeout limit."""
