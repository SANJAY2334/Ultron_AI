"""Audio Capture Domain Models and Exception Taxonomy (Phase 4C.2).

Defines AudioDeviceInfo, AudioCaptureState, and sanitized domain exceptions for audio capture hardware.
"""

from enum import StrEnum

from pydantic import BaseModel, Field


class AudioCaptureState(StrEnum):
    """Lifecycle states of AudioCapture adapters."""

    STOPPED = "STOPPED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    ERROR = "ERROR"


class AudioDeviceInfo(BaseModel):
    """Information model representing a physical or virtual audio input device."""

    device_id: str = Field(description="Unique device string identifier or index")
    name: str = Field(description="Human-readable device name")
    input_channels: int = Field(ge=0, description="Available input channels count")
    sample_rates: list[int] = Field(
        default_factory=list, description="Supported sample rates in Hz"
    )
    default_sample_rate: int = Field(default=16000, description="Default device sample rate")
    is_default: bool = Field(default=False, description="True if device is system default input")
    is_available: bool = Field(default=True, description="True if device is currently operational")


# Domain Exception Taxonomy for Audio Capture
class AudioCaptureError(Exception):
    """Base application exception for all audio capture subsystem errors."""


class AudioDeviceNotFoundError(AudioCaptureError):
    """Raised when a requested audio input device_id cannot be located."""


class AudioCaptureUnavailableError(AudioCaptureError):
    """Raised when no hardware microphone device is available on the system."""


class AudioCaptureConfigurationError(AudioCaptureError):
    """Raised when hardware cannot support the requested AudioStreamConfig or format."""


class AudioCaptureDataError(AudioCaptureError):
    """Raised when captured raw audio bytes violate payload integrity or frame calculations."""
