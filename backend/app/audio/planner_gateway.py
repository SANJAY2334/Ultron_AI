"""Voice-to-Autonomous Planner Gateway Contracts and Event Definitions (Phase 4D).

Defines IVoicePlannerGateway interface, PlannerStreamEvent, VoicePlannerResponse,
and sanitized voice gateway exception hierarchy.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterable
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.audio.models import Transcript


class PlannerStreamEventType(StrEnum):
    """Event types for voice planner streaming responses."""

    THINKING = "THINKING"  # Operational status indicator (NO chain-of-thought)
    TOOL_STARTED = "TOOL_STARTED"  # Sanitized tool execution start indicator
    TOOL_COMPLETED = "TOOL_COMPLETED"  # Sanitized tool completion indicator
    PARTIAL_RESPONSE = "PARTIAL_RESPONSE"  # Incremental text chunk
    COMPLETED = "COMPLETED"  # Final response completed
    ERROR = "ERROR"  # Sanitized error message


class PlannerStreamEvent(BaseModel):
    """Sanitized event emitted during voice planner execution streaming."""

    event_type: PlannerStreamEventType = Field(description="Stream event type discriminator")
    payload: str = Field(description="Sanitized user-facing operational text or partial response")
    request_id: str = Field(description="Unique request tracking identifier")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Event timestamp"
    )

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware.")
        return v


class VoicePlannerResponse(BaseModel):
    """Canonical voice-safe response object returned by the planner gateway."""

    request_id: str = Field(description="Request identifier matching the input transcript")
    session_id: str = Field(description="Audio session identifier")
    correlation_id: str = Field(description="Tracing correlation identifier")
    text_response: str = Field(description="Voice-safe text response for TTS synthesis")
    success: bool = Field(default=True, description="True if planner execution succeeded")
    error_code: str | None = Field(
        default=None, description="Sanitized error code string if failed"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Response timestamp"
    )

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware.")
        return v


# Exception Taxonomy for Voice Planner Gateway
class VoiceGatewayError(Exception):
    """Base application exception for all Voice Planner Gateway failures."""


class VoiceGatewayConfigurationError(VoiceGatewayError):
    """Raised when VoicePlannerGateway configuration parameters are invalid."""


class VoiceGatewayPlannerError(VoiceGatewayError):
    """Raised when the underlying planner fails during voice request processing."""


class VoiceGatewayTimeoutError(VoiceGatewayError):
    """Raised when planner execution exceeds the voice gateway timeout limit."""


class VoiceGatewayCancelledError(VoiceGatewayError):
    """Raised when a voice planner request is cancelled due to barge-in or session termination."""


class IVoicePlannerGateway(ABC):
    """Abstract interface defining the boundary between Audio Session Manager and Autonomous Planner."""

    @abstractmethod
    async def submit_transcript(
        self,
        transcript: Transcript,
        user_id: str = "default_user",
        context: Any = None,
    ) -> VoicePlannerResponse:
        """Submits a voice transcript to the autonomous planner for execution and returns a voice response."""

    @abstractmethod
    def stream_response(
        self,
        transcript: Transcript,
        user_id: str = "default_user",
        context: Any = None,
    ) -> AsyncIterable[PlannerStreamEvent]:
        """Submits a transcript and streams incremental PlannerStreamEvent notifications."""

    @abstractmethod
    async def cancel_request(self, request_id: str) -> None:
        """Cancels an active voice planner request cleanly."""

    @abstractmethod
    async def health(self) -> dict[str, Any]:
        """Probes health and readiness status of the planner gateway."""
