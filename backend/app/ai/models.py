"""Canonical AI Domain Models.

Provides provider-independent, strongly-typed Pydantic v2 schemas used across ULTRON.
Ensures zero provider-specific schemas leak outside of provider implementations.
"""

import uuid
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """Canonical representation of an LLM tool/function call request."""

    id: str = Field(default_factory=lambda: f"call_{uuid.uuid4().hex[:12]}")
    type: Literal["function"] = "function"
    function_name: str = Field(description="Name of the target tool to invoke")
    arguments: dict[str, Any] = Field(
        default_factory=dict, description="Parsed keyword arguments for tool execution"
    )


class ToolResultReference(BaseModel):
    """Canonical reference to the output of an executed tool call."""

    tool_call_id: str = Field(description="Associated ToolCall identifier")
    tool_name: str = Field(description="Name of executed tool")
    status: Literal["success", "error", "denied"] = Field(description="Tool execution status")
    output: Any = Field(description="Serialized tool result output")
    execution_time_ms: float = Field(default=0.0, description="Tool execution duration in ms")


class Message(BaseModel):
    """Canonical conversation message representation."""

    id: str = Field(default_factory=lambda: f"msg_{uuid.uuid4().hex[:12]}")
    role: Literal["system", "user", "assistant", "tool"] = Field(description="Message author role")
    content: str = Field(default="", description="Text content payload of message")
    name: str | None = Field(default=None, description="Optional name identifier of author")
    tool_call_id: str | None = Field(
        default=None, description="Associated tool_call_id if role=='tool'"
    )
    tool_calls: list[ToolCall] = Field(
        default_factory=list, description="Requested tool calls if role=='assistant'"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Message creation timestamp"
    )


class Usage(BaseModel):
    """Token usage metrics and financial cost estimation for an LLM call."""

    prompt_tokens: int = Field(default=0, ge=0, description="Input prompt tokens consumed")
    completion_tokens: int = Field(
        default=0, ge=0, description="Output completion tokens generated"
    )
    total_tokens: int = Field(default=0, ge=0, description="Total tokens consumed")
    estimated_cost_usd: float = Field(
        default=0.0, ge=0.0, description="Estimated financial cost in USD"
    )


class ProviderMetadata(BaseModel):
    """Telemetry and execution metadata returned by an AI Provider."""

    provider_name: str = Field(description="Name of provider engine (e.g., 'openai', 'anthropic')")
    model_name: str = Field(description="Name of specific LLM model used")
    latency_ms: float = Field(ge=0.0, description="End-to-end provider API response latency in ms")
    retries: int = Field(default=0, ge=0, description="Number of execution retry attempts")
    finish_reason: str | None = Field(
        default="stop", description="Provider finish reason (stop, length, tool_calls)"
    )


class GenerationRequest(BaseModel):
    """Canonical request payload for LLM text generation or tool selection."""

    messages: list[Message] = Field(description="Ordered list of conversation messages")
    model: str | None = Field(default=None, description="Target model override")
    temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="Sampling temperature randomness"
    )
    max_tokens: int | None = Field(default=None, ge=1, description="Maximum tokens to generate")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Nucleus sampling threshold")
    stop_sequences: list[str] = Field(
        default_factory=list, description="Optional stop sequence triggers"
    )
    tools: list[dict[str, Any]] = Field(
        default_factory=list, description="Available tool JSON schemas for function calling"
    )
    stream: bool = Field(default=False, description="Enable real-time token streaming")
    correlation_id: str | None = Field(
        default=None, description="Context correlation ID for request tracing"
    )


class GenerationResponse(BaseModel):
    """Canonical response payload returned by LLM completion generators."""

    id: str = Field(default_factory=lambda: f"gen_{uuid.uuid4().hex[:12]}")
    message: Message = Field(description="Generated assistant response message")
    usage: Usage = Field(default_factory=Usage, description="Token consumption metrics")
    metadata: ProviderMetadata = Field(description="Provider telemetry and timing metadata")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class StreamChunk(BaseModel):
    """Canonical chunk yielded during real-time Server-Sent Events (SSE) streaming."""

    id: str = Field(default_factory=lambda: f"chunk_{uuid.uuid4().hex[:12]}")
    content_delta: str = Field(default="", description="Incremental text content snippet")
    tool_call_delta: ToolCall | None = Field(
        default=None, description="Incremental tool call fragment"
    )
    finish_reason: str | None = Field(default=None, description="Finish reason if chunk is final")
    usage: Usage | None = Field(default=None, description="Final token usage metrics if complete")


class Conversation(BaseModel):
    """Canonical multi-turn conversation session object."""

    id: str = Field(default_factory=lambda: f"conv_{uuid.uuid4().hex[:12]}")
    messages: list[Message] = Field(default_factory=list, description="Ordered conversation turns")
    system_prompt: str | None = Field(
        default=None, description="System instructions guiding persona"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Session state variables and user metadata"
    )
