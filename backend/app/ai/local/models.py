"""Canonical Local Model Management & LLM Runtime Schemas (Phase 4H.8).

Defines strongly typed, framework-agnostic Pydantic v2 domain models and enums for
ModelFormat, ModelLifecycleState, ModelMetadata, LLMRequest, and LLMResponse.
Enforces strict validation, immutable boundaries, and an actionable exception taxonomy.
"""

from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.ai.models import Message, ProviderMetadata, Usage


class ModelFormat(StrEnum):
    """Supported local model file and serialization formats."""

    GGUF = "GGUF"
    SAFETENSORS = "SAFETENSORS"
    ONNX = "ONNX"
    PYTORCH = "PYTORCH"
    CTRANSLATE2 = "CTRANSLATE2"


class ModelLifecycleState(StrEnum):
    """Deterministic lifecycle state of a managed local model."""

    DISCOVERED = "DISCOVERED"
    VALIDATING = "VALIDATING"
    VALID = "VALID"
    LOADING = "LOADING"
    LOADED = "LOADED"
    UNLOADING = "UNLOADING"
    UNAVAILABLE = "UNAVAILABLE"
    INVALID = "INVALID"
    FAILED = "FAILED"


class ModelMetadata(BaseModel):
    """Immutable metadata representation of a registered local language model."""

    model_config = ConfigDict(extra="forbid")

    model_id: str = Field(
        min_length=1, description="Unique, safe model identifier (e.g. 'ultron-local-tiny')"
    )
    name: str = Field(min_length=1, description="Human-readable model name")
    path: str = Field(
        min_length=1, description="Normalized filesystem path within approved model directory"
    )
    format: ModelFormat = Field(description="Model format classification")
    size_bytes: int = Field(ge=0, description="Total size of model weights on disk in bytes")
    quantization: str = Field(
        default="none", description="Quantization scheme (e.g. 'q4_k_m', 'int8', 'none')"
    )
    context_length: int = Field(gt=0, description="Maximum context window token capacity")
    sha256: str | None = Field(
        default=None, description="Expected or calculated SHA-256 checksum hex string"
    )
    status: ModelLifecycleState = Field(
        default=ModelLifecycleState.DISCOVERED, description="Current model lifecycle state"
    )
    runtime: str = Field(
        default="cpu", description="Target inference runtime engine (e.g. 'transformers-cpu')"
    )
    estimated_ram_mb: float = Field(
        gt=0.0, description="Estimated resident RAM footprint in megabytes"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Model registration timestamp"
    )

    @field_validator("created_at")
    @classmethod
    def validate_timezone(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware.")
        return v


class LLMRequest(BaseModel):
    """Strongly typed input request contract for local LLM inference."""

    model_config = ConfigDict(extra="forbid")

    prompt: str | None = Field(
        default=None, description="Raw text prompt if single-turn generation"
    )
    messages: list[Message] = Field(
        default_factory=list, description="Structured conversation message history"
    )
    max_tokens: int = Field(default=256, ge=1, le=4096, description="Maximum tokens to generate")
    temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="Sampling randomness temperature"
    )
    stop_sequences: list[str] = Field(
        default_factory=list, description="Sequences upon which generation terminates"
    )
    timeout_sec: float = Field(
        default=30.0, gt=0.0, le=300.0, description="Inference timeout ceiling"
    )
    correlation_id: str | None = Field(
        default=None, description="Context correlation ID for request tracing"
    )

    def get_effective_prompt(self) -> str:
        """Renders effective text prompt from prompt string or message sequence."""
        if self.prompt is not None and len(self.prompt.strip()) > 0:
            return self.prompt
        if self.messages:
            lines: list[str] = []
            for msg in self.messages:
                lines.append(f"{msg.role.upper()}: {msg.content}")
            return "\n".join(lines)
        return ""


class LLMResponse(BaseModel):
    """Strongly typed output completion contract returned from local LLM inference."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: f"llm_resp_{uuid4().hex[:12]}")
    text: str = Field(description="Generated completion text (UNTRUSTED MODEL OUTPUT)")
    model_id: str = Field(description="Identifier of model that produced the completion")
    usage: Usage = Field(default_factory=Usage, description="Token consumption metrics")
    metadata: ProviderMetadata = Field(description="Operational telemetry and execution metadata")
    finish_reason: str = Field(
        default="stop", description="Reason inference halted (e.g. 'stop', 'length')"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Response generation timestamp"
    )

    @field_validator("created_at")
    @classmethod
    def validate_timezone(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware.")
        return v


# Actionable Exception Hierarchy for Local Model Subsystem
class LocalModelError(Exception):
    """Base exception for all local model management and runtime failures."""


class ModelNotFoundError(LocalModelError):
    """Raised when a requested model_id does not exist in the registry."""


class ModelUnavailableError(LocalModelError):
    """Raised when a model exists in registry but is unloaded or physically unavailable."""


class ModelIntegrityError(LocalModelError):
    """Raised when model file checksum verification fails or model weights are corrupt."""


class ModelLoadError(LocalModelError):
    """Raised when loading model weights into memory fails."""


class ModelValidationError(LocalModelError):
    """Raised when model configuration, path, or format validation fails."""


class LLMInferenceError(LocalModelError):
    """Raised when execution of local model inference encounters a failure."""


class LLMTimeoutError(LocalModelError):
    """Raised when local model inference exceeds the configured timeout duration."""


class LLMContextLimitExceededError(LocalModelError):
    """Raised when input context exceeds the model or runtime context token capacity."""
