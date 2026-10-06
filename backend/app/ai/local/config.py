"""Configuration Settings for Local Model Manager and LLM Runtime (Phase 4H.8).

Defines LocalModelConfig with explicit validations, sensible defaults, and safety bounds
preventing resource over-commitment or unsafe filesystem path handling.
"""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator


class LocalModelConfig(BaseModel):
    """Configuration schema governing local model management and runtime execution."""

    model_config = ConfigDict(extra="forbid")

    model_directory: str = Field(
        default="app/ai/local_models",
        min_length=1,
        description="Relative or absolute path to approved local model storage directory",
    )
    default_model_id: str = Field(
        default="ultron_distilgpt2",
        min_length=1,
        description="Default model identifier to load if unspecified",
    )
    max_context_tokens: int = Field(
        default=4096,
        ge=64,
        le=32768,
        description="Maximum permissible context length in tokens before rejecting request",
    )
    max_output_tokens: int = Field(
        default=1024,
        ge=1,
        le=4096,
        description="Maximum tokens generated per individual inference request",
    )
    max_concurrent_requests: int = Field(
        default=2,
        ge=1,
        le=8,
        description="Concurrency ceiling for simultaneous local LLM inference tasks",
    )
    model_load_timeout_sec: float = Field(
        default=30.0,
        gt=0.0,
        le=300.0,
        description="Maximum seconds allowed for loading model weights into memory",
    )
    inference_timeout_sec: float = Field(
        default=30.0,
        gt=0.0,
        le=300.0,
        description="Maximum seconds allowed for generation before timing out",
    )
    checksum_required: bool = Field(
        default=False,
        description="If True, any model missing an explicit verified sha256 checksum is rejected",
    )
    max_model_size_mb: float = Field(
        default=8192.0,
        ge=1.0,
        le=32768.0,
        description="Maximum single model file size on disk allowed to be loaded (MB)",
    )

    @field_validator("model_directory")
    @classmethod
    def validate_model_directory(cls, v: str) -> str:
        """Ensures model directory string is not empty or malformed."""
        clean = v.strip()
        if not clean:
            raise ValueError("model_directory cannot be empty.")
        return clean

    def resolve_model_dir(self, base_dir: Path | None = None) -> Path:
        """Resolves model_directory to an absolute, normalized Path object."""
        p = Path(self.model_directory)
        if p.is_absolute():
            return p.resolve()
        base = base_dir or Path.cwd()
        return (base / p).resolve()
