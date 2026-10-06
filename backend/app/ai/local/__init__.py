"""Local Model Management and LLM Runtime Subsystem (Phase 4H.8).

Provides secure local model discovery, integrity verification, lifecycle control,
and hardware-aware non-blocking local inference.
"""

from app.ai.local.config import LocalModelConfig
from app.ai.local.model_manager import IModelManager, ModelManager
from app.ai.local.models import (
    LLMContextLimitExceededError,
    LLMInferenceError,
    LLMRequest,
    LLMResponse,
    LLMTimeoutError,
    LocalModelError,
    ModelFormat,
    ModelIntegrityError,
    ModelLifecycleState,
    ModelLoadError,
    ModelMetadata,
    ModelNotFoundError,
    ModelUnavailableError,
    ModelValidationError,
)
from app.ai.local.runtime_provider import (
    ILocalLLMProvider,
    LocalRuntimeProvider,
    MockLocalLLMProvider,
)

__all__ = [
    "IModelManager",
    "ModelManager",
    "ILocalLLMProvider",
    "LocalRuntimeProvider",
    "MockLocalLLMProvider",
    "LocalModelConfig",
    "ModelFormat",
    "ModelLifecycleState",
    "ModelMetadata",
    "LLMRequest",
    "LLMResponse",
    "LocalModelError",
    "ModelNotFoundError",
    "ModelUnavailableError",
    "ModelIntegrityError",
    "ModelLoadError",
    "ModelValidationError",
    "LLMInferenceError",
    "LLMTimeoutError",
    "LLMContextLimitExceededError",
]
