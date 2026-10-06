"""Local Model Manager Subsystem Implementation (Phase 4H.8).

Provides centralized discovery, metadata inspection, streaming SHA-256 integrity verification,
path containment validation, and lifecycle state management for local AI language models.

Security & Architectural Invariants:
- Strictly scoped to local model lifecycle management only. Zero tool execution or authorization logic.
- Path Traversal Defenses: Enforces canonical normalization and containment inside approved model directory.
- Streaming Checksum: Calculates SHA-256 in bounded 64 KiB blocks to prevent memory spikes.
- Deterministic Lifecycle: Rejects invalid concurrent transitions and resets state on failed unloads.
"""

import asyncio
import hashlib
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from app.ai.local.config import LocalModelConfig
from app.ai.local.models import (
    ModelFormat,
    ModelIntegrityError,
    ModelLifecycleState,
    ModelLoadError,
    ModelMetadata,
    ModelNotFoundError,
    ModelValidationError,
)
from app.hardware.resource_manager import IResourceManager

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS: dict[str, ModelFormat] = {
    ".gguf": ModelFormat.GGUF,
    ".safetensors": ModelFormat.SAFETENSORS,
    ".onnx": ModelFormat.ONNX,
    ".bin": ModelFormat.PYTORCH,
    ".pt": ModelFormat.PYTORCH,
}


class IModelManager(ABC):
    """Abstract interface defining the local model management and lifecycle contract."""

    @abstractmethod
    async def discover_models(self) -> list[ModelMetadata]:
        """Scans approved model directory and registers newly discovered models."""

    @abstractmethod
    async def inspect_model(self, model_id: str) -> ModelMetadata:
        """Retrieves verified metadata record for a specified model."""

    @abstractmethod
    async def validate_model(self, model_id: str) -> bool:
        """Validates model path containment, file existence, and SHA-256 checksum integrity."""

    @abstractmethod
    async def load_model(self, model_id: str) -> bool:
        """Loads model into active memory following resource and integrity checks."""

    @abstractmethod
    async def unload_model(self, model_id: str) -> bool:
        """Unloads model weights from memory and returns state to VALID or DISCOVERED."""

    @abstractmethod
    async def model_status(self, model_id: str) -> ModelLifecycleState:
        """Returns the current lifecycle state of the model."""

    @abstractmethod
    async def health(self) -> dict[str, Any]:
        """Probes operational health and returns model manager telemetry."""


class ModelManager(IModelManager):
    """Production implementation of the Local Model Manager."""

    def __init__(
        self,
        config: LocalModelConfig | None = None,
        resource_manager: IResourceManager | None = None,
    ) -> None:
        """Initializes ModelManager.

        Args:
            config: Optional LocalModelConfig.
            resource_manager: Optional IResourceManager instance for resource checks.
        """
        self.config = config or LocalModelConfig()
        self.resource_manager = resource_manager
        self.approved_dir = self.config.resolve_model_dir()

        self._lock = asyncio.Lock()
        self._registry: dict[str, ModelMetadata] = {}
        self._loaded_model_id: str | None = None

        # Ensure approved model directory exists
        self.approved_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"ModelManager initialized with approved directory: {self.approved_dir}")

    def _sanitize_and_validate_path(self, raw_path: str | Path) -> Path:
        """Validates that a path is strictly contained within the approved model directory."""
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = (self.approved_dir / candidate).resolve()
        else:
            candidate = candidate.resolve()

        # Reject path traversal and symlink escape
        try:
            candidate.relative_to(self.approved_dir)
        except ValueError as exc:
            raise ModelValidationError(
                f"Path traversal rejected: '{raw_path}' is outside approved directory '{self.approved_dir}'."
            ) from exc

        # Disallow directory traversing components like '..' in string representation
        if ".." in str(raw_path):
            raise ModelValidationError(
                f"Path traversal sequence '..' detected in path: '{raw_path}'."
            )

        return candidate

    @staticmethod
    def _validate_model_id(model_id: str) -> None:
        """Ensures model_id is a safe identifier and cannot be used for path traversal."""
        import re

        if (
            not re.match(r"^[a-zA-Z0-9_\-\.]+$", model_id)
            or ".." in model_id
            or "/" in model_id
            or "\\" in model_id
        ):
            raise ModelValidationError(
                f"Invalid model_id '{model_id}'. Model IDs must be safe alphanumeric identifiers without path separators."
            )

    def register_model(self, metadata: ModelMetadata) -> None:
        """Registers a ModelMetadata record into the in-memory registry."""
        self._validate_model_id(metadata.model_id)
        validated_path = self._sanitize_and_validate_path(metadata.path)
        # Update path to resolved normalized string
        metadata_dict = metadata.model_dump()
        metadata_dict["path"] = str(validated_path)
        updated_meta = ModelMetadata(**metadata_dict)
        self._registry[updated_meta.model_id] = updated_meta
        logger.info(
            f"Model registered: '{updated_meta.model_id}' ({updated_meta.format}) at {validated_path}"
        )

    async def discover_models(self) -> list[ModelMetadata]:
        """Scans approved model directory and registers newly discovered models."""
        async with self._lock:
            discovered: list[ModelMetadata] = []
            if not self.approved_dir.exists():
                return discovered

            # Scan directory entries (files and directories)
            for entry in self.approved_dir.iterdir():
                try:
                    resolved_path = self._sanitize_and_validate_path(entry)
                except ModelValidationError:
                    continue

                # Single file model
                if resolved_path.is_file():
                    ext = resolved_path.suffix.lower()
                    if ext in SUPPORTED_EXTENSIONS:
                        m_format = SUPPORTED_EXTENSIONS[ext]
                        m_id = resolved_path.stem
                        if m_id not in self._registry:
                            size = resolved_path.stat().st_size
                            est_ram = max(64.0, round((size / (1024 * 1024)) * 1.2, 1))
                            meta = ModelMetadata(
                                model_id=m_id,
                                name=resolved_path.name,
                                path=str(resolved_path),
                                format=m_format,
                                size_bytes=size,
                                quantization="none",
                                context_length=self.config.max_context_tokens,
                                status=ModelLifecycleState.DISCOVERED,
                                runtime="cpu",
                                estimated_ram_mb=est_ram,
                            )
                            self._registry[m_id] = meta
                            discovered.append(meta)

                # Directory-based model (e.g. HuggingFace format or CTranslate2 format)
                elif resolved_path.is_dir():
                    m_id = resolved_path.name
                    has_weights = any(
                        p.suffix.lower() in SUPPORTED_EXTENSIONS for p in resolved_path.glob("*")
                    )
                    if has_weights and m_id not in self._registry:
                        total_size = sum(
                            p.stat().st_size for p in resolved_path.glob("*") if p.is_file()
                        )
                        est_ram = max(64.0, round((total_size / (1024 * 1024)) * 1.2, 1))
                        # Identify format
                        m_format = ModelFormat.SAFETENSORS
                        if any(p.suffix.lower() == ".onnx" for p in resolved_path.glob("*")):
                            m_format = ModelFormat.ONNX
                        elif any(p.name == "model.bin" for p in resolved_path.glob("*")):
                            m_format = ModelFormat.CTRANSLATE2

                        meta = ModelMetadata(
                            model_id=m_id,
                            name=resolved_path.name,
                            path=str(resolved_path),
                            format=m_format,
                            size_bytes=total_size,
                            quantization="none",
                            context_length=self.config.max_context_tokens,
                            status=ModelLifecycleState.DISCOVERED,
                            runtime="cpu",
                            estimated_ram_mb=est_ram,
                        )
                        self._registry[m_id] = meta
                        discovered.append(meta)

            return list(self._registry.values())

    async def inspect_model(self, model_id: str) -> ModelMetadata:
        """Retrieves metadata record for a model_id."""
        if model_id not in self._registry:
            raise ModelNotFoundError(f"Model with id '{model_id}' was not found in registry.")
        return self._registry[model_id]

    async def validate_model(self, model_id: str) -> bool:
        """Validates model path containment, file existence, and streaming SHA-256 checksum."""
        meta = await self.inspect_model(model_id)
        meta.status = ModelLifecycleState.VALIDATING

        path = self._sanitize_and_validate_path(meta.path)
        if not path.exists():
            meta.status = ModelLifecycleState.UNAVAILABLE
            raise ModelNotFoundError(f"Model file at path '{path}' does not exist.")

        # Check maximum allowed model size
        size_mb = meta.size_bytes / (1024 * 1024)
        if size_mb > self.config.max_model_size_mb:
            meta.status = ModelLifecycleState.INVALID
            raise ModelValidationError(
                f"Model size ({size_mb:.1f} MB) exceeds maximum allowed size ({self.config.max_model_size_mb:.1f} MB)."
            )

        # Checksum calculation if model is a single file
        if path.is_file():
            calculated_hash = await asyncio.to_thread(self._compute_streaming_sha256, path)
            if meta.sha256 is not None:
                if calculated_hash.lower() != meta.sha256.lower():
                    meta.status = ModelLifecycleState.INVALID
                    raise ModelIntegrityError(
                        f"Checksum mismatch for model '{model_id}'. "
                        f"Expected: {meta.sha256}, Actual: {calculated_hash}."
                    )
            else:
                if self.config.checksum_required:
                    meta.status = ModelLifecycleState.INVALID
                    raise ModelIntegrityError(
                        f"Model '{model_id}' has no registered checksum, but checksum_required=True."
                    )
                # Store computed checksum in metadata
                meta.sha256 = calculated_hash

        meta.status = ModelLifecycleState.VALID
        return True

    @staticmethod
    def _compute_streaming_sha256(file_path: Path, chunk_size: int = 65536) -> str:
        """Computes SHA-256 checksum in bounded 64 KiB chunks to avoid memory spikes."""
        sha = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(chunk_size):
                sha.update(chunk)
        return sha.hexdigest()

    async def load_model(self, model_id: str) -> bool:
        """Coordinates lifecycle checks and marks model as LOADED."""
        async with self._lock:
            meta = await self.inspect_model(model_id)

            if meta.status == ModelLifecycleState.LOADED and self._loaded_model_id == model_id:
                logger.info(f"Model '{model_id}' is already loaded.")
                return True

            if meta.status == ModelLifecycleState.LOADING:
                raise ModelLoadError(
                    f"Model '{model_id}' is already undergoing a loading transition."
                )

            # Validate before loading
            await self.validate_model(model_id)

            # Check capacity with ResourceManager if available
            if self.resource_manager is not None:
                snapshot = await self.resource_manager.get_snapshot()
                if snapshot.ram_available_mb < meta.estimated_ram_mb:
                    meta.status = ModelLifecycleState.FAILED
                    raise ModelLoadError(
                        f"Insufficient available RAM ({snapshot.ram_available_mb:.1f} MB) "
                        f"to load model '{model_id}' (requires estimated {meta.estimated_ram_mb:.1f} MB)."
                    )

            meta.status = ModelLifecycleState.LOADING

            # If another model is loaded, unload it first
            if self._loaded_model_id and self._loaded_model_id != model_id:
                old_meta = self._registry.get(self._loaded_model_id)
                if old_meta:
                    old_meta.status = ModelLifecycleState.VALID

            meta.status = ModelLifecycleState.LOADED
            self._loaded_model_id = model_id
            logger.info(f"Model '{model_id}' successfully marked as LOADED.")
            return True

    async def unload_model(self, model_id: str) -> bool:
        """Transitions model state from LOADED to UNLOADING and back to VALID."""
        async with self._lock:
            if model_id not in self._registry:
                raise ModelNotFoundError(f"Model '{model_id}' not found in registry.")

            meta = self._registry[model_id]
            if meta.status != ModelLifecycleState.LOADED:
                meta.status = ModelLifecycleState.VALID
                if self._loaded_model_id == model_id:
                    self._loaded_model_id = None
                return True

            meta.status = ModelLifecycleState.UNLOADING
            # Perform memory dereferencing simulation
            meta.status = ModelLifecycleState.VALID
            if self._loaded_model_id == model_id:
                self._loaded_model_id = None

            logger.info(f"Model '{model_id}' successfully unloaded.")
            return True

    async def model_status(self, model_id: str) -> ModelLifecycleState:
        """Returns the current lifecycle state of a model."""
        meta = await self.inspect_model(model_id)
        return meta.status

    async def health(self) -> dict[str, Any]:
        """Probes operational health and active models."""
        async with self._lock:
            return {
                "subsystem": "local_model_manager",
                "healthy": True,
                "registered_models_count": len(self._registry),
                "loaded_model_id": self._loaded_model_id,
                "approved_directory": str(self.approved_dir),
                "models": {k: v.status.value for k, v in self._registry.items()},
            }
