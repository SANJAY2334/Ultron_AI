"""Framework-Agnostic Embedding Provider Interface & Pipeline.

Provides abstract BaseEmbeddingProvider interface and concrete RouterEmbeddingProvider
utilizing the existing AIProviderRouter subsystem.
"""

import logging
from abc import ABC, abstractmethod

from app.ai.router import AIProviderRouter, ai_router
from app.memory.models import VectorEmbedding

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Raised when dense vector embedding generation fails."""


class BaseEmbeddingProvider(ABC):
    """Framework-agnostic abstract base class for vector embedding generation providers."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Returns the target embedding model identifier name."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Returns the expected vector embedding dimension."""

    @property
    @abstractmethod
    def version(self) -> str:
        """Returns the embedding model version string."""

    @abstractmethod
    async def embed_text(self, text: str) -> VectorEmbedding:
        """Generates a dense vector embedding for a single text string.

        Raises:
            EmbeddingError: If embedding generation fails.
        """

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[VectorEmbedding]:
        """Generates dense vector embeddings for a list of text strings.

        Raises:
            EmbeddingError: If batch embedding generation fails.
        """

    @abstractmethod
    async def health(self) -> bool:
        """Checks operational health of the embedding provider."""


class RouterEmbeddingProvider(BaseEmbeddingProvider):
    """Concrete BaseEmbeddingProvider adapter routing through AIProviderRouter."""

    def __init__(
        self,
        router: AIProviderRouter = ai_router,
        model_name: str = "text-embedding-3-small",
        dimension: int = 1536,
        version: str = "v1",
    ) -> None:
        """Initializes RouterEmbeddingProvider adapter."""
        self.router = router
        self._model_name = model_name
        self._dimension = dimension
        self._version = version

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def version(self) -> str:
        return self._version

    async def embed_text(self, text: str) -> VectorEmbedding:
        """Generates embedding vector using AIProviderRouter."""
        if not text or not text.strip():
            raise EmbeddingError("Cannot generate embedding for empty text payload.")

        try:
            vectors = await self.router.embed([text])
            if not vectors or not vectors[0]:
                raise EmbeddingError("AIProviderRouter returned empty embedding response.")

            vec = vectors[0]
            if len(vec) != self._dimension:
                raise EmbeddingError(
                    f"Generated vector dimension ({len(vec)}) does not match provider dimension ({self._dimension})."
                )

            return VectorEmbedding(
                vector=vec,
                model_name=self._model_name,
                dimension=self._dimension,
                version=self._version,
            )
        except Exception as exc:
            logger.error(f"RouterEmbeddingProvider failed to embed text: {exc}", exc_info=True)
            raise EmbeddingError(f"Embedding pipeline error: {exc}") from exc

    async def embed_batch(self, texts: list[str]) -> list[VectorEmbedding]:
        """Generates batch embedding vectors using AIProviderRouter."""
        if not texts:
            return []

        for i, txt in enumerate(texts):
            if not txt or not txt.strip():
                raise EmbeddingError(f"Text payload at index {i} is empty.")

        try:
            vectors = await self.router.embed(texts)
            if len(vectors) != len(texts):
                raise EmbeddingError(
                    f"Batch output count ({len(vectors)}) does not match input count ({len(texts)})."
                )

            results: list[VectorEmbedding] = []
            for vec in vectors:
                if len(vec) != self._dimension:
                    raise EmbeddingError(
                        f"Generated vector dimension ({len(vec)}) does not match provider dimension ({self._dimension})."
                    )
                results.append(
                    VectorEmbedding(
                        vector=vec,
                        model_name=self._model_name,
                        dimension=self._dimension,
                        version=self._version,
                    )
                )

            return results
        except Exception as exc:
            logger.error(f"RouterEmbeddingProvider failed batch embed: {exc}", exc_info=True)
            raise EmbeddingError(f"Batch embedding pipeline error: {exc}") from exc

    async def health(self) -> bool:
        """Probes health of registered embedding providers via AIProviderRouter."""
        try:
            health_map = await self.router.health()
            return any(health_map.values()) if health_map else True
        except Exception:
            return False
