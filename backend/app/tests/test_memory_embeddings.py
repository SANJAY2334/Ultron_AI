"""Unit Tests for Embedding Provider Interface & Pipeline.

Validates single text embedding, batch embedding, provider failure handling,
vector dimension validation, model/version metadata, and health probes.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from app.ai.router import AIProviderRouter
from app.memory.embeddings import (
    BaseEmbeddingProvider,
    EmbeddingError,
    RouterEmbeddingProvider,
)
from app.memory.models import VectorEmbedding


class MockEmbeddingProvider(BaseEmbeddingProvider):
    """Mock implementation of BaseEmbeddingProvider for testing."""

    def __init__(
        self,
        dimension: int = 4,
        model_name: str = "mock-embed",
        version: str = "v1",
        should_fail: bool = False,
    ) -> None:
        self._dimension = dimension
        self._model_name = model_name
        self._version = version
        self.should_fail = should_fail

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
        if self.should_fail:
            raise EmbeddingError("Simulated embedding provider failure")
        if not text or not text.strip():
            raise EmbeddingError("Empty text payload")
        return VectorEmbedding(
            vector=[0.1] * self._dimension,
            model_name=self._model_name,
            dimension=self._dimension,
            version=self._version,
        )

    async def embed_batch(self, texts: list[str]) -> list[VectorEmbedding]:
        if self.should_fail:
            raise EmbeddingError("Simulated embedding provider failure")
        results = []
        for txt in texts:
            results.append(await self.embed_text(txt))
        return results

    async def health(self) -> bool:
        return not self.should_fail


def test_mock_embedding_provider_single_and_batch() -> None:
    """Verify single and batch embedding generation."""

    async def _test() -> None:
        provider = MockEmbeddingProvider(dimension=4)

        # Single
        vec = await provider.embed_text("Test sentence")
        assert vec.dimension == 4
        assert vec.model_name == "mock-embed"
        assert vec.version == "v1"
        assert len(vec.vector) == 4

        # Batch
        batch = await provider.embed_batch(["Text 1", "Text 2"])
        assert len(batch) == 2
        assert batch[0].vector == [0.1, 0.1, 0.1, 0.1]

    asyncio.run(_test())


def test_embedding_provider_failure_and_dimension_validation() -> None:
    """Verify provider raises EmbeddingError on failure or empty text."""

    async def _test() -> None:
        failing_provider = MockEmbeddingProvider(should_fail=True)

        with pytest.raises(EmbeddingError, match="Simulated embedding provider failure"):
            await failing_provider.embed_text("Fail text")

        with pytest.raises(EmbeddingError, match="Empty text payload"):
            provider = MockEmbeddingProvider()
            await provider.embed_text("   ")

    asyncio.run(_test())


def test_router_embedding_provider_integration() -> None:
    """Verify RouterEmbeddingProvider integrates cleanly with AIProviderRouter mock."""

    async def _test() -> None:
        mock_router = AsyncMock(spec=AIProviderRouter)
        mock_router.embed.return_value = [[0.1, 0.2, 0.3, 0.4]]

        router_provider = RouterEmbeddingProvider(
            router=mock_router,
            model_name="test-model",
            dimension=4,
            version="v1.1",
        )

        assert router_provider.model_name == "test-model"
        assert router_provider.dimension == 4
        assert router_provider.version == "v1.1"

        vec = await router_provider.embed_text("Router text payload")
        assert vec.dimension == 4
        assert vec.vector == [0.1, 0.2, 0.3, 0.4]

        # Dimension mismatch error
        mock_router.embed.return_value = [[0.1, 0.2, 0.3]]  # 3 elements instead of 4
        with pytest.raises(
            EmbeddingError, match="Generated vector dimension \\(3\\) does not match"
        ):
            await router_provider.embed_text("Mismatch payload")

    asyncio.run(_test())
