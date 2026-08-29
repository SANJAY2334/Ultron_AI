"""Unit Tests for Abstract Base LLM Provider Interface.

Validates that concrete drivers correctly inherit and implement BaseLLMProvider contracts.
"""

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

from app.ai.base import BaseLLMProvider
from app.ai.models import (
    GenerationRequest,
    GenerationResponse,
    Message,
    ProviderMetadata,
    StreamChunk,
    Usage,
)


class ConcreteTestProvider(BaseLLMProvider):
    """Dummy concrete provider for interface testing."""

    @property
    def provider_name(self) -> str:
        return "test_provider"

    @property
    def default_model(self) -> str:
        return "test-model-v1"

    async def complete(self, request: GenerationRequest) -> GenerationResponse:
        return GenerationResponse(
            message=Message(role="assistant", content="Test response"),
            usage=Usage(prompt_tokens=5, completion_tokens=2, total_tokens=7),
            metadata=ProviderMetadata(
                provider_name=self.provider_name,
                model_name=self.default_model,
                latency_ms=10.0,
            ),
        )

    async def stream(self, request: GenerationRequest) -> AsyncGenerator[StreamChunk, None]:
        yield StreamChunk(content_delta="Test ")
        yield StreamChunk(content_delta="response")

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]

    async def count_tokens(self, input_data: str | list[Message]) -> int:
        if isinstance(input_data, str):
            return len(input_data) // 4
        return sum(len(m.content) // 4 for m in input_data)

    async def health(self) -> bool:
        return True

    def capabilities(self) -> dict[str, Any]:
        return {"streaming": True, "function_calling": True, "vision": False}


def test_concrete_provider_implementation() -> None:
    """Verify concrete provider implementation fulfills interface methods."""

    async def _test() -> None:
        provider = ConcreteTestProvider()
        assert provider.provider_name == "test_provider"
        assert provider.default_model == "test-model-v1"

        # Test Complete
        req = GenerationRequest(messages=[Message(role="user", content="Hi")])
        res = await provider.complete(req)
        assert res.message.content == "Test response"
        assert res.usage.total_tokens == 7

        # Test Stream
        chunks = []
        async for chunk in provider.stream(req):
            chunks.append(chunk.content_delta)
        assert "".join(chunks) == "Test response"

        # Test Embed
        embeddings = await provider.embed(["hello", "world"])
        assert len(embeddings) == 2
        assert embeddings[0] == [0.1, 0.2, 0.3]

        # Test Count Tokens
        token_count = await provider.count_tokens("Hello world token count test")
        assert token_count > 0

        # Test Health & Capabilities
        healthy = await provider.health()
        assert healthy is True
        caps = provider.capabilities()
        assert caps["streaming"] is True

    asyncio.run(_test())
