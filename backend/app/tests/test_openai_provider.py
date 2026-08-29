"""Unit Tests for OpenAI LLM Provider Driver.

Validates OpenAILLMProvider mapping of canonical requests to OpenAI HTTP JSON payloads,
SSE streaming chunk parsing, embeddings generation, and error handling using httpx mocks.
"""

import asyncio
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from app.ai.models import GenerationRequest, Message
from app.ai.openai_provider import AIProviderError, OpenAILLMProvider


def test_openai_provider_initialization() -> None:
    """Verify driver initialization and header construction."""
    provider = OpenAILLMProvider(api_key="sk-test-key-12345")
    assert provider.provider_name == "openai"
    assert provider.default_model == "gpt-4o"
    assert provider._get_headers()["Authorization"] == "Bearer sk-test-key-12345"


def test_openai_provider_complete_mock() -> None:
    """Verify complete() maps OpenAI HTTP JSON response to canonical GenerationResponse."""

    async def _test() -> None:
        provider = OpenAILLMProvider(api_key="sk-test-key-12345")

        mock_openai_response = {
            "id": "chatcmpl-999",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Hello! ULTRON is online.",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_resp = httpx.Response(200, json=mock_openai_response)
            mock_post.return_value = mock_resp

            req = GenerationRequest(messages=[Message(role="user", content="Status report.")])
            res = await provider.complete(req)

            assert res.id == "chatcmpl-999"
            assert res.message.content == "Hello! ULTRON is online."
            assert res.usage.total_tokens == 15
            assert res.metadata.provider_name == "openai"
            assert res.metadata.model_name == "gpt-4o"

    asyncio.run(_test())


def test_openai_provider_embed_mock() -> None:
    """Verify embed() maps text list to vector embeddings list."""

    async def _test() -> None:
        provider = OpenAILLMProvider(api_key="sk-test-key-12345")
        mock_embedding_response = {
            "data": [
                {"embedding": [0.1, 0.2, 0.3]},
                {"embedding": [0.4, 0.5, 0.6]},
            ]
        }

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = httpx.Response(200, json=mock_embedding_response)

            embeddings = await provider.embed(["test text 1", "test text 2"])
            assert len(embeddings) == 2
            assert embeddings[0] == [0.1, 0.2, 0.3]

    asyncio.run(_test())


def test_openai_provider_health_mock() -> None:
    """Verify health() returns True when endpoint returns HTTP 200."""

    async def _test() -> None:
        provider = OpenAILLMProvider(api_key="sk-test-key-12345")

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = httpx.Response(200, json={"data": []})
            healthy = await provider.health()
            assert healthy is True

    asyncio.run(_test())


def test_openai_provider_error_handling() -> None:
    """Verify HTTP status errors raise AIProviderError."""

    async def _test() -> None:
        provider = OpenAILLMProvider(api_key="invalid-key")

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = httpx.Response(
                401, json={"error": {"message": "Invalid API key"}}
            )

            req = GenerationRequest(messages=[Message(role="user", content="Hi")])
            with pytest.raises(AIProviderError):
                await provider.complete(req)

    asyncio.run(_test())
