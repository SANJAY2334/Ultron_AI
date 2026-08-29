"""OpenAI LLM Provider Driver.

Concrete implementation of BaseLLMProvider communicating with OpenAI API endpoints using httpx.
Translates provider-specific JSON requests and responses to/from canonical AI domain models.
"""

import json
import time
from collections.abc import AsyncGenerator
from typing import Any

import httpx
from pydantic import SecretStr

from app.ai.base import BaseLLMProvider
from app.ai.models import (
    GenerationRequest,
    GenerationResponse,
    Message,
    ProviderMetadata,
    StreamChunk,
    ToolCall,
    Usage,
)


class AIProviderError(Exception):
    """Raised when an AI provider call fails or returns an HTTP error status."""


class OpenAILLMProvider(BaseLLMProvider):
    """OpenAI LLM Provider Driver implementing BaseLLMProvider interface."""

    def __init__(
        self,
        api_key: str | SecretStr,
        base_url: str = "https://api.openai.com/v1",
        default_model: str = "gpt-4o",
        timeout_seconds: float = 60.0,
    ) -> None:
        """Initializes the OpenAI LLM Provider Driver.

        Args:
            api_key: OpenAI API key string or SecretStr.
            base_url: Base API URL endpoint.
            default_model: Default OpenAI model identifier.
            timeout_seconds: HTTP request timeout in seconds.
        """
        self._api_key = api_key.get_secret_value() if isinstance(api_key, SecretStr) else api_key
        self._base_url = base_url.rstrip("/")
        self._default_model = default_model
        self._timeout = timeout_seconds

    @property
    def provider_name(self) -> str:
        """Returns the provider driver identifier."""
        return "openai"

    @property
    def default_model(self) -> str:
        """Returns the default model identifier."""
        return self._default_model

    def _get_headers(self) -> dict[str, str]:
        """Constructs HTTP headers required for OpenAI API calls."""
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _convert_messages_to_provider_format(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Converts canonical Message objects to OpenAI message dictionaries."""
        formatted: list[dict[str, Any]] = []
        for msg in messages:
            item: dict[str, Any] = {"role": msg.role, "content": msg.content}
            if msg.name:
                item["name"] = msg.name
            if msg.tool_call_id:
                item["tool_call_id"] = msg.tool_call_id
            if msg.tool_calls:
                item["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function_name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in msg.tool_calls
                ]
            formatted.append(item)
        return formatted

    async def complete(self, request: GenerationRequest) -> GenerationResponse:
        """Asynchronously executes an OpenAI chat completion request."""
        start_time = time.perf_counter()
        target_model = request.model or self._default_model

        payload: dict[str, Any] = {
            "model": target_model,
            "messages": self._convert_messages_to_provider_format(request.messages),
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stream": False,
        }
        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens
        if request.stop_sequences:
            payload["stop"] = request.stop_sequences
        if request.tools:
            payload["tools"] = request.tools

        url = f"{self._base_url}/chat/completions"

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                response = await client.post(url, headers=self._get_headers(), json=payload)
                if response.is_error:
                    raise AIProviderError(
                        f"OpenAI HTTP error status {response.status_code}: {response.text}"
                    )
                data = response.json()
            except AIProviderError:
                raise
            except Exception as exc:
                raise AIProviderError(f"OpenAI connection error: {exc}") from exc

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        choice = data["choices"][0]
        msg_data = choice["message"]

        tool_calls: list[ToolCall] = []
        if "tool_calls" in msg_data and msg_data["tool_calls"]:
            for tc in msg_data["tool_calls"]:
                fn = tc.get("function", {})
                args_str = fn.get("arguments", "{}")
                try:
                    args = json.loads(args_str) if isinstance(args_str, str) else args_str
                except Exception:
                    args = {}
                tool_calls.append(
                    ToolCall(
                        id=tc.get("id", ""),
                        function_name=fn.get("name", ""),
                        arguments=args,
                    )
                )

        response_message = Message(
            role="assistant",
            content=msg_data.get("content") or "",
            tool_calls=tool_calls,
        )

        usage_data = data.get("usage", {})
        prompt_tokens = usage_data.get("prompt_tokens", 0)
        completion_tokens = usage_data.get("completion_tokens", 0)
        total_tokens = usage_data.get("total_tokens", prompt_tokens + completion_tokens)

        # Basic cost calculation estimate ($0.005 per 1k prompt, $0.015 per 1k completion for gpt-4o)
        cost_usd = (prompt_tokens * 0.000005) + (completion_tokens * 0.000015)

        return GenerationResponse(
            id=data.get("id", ""),
            message=response_message,
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                estimated_cost_usd=cost_usd,
            ),
            metadata=ProviderMetadata(
                provider_name=self.provider_name,
                model_name=target_model,
                latency_ms=elapsed_ms,
                finish_reason=choice.get("finish_reason", "stop"),
            ),
        )

    async def stream(self, request: GenerationRequest) -> AsyncGenerator[StreamChunk, None]:
        """Asynchronously streams generated tokens from OpenAI SSE endpoint."""
        target_model = request.model or self._default_model

        payload: dict[str, Any] = {
            "model": target_model,
            "messages": self._convert_messages_to_provider_format(request.messages),
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stream": True,
        }
        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens
        if request.stop_sequences:
            payload["stop"] = request.stop_sequences

        url = f"{self._base_url}/chat/completions"

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                async with client.stream(
                    "POST", url, headers=self._get_headers(), json=payload
                ) as response:
                    if response.is_error:
                        raise AIProviderError(f"OpenAI SSE error status {response.status_code}")
                    async for line in response.aiter_lines():
                        if not line or line.startswith(":"):
                            continue
                        if line.startswith("data: "):
                            data_str = line[6:].strip()
                            if data_str == "[DONE]":
                                break
                            try:
                                chunk_json = json.loads(data_str)
                                choice = chunk_json["choices"][0]
                                delta = choice.get("delta", {})
                                content = delta.get("content") or ""
                                finish_reason = choice.get("finish_reason")

                                yield StreamChunk(
                                    id=chunk_json.get("id", ""),
                                    content_delta=content,
                                    finish_reason=finish_reason,
                                )
                            except Exception:
                                continue
            except AIProviderError:
                raise
            except Exception as exc:
                raise AIProviderError(f"OpenAI stream error: {exc}") from exc

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Asynchronously generates vector embeddings using text-embedding-3-small."""
        url = f"{self._base_url}/embeddings"
        payload = {
            "model": "text-embedding-3-small",
            "input": texts,
        }

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                response = await client.post(url, headers=self._get_headers(), json=payload)
                if response.is_error:
                    raise AIProviderError(
                        f"OpenAI embeddings error status {response.status_code}: {response.text}"
                    )
                data = response.json()
                return [item["embedding"] for item in data["data"]]
            except AIProviderError:
                raise
            except Exception as exc:
                raise AIProviderError(f"OpenAI embeddings error: {exc}") from exc

    async def count_tokens(self, input_data: str | list[Message]) -> int:
        """Estimates token count for input text or list of canonical messages."""
        if isinstance(input_data, str):
            # Fallback estimation (~4 chars per token)
            return len(input_data) // 4
        total_chars = sum(len(m.content) for m in input_data)
        return total_chars // 4

    async def health(self) -> bool:
        """Probes OpenAI API health connectivity."""
        url = f"{self._base_url}/models"
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.get(url, headers=self._get_headers())
                return response.status_code == 200
            except Exception:
                return False

    def capabilities(self) -> dict[str, Any]:
        """Returns capabilities matrix for OpenAI driver."""
        return {
            "streaming": True,
            "function_calling": True,
            "embeddings": True,
            "vision": True,
            "json_mode": True,
        }
