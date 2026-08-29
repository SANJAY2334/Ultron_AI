"""Resilient AI Provider Router.

Manages primary, secondary, and local fallback BaseLLMProvider drivers wrapped in CircuitBreakers.
Guarantees uninterrupted AI availability with automated failover and telemetry logging.
"""

import logging
from collections.abc import AsyncGenerator

from app.ai.base import BaseLLMProvider
from app.ai.models import (
    GenerationRequest,
    GenerationResponse,
    Message,
    StreamChunk,
)
from app.ai.openai_provider import AIProviderError
from app.core.circuit_breaker import CircuitBreaker, CircuitBreakerOpenError

logger = logging.getLogger(__name__)


class AllAIProvidersFailedError(Exception):
    """Raised when all registered AI Provider drivers fail or are circuit-broken."""


class AIProviderRouter:
    """Resilient router managing multi-provider failover and circuit breaker protection."""

    def __init__(self) -> None:
        """Initializes an empty AI Provider Router."""
        self._providers: list[BaseLLMProvider] = []
        self._circuit_breakers: dict[str, CircuitBreaker] = {}

    def register_provider(
        self,
        provider: BaseLLMProvider,
        failure_threshold: int = 3,
        recovery_timeout: float = 30.0,
    ) -> None:
        """Registers an AI LLM Provider driver in the router priority chain.

        Args:
            provider: Concrete BaseLLMProvider instance.
            failure_threshold: Circuit breaker failure threshold.
            recovery_timeout: Circuit breaker recovery timeout in seconds.
        """
        name = provider.provider_name
        self._providers.append(provider)
        self._circuit_breakers[name] = CircuitBreaker(
            name=f"cb_{name}",
            failure_threshold=failure_threshold,
            recovery_timeout_seconds=recovery_timeout,
        )
        logger.info(f"Registered AI Provider driver '{name}' (Priority #{len(self._providers)}).")

    async def complete(self, request: GenerationRequest) -> GenerationResponse:
        """Executes a generation request with automatic multi-provider failover.

        Args:
            request: Canonical GenerationRequest object.

        Returns:
            GenerationResponse: Canonical response from the first healthy provider.

        Raises:
            AllAIProvidersFailedError: If every registered provider fails.
        """
        if not self._providers:
            raise AllAIProvidersFailedError(
                "No AI Provider drivers registered in AIProviderRouter."
            )

        last_error: Exception | None = None
        retries = 0

        for provider in self._providers:
            name = provider.provider_name
            cb = self._circuit_breakers[name]

            try:
                logger.info(f"Routing generation request to provider '{name}'...")
                target_p = provider

                async def _call_complete(p: BaseLLMProvider = target_p) -> GenerationResponse:
                    return await p.complete(request)

                response = await cb.call(_call_complete)
                response.metadata.retries = retries
                return response
            except (CircuitBreakerOpenError, AIProviderError, Exception) as exc:
                logger.warning(
                    f"Provider '{name}' call failed ({exc}). Failing over to next provider..."
                )
                last_error = exc
                retries += 1

        raise AllAIProvidersFailedError(
            f"All registered AI providers failed. Last exception: {last_error}"
        ) from last_error

    async def stream(self, request: GenerationRequest) -> AsyncGenerator[StreamChunk, None]:
        """Streams generation response chunks with multi-provider failover."""
        if not self._providers:
            raise AllAIProvidersFailedError(
                "No AI Provider drivers registered in AIProviderRouter."
            )

        last_error: Exception | None = None

        for provider in self._providers:
            name = provider.provider_name
            cb = self._circuit_breakers[name]

            try:
                if cb.state == cb.state.OPEN:
                    logger.warning(f"CircuitBreaker '{name}' is OPEN. Skipping in stream chain.")
                    continue

                async for chunk in provider.stream(request):
                    yield chunk
                return
            except Exception as exc:
                logger.warning(
                    f"Provider '{name}' streaming failed ({exc}). Failing over to next provider..."
                )
                last_error = exc

        raise AllAIProvidersFailedError(
            f"All registered AI providers failed streaming. Last exception: {last_error}"
        ) from last_error

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generates vector embeddings using the first available provider driver."""
        if not self._providers:
            raise AllAIProvidersFailedError(
                "No AI Provider drivers registered in AIProviderRouter."
            )

        for provider in self._providers:
            name = provider.provider_name
            cb = self._circuit_breakers[name]

            try:
                target_p = provider

                async def _call_embed(p: BaseLLMProvider = target_p) -> list[list[float]]:
                    return await p.embed(texts)

                return await cb.call(_call_embed)
            except Exception as exc:
                logger.warning(f"Provider '{name}' embedding failed ({exc}). Failing over...")

        raise AllAIProvidersFailedError("All AI providers failed embedding generation.")

    async def count_tokens(self, input_data: str | list[Message]) -> int:
        """Estimates token count using the primary provider driver."""
        if self._providers:
            return await self._providers[0].count_tokens(input_data)
        if isinstance(input_data, str):
            return len(input_data) // 4
        return sum(len(m.content) // 4 for m in input_data)

    async def health(self) -> dict[str, bool]:
        """Probes health status of all registered provider drivers.

        Returns:
            dict[str, bool]: Map of provider name to health boolean.
        """
        health_status: dict[str, bool] = {}
        for provider in self._providers:
            health_status[provider.provider_name] = await provider.health()
        return health_status


# Global AI Provider Router singleton
ai_router = AIProviderRouter()
