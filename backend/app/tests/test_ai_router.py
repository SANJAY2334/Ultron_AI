"""Unit Tests for Circuit Breaker and AI Provider Router.

Validates CircuitBreaker state transitions, fast-fail rejection, automated recovery timeout,
and AIProviderRouter primary-to-secondary failover.
"""

import asyncio
from typing import Any

import pytest

from app.ai.base import BaseLLMProvider
from app.ai.models import GenerationRequest, GenerationResponse, Message, ProviderMetadata, Usage
from app.ai.openai_provider import AIProviderError
from app.ai.router import AIProviderRouter, AllAIProvidersFailedError
from app.core.circuit_breaker import CircuitBreaker, CircuitBreakerOpenError, CircuitState


class FailingProvider(BaseLLMProvider):
    """Provider driver that always raises AIProviderError."""

    def __init__(self, name: str = "failing_provider") -> None:
        self._name = name

    @property
    def provider_name(self) -> str:
        return self._name

    @property
    def default_model(self) -> str:
        return "fail-model"

    async def complete(self, request: GenerationRequest) -> GenerationResponse:
        raise AIProviderError(f"Provider '{self._name}' forced failure.")

    async def stream(self, request: GenerationRequest):
        raise AIProviderError(f"Provider '{self._name}' forced stream failure.")
        yield

    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise AIProviderError(f"Provider '{self._name}' forced embed failure.")

    async def count_tokens(self, input_data: str | list[Message]) -> int:
        return 0

    async def health(self) -> bool:
        return False

    def capabilities(self) -> dict[str, Any]:
        return {}


class WorkingProvider(BaseLLMProvider):
    """Provider driver that succeeds."""

    def __init__(self, name: str = "working_provider") -> None:
        self._name = name

    @property
    def provider_name(self) -> str:
        return self._name

    @property
    def default_model(self) -> str:
        return "work-model"

    async def complete(self, request: GenerationRequest) -> GenerationResponse:
        return GenerationResponse(
            message=Message(role="assistant", content=f"Response from {self._name}"),
            usage=Usage(total_tokens=10),
            metadata=ProviderMetadata(
                provider_name=self._name,
                model_name=self.default_model,
                latency_ms=15.0,
            ),
        )

    async def stream(self, request: GenerationRequest):
        yield None

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2] for _ in texts]

    async def count_tokens(self, input_data: str | list[Message]) -> int:
        return 5

    async def health(self) -> bool:
        return True

    def capabilities(self) -> dict[str, Any]:
        return {"streaming": True}


def test_circuit_breaker_state_transitions() -> None:
    """Verify CircuitBreaker CLOSED -> OPEN on failures, and fast-fail when OPEN."""

    async def _test() -> None:
        cb = CircuitBreaker("test_cb", failure_threshold=2, recovery_timeout_seconds=0.2)
        assert cb.state == CircuitState.CLOSED

        async def fail_task() -> None:
            raise ValueError("Failure")

        # Failure 1
        with pytest.raises(ValueError):
            await cb.call(fail_task)
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 1

        # Failure 2 -> Trips to OPEN
        with pytest.raises(ValueError):
            await cb.call(fail_task)
        assert cb.state == CircuitState.OPEN

        # Immediate third call should fast-fail with CircuitBreakerOpenError
        with pytest.raises(CircuitBreakerOpenError):
            await cb.call(fail_task)

        # Wait for recovery timeout
        await asyncio.sleep(0.25)
        assert cb.state == CircuitState.HALF_OPEN

    asyncio.run(_test())


def test_ai_router_failover() -> None:
    """Verify AIProviderRouter fails over from primary failing provider to working secondary."""

    async def _test() -> None:
        router = AIProviderRouter()
        p1 = FailingProvider("primary_failing")
        p2 = WorkingProvider("secondary_working")

        router.register_provider(p1, failure_threshold=2)
        router.register_provider(p2, failure_threshold=2)

        req = GenerationRequest(messages=[Message(role="user", content="Ping")])
        res = await router.complete(req)

        assert res.message.content == "Response from secondary_working"
        assert res.metadata.provider_name == "secondary_working"
        assert res.metadata.retries == 1

    asyncio.run(_test())


def test_ai_router_all_failed_raises_exception() -> None:
    """Verify AllAIProvidersFailedError raised when all registered providers fail."""

    async def _test() -> None:
        router = AIProviderRouter()
        p1 = FailingProvider("p1")
        p2 = FailingProvider("p2")

        router.register_provider(p1)
        router.register_provider(p2)

        req = GenerationRequest(messages=[Message(role="user", content="Ping")])
        with pytest.raises(AllAIProvidersFailedError):
            await router.complete(req)

    asyncio.run(_test())


def test_ai_router_health() -> None:
    """Verify probing router provider health map."""

    async def _test() -> None:
        router = AIProviderRouter()
        router.register_provider(FailingProvider("p1"))
        router.register_provider(WorkingProvider("p2"))

        health = await router.health()
        assert health["p1"] is False
        assert health["p2"] is True

    asyncio.run(_test())
