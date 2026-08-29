"""Resilience Circuit Breaker Subsystem.

Provides a thread-safe Async Circuit Breaker protecting third-party LLM APIs and external resources.
Prevents cascading failures by transitioning between CLOSED, OPEN, and HALF_OPEN states.
"""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from enum import Enum, auto
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit Breaker Operational States."""

    CLOSED = auto()  # Normal operations: Traffic flows freely.
    OPEN = auto()  # Tripped state: Requests fail fast without hitting remote endpoint.
    HALF_OPEN = auto()  # Trial state: Allows limited test requests to probe endpoint recovery.


class CircuitBreakerOpenError(Exception):
    """Raised when attempting execution through an OPEN Circuit Breaker."""


class CircuitBreaker:
    """Async Circuit Breaker enforcing failure thresholds and automated recovery timeouts."""

    def __init__(
        self,
        name: str,
        failure_threshold: int = 3,
        recovery_timeout_seconds: float = 30.0,
    ) -> None:
        """Initializes the Circuit Breaker instance.

        Args:
            name: Identifier name for the circuit breaker.
            failure_threshold: Number of consecutive failures before tripping to OPEN.
            recovery_timeout_seconds: Duration in seconds to remain OPEN before probing HALF_OPEN.
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout_seconds

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float = 0.0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        """Returns current state, evaluating recovery timeout if OPEN."""
        if self._state == CircuitState.OPEN:
            if (time.perf_counter() - self._last_failure_time) >= self.recovery_timeout:
                logger.info(
                    f"CircuitBreaker '{self.name}' recovery timeout elapsed. Transitioning OPEN -> HALF_OPEN."
                )
                self._state = CircuitState.HALF_OPEN
        return self._state

    @property
    def failure_count(self) -> int:
        """Returns current consecutive failure count."""
        return self._failure_count

    async def call(self, func: Callable[[], Awaitable[T]]) -> T:
        """Executes an async callable through the Circuit Breaker safety wrapper.

        Args:
            func: Async coroutine function to execute.

        Returns:
            T: Function result on success.

        Raises:
            CircuitBreakerOpenError: If state is OPEN.
            Exception: Re-raises original exception on failure and updates failure counters.
        """
        current_state = self.state

        if current_state == CircuitState.OPEN:
            raise CircuitBreakerOpenError(
                f"CircuitBreaker '{self.name}' is OPEN. Fast-failing request."
            )

        async with self._lock:
            try:
                result = await func()
                self._on_success()
                return result
            except Exception as exc:
                self._on_failure()
                raise exc

    def _on_success(self) -> None:
        """Resets failure counter and transitions HALF_OPEN -> CLOSED on successful call."""
        if self._state != CircuitState.CLOSED:
            logger.info(
                f"CircuitBreaker '{self.name}' call succeeded. Transitioning {self._state.name} -> CLOSED."
            )
        self._state = CircuitState.CLOSED
        self._failure_count = 0

    def _on_failure(self) -> None:
        """Increments failure counter and trips circuit to OPEN if threshold is reached."""
        self._failure_count += 1
        self._last_failure_time = time.perf_counter()

        logger.warning(
            f"CircuitBreaker '{self.name}' failure recorded ({self._failure_count}/{self.failure_threshold})."
        )

        if self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN
            logger.error(
                f"CircuitBreaker '{self.name}' failure threshold reached ({self._failure_count}). Transitioning -> OPEN."
            )

    def reset(self) -> None:
        """Resets Circuit Breaker manually to CLOSED state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0
