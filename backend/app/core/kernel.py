"""ULTRON Kernel State Machine & Lifecycle Orchestrator.

Manages application lifecycle states, startup/shutdown hook registrations, and graceful OS signal
interception (SIGINT, SIGTERM) according to SDS v2.0 specifications.
"""

import logging
import signal
import sys
from collections.abc import Awaitable, Callable
from enum import Enum, auto

logger = logging.getLogger(__name__)


class KernelState(Enum):
    """ULTRON Kernel Operational Lifecycle States."""

    UNINITIALIZED = auto()
    CONFIGURING = auto()
    INITIALIZING_SERVICES = auto()
    RUNNING = auto()
    SHUTTING_DOWN = auto()
    TERMINATED = auto()


class KernelStateError(Exception):
    """Raised when attempting an illegal Kernel state transition."""


# Valid state transitions lookup table
VALID_TRANSITIONS: dict[KernelState, list[KernelState]] = {
    KernelState.UNINITIALIZED: [KernelState.CONFIGURING],
    KernelState.CONFIGURING: [KernelState.INITIALIZING_SERVICES, KernelState.SHUTTING_DOWN],
    KernelState.INITIALIZING_SERVICES: [KernelState.RUNNING, KernelState.SHUTTING_DOWN],
    KernelState.RUNNING: [KernelState.SHUTTING_DOWN],
    KernelState.SHUTTING_DOWN: [KernelState.TERMINATED],
    KernelState.TERMINATED: [],
}

AsyncHook = Callable[[], Awaitable[None]]


class UltronKernel:
    """Core Kernel Orchestrator managing application state machine and hooks."""

    def __init__(self) -> None:
        """Initializes the Kernel in UNINITIALIZED state."""
        self._state: KernelState = KernelState.UNINITIALIZED
        self._startup_hooks: list[AsyncHook] = []
        self._shutdown_hooks: list[AsyncHook] = []

    @property
    def state(self) -> KernelState:
        """Returns the current operational state of the Kernel."""
        return self._state

    def _transition_to(self, new_state: KernelState) -> None:
        """Transitions the Kernel to a new state if allowed.

        Args:
            new_state: Target state to transition to.

        Raises:
            KernelStateError: If transition is invalid.
        """
        allowed_states = VALID_TRANSITIONS.get(self._state, [])
        if new_state not in allowed_states:
            raise KernelStateError(
                f"Cannot transition Kernel from '{self._state.name}' to '{new_state.name}'."
            )
        logger.info(f"Kernel state transition: {self._state.name} -> {new_state.name}")
        self._state = new_state

    def add_startup_hook(self, hook: AsyncHook) -> None:
        """Registers an asynchronous callback function to be executed during startup.

        Args:
            hook: Async coroutine function with signature async def ().
        """
        self._startup_hooks.append(hook)

    def add_shutdown_hook(self, hook: AsyncHook) -> None:
        """Registers an asynchronous callback function to be executed during shutdown.

        Args:
            hook: Async coroutine function with signature async def ().
        """
        self._shutdown_hooks.append(hook)

    async def boot(self) -> None:
        """Executes the complete Kernel boot sequence.

        Transitions: UNINITIALIZED -> CONFIGURING -> INITIALIZING_SERVICES -> RUNNING.
        """
        self._transition_to(KernelState.CONFIGURING)
        # Configuration is loaded via app.core.config

        self._transition_to(KernelState.INITIALIZING_SERVICES)
        for hook in self._startup_hooks:
            try:
                await hook()
            except Exception as e:
                logger.error(f"Startup hook execution failed: {e}", exc_info=True)
                await self.shutdown()
                raise

        self._transition_to(KernelState.RUNNING)

    async def shutdown(self) -> None:
        """Executes graceful Kernel shutdown sequence.

        Transitions: (CONFIGURING|INITIALIZING_SERVICES|RUNNING) -> SHUTTING_DOWN -> TERMINATED.
        """
        if self._state in (KernelState.SHUTTING_DOWN, KernelState.TERMINATED):
            return

        self._transition_to(KernelState.SHUTTING_DOWN)

        for hook in reversed(self._shutdown_hooks):
            try:
                await hook()
            except Exception as e:
                logger.error(f"Shutdown hook execution failed: {e}", exc_info=True)

        self._transition_to(KernelState.TERMINATED)

    def register_signal_handlers(self) -> None:
        """Registers OS signal handlers (SIGINT, SIGTERM) for graceful shutdown."""
        if sys.platform != "win32":
            import asyncio

            loop = asyncio.get_event_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, lambda: asyncio.create_task(self.shutdown()))


# Global kernel singleton instance
kernel = UltronKernel()
