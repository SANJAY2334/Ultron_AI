"""Unit Tests for Kernel Lifecycle & State Machine Orchestrator.

Validates state machine transitions, startup/shutdown hook execution order,
and state error rejection.
"""

import asyncio

import pytest

from app.core.kernel import KernelState, KernelStateError, UltronKernel


def test_kernel_boot_and_shutdown_lifecycle() -> None:
    """Verify clean state transitions across boot and shutdown sequence."""

    async def _test() -> None:
        k = UltronKernel()
        assert k.state == KernelState.UNINITIALIZED

        await k.boot()
        assert k.state == KernelState.RUNNING

        await k.shutdown()
        assert k.state == KernelState.TERMINATED

    asyncio.run(_test())


def test_kernel_startup_and_shutdown_hooks() -> None:
    """Verify registered startup and shutdown async hooks execute."""

    async def _test() -> None:
        k = UltronKernel()
        events: list[str] = []

        async def startup_1() -> None:
            events.append("startup_1")

        async def startup_2() -> None:
            events.append("startup_2")

        async def shutdown_1() -> None:
            events.append("shutdown_1")

        k.add_startup_hook(startup_1)
        k.add_startup_hook(startup_2)
        k.add_shutdown_hook(shutdown_1)

        await k.boot()
        assert events == ["startup_1", "startup_2"]

        await k.shutdown()
        assert events == ["startup_1", "startup_2", "shutdown_1"]

    asyncio.run(_test())


def test_invalid_state_transition_raises_error() -> None:
    """Verify attempting illegal state transition raises KernelStateError."""
    k = UltronKernel()
    assert k.state == KernelState.UNINITIALIZED

    with pytest.raises(KernelStateError):
        k._transition_to(KernelState.RUNNING)
