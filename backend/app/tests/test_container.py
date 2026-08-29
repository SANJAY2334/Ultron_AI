"""Unit Tests for Async Dependency Injection Container.

Validates singleton instance resolution, transient factories, async factory resolution,
duplicate registration guards, and container resets.
"""

import asyncio

import pytest

from app.core.container import (
    Container,
    ContainerKeyError,
    ContainerRegistrationError,
)


class DummyService:
    def __init__(self, name: str = "dummy") -> None:
        self.name = name


def test_singleton_instance_resolution() -> None:
    """Verify explicit singleton instance resolution."""

    async def _test() -> None:
        ctn = Container()
        service = DummyService("singleton_1")
        ctn.register_singleton(DummyService, service)

        resolved: DummyService = await ctn.resolve(DummyService)
        assert resolved is service
        assert resolved.name == "singleton_1"

    asyncio.run(_test())


def test_singleton_factory_lazy_resolution() -> None:
    """Verify lazy singleton factory creates instance once and caches it."""

    async def _test() -> None:
        ctn = Container()
        counter = 0

        def factory() -> DummyService:
            nonlocal counter
            counter += 1
            return DummyService(f"count_{counter}")

        ctn.register_singleton(DummyService, factory)

        r1: DummyService = await ctn.resolve(DummyService)
        r2: DummyService = await ctn.resolve(DummyService)

        assert r1 is r2
        assert r1.name == "count_1"
        assert counter == 1

    asyncio.run(_test())


def test_transient_factory_resolution() -> None:
    """Verify transient factory creates a new instance on every resolve call."""

    async def _test() -> None:
        ctn = Container()
        counter = 0

        def factory() -> DummyService:
            nonlocal counter
            counter += 1
            return DummyService(f"transient_{counter}")

        ctn.register_factory(DummyService, factory)

        r1: DummyService = await ctn.resolve(DummyService)
        r2: DummyService = await ctn.resolve(DummyService)

        assert r1 is not r2
        assert r1.name == "transient_1"
        assert r2.name == "transient_2"
        assert counter == 2

    asyncio.run(_test())


def test_async_factory_resolution() -> None:
    """Verify async coroutine factory functions can be resolved."""

    async def _test() -> None:
        ctn = Container()

        async def async_factory() -> DummyService:
            return DummyService("async_service")

        ctn.register_singleton(DummyService, async_factory)

        resolved: DummyService = await ctn.resolve(DummyService)
        assert resolved.name == "async_service"

    asyncio.run(_test())


def test_unregistered_key_raises_error() -> None:
    """Verify requesting an unregistered key raises ContainerKeyError."""

    async def _test() -> None:
        ctn = Container()
        with pytest.raises(ContainerKeyError):
            await ctn.resolve("unregistered_service_key")

    asyncio.run(_test())


def test_duplicate_registration_protection() -> None:
    """Verify duplicate key registration raises error unless override=True."""
    ctn = Container()
    ctn.register_singleton("key1", "val1")

    with pytest.raises(ContainerRegistrationError):
        ctn.register_singleton("key1", "val2")

    # Should succeed when override=True
    ctn.register_singleton("key1", "val2", override=True)
    assert ctn.resolve_sync("key1") == "val2"


def test_container_reset() -> None:
    """Verify container reset clears all registered services."""
    ctn = Container()
    ctn.register_singleton("key1", "val1")
    assert ctn.resolve_sync("key1") == "val1"

    ctn.reset()

    with pytest.raises(ContainerKeyError):
        ctn.resolve_sync("key1")
