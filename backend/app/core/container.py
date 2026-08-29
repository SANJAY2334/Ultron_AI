"""Async Dependency Injection Container & Service Registry.

Provides a thread-safe, interface-driven Async Dependency Injection Container
for registering and resolving singletons and transient service factories, adhering to
Clean Architecture & Inversion of Control (SOLID Principle #5).
"""

import asyncio
import inspect
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


class ContainerKeyError(KeyError):
    """Raised when requesting a service key that has not been registered in the Container."""


class ContainerRegistrationError(Exception):
    """Raised when attempting an invalid service registration in the Container."""


class Container:
    """Thread-safe Asynchronous Dependency Injection Container.

    Manages singletons, factories, and interface mappings across the application lifecycle.
    """

    def __init__(self) -> None:
        """Initializes an empty DI Container instance."""
        self._singletons: dict[Any, Any] = {}
        self._factories: dict[Any, Callable[..., Any]] = {}
        self._instances: dict[Any, Any] = {}
        self._lock = asyncio.Lock()

    def register_singleton(
        self, key: Any, instance_or_factory: Any, override: bool = False
    ) -> None:
        """Registers a singleton instance or factory function in the container.

        Args:
            key: Service key or interface class (e.g. IMemoryStore).
            instance_or_factory: Concrete object instance or factory callable.
            override: Set to True to allow overwriting an existing registration.

        Raises:
            ContainerRegistrationError: If key is already registered and override is False.
        """
        if (key in self._singletons or key in self._factories) and not override:
            raise ContainerRegistrationError(
                f"Service key '{key}' is already registered in Container. Set override=True to replace."
            )

        if callable(instance_or_factory) and not inspect.isclass(instance_or_factory):
            self._factories[key] = instance_or_factory
            self._singletons[key] = True  # Flagged as singleton factory
        else:
            self._instances[key] = instance_or_factory
            self._singletons[key] = False

    def register_factory(
        self, key: Any, factory: Callable[..., Any], override: bool = False
    ) -> None:
        """Registers a transient factory function that creates a new instance on every resolve.

        Args:
            key: Service key or interface class.
            factory: Callable producing a new instance.
            override: Set to True to allow overwriting an existing registration.

        Raises:
            ContainerRegistrationError: If key is already registered and override is False.
        """
        if (key in self._singletons or key in self._factories) and not override:
            raise ContainerRegistrationError(
                f"Service key '{key}' is already registered in Container. Set override=True to replace."
            )

        if not callable(factory):
            raise ContainerRegistrationError(f"Factory for key '{key}' must be callable.")

        self._factories[key] = factory

    async def resolve(self, key: type[T] | Any) -> T:
        """Asynchronously resolves a registered service by its key or interface type.

        Args:
            key: Registered service key or interface class.

        Returns:
            T: Resolved service instance.

        Raises:
            ContainerKeyError: If key is not registered in the Container.
        """
        async with self._lock:
            # Case 1: Pre-existing singleton instance
            if key in self._instances:
                return self._instances[key]

            # Case 2: Factory function (singleton or transient)
            if key in self._factories:
                factory = self._factories[key]
                is_singleton = self._singletons.get(key, False)

                if inspect.iscoroutinefunction(factory):
                    instance = await factory()
                else:
                    instance = factory()

                if is_singleton:
                    self._instances[key] = instance
                    del self._factories[key]

                return instance

            raise ContainerKeyError(f"No service registered for key: '{key}'")

    def resolve_sync(self, key: type[T] | Any) -> T:
        """Synchronously resolves a pre-instantiated singleton or synchronous factory.

        Args:
            key: Registered service key or interface class.

        Returns:
            T: Resolved service instance.

        Raises:
            ContainerKeyError: If key is not registered.
            RuntimeError: If resolving an async factory synchronously.
        """
        if key in self._instances:
            return self._instances[key]

        if key in self._factories:
            factory = self._factories[key]
            if inspect.iscoroutinefunction(factory):
                raise RuntimeError(
                    f"Service '{key}' uses an async factory and must be resolved using 'await container.resolve()'."
                )
            instance = factory()
            if self._singletons.get(key, False):
                self._instances[key] = instance
                del self._factories[key]
            return instance

        raise ContainerKeyError(f"No service registered for key: '{key}'")

    def reset(self) -> None:
        """Clears all registered services and instances from the Container."""
        self._singletons.clear()
        self._factories.clear()
        self._instances.clear()


# Global default container instance
container = Container()
