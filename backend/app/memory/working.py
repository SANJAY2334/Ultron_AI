"""Working Memory Module.

Provides singleton instance factory and DI resolution helpers for Working Memory.
"""

from app.core.container import ContainerKeyError, container
from app.memory.adapters.redis_working_memory import RedisWorkingMemory
from app.memory.base import IWorkingMemory


def get_working_memory() -> IWorkingMemory:
    """Resolves or instantiates the IWorkingMemory singleton via DI container.

    Returns:
        IWorkingMemory: Concrete RedisWorkingMemory instance registered in container.
    """
    try:
        return container.resolve_sync(IWorkingMemory)
    except ContainerKeyError:
        instance = RedisWorkingMemory()
        container.register_singleton(IWorkingMemory, instance)
        return instance
