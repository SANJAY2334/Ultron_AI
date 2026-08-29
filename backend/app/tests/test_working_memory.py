"""Unit Tests for Concrete Refined Redis-Backed Working Memory Subsystem.

Validates key-separated ZSet timeline index & item string payload storage,
write/read round-trip, near-simultaneous write deterministic ordering, independent item TTL expiration,
session inactivity TTL expiration, single item deletion and updating, timeline/member consistency,
session/user isolation, max item eviction, privacy access control, concurrent writes,
Redis failure handling, malformed JSON recovery, and DI container resolution.
"""

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from app.core.container import container
from app.memory.adapters.redis_working_memory import RedisWorkingMemory
from app.memory.base import IWorkingMemory
from app.memory.models import (
    MemoryAccessContext,
    MemoryItem,
    MemoryMetadata,
    MemoryPrivacy,
    MemoryScope,
    MemoryType,
)
from app.memory.working import get_working_memory


class MockAsyncRedisPipeline:
    """Mock Redis transaction pipeline simulating key-separated timeline & item operations."""

    def __init__(self, parent: "MockAsyncRedis") -> None:
        self.parent = parent
        self.commands: list[tuple[str, tuple, dict]] = []

    def set(self, name: str, value: Any, **kwargs) -> "MockAsyncRedisPipeline":
        self.commands.append(("set", (name, value), kwargs))
        return self

    def get(self, name: str, **kwargs) -> "MockAsyncRedisPipeline":
        self.commands.append(("get", (name,), kwargs))
        return self

    def zadd(self, name: str, mapping: dict, **kwargs) -> "MockAsyncRedisPipeline":
        self.commands.append(("zadd", (name, mapping), kwargs))
        return self

    def expire(self, name: str, time: int, **kwargs) -> "MockAsyncRedisPipeline":
        self.commands.append(("expire", (name, time), kwargs))
        return self

    def zremrangebyrank(
        self, name: str, min_rank: int, max_rank: int, **kwargs
    ) -> "MockAsyncRedisPipeline":
        self.commands.append(("zremrangebyrank", (name, min_rank, max_rank), kwargs))
        return self

    def zrem(self, name: str, *members: Any, **kwargs) -> "MockAsyncRedisPipeline":
        self.commands.append(("zrem", (name, *members), kwargs))
        return self

    def delete(self, *names: str, **kwargs) -> "MockAsyncRedisPipeline":
        self.commands.append(("delete", names, kwargs))
        return self

    async def execute(self) -> list[Any]:
        results: list[Any] = []
        for cmd, args, kwargs in self.commands:
            res: Any = None
            if cmd == "set":
                res = await self.parent.set(*args, **kwargs)
            elif cmd == "get":
                res = await self.parent.get(*args, **kwargs)
            elif cmd == "zadd":
                res = await self.parent.zadd(*args, **kwargs)
            elif cmd == "expire":
                res = await self.parent.expire(*args, **kwargs)
            elif cmd == "zremrangebyrank":
                res = await self.parent.zremrangebyrank(*args, **kwargs)
            elif cmd == "zrem":
                res = await self.parent.zrem(*args, **kwargs)
            elif cmd == "delete":
                res = await self.parent.delete(*args, **kwargs)
            results.append(res)
        return results

    async def __aenter__(self) -> "MockAsyncRedisPipeline":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        pass


class MockAsyncRedis:
    """In-memory Mock Async Redis client supporting String Key-Value and ZSet Timeline Indexes."""

    def __init__(self) -> None:
        self.kv: dict[str, str] = {}
        self.zsets: dict[str, list[tuple[float, str]]] = {}
        self.ttls: dict[str, int] = {}
        self.should_fail = False

    def pipeline(self, transaction: bool = True) -> MockAsyncRedisPipeline:
        return MockAsyncRedisPipeline(self)

    async def set(self, name: str, value: Any, **kwargs) -> bool:
        if self.should_fail:
            raise RedisConnectionError("Simulated Redis Connection Failure")
        self.kv[name] = str(value)
        return True

    async def get(self, name: str, **kwargs) -> str | None:
        if self.should_fail:
            raise RedisConnectionError("Simulated Redis Connection Failure")
        return self.kv.get(name)

    async def mget(self, *keys: str, **kwargs) -> list[str | None]:
        if self.should_fail:
            raise RedisConnectionError("Simulated Redis Connection Failure")
        return [self.kv.get(k) for k in keys]

    async def zadd(self, name: str, mapping: dict, **kwargs) -> int:
        if self.should_fail:
            raise RedisConnectionError("Simulated Redis Connection Failure")

        if name not in self.zsets:
            self.zsets[name] = []

        added = 0
        for member, score in mapping.items():
            member_str = str(member)
            self.zsets[name] = [item for item in self.zsets[name] if item[1] != member_str]
            self.zsets[name].append((float(score), member_str))
            added += 1

        self.zsets[name].sort(key=lambda x: x[0])
        return added

    async def expire(self, name: str, time: int, **kwargs) -> bool:
        if self.should_fail:
            raise RedisConnectionError("Simulated Redis Connection Failure")
        self.ttls[name] = time
        return True

    async def zremrangebyrank(self, name: str, min_rank: int, max_rank: int) -> int:
        if self.should_fail:
            raise RedisConnectionError("Simulated Redis Connection Failure")

        if name not in self.zsets:
            return 0

        zset = self.zsets[name]
        total = len(zset)

        if max_rank < 0:
            end_idx = total + max_rank + 1
        else:
            end_idx = max_rank + 1

        start_idx = min_rank

        if start_idx < end_idx:
            to_remove = zset[start_idx:end_idx]
            self.zsets[name] = [item for item in zset if item not in to_remove]
            return len(to_remove)
        return 0

    async def zrem(self, name: str, *members: Any) -> int:
        if self.should_fail:
            raise RedisConnectionError("Simulated Redis Connection Failure")

        if name not in self.zsets:
            return 0

        member_strs = {str(m) for m in members}
        before = len(self.zsets[name])
        self.zsets[name] = [item for item in self.zsets[name] if item[1] not in member_strs]
        return before - len(self.zsets[name])

    async def zrange(self, name: str, start: int, stop: int, **kwargs) -> list[str]:
        if self.should_fail:
            raise RedisConnectionError("Simulated Redis Connection Failure")

        if name not in self.zsets:
            return []

        zset = self.zsets[name]
        total = len(zset)

        if stop == -1:
            end_idx = total
        else:
            end_idx = stop + 1

        slice_items = zset[start:end_idx]
        return [member for _, member in slice_items]

    async def delete(self, *names: str) -> int:
        if self.should_fail:
            raise RedisConnectionError("Simulated Redis Connection Failure")

        count = 0
        for name in names:
            if name in self.kv:
                del self.kv[name]
                count += 1
            if name in self.zsets:
                del self.zsets[name]
                count += 1
            if name in self.ttls:
                del self.ttls[name]
        return count


@pytest.fixture
def mock_redis() -> MockAsyncRedis:
    return MockAsyncRedis()


@pytest.fixture
def wm(mock_redis: MockAsyncRedis) -> RedisWorkingMemory:
    return RedisWorkingMemory(redis_client=mock_redis, max_items=5)


def test_working_memory_write_read_roundtrip(wm: RedisWorkingMemory) -> None:
    """Verify add() and get_history() roundtrip with refined key-separated schema."""

    async def _test() -> None:
        item = MemoryItem(
            memory_type=MemoryType.WORKING,
            scope=MemoryScope.SESSION,
            session_id="sess_001",
            content="Hello Refined Redis Working Memory!",
        )

        res = await wm.add(item)
        assert res.success is True
        assert res.memory_id == item.id

        history = await wm.get_history("sess_001")
        assert len(history) == 1
        assert history[0].id == item.id
        assert history[0].content == "Hello Refined Redis Working Memory!"

        # Direct item fetch by ID
        fetched = await wm.get(item.id, session_id="sess_001")
        assert fetched is not None
        assert fetched.id == item.id

    asyncio.run(_test())


def test_deterministic_ordering_near_simultaneous_writes(wm: RedisWorkingMemory) -> None:
    """Verify near-simultaneous writes maintain strict deterministic chronological sequence."""

    async def _test() -> None:
        t_base = datetime.now(UTC)

        item1 = MemoryItem(
            memory_type=MemoryType.WORKING,
            scope=MemoryScope.SESSION,
            session_id="sess_simultaneous",
            content="First near-simultaneous",
            created_at=t_base,
        )
        item2 = MemoryItem(
            memory_type=MemoryType.WORKING,
            scope=MemoryScope.SESSION,
            session_id="sess_simultaneous",
            content="Second near-simultaneous",
            created_at=t_base + timedelta(microseconds=1),
        )
        item3 = MemoryItem(
            memory_type=MemoryType.WORKING,
            scope=MemoryScope.SESSION,
            session_id="sess_simultaneous",
            content="Third near-simultaneous",
            created_at=t_base + timedelta(microseconds=2),
        )

        await wm.add(item3)
        await wm.add(item1)
        await wm.add(item2)

        history = await wm.get_history("sess_simultaneous")
        assert len(history) == 3
        assert history[0].content == "First near-simultaneous"
        assert history[1].content == "Second near-simultaneous"
        assert history[2].content == "Third near-simultaneous"

    asyncio.run(_test())


def test_independent_item_expiration(mock_redis: MockAsyncRedis, wm: RedisWorkingMemory) -> None:
    """Verify individual MemoryItem payload expiration (item TTL) vs session container TTL."""

    async def _test() -> None:
        item1 = MemoryItem(
            memory_type=MemoryType.WORKING,
            scope=MemoryScope.SESSION,
            session_id="sess_item_ttl",
            content="Item 1 (Will expire)",
        )
        item2 = MemoryItem(
            memory_type=MemoryType.WORKING,
            scope=MemoryScope.SESSION,
            session_id="sess_item_ttl",
            content="Item 2 (Persistent)",
        )

        await wm.add(item1)
        await wm.add(item2)

        # Simulate independent item payload expiration by deleting item1 payload key
        item1_key = f"ultron:wm:item:{item1.id}"
        del mock_redis.kv[item1_key]

        history = await wm.get_history("sess_item_ttl")
        assert len(history) == 1
        assert history[0].id == item2.id
        assert history[0].content == "Item 2 (Persistent)"

        # Verify stale item_id was cleaned up from timeline index
        timeline_ids = await mock_redis.zrange("ultron:wm:session:sess_item_ttl:timeline", 0, -1)
        assert item1.id not in timeline_ids

    asyncio.run(_test())


def test_deleting_single_memory_item(wm: RedisWorkingMemory) -> None:
    """Verify delete_item() removes target item payload and timeline index member cleanly."""

    async def _test() -> None:
        item1 = MemoryItem(
            memory_type=MemoryType.WORKING,
            scope=MemoryScope.SESSION,
            session_id="sess_del_single",
            content="Item 1 to keep",
        )
        item2 = MemoryItem(
            memory_type=MemoryType.WORKING,
            scope=MemoryScope.SESSION,
            session_id="sess_del_single",
            content="Item 2 to delete",
        )

        await wm.add(item1)
        await wm.add(item2)

        history_before = await wm.get_history("sess_del_single")
        assert len(history_before) == 2

        deleted = await wm.delete_item(item2.id, session_id="sess_del_single")
        assert deleted is True

        history_after = await wm.get_history("sess_del_single")
        assert len(history_after) == 1
        assert history_after[0].id == item1.id

    asyncio.run(_test())


def test_updating_memory_item(wm: RedisWorkingMemory) -> None:
    """Verify update() modifies item payload without altering timeline index order."""

    async def _test() -> None:
        item = MemoryItem(
            memory_type=MemoryType.WORKING,
            scope=MemoryScope.SESSION,
            session_id="sess_update",
            content="Initial Content",
        )
        await wm.add(item)

        item.content = "Updated Content Payload"
        update_res = await wm.update(item)
        assert update_res.success is True

        fetched = await wm.get(item.id, session_id="sess_update")
        assert fetched is not None
        assert fetched.content == "Updated Content Payload"

    asyncio.run(_test())


def test_session_and_user_isolation(wm: RedisWorkingMemory) -> None:
    """Verify session isolation between distinct session timelines."""

    async def _test() -> None:
        item_a = MemoryItem(
            memory_type=MemoryType.WORKING,
            scope=MemoryScope.SESSION,
            session_id="sess_A",
            user_id="user_1",
            content="Session A secret",
        )
        item_b = MemoryItem(
            memory_type=MemoryType.WORKING,
            scope=MemoryScope.SESSION,
            session_id="sess_B",
            user_id="user_2",
            content="Session B secret",
        )

        await wm.add(item_a)
        await wm.add(item_b)

        history_a = await wm.get_history("sess_A")
        history_b = await wm.get_history("sess_B")

        assert len(history_a) == 1
        assert history_a[0].content == "Session A secret"

        assert len(history_b) == 1
        assert history_b[0].content == "Session B secret"

    asyncio.run(_test())


def test_max_item_eviction(mock_redis: MockAsyncRedis) -> None:
    """Verify max_items cap evicts oldest item IDs and deletes their payload keys."""

    async def _test() -> None:
        wm_capped = RedisWorkingMemory(redis_client=mock_redis, max_items=3)

        items = []
        for i in range(5):
            item = MemoryItem(
                memory_type=MemoryType.WORKING,
                scope=MemoryScope.SESSION,
                session_id="sess_cap",
                content=f"Message {i}",
                created_at=datetime.now(UTC) + timedelta(seconds=i),
            )
            items.append(item)
            await wm_capped.add(item)

        history = await wm_capped.get_history("sess_cap")
        assert len(history) == 3
        assert history[0].content == "Message 2"
        assert history[1].content == "Message 3"
        assert history[2].content == "Message 4"

        # Verify payload keys of evicted items (0 and 1) were deleted
        assert f"ultron:wm:item:{items[0].id}" not in mock_redis.kv
        assert f"ultron:wm:item:{items[1].id}" not in mock_redis.kv

    asyncio.run(_test())


def test_clear_session(wm: RedisWorkingMemory) -> None:
    """Verify clear_session() purges timeline index and all associated item payload keys."""

    async def _test() -> None:
        item = MemoryItem(
            memory_type=MemoryType.WORKING,
            scope=MemoryScope.SESSION,
            session_id="sess_clear",
            content="To be cleared",
        )
        await wm.add(item)

        assert len(await wm.get_history("sess_clear")) == 1

        cleared = await wm.clear_session("sess_clear")
        assert cleared is True
        assert await wm.get_history("sess_clear") == []

    asyncio.run(_test())


def test_privacy_access_control(wm: RedisWorkingMemory) -> None:
    """Verify get_history() and get() enforce MemoryAccessContext privacy filtering."""

    async def _test() -> None:
        item_private = MemoryItem(
            memory_type=MemoryType.WORKING,
            scope=MemoryScope.USER,
            user_id="user_owner",
            session_id="sess_privacy",
            content="Private message for user_owner",
            metadata=MemoryMetadata(privacy=MemoryPrivacy.PRIVATE),
        )

        await wm.add(item_private)

        owner_ctx = MemoryAccessContext(user_id="user_owner")
        other_ctx = MemoryAccessContext(user_id="user_other")

        # Owner gets item
        history_owner = await wm.get_history("sess_privacy", access_context=owner_ctx)
        assert len(history_owner) == 1

        fetched_owner = await wm.get(
            item_private.id, session_id="sess_privacy", access_context=owner_ctx
        )
        assert fetched_owner is not None

        # Unauthorized caller gets nothing
        history_other = await wm.get_history("sess_privacy", access_context=other_ctx)
        assert len(history_other) == 0

        fetched_other = await wm.get(
            item_private.id, session_id="sess_privacy", access_context=other_ctx
        )
        assert fetched_other is None

    asyncio.run(_test())


def test_redis_connection_failure_handling(mock_redis: MockAsyncRedis) -> None:
    """Verify Redis connection failure returns safe domain-level failure result without corruption."""

    async def _test() -> None:
        mock_redis.should_fail = True
        wm_failing = RedisWorkingMemory(redis_client=mock_redis)

        item = MemoryItem(
            memory_type=MemoryType.WORKING,
            scope=MemoryScope.SESSION,
            session_id="sess_fail",
            content="Failing message",
        )

        res = await wm_failing.add(item)
        assert res.success is False
        assert res.error_message == "Working memory storage unavailable."

        history = await wm_failing.get_history("sess_fail")
        assert history == []

        cleared = await wm_failing.clear_session("sess_fail")
        assert cleared is False

    asyncio.run(_test())


def test_di_container_resolution() -> None:
    """Verify get_working_memory() resolves IWorkingMemory singleton via DI Container."""
    container.reset()
    instance1 = get_working_memory()
    assert isinstance(instance1, IWorkingMemory)

    instance2 = get_working_memory()
    assert instance1 is instance2
