"""Redis-Backed Working Memory Subsystem Adapter (Refined Storage).

Provides concrete Redis storage implementation for session-scoped working memory using:
- Timeline ZSet Index: `ultron:wm:session:{session_id}:timeline` (member = memory_id, score = timestamp)
- Item Payload Hash/String: `ultron:wm:item:{memory_id}` (value = serialized MemoryItem JSON)

Enforces separate item vs session TTLs, deterministic ordering, eviction caps, privacy access control,
item-level deletion/updates, and atomic transaction pipelines.
"""

import logging
from datetime import UTC, datetime
from typing import Any

from redis.asyncio import Redis, RedisError

from app.core.redis import get_redis_client
from app.memory.base import IWorkingMemory
from app.memory.models import (
    MemoryAccessContext,
    MemoryItem,
    MemoryWriteResult,
)

logger = logging.getLogger(__name__)


class RedisWorkingMemory(IWorkingMemory):
    """Concrete Redis-backed implementation of IWorkingMemory using refined Key-Separated Schema.

    Redis Key Schema:
    - Session Timeline ZSet Index: `ultron:wm:session:{session_id}:timeline`
      Member: `memory_id` (unique string)
      Score: Deterministic timestamp score (seconds since epoch + microsecond sequence)
    - Item Payload Key: `ultron:wm:item:{memory_id}`
      Value: Serialized MemoryItem JSON string
    """

    def __init__(
        self,
        redis_client: Any = None,
        max_items: int = 100,
        max_bytes: int = 1048576,  # 1MB per session limit
        session_ttl_seconds: int = 86400,  # 24 hours container TTL
        default_item_ttl_seconds: int = 86400,  # 24 hours item TTL
    ) -> None:
        """Initializes RedisWorkingMemory adapter with separate session & item TTLs and eviction caps."""
        self._redis = redis_client
        self.max_items = max_items
        self.max_bytes = max_bytes
        self.session_ttl_seconds = session_ttl_seconds
        self.default_item_ttl_seconds = default_item_ttl_seconds

    @property
    def redis(self) -> Redis:
        """Lazy-resolves Redis client if not injected."""
        if self._redis is None:
            self._redis = get_redis_client()
        return self._redis

    def _get_timeline_key(self, session_id: str) -> str:
        """Formats Redis ZSet timeline key for a given session ID."""
        return f"ultron:wm:session:{session_id}:timeline"

    def _get_item_key(self, memory_id: str) -> str:
        """Formats Redis String item key for a given memory ID."""
        return f"ultron:wm:item:{memory_id}"

    async def add(
        self,
        item: MemoryItem,
        access_context: MemoryAccessContext | None = None,
    ) -> MemoryWriteResult:
        """Adds a session-scoped memory item using key-separated ZSet timeline & item string storage.

        Args:
            item: Canonical MemoryItem instance.
            access_context: Optional caller privacy authorization context.

        Returns:
            MemoryWriteResult: Write execution result status.
        """
        if not item.session_id:
            return MemoryWriteResult(
                memory_id=item.id,
                success=False,
                memory_type=item.memory_type,
                error_message="Working memory item requires a valid session_id.",
            )

        if access_context and not access_context.can_access(item.metadata.privacy, item.user_id):
            return MemoryWriteResult(
                memory_id=item.id,
                success=False,
                memory_type=item.memory_type,
                error_message="Access Denied: Caller context cannot write item with specified privacy level.",
            )

        timeline_key = self._get_timeline_key(item.session_id)
        item_key = self._get_item_key(item.id)
        json_str = item.model_dump_json()

        # Deterministic microsecond precision timestamp score
        score = item.created_at.timestamp()
        item_ttl = item.ttl_seconds or self.default_item_ttl_seconds

        try:
            # First fetch current timeline count to handle eviction cleanup
            current_ids: Any = await self.redis.zrange(timeline_key, 0, -1)
            evicted_item_ids: list[str] = []
            if len(current_ids) >= self.max_items:
                overflow_count = len(current_ids) - self.max_items + 1
                raw_evicted = current_ids[:overflow_count]
                for raw_id in raw_evicted:
                    evicted_item_ids.append(
                        raw_id.decode("utf-8") if isinstance(raw_id, bytes) else str(raw_id)
                    )

            async with self.redis.pipeline(transaction=True) as pipe:
                # 1. Store individual MemoryItem payload
                pipe.set(item_key, json_str)
                pipe.expire(item_key, item_ttl)

                # 2. Add memory_id member to timeline ZSet
                pipe.zadd(timeline_key, {item.id: score})
                pipe.expire(timeline_key, self.session_ttl_seconds)

                # 3. Trim timeline index
                pipe.zremrangebyrank(timeline_key, 0, -(self.max_items + 1))
                await pipe.execute()

            # Clean up payload keys of evicted items
            if evicted_item_ids:
                del_keys = [self._get_item_key(eid) for eid in evicted_item_ids]
                await self.redis.delete(*del_keys)

            logger.debug(f"Stored working memory item '{item.id}' in timeline '{item.session_id}'.")
            return MemoryWriteResult(
                memory_id=item.id,
                success=True,
                memory_type=item.memory_type,
            )
        except RedisError as exc:
            logger.error(
                f"Redis working memory add operation failed for session '{item.session_id}': {exc}",
                exc_info=True,
            )
            return MemoryWriteResult(
                memory_id=item.id,
                success=False,
                memory_type=item.memory_type,
                error_message="Working memory storage unavailable.",
            )
        except Exception as exc:
            logger.error(
                f"Unexpected failure writing working memory item '{item.id}': {exc}",
                exc_info=True,
            )
            return MemoryWriteResult(
                memory_id=item.id,
                success=False,
                memory_type=item.memory_type,
                error_message="Internal working memory processing error.",
            )

    async def get(
        self,
        item_id: str,
        session_id: str | None = None,
        access_context: MemoryAccessContext | None = None,
    ) -> MemoryItem | None:
        """Retrieves a single working memory item directly by item key."""
        item_key = self._get_item_key(item_id)
        try:
            raw_payload = await self.redis.get(item_key)
            if not raw_payload:
                return None

            raw_str = (
                raw_payload.decode("utf-8") if isinstance(raw_payload, bytes) else str(raw_payload)
            )
            item = MemoryItem.model_validate_json(raw_str)

            # Active check
            if not item.is_active:
                return None

            # TTL check
            if item.expires_at and item.expires_at < datetime.now(UTC):
                return None

            # Privacy access check
            if access_context and not access_context.can_access(
                item.metadata.privacy, item.user_id
            ):
                return None

            return item
        except RedisError as exc:
            logger.error(f"Redis get failed for item '{item_id}': {exc}", exc_info=True)
            return None

    async def update(
        self,
        item: MemoryItem,
        access_context: MemoryAccessContext | None = None,
    ) -> MemoryWriteResult:
        """Updates an existing working memory item payload without corrupting timeline ordering."""
        if not item.session_id:
            return MemoryWriteResult(
                memory_id=item.id,
                success=False,
                memory_type=item.memory_type,
                error_message="Working memory item requires a valid session_id.",
            )

        if access_context and not access_context.can_access(item.metadata.privacy, item.user_id):
            return MemoryWriteResult(
                memory_id=item.id,
                success=False,
                memory_type=item.memory_type,
                error_message="Access Denied: Caller context cannot update item.",
            )

        item_key = self._get_item_key(item.id)
        item.updated_at = datetime.now(UTC)
        json_str = item.model_dump_json()
        item_ttl = item.ttl_seconds or self.default_item_ttl_seconds

        try:
            async with self.redis.pipeline(transaction=True) as pipe:
                pipe.set(item_key, json_str)
                pipe.expire(item_key, item_ttl)
                await pipe.execute()

            return MemoryWriteResult(
                memory_id=item.id,
                success=True,
                memory_type=item.memory_type,
            )
        except RedisError as exc:
            logger.error(f"Redis update failed for item '{item.id}': {exc}", exc_info=True)
            return MemoryWriteResult(
                memory_id=item.id,
                success=False,
                memory_type=item.memory_type,
                error_message="Working memory update failed.",
            )

    async def delete_item(self, memory_id: str, session_id: str) -> bool:
        """Deletes a single memory item from item payload storage and timeline index."""
        if not memory_id or not session_id:
            return False

        timeline_key = self._get_timeline_key(session_id)
        item_key = self._get_item_key(memory_id)

        try:
            async with self.redis.pipeline(transaction=True) as pipe:
                pipe.delete(item_key)
                pipe.zrem(timeline_key, memory_id)
                await pipe.execute()

            logger.info(f"Deleted working memory item '{memory_id}' from session '{session_id}'.")
            return True
        except RedisError as exc:
            logger.error(
                f"Redis delete_item failed for memory_id '{memory_id}': {exc}",
                exc_info=True,
            )
            return False

    async def get_history(
        self,
        session_id: str,
        limit: int = 50,
        access_context: MemoryAccessContext | None = None,
    ) -> list[MemoryItem]:
        """Retrieves ordered conversation working memory history for a session."""
        if not session_id:
            return []

        timeline_key = self._get_timeline_key(session_id)
        now = datetime.now(UTC)

        try:
            # Fetch member memory_ids in timestamp score ascending order
            raw_ids = await self.redis.zrange(timeline_key, 0, -1)
            if not raw_ids:
                return []

            memory_ids = [
                rid.decode("utf-8") if isinstance(rid, bytes) else str(rid) for rid in raw_ids
            ]
            item_keys = [self._get_item_key(mid) for mid in memory_ids]

            # Batch fetch payloads
            raw_payloads = await self.redis.mget(*item_keys)
            valid_items: list[MemoryItem] = []
            stale_ids_to_clean: list[str] = []

            for mid, raw_payload in zip(memory_ids, raw_payloads, strict=False):
                if raw_payload is None:
                    # Item payload expired independently (Item TTL) or was deleted
                    stale_ids_to_clean.append(mid)
                    continue

                try:
                    raw_str = (
                        raw_payload.decode("utf-8")
                        if isinstance(raw_payload, bytes)
                        else str(raw_payload)
                    )
                    item = MemoryItem.model_validate_json(raw_str)

                    # Filter tombstoned items
                    if not item.is_active:
                        continue

                    # Filter item TTL expiration
                    if item.expires_at and item.expires_at < now:
                        stale_ids_to_clean.append(mid)
                        continue

                    # Filter privacy access control
                    if access_context and not access_context.can_access(
                        item.metadata.privacy, item.user_id
                    ):
                        continue

                    valid_items.append(item)
                except Exception as parse_exc:
                    logger.warning(
                        f"Skipping malformed working memory JSON payload for '{mid}': {parse_exc}"
                    )
                    stale_ids_to_clean.append(mid)
                    continue

            # Clean up stale memory IDs from timeline index
            if stale_ids_to_clean:
                await self.redis.zrem(timeline_key, *stale_ids_to_clean)

            # Return requested limit slice (newest N items in chronological order)
            return valid_items[-limit:] if limit > 0 else valid_items
        except RedisError as exc:
            logger.error(
                f"Redis working memory get_history failed for session '{session_id}': {exc}",
                exc_info=True,
            )
            return []

    async def clear_session(self, session_id: str) -> bool:
        """Clears all transient working memory items belonging to a session."""
        if not session_id:
            return False

        timeline_key = self._get_timeline_key(session_id)
        try:
            raw_ids = await self.redis.zrange(timeline_key, 0, -1)
            item_keys = [
                self._get_item_key(rid.decode("utf-8") if isinstance(rid, bytes) else str(rid))
                for rid in raw_ids
            ]

            keys_to_del = [timeline_key] + item_keys
            await self.redis.delete(*keys_to_del)
            logger.info(f"Cleared working memory session '{session_id}'.")
            return True
        except RedisError as exc:
            logger.error(
                f"Redis clear_session failed for session '{session_id}': {exc}",
                exc_info=True,
            )
            return False
