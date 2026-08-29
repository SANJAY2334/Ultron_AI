"""Unit Tests for PgVectorStore Adapter & EmbeddingProfile Contract.

Validates upsert, update, get, cosine vector similarity search, embedding dimension/profile mismatch errors,
metadata persistence, query filtering, privacy access control, user/project isolation,
soft vs physical deletion, expiration filtering, health probes, and database failure handling.
"""

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy.exc import SQLAlchemyError

from app.memory.adapters.embedding_profile import (
    DistanceMetric,
    EmbeddingProfile,
    EmbeddingProfileMismatchError,
)
from app.memory.adapters.pgvector_store import MemoryItemModel, PgVectorStore
from app.memory.models import (
    MemoryAccessContext,
    MemoryItem,
    MemoryMetadata,
    MemoryPrivacy,
    MemoryQuery,
    MemoryScope,
    MemoryType,
    VectorEmbedding,
)


class MockDatabaseSession:
    """In-memory AsyncSession mock for unit testing PgVectorStore without live PostgreSQL."""

    def __init__(self, parent: "MockSessionMaker") -> None:
        self.parent = parent
        self.in_transaction = False

    async def get(self, entity: type, ident: Any) -> Any:
        if self.parent.should_fail:
            raise SQLAlchemyError("Simulated Database Error")
        return self.parent.records.get(ident)

    def add(self, instance: Any) -> None:
        if self.parent.should_fail:
            raise SQLAlchemyError("Simulated Database Error")
        self.parent.records[instance.id] = instance

    async def delete(self, instance: Any) -> None:
        if self.parent.should_fail:
            raise SQLAlchemyError("Simulated Database Error")
        if hasattr(instance, "id") and instance.id in self.parent.records:
            del self.parent.records[instance.id]

    async def execute(self, statement: Any) -> Any:
        if self.parent.should_fail:
            raise SQLAlchemyError("Simulated Database Error")

        records_list = list(self.parent.records.values())

        class ScalarResult:
            def scalars(self):
                class ScalarsList:
                    def all(self):
                        return records_list

                return ScalarsList()

        return ScalarResult()

    def begin(self) -> "MockDatabaseTransaction":
        return MockDatabaseTransaction(self)

    async def __aenter__(self) -> "MockDatabaseSession":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        pass


class MockDatabaseTransaction:
    """Mock database transaction context manager."""

    def __init__(self, session: MockDatabaseSession) -> None:
        self.session = session

    async def __aenter__(self) -> "MockDatabaseTransaction":
        self.session.in_transaction = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self.session.in_transaction = False


class MockSessionMaker:
    """Factory for MockDatabaseSession instances."""

    def __init__(self) -> None:
        self.records: dict[str, MemoryItemModel] = {}
        self.should_fail = False

    def __call__(self) -> MockDatabaseSession:
        return MockDatabaseSession(self)


@pytest.fixture
def mock_db() -> MockSessionMaker:
    return MockSessionMaker()


@pytest.fixture
def store(mock_db: MockSessionMaker) -> PgVectorStore:
    profile = EmbeddingProfile(
        provider="openai",
        model_name="text-embedding-3-small",
        dimension=4,
        version="v1",
        distance_metric=DistanceMetric.COSINE,
    )
    return PgVectorStore(session_factory=mock_db, profile=profile)


def test_embedding_profile_compatibility_contract() -> None:
    """Verify EmbeddingProfile compatibility checks and vector validation."""
    p1 = EmbeddingProfile(
        provider="openai",
        model_name="text-embedding-3-small",
        dimension=4,
        version="v1",
    )
    p2 = EmbeddingProfile(
        provider="openai",
        model_name="text-embedding-3-small",
        dimension=4,
        version="v1",
    )
    p3 = EmbeddingProfile(
        provider="openai",
        model_name="text-embedding-3-large",
        dimension=4,
        version="v1",
    )

    assert p1.is_compatible(p2) is True
    assert p1.is_compatible(p3) is False

    valid_vec = VectorEmbedding(
        vector=[0.1, 0.2, 0.3, 0.4],
        model_name="text-embedding-3-small",
        dimension=4,
        version="v1",
    )
    p1.validate_vector(valid_vec)  # Should pass cleanly

    bad_dim_vec = VectorEmbedding(
        vector=[0.1, 0.2],
        model_name="text-embedding-3-small",
        dimension=2,
        version="v1",
    )
    with pytest.raises(EmbeddingProfileMismatchError, match="dimension mismatch"):
        p1.validate_vector(bad_dim_vec)


def test_pgvector_store_upsert_and_get(store: PgVectorStore) -> None:
    """Verify upsert, update, and get operations in PgVectorStore."""

    async def _test() -> None:
        emb = VectorEmbedding(
            vector=[1.0, 0.0, 0.0, 0.0],
            model_name="text-embedding-3-small",
            dimension=4,
            version="v1",
        )
        item = MemoryItem(
            memory_type=MemoryType.SEMANTIC,
            scope=MemoryScope.USER,
            user_id="user_pg_1",
            content="Persistent PG concept",
            embeddings=[emb],
            metadata=MemoryMetadata(privacy=MemoryPrivacy.PUBLIC),
        )

        res = await store.upsert(item)
        assert res.success is True
        assert res.memory_id == item.id

        access_owner = MemoryAccessContext(user_id="user_pg_1")
        fetched = await store.get(item.id, access_owner)
        assert fetched is not None
        assert fetched.id == item.id
        assert fetched.content == "Persistent PG concept"

        # Update
        item.content = "Updated Persistent PG concept"
        up_res = await store.upsert(item)
        assert up_res.success is True

        fetched_updated = await store.get(item.id, access_owner)
        assert fetched_updated is not None
        assert fetched_updated.content == "Updated Persistent PG concept"

    asyncio.run(_test())


def test_embedding_profile_mismatch_rejection(store: PgVectorStore) -> None:
    """Verify vector with incompatible model name or dimension is rejected on upsert."""

    async def _test() -> None:
        incompatible_emb = VectorEmbedding(
            vector=[0.1, 0.2, 0.3, 0.4],
            model_name="wrong-model-name",
            dimension=4,
            version="v1",
        )
        item = MemoryItem(
            memory_type=MemoryType.SEMANTIC,
            scope=MemoryScope.USER,
            user_id="user_1",
            content="Incompatible model item",
            embeddings=[incompatible_emb],
        )

        res = await store.upsert(item)
        assert res.success is False
        assert "Embedding profile compatibility error" in (res.error_message or "")

    asyncio.run(_test())


def test_pgvector_search_and_cosine_ranking(store: PgVectorStore) -> None:
    """Verify similarity search, cosine ranking, and privacy enforcement in PgVectorStore."""

    async def _test() -> None:
        emb1 = VectorEmbedding(
            vector=[1.0, 0.0, 0.0, 0.0],
            model_name="text-embedding-3-small",
            dimension=4,
            version="v1",
        )
        emb2 = VectorEmbedding(
            vector=[0.0, 1.0, 0.0, 0.0],
            model_name="text-embedding-3-small",
            dimension=4,
            version="v1",
        )

        item1 = MemoryItem(
            memory_type=MemoryType.SEMANTIC,
            scope=MemoryScope.USER,
            user_id="user_owner",
            content="Matching concept 1",
            embeddings=[emb1],
            metadata=MemoryMetadata(privacy=MemoryPrivacy.PUBLIC),
        )
        item2 = MemoryItem(
            memory_type=MemoryType.SEMANTIC,
            scope=MemoryScope.USER,
            user_id="user_owner",
            content="Matching concept 2",
            embeddings=[emb2],
            metadata=MemoryMetadata(privacy=MemoryPrivacy.PRIVATE),
        )

        await store.upsert(item1)
        await store.upsert(item2)

        owner_ctx = MemoryAccessContext(user_id="user_owner")
        other_ctx = MemoryAccessContext(user_id="user_other")

        query_emb = VectorEmbedding(
            vector=[0.9, 0.1, 0.0, 0.0],
            model_name="text-embedding-3-small",
            dimension=4,
            version="v1",
        )
        query = MemoryQuery(query_embedding=query_emb, limit=10)

        # Owner gets both items
        search_owner = await store.search(query, owner_ctx)
        assert search_owner.total_found == 2
        assert search_owner.items[0].id == item1.id
        assert search_owner.scores[0] > search_owner.scores[1]

        # Other caller only gets PUBLIC item
        search_other = await store.search(query, other_ctx)
        assert search_other.total_found == 1
        assert search_other.items[0].id == item1.id

    asyncio.run(_test())


def test_soft_and_physical_deletion(store: PgVectorStore) -> None:
    """Verify soft_delete=True sets is_active=False and soft_delete=False removes record."""

    async def _test() -> None:
        emb = VectorEmbedding(
            vector=[1.0, 0.0, 0.0, 0.0],
            model_name="text-embedding-3-small",
            dimension=4,
            version="v1",
        )
        item1 = MemoryItem(
            memory_type=MemoryType.SEMANTIC,
            scope=MemoryScope.USER,
            user_id="user_del",
            content="Item to soft delete",
            embeddings=[emb],
        )
        item2 = MemoryItem(
            memory_type=MemoryType.SEMANTIC,
            scope=MemoryScope.USER,
            user_id="user_del",
            content="Item to hard delete",
            embeddings=[emb],
        )

        await store.upsert(item1)
        await store.upsert(item2)

        owner_ctx = MemoryAccessContext(user_id="user_del")

        # Soft Delete
        del_soft = await store.delete(item1.id, owner_ctx, soft_delete=True)
        assert del_soft.success is True
        assert await store.get(item1.id, owner_ctx) is None

        # Physical Delete
        del_hard = await store.delete(item2.id, owner_ctx, soft_delete=False)
        assert del_hard.success is True
        assert await store.get(item2.id, owner_ctx) is None

    asyncio.run(_test())


def test_expiration_filtering(store: PgVectorStore) -> None:
    """Verify expired memory items are excluded from get() and search()."""

    async def _test() -> None:
        emb = VectorEmbedding(
            vector=[1.0, 0.0, 0.0, 0.0],
            model_name="text-embedding-3-small",
            dimension=4,
            version="v1",
        )
        past_time = datetime.now(UTC) - timedelta(hours=1)

        expired_item = MemoryItem(
            memory_type=MemoryType.SEMANTIC,
            scope=MemoryScope.USER,
            user_id="user_ttl",
            content="Expired item",
            embeddings=[emb],
            expires_at=past_time,
        )

        await store.upsert(expired_item)
        owner_ctx = MemoryAccessContext(user_id="user_ttl")

        assert await store.get(expired_item.id, owner_ctx) is None

        search_res = await store.search(MemoryQuery(user_id="user_ttl"), owner_ctx)
        assert search_res.total_found == 0

    asyncio.run(_test())


def test_database_failure_handling(mock_db: MockSessionMaker, store: PgVectorStore) -> None:
    """Verify database errors return safe domain-level failure responses without exposing credentials."""

    async def _test() -> None:
        mock_db.should_fail = True
        emb = VectorEmbedding(
            vector=[1.0, 0.0, 0.0, 0.0],
            model_name="text-embedding-3-small",
            dimension=4,
            version="v1",
        )
        item = MemoryItem(
            memory_type=MemoryType.SEMANTIC,
            scope=MemoryScope.USER,
            user_id="user_fail",
            content="Failing item",
            embeddings=[emb],
        )

        up_res = await store.upsert(item)
        assert up_res.success is False
        assert up_res.error_message == "Database persistent storage unavailable."

        owner_ctx = MemoryAccessContext(user_id="user_fail")
        assert await store.get(item.id, owner_ctx) is None

        del_res = await store.delete(item.id, owner_ctx)
        assert del_res.success is False
        assert del_res.error_message == "Database deletion operation failed."

        assert await store.health() is False

    asyncio.run(_test())
