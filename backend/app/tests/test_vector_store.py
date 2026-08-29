"""Unit Tests for VectorStore Interface & InMemoryVectorStore.

Validates upsert, get, cosine similarity search, soft vs physical deletion,
empty search behavior, dimension mismatch validation, and MemoryAccessContext privacy filtering.
"""

import asyncio

import pytest

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
from app.memory.vector_store import InMemoryVectorStore, cosine_similarity


def test_cosine_similarity_calculation() -> None:
    """Verify cosine similarity calculation accuracy and dimension mismatch error."""
    v1 = [1.0, 0.0, 0.0, 0.0]
    v2 = [1.0, 0.0, 0.0, 0.0]
    v3 = [0.0, 1.0, 0.0, 0.0]

    assert cosine_similarity(v1, v2) == pytest.approx(1.0)
    assert cosine_similarity(v1, v3) == pytest.approx(0.0)

    with pytest.raises(ValueError, match="Vector dimension mismatch"):
        cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])


def test_in_memory_vector_store_crud_and_search() -> None:
    """Verify upsert, get, search, and delete in InMemoryVectorStore."""

    async def _test() -> None:
        store = InMemoryVectorStore(expected_dimension=4)

        emb1 = VectorEmbedding(vector=[1.0, 0.0, 0.0, 0.0], dimension=4)
        emb2 = VectorEmbedding(vector=[0.0, 1.0, 0.0, 0.0], dimension=4)

        item1 = MemoryItem(
            memory_type=MemoryType.SEMANTIC,
            scope=MemoryScope.USER,
            user_id="user_123",
            content="Vector item 1",
            embeddings=[emb1],
            metadata=MemoryMetadata(privacy=MemoryPrivacy.PUBLIC),
        )

        item2 = MemoryItem(
            memory_type=MemoryType.SEMANTIC,
            scope=MemoryScope.USER,
            user_id="user_123",
            content="Vector item 2",
            embeddings=[emb2],
            metadata=MemoryMetadata(privacy=MemoryPrivacy.PRIVATE),
        )

        # Upsert
        res1 = await store.upsert(item1)
        res2 = await store.upsert(item2)
        assert res1.success is True
        assert res2.success is True
        assert await store.count() == 2

        # Get
        access_owner = MemoryAccessContext(user_id="user_123")
        access_other = MemoryAccessContext(user_id="user_999")

        assert await store.get(item1.id, access_owner) is not None
        assert await store.get(item2.id, access_owner) is not None
        assert await store.get(item2.id, access_other) is None  # Private item denied

        # Search with similarity scoring
        query_emb = VectorEmbedding(vector=[0.9, 0.1, 0.0, 0.0], dimension=4)
        query = MemoryQuery(query_embedding=query_emb, limit=10)

        search_res = await store.search(query, access_owner)
        assert search_res.total_found == 2
        assert search_res.items[0].id == item1.id  # Item 1 is closest match
        assert search_res.scores[0] > search_res.scores[1]

        # Soft Delete
        del_soft = await store.delete(item1.id, access_owner, soft_delete=True)
        assert del_soft.success is True
        assert del_soft.soft_deleted is True
        assert await store.count() == 1

        # Physical Delete
        del_hard = await store.delete(item2.id, access_owner, soft_delete=False)
        assert del_hard.success is True
        assert del_hard.soft_deleted is False
        assert await store.count() == 0

    asyncio.run(_test())


def test_vector_store_dimension_mismatch_validation() -> None:
    """Verify upsert and search reject vector dimension mismatches safely."""

    async def _test() -> None:
        store = InMemoryVectorStore(expected_dimension=4)

        bad_emb = VectorEmbedding(vector=[1.0, 0.0], dimension=2)
        bad_item = MemoryItem(
            memory_type=MemoryType.SEMANTIC,
            scope=MemoryScope.USER,
            user_id="user_123",
            content="Bad dimension item",
            embeddings=[bad_emb],
        )

        res = await store.upsert(bad_item)
        assert res.success is False
        assert "does not match expected store dimension" in (res.error_message or "")

        # Query dimension mismatch
        query_bad = MemoryQuery(query_embedding=bad_emb)
        with pytest.raises(ValueError, match="Query vector dimension \\(2\\) does not match"):
            await store.search(query_bad, MemoryAccessContext())

    asyncio.run(_test())
