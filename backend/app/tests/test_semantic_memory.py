"""Unit Tests for SemanticMemory Manager.

Validates concept storage with automatic embedding generation, semantic similarity search,
embedding failure handling, deletion, and MemoryAccessContext privacy enforcement.
"""

import asyncio

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
from app.memory.semantic import SemanticMemory
from app.memory.vector_store import InMemoryVectorStore
from app.tests.test_memory_embeddings import MockEmbeddingProvider


def test_semantic_memory_store_and_search() -> None:
    """Verify concept storage with automatic embedding generation and search recall."""

    async def _test() -> None:
        vector_store = InMemoryVectorStore(expected_dimension=4)
        embedding_provider = MockEmbeddingProvider(dimension=4)
        semantic = SemanticMemory(vector_store=vector_store, embedding_provider=embedding_provider)

        item = MemoryItem(
            memory_type=MemoryType.SEMANTIC,
            scope=MemoryScope.USER,
            user_id="user_sem",
            content="Concept item without initial embedding",
            metadata=MemoryMetadata(privacy=MemoryPrivacy.PUBLIC),
        )

        # Store concept (generates embedding automatically)
        res = await semantic.store_concept(item)
        assert res.success is True
        assert len(item.embeddings) == 1
        assert item.embeddings[0].dimension == 4

        # Search concepts
        query = MemoryQuery(query_text="Concept item")
        access_ctx = MemoryAccessContext(user_id="user_sem")

        search_res = await semantic.search_concepts(query, access_ctx)
        assert search_res.total_found == 1
        assert search_res.items[0].id == item.id

        # Delete concept
        del_res = await semantic.delete_concept(item.id, access_ctx, soft_delete=True)
        assert del_res.success is True

    asyncio.run(_test())


def test_semantic_memory_embedding_failure_and_privacy_denial() -> None:
    """Verify embedding failure is explicitly reported and privacy access is enforced."""

    async def _test() -> None:
        vector_store = InMemoryVectorStore(expected_dimension=4)
        failing_provider = MockEmbeddingProvider(dimension=4, should_fail=True)
        semantic = SemanticMemory(vector_store=vector_store, embedding_provider=failing_provider)

        item = MemoryItem(
            memory_type=MemoryType.SEMANTIC,
            scope=MemoryScope.USER,
            user_id="user_owner",
            content="Failing concept item",
            metadata=MemoryMetadata(privacy=MemoryPrivacy.PRIVATE),
        )

        # Embedding failure does not silently succeed
        res = await semantic.store_concept(item)
        assert res.success is False
        assert "Embedding generation failed" in (res.error_message or "")

        # Privacy denial
        owner_ctx = MemoryAccessContext(user_id="user_owner")
        other_ctx = MemoryAccessContext(user_id="user_other")

        item.embeddings.append(VectorEmbedding(vector=[0.1, 0.1, 0.1, 0.1], dimension=4))
        denied_write = await semantic.store_concept(item, access_context=other_ctx)
        assert denied_write.success is False
        assert denied_write.error_message == "Access Denied: Caller context cannot write item."

        ok_write = await semantic.store_concept(item, access_context=owner_ctx)
        assert ok_write.success is True

    asyncio.run(_test())
