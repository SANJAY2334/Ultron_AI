"""Episodic Memory Implementation.

Provides concrete IEpisodicMemory implementation managing temporal interaction episodes,
preserving timestamps, session/user/project/planner attributes, provenance, and MemoryAccessContext privacy filtering.
"""

import logging

from app.memory.base import IEpisodicMemory
from app.memory.embeddings import BaseEmbeddingProvider, EmbeddingError
from app.memory.models import (
    MemoryAccessContext,
    MemoryDeleteResult,
    MemoryItem,
    MemoryQuery,
    MemorySearchResult,
    MemoryType,
    MemoryWriteResult,
    VectorEmbedding,
)
from app.memory.vector_store import BaseVectorStore

logger = logging.getLogger(__name__)


class EpisodicMemory(IEpisodicMemory):
    """Concrete implementation of IEpisodicMemory for temporal episode storage and recall."""

    def __init__(
        self,
        vector_store: BaseVectorStore,
        embedding_provider: BaseEmbeddingProvider | None = None,
    ) -> None:
        """Initializes EpisodicMemory manager."""
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider

    async def store_episode(
        self,
        item: MemoryItem,
        access_context: MemoryAccessContext | None = None,
    ) -> MemoryWriteResult:
        """Stores a temporal interaction episode in Episodic Memory."""
        if access_context and not access_context.can_access(item.metadata.privacy, item.user_id):
            return MemoryWriteResult(
                memory_id=item.id,
                success=False,
                memory_type=item.memory_type,
                error_message="Access Denied: Caller context cannot write episode item.",
            )

        item.memory_type = MemoryType.EPISODIC

        # Optionally generate vector embedding if embedding provider is supplied
        if not item.embeddings and self.embedding_provider:
            try:
                embedding = await self.embedding_provider.embed_text(item.content)
                item.embeddings.append(embedding)
            except EmbeddingError as exc:
                logger.warning(
                    f"EpisodicMemory embedding generation skipped for episode '{item.id}': {exc}"
                )

        # Fallback dummy embedding for vector store persistence if none generated
        if not item.embeddings:
            item.embeddings.append(
                VectorEmbedding(
                    vector=[0.0] * 4,
                    dimension=4,
                    model_name="fallback-episodic",
                    version="v1",
                )
            )

        return await self.vector_store.upsert(item)

    async def recall_episodes(
        self, query: MemoryQuery, access_context: MemoryAccessContext
    ) -> MemorySearchResult:
        """Recalls relevant temporal episode items matching search parameters and authorization context."""
        query.memory_types = [MemoryType.EPISODIC]

        # Generate query embedding if search query text and embedding provider are supplied
        if self.embedding_provider and query.query_text and not query.query_embedding:
            try:
                query.query_embedding = await self.embedding_provider.embed_text(query.query_text)
            except EmbeddingError as exc:
                logger.warning(f"EpisodicMemory query embedding failed: {exc}")

        result = await self.vector_store.search(query, access_context)

        # Sort recalled episodes chronologically by created_at timestamp
        sorted_pairs = sorted(
            zip(result.items, result.scores, strict=True),
            key=lambda x: x[0].created_at,
        )

        result.items = [item for item, _ in sorted_pairs]
        result.scores = [score for _, score in sorted_pairs]

        return result

    async def delete_episode(
        self,
        memory_id: str,
        access_context: MemoryAccessContext,
        soft_delete: bool = True,
    ) -> MemoryDeleteResult:
        """Deletes or tombstones an episode item by ID."""
        return await self.vector_store.delete(memory_id, access_context, soft_delete=soft_delete)
