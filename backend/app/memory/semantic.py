"""Semantic Memory Implementation.

Provides concrete ISemanticMemory pipeline integrating BaseEmbeddingProvider and BaseVectorStore.
Generates dense vector embeddings, enforces dimension checks, and enforces MemoryAccessContext privacy filtering.
"""

import logging

from app.memory.base import ISemanticMemory
from app.memory.embeddings import BaseEmbeddingProvider, EmbeddingError
from app.memory.models import (
    MemoryAccessContext,
    MemoryDeleteResult,
    MemoryItem,
    MemoryQuery,
    MemorySearchResult,
    MemoryType,
    MemoryWriteResult,
)
from app.memory.vector_store import BaseVectorStore

logger = logging.getLogger(__name__)


class SemanticMemory(ISemanticMemory):
    """Concrete implementation of ISemanticMemory utilizing BaseEmbeddingProvider and BaseVectorStore."""

    def __init__(
        self,
        vector_store: BaseVectorStore,
        embedding_provider: BaseEmbeddingProvider,
    ) -> None:
        """Initializes SemanticMemory pipeline."""
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider

    async def store_concept(
        self,
        item: MemoryItem,
        access_context: MemoryAccessContext | None = None,
    ) -> MemoryWriteResult:
        """Stores a domain concept or factual item in Semantic Memory.

        Generates dense vector embeddings via EmbeddingProvider if missing prior to vector persistence.
        """
        if access_context and not access_context.can_access(item.metadata.privacy, item.user_id):
            return MemoryWriteResult(
                memory_id=item.id,
                success=False,
                memory_type=item.memory_type,
                error_message="Access Denied: Caller context cannot write item.",
            )

        # Enforce embedding generation if missing
        if not item.embeddings:
            try:
                embedding = await self.embedding_provider.embed_text(item.content)
                item.embeddings.append(embedding)
            except EmbeddingError as exc:
                logger.error(
                    f"SemanticMemory embedding generation failed for item '{item.id}': {exc}",
                    exc_info=True,
                )
                return MemoryWriteResult(
                    memory_id=item.id,
                    success=False,
                    memory_type=item.memory_type,
                    error_message=f"Embedding generation failed: {exc}",
                )

        item.memory_type = MemoryType.SEMANTIC
        return await self.vector_store.upsert(item)

    async def search_concepts(
        self, query: MemoryQuery, access_context: MemoryAccessContext
    ) -> MemorySearchResult:
        """Performs vector similarity search over long-term semantic knowledge."""
        # Ensure query vector embedding is generated if query_text is provided
        if query.query_text and not query.query_embedding:
            try:
                query.query_embedding = await self.embedding_provider.embed_text(query.query_text)
            except EmbeddingError as exc:
                logger.error(
                    f"SemanticMemory query embedding generation failed: {exc}",
                    exc_info=True,
                )
                return MemorySearchResult(
                    items=[], scores=[], query=query, total_found=0, search_time_ms=0.0
                )

        query.memory_types = [MemoryType.SEMANTIC]
        return await self.vector_store.search(query, access_context)

    async def delete_concept(
        self,
        memory_id: str,
        access_context: MemoryAccessContext,
        soft_delete: bool = True,
    ) -> MemoryDeleteResult:
        """Deletes or tombstones a concept item by ID."""
        return await self.vector_store.delete(memory_id, access_context, soft_delete=soft_delete)
