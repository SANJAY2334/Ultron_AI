"""Framework-Agnostic Vector Store Interface & In-Memory Implementation.

Provides abstract BaseVectorStore contract and concrete InMemoryVectorStore implementation
supporting cosine similarity search, dimension validation, and MemoryAccessContext privacy filtering.
"""

import logging
import math
import time
from abc import ABC, abstractmethod
from datetime import UTC, datetime

from app.memory.models import (
    MemoryAccessContext,
    MemoryDeleteResult,
    MemoryItem,
    MemoryQuery,
    MemorySearchResult,
    MemoryWriteResult,
    VectorEmbedding,
)

logger = logging.getLogger(__name__)


class BaseVectorStore(ABC):
    """Framework-agnostic abstract base class for vector store backends."""

    @abstractmethod
    async def upsert(self, item: MemoryItem) -> MemoryWriteResult:
        """Inserts or updates a memory item with its vector embedding."""

    @abstractmethod
    async def search(
        self, query: MemoryQuery, access_context: MemoryAccessContext
    ) -> MemorySearchResult:
        """Executes vector similarity search enforcing privacy access filtering."""

    @abstractmethod
    async def delete(
        self,
        memory_id: str,
        access_context: MemoryAccessContext,
        soft_delete: bool = True,
    ) -> MemoryDeleteResult:
        """Deletes or tombstones a vector item by ID."""

    @abstractmethod
    async def get(self, memory_id: str, access_context: MemoryAccessContext) -> MemoryItem | None:
        """Retrieves a single vector item by ID."""

    @abstractmethod
    async def count(self) -> int:
        """Returns total active count of stored items."""

    @abstractmethod
    async def health(self) -> bool:
        """Checks operational health of vector store backend."""


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Computes exact cosine similarity score between two float vectors."""
    if len(v1) != len(v2):
        raise ValueError(
            f"Vector dimension mismatch for similarity calculation: {len(v1)} vs {len(v2)}."
        )

    dot_product = sum(a * b for a, b in zip(v1, v2, strict=True))
    norm_v1 = math.sqrt(sum(a * a for a in v1))
    norm_v2 = math.sqrt(sum(b * b for b in v2))

    if norm_v1 == 0.0 or norm_v2 == 0.0:
        return 0.0

    return dot_product / (norm_v1 * norm_v2)


class InMemoryVectorStore(BaseVectorStore):
    """Concrete in-memory implementation of BaseVectorStore for vector storage and cosine search."""

    def __init__(self, expected_dimension: int = 1536) -> None:
        self.expected_dimension = expected_dimension
        # items[memory_id] = MemoryItem
        self._store: dict[str, MemoryItem] = {}

    async def upsert(self, item: MemoryItem) -> MemoryWriteResult:
        """Inserts or updates a MemoryItem in the vector store after dimension validation."""
        if not item.embeddings:
            return MemoryWriteResult(
                memory_id=item.id,
                success=False,
                memory_type=item.memory_type,
                error_message="MemoryItem requires at least one VectorEmbedding for vector store persistence.",
            )

        # Validate vector dimension
        embedding = item.embeddings[0]
        if embedding.dimension != self.expected_dimension:
            return MemoryWriteResult(
                memory_id=item.id,
                success=False,
                memory_type=item.memory_type,
                error_message=f"Embedding dimension ({embedding.dimension}) does not match expected store dimension ({self.expected_dimension}).",
            )

        if len(embedding.vector) != self.expected_dimension:
            return MemoryWriteResult(
                memory_id=item.id,
                success=False,
                memory_type=item.memory_type,
                error_message=f"Vector length ({len(embedding.vector)}) does not match expected store dimension ({self.expected_dimension}).",
            )

        item.updated_at = datetime.now(UTC)
        self._store[item.id] = item

        return MemoryWriteResult(
            memory_id=item.id,
            success=True,
            memory_type=item.memory_type,
        )

    async def search(
        self, query: MemoryQuery, access_context: MemoryAccessContext
    ) -> MemorySearchResult:
        """Executes similarity search over stored embeddings filtering by privacy, confidence, and importance."""
        start_time = time.perf_counter()
        now = datetime.now(UTC)

        query_vec: VectorEmbedding | None = query.query_embedding
        if query_vec:
            if query_vec.dimension != self.expected_dimension:
                raise ValueError(
                    f"Query vector dimension ({query_vec.dimension}) does not match store dimension ({self.expected_dimension})."
                )
            if len(query_vec.vector) != self.expected_dimension:
                raise ValueError(
                    f"Query vector length ({len(query_vec.vector)}) does not match store dimension ({self.expected_dimension})."
                )

        scored_items: list[tuple[MemoryItem, float]] = []

        for item in self._store.values():
            # Inactive check
            if not item.is_active and not query.include_inactive:
                continue

            # Expiration check
            if item.expires_at and item.expires_at < now:
                continue

            # Memory type filter
            if query.memory_types and item.memory_type not in query.memory_types:
                continue

            # Scope filter
            if query.scope and item.scope != query.scope:
                continue

            # Session / User / Project filter
            if query.session_id and item.session_id != query.session_id:
                continue
            if query.user_id and item.user_id != query.user_id:
                continue
            if query.project_id and item.project_id != query.project_id:
                continue

            # Confidence and importance thresholds
            if item.metadata.confidence < query.min_confidence:
                continue
            if item.importance < query.min_importance:
                continue

            # Privacy access control
            if not access_context.can_access(item.metadata.privacy, item.user_id):
                continue

            # Similarity score calculation
            score = 1.0
            if query_vec and item.embeddings:
                score = cosine_similarity(query_vec.vector, item.embeddings[0].vector)

            scored_items.append((item, score))

        # Sort by similarity score descending
        scored_items.sort(key=lambda x: x[1], reverse=True)

        # Slice limit
        sliced = scored_items[: query.limit]
        res_items = [item for item, _ in sliced]
        res_scores = [score for _, score in sliced]

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return MemorySearchResult(
            items=res_items,
            scores=res_scores,
            query=query,
            total_found=len(scored_items),
            search_time_ms=elapsed_ms,
        )

    async def get(self, memory_id: str, access_context: MemoryAccessContext) -> MemoryItem | None:
        """Retrieves a single MemoryItem by ID if accessible."""
        item = self._store.get(memory_id)
        if not item or not item.is_active:
            return None

        if not access_context.can_access(item.metadata.privacy, item.user_id):
            return None

        return item

    async def delete(
        self,
        memory_id: str,
        access_context: MemoryAccessContext,
        soft_delete: bool = True,
    ) -> MemoryDeleteResult:
        """Deletes or tombstones a MemoryItem by ID."""
        item = self._store.get(memory_id)
        if not item:
            return MemoryDeleteResult(
                memory_id=memory_id,
                success=False,
                soft_deleted=soft_delete,
                deleted_count=0,
                error_message="Item not found.",
            )

        if not access_context.can_access(item.metadata.privacy, item.user_id):
            return MemoryDeleteResult(
                memory_id=memory_id,
                success=False,
                soft_deleted=soft_delete,
                deleted_count=0,
                error_message="Access Denied.",
            )

        if soft_delete:
            item.is_active = False
            item.updated_at = datetime.now(UTC)
        else:
            del self._store[memory_id]

        return MemoryDeleteResult(
            memory_id=memory_id,
            success=True,
            soft_deleted=soft_delete,
            deleted_count=1,
        )

    async def count(self) -> int:
        """Returns count of active items stored."""
        return sum(1 for item in self._store.values() if item.is_active)

    async def health(self) -> bool:
        """Health check returns true for in-memory store."""
        return True
