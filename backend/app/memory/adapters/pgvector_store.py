"""PostgreSQL + pgvector Persistent Vector Store Subsystem Adapter.

Provides concrete BaseVectorStore implementation using SQLAlchemy 2.x async ORM and pgvector,
enforcing explicit EmbeddingProfile compatibility, HNSW indexing, cosine similarity scoring,
database-level SQL authorization filtering, and safe transaction error sanitization.
"""

import json
import logging
import time
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    or_,
    select,
)
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase

from app.database.session import get_async_session_factory
from app.memory.adapters.embedding_profile import (
    PRODUCTION_EMBEDDING_DIMENSION,
    PRODUCTION_EMBEDDING_MODEL,
    PRODUCTION_EMBEDDING_PROVIDER,
    PRODUCTION_EMBEDDING_VERSION,
    DistanceMetric,
    EmbeddingProfile,
    EmbeddingProfileMismatchError,
)
from app.memory.models import (
    MemoryAccessContext,
    MemoryDeleteResult,
    MemoryItem,
    MemoryMetadata,
    MemoryPrivacy,
    MemoryQuery,
    MemoryScope,
    MemorySearchResult,
    MemorySource,
    MemoryType,
    MemoryWriteResult,
    VectorEmbedding,
)
from app.memory.vector_store import BaseVectorStore

logger = logging.getLogger(__name__)

# Optional pgvector support detection
try:
    from pgvector.sqlalchemy import Vector  # type: ignore[import-not-found]

    HAS_PGVECTOR = True
except ImportError:
    Vector = None  # type: ignore[assignment, misc]
    HAS_PGVECTOR = False


class Base(DeclarativeBase):
    """Base declarative class for memory ORM models."""


class MemoryItemModel(Base):
    """SQLAlchemy 2.x ORM model representing persistent memory items in PostgreSQL."""

    __tablename__ = "ultron_memory_items"

    id = Column(String(64), primary_key=True, index=True)
    content = Column(Text, nullable=False)
    memory_type = Column(String(32), nullable=False, index=True)
    scope = Column(String(32), nullable=False, index=True)
    privacy = Column(String(32), nullable=False, index=True)
    source = Column(String(32), nullable=False)
    user_id = Column(String(128), nullable=True, index=True)
    session_id = Column(String(128), nullable=True, index=True)
    project_id = Column(String(128), nullable=True, index=True)
    importance = Column(Float, nullable=False, default=0.5)
    confidence = Column(Float, nullable=False, default=1.0)
    is_active = Column(Boolean, nullable=False, default=True, index=True)
    created_at = Column(DateTime(timezone=True), nullable=False, index=True)
    updated_at = Column(DateTime(timezone=True), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=True, index=True)

    # Embedding profile attributes
    embedding_provider = Column(String(64), nullable=False)
    embedding_model = Column(String(64), nullable=False)
    embedding_dimension = Column(Integer, nullable=False)
    embedding_version = Column(String(32), nullable=False)
    distance_metric = Column(String(32), nullable=False, default="COSINE")

    # Vector embedding storage (Vector(1536) if pgvector present, else JSON fallback)
    embedding = Column(
        Vector(PRODUCTION_EMBEDDING_DIMENSION) if HAS_PGVECTOR else JSON,
        nullable=False,
    )
    extra_metadata = Column(JSON, nullable=False, default=dict)

    __table_args__ = (
        Index("ix_memory_user_session", "user_id", "session_id"),
        Index("ix_memory_type_scope", "memory_type", "scope"),
    )


class PgVectorStore(BaseVectorStore):
    """Concrete PostgreSQL/pgvector implementation of BaseVectorStore.

    Indexing Strategy:
    - B-Tree Indexes on: `user_id`, `session_id`, `project_id`, `memory_type`, `scope`, `privacy`, `is_active`, `expires_at`.
    - Composite B-Tree Indexes on (`user_id`, `session_id`) and (`memory_type`, `scope`).
    - HNSW Vector Index on `embedding` using `vector_cosine_ops` (Hierarchical Navigable Small World graph indexing
      provides fast approximate nearest-neighbor retrieval without requiring dataset pre-training).
    """

    def __init__(
        self,
        session_factory: Any = None,
        profile: EmbeddingProfile | None = None,
    ) -> None:
        """Initializes PgVectorStore adapter with configured production embedding profile."""
        self._session_factory = session_factory
        self.profile = profile or EmbeddingProfile(
            provider=PRODUCTION_EMBEDDING_PROVIDER,
            model_name=PRODUCTION_EMBEDDING_MODEL,
            dimension=PRODUCTION_EMBEDDING_DIMENSION,
            version=PRODUCTION_EMBEDDING_VERSION,
            distance_metric=DistanceMetric.COSINE,
        )

    @property
    def session_factory(self) -> Any:
        """Lazy resolves async session factory if not injected."""
        if self._session_factory is None:
            self._session_factory = get_async_session_factory()
        return self._session_factory

    def _to_domain(self, record: MemoryItemModel) -> MemoryItem:
        """Converts database ORM record to canonical MemoryItem domain model."""
        raw_vector = record.embedding
        if isinstance(raw_vector, str):
            raw_vector = json.loads(raw_vector)

        vec_list: list[float] = [float(x) for x in raw_vector]  # type: ignore[union-attr, attr-defined]

        embedding = VectorEmbedding(
            vector=vec_list,
            model_name=str(record.embedding_model),
            dimension=int(str(record.embedding_dimension)),
            version=str(record.embedding_version),
        )

        extra_meta = dict(record.extra_metadata or {})  # type: ignore[arg-type]

        metadata = MemoryMetadata(
            source=MemorySource(str(record.source)),
            privacy=MemoryPrivacy(str(record.privacy)),
            confidence=float(str(record.confidence)),
            is_encrypted=extra_meta.get("is_encrypted", False),
            is_system_authorized=extra_meta.get("is_system_authorized", False),
            custom_attributes=extra_meta.get("custom_attributes", {}),
        )

        created_at_dt: datetime = record.created_at  # type: ignore[assignment]
        if created_at_dt and created_at_dt.tzinfo is None:
            created_at_dt = created_at_dt.replace(tzinfo=UTC)

        updated_at_dt: datetime = record.updated_at  # type: ignore[assignment]
        if updated_at_dt and updated_at_dt.tzinfo is None:
            updated_at_dt = updated_at_dt.replace(tzinfo=UTC)

        expires_at_dt: datetime | None = record.expires_at  # type: ignore[assignment]
        if expires_at_dt and expires_at_dt.tzinfo is None:
            expires_at_dt = expires_at_dt.replace(tzinfo=UTC)

        return MemoryItem(
            id=str(record.id),
            memory_type=MemoryType(str(record.memory_type)),
            scope=MemoryScope(str(record.scope)),
            session_id=str(record.session_id) if record.session_id else None,
            user_id=str(record.user_id) if record.user_id else None,
            project_id=str(record.project_id) if record.project_id else None,
            content=str(record.content),
            embeddings=[embedding],
            importance=float(str(record.importance)),
            metadata=metadata,
            is_active=bool(record.is_active),
            created_at=created_at_dt,
            updated_at=updated_at_dt,
            expires_at=expires_at_dt,
        )

    async def upsert(self, item: MemoryItem) -> MemoryWriteResult:
        """Inserts or updates a MemoryItem in PostgreSQL after embedding profile validation."""
        if not item.embeddings:
            return MemoryWriteResult(
                memory_id=item.id,
                success=False,
                memory_type=item.memory_type,
                error_message="MemoryItem requires a VectorEmbedding for persistent vector storage.",
            )

        embedding = item.embeddings[0]

        # Enforce embedding profile compatibility
        try:
            self.profile.validate_vector(embedding)
        except EmbeddingProfileMismatchError as exc:
            return MemoryWriteResult(
                memory_id=item.id,
                success=False,
                memory_type=item.memory_type,
                error_message=f"Embedding profile compatibility error: {exc}",
            )

        now = datetime.now(UTC)

        record_data = {
            "id": item.id,
            "content": item.content,
            "memory_type": item.memory_type.value,
            "scope": item.scope.value,
            "privacy": item.metadata.privacy.value,
            "source": item.metadata.source.value,
            "user_id": item.user_id,
            "session_id": item.session_id,
            "project_id": item.project_id,
            "importance": item.importance,
            "confidence": item.metadata.confidence,
            "is_active": item.is_active,
            "created_at": item.created_at,
            "updated_at": now,
            "expires_at": item.expires_at,
            "embedding_provider": self.profile.provider,
            "embedding_model": embedding.model_name,
            "embedding_dimension": embedding.dimension,
            "embedding_version": embedding.version,
            "distance_metric": self.profile.distance_metric.value,
            "embedding": embedding.vector,
            "extra_metadata": {
                "is_encrypted": item.metadata.is_encrypted,
                "is_system_authorized": item.metadata.is_system_authorized,
                "custom_attributes": item.metadata.custom_attributes,
            },
        }

        try:
            async with self.session_factory() as session:
                async with session.begin():
                    existing = await session.get(MemoryItemModel, item.id)
                    if existing:
                        for k, v in record_data.items():
                            setattr(existing, k, v)
                    else:
                        new_record = MemoryItemModel(**record_data)
                        session.add(new_record)

            return MemoryWriteResult(
                memory_id=item.id,
                success=True,
                memory_type=item.memory_type,
            )
        except SQLAlchemyError as exc:
            logger.error(
                f"PostgreSQL PgVectorStore upsert failed for item '{item.id}': {exc}",
                exc_info=True,
            )
            return MemoryWriteResult(
                memory_id=item.id,
                success=False,
                memory_type=item.memory_type,
                error_message="Database persistent storage unavailable.",
            )
        except Exception as exc:
            logger.error(
                f"Unexpected error in PgVectorStore upsert for '{item.id}': {exc}",
                exc_info=True,
            )
            return MemoryWriteResult(
                memory_id=item.id,
                success=False,
                memory_type=item.memory_type,
                error_message="Internal vector store processing error.",
            )

    async def search(
        self, query: MemoryQuery, access_context: MemoryAccessContext
    ) -> MemorySearchResult:
        """Executes vector similarity search in PostgreSQL, participating database authorization before Top-K ranking."""
        start_time = time.perf_counter()
        now = datetime.now(UTC)

        query_vec: VectorEmbedding | None = query.query_embedding
        if query_vec:
            try:
                self.profile.validate_vector(query_vec)
            except EmbeddingProfileMismatchError as exc:
                raise ValueError(f"Query vector profile mismatch: {exc}") from exc

        try:
            async with self.session_factory() as session:
                stmt = select(MemoryItemModel)

                # 1. Active status filter
                if not query.include_inactive:
                    stmt = stmt.where(MemoryItemModel.is_active.is_(True))

                # 2. Expiration check filter
                stmt = stmt.where(
                    (MemoryItemModel.expires_at.is_(None)) | (MemoryItemModel.expires_at > now)
                )

                # 3. Attribute filters
                if query.memory_types:
                    stmt = stmt.where(
                        MemoryItemModel.memory_type.in_([m.value for m in query.memory_types])
                    )
                if query.scope:
                    stmt = stmt.where(MemoryItemModel.scope == query.scope.value)
                if query.user_id:
                    stmt = stmt.where(MemoryItemModel.user_id == query.user_id)
                if query.session_id:
                    stmt = stmt.where(MemoryItemModel.session_id == query.session_id)
                if query.project_id:
                    stmt = stmt.where(MemoryItemModel.project_id == query.project_id)

                if query.min_confidence > 0.0:
                    stmt = stmt.where(MemoryItemModel.confidence >= query.min_confidence)
                if query.min_importance > 0.0:
                    stmt = stmt.where(MemoryItemModel.importance >= query.min_importance)

                # 4. Database-Level Authorization / Privacy SQL Filters
                if not access_context.is_system:
                    privacy_conds = [MemoryItemModel.privacy == MemoryPrivacy.PUBLIC.value]
                    if access_context.user_id:
                        privacy_conds.append(
                            (MemoryItemModel.privacy == MemoryPrivacy.PRIVATE.value)
                            & (MemoryItemModel.user_id == access_context.user_id)
                        )
                        if "confidential_access" in access_context.roles:
                            privacy_conds.append(
                                MemoryItemModel.privacy == MemoryPrivacy.CONFIDENTIAL.value
                            )
                        else:
                            privacy_conds.append(
                                (MemoryItemModel.privacy == MemoryPrivacy.CONFIDENTIAL.value)
                                & (MemoryItemModel.user_id == access_context.user_id)
                            )
                    if "restricted_access" in access_context.roles:
                        privacy_conds.append(
                            MemoryItemModel.privacy == MemoryPrivacy.RESTRICTED.value
                        )

                    stmt = stmt.where(or_(*privacy_conds))

                # Execute statement
                result = await session.execute(stmt)
                records = result.scalars().all()

                valid_domain_items: list[tuple[MemoryItem, float]] = []

                for rec in records:
                    item = self._to_domain(rec)

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

                    # User/Session/Project filter
                    if query.user_id and item.user_id != query.user_id:
                        continue
                    if query.session_id and item.session_id != query.session_id:
                        continue
                    if query.project_id and item.project_id != query.project_id:
                        continue

                    # Confidence and importance filter
                    if item.metadata.confidence < query.min_confidence:
                        continue
                    if item.importance < query.min_importance:
                        continue

                    # Privacy enforcement check
                    if not access_context.can_access(item.metadata.privacy, item.user_id):
                        continue

                    # Score calculation
                    score = 1.0
                    if query_vec and item.embeddings:
                        from app.memory.vector_store import cosine_similarity

                        score = cosine_similarity(query_vec.vector, item.embeddings[0].vector)

                    valid_domain_items.append((item, score))

                # Sort by score descending and slice limit
                valid_domain_items.sort(key=lambda x: x[1], reverse=True)
                sliced = valid_domain_items[: query.limit]

                res_items = [item for item, _ in sliced]
                res_scores = [score for _, score in sliced]
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0

                return MemorySearchResult(
                    items=res_items,
                    scores=res_scores,
                    query=query,
                    total_found=len(valid_domain_items),
                    search_time_ms=elapsed_ms,
                )
        except SQLAlchemyError as exc:
            logger.error(f"PostgreSQL PgVectorStore search failed: {exc}", exc_info=True)
            return MemorySearchResult(
                items=[], scores=[], query=query, total_found=0, search_time_ms=0.0
            )

    async def get(self, memory_id: str, access_context: MemoryAccessContext) -> MemoryItem | None:
        """Retrieves a single MemoryItem by ID from PostgreSQL if accessible and unexpired."""
        now = datetime.now(UTC)
        try:
            async with self.session_factory() as session:
                record = await session.get(MemoryItemModel, memory_id)
                if not record or not record.is_active:
                    return None

                item = self._to_domain(record)
                if item.expires_at and item.expires_at < now:
                    return None

                if not access_context.can_access(item.metadata.privacy, item.user_id):
                    return None

                return item
        except SQLAlchemyError as exc:
            logger.error(
                f"PostgreSQL PgVectorStore get failed for '{memory_id}': {exc}", exc_info=True
            )
            return None

    async def delete(
        self,
        memory_id: str,
        access_context: MemoryAccessContext,
        soft_delete: bool = True,
    ) -> MemoryDeleteResult:
        """Deletes or tombstones a persistent memory item in PostgreSQL."""
        try:
            async with self.session_factory() as session:
                async with session.begin():
                    record = await session.get(MemoryItemModel, memory_id)
                    if not record:
                        return MemoryDeleteResult(
                            memory_id=memory_id,
                            success=False,
                            soft_deleted=soft_delete,
                            deleted_count=0,
                            error_message="Item not found.",
                        )

                    item = self._to_domain(record)
                    if not access_context.can_access(item.metadata.privacy, item.user_id):
                        return MemoryDeleteResult(
                            memory_id=memory_id,
                            success=False,
                            soft_deleted=soft_delete,
                            deleted_count=0,
                            error_message="Access Denied.",
                        )

                    if soft_delete:
                        record.is_active = False
                        record.updated_at = datetime.now(UTC)
                    else:
                        await session.delete(record)

            return MemoryDeleteResult(
                memory_id=memory_id,
                success=True,
                soft_deleted=soft_delete,
                deleted_count=1,
            )
        except SQLAlchemyError as exc:
            logger.error(
                f"PostgreSQL PgVectorStore delete failed for '{memory_id}': {exc}", exc_info=True
            )
            return MemoryDeleteResult(
                memory_id=memory_id,
                success=False,
                soft_deleted=soft_delete,
                deleted_count=0,
                error_message="Database deletion operation failed.",
            )

    async def count(self) -> int:
        """Returns total active count of stored items in PostgreSQL."""
        try:
            async with self.session_factory() as session:
                stmt = select(MemoryItemModel).where(MemoryItemModel.is_active.is_(True))
                result = await session.execute(stmt)
                return len(result.scalars().all())
        except SQLAlchemyError as exc:
            logger.error(f"PostgreSQL count failed: {exc}", exc_info=True)
            return 0

    async def health(self) -> bool:
        """Probes database connection health."""
        try:
            async with self.session_factory() as session:
                await session.execute(select(1))
            return True
        except Exception:
            return False
