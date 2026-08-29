"""Memory Subsystem Domain Models & Enums.

Defines framework-agnostic Pydantic models for memory items, VectorEmbedding schemas,
MemoryAccessContext authorization filters, search queries, write/delete results,
scope level invariants, privacy classifications, and provenance sources.
"""

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class MemoryType(StrEnum):
    """Semantic classification types for hybrid memory items.

    Reflects cognitive domain categories rather than physical storage engines.
    """

    WORKING = "WORKING"
    EPISODIC = "EPISODIC"
    SEMANTIC = "SEMANTIC"
    RELATIONAL = "RELATIONAL"  # Formerly GRAPH, representing structural knowledge relations


class MemoryScope(StrEnum):
    """Scoped visibility boundaries for memory items."""

    SESSION = "SESSION"
    USER = "USER"
    PROJECT = "PROJECT"
    GLOBAL = "GLOBAL"


class MemoryPrivacy(StrEnum):
    """Security and privacy sensitivity levels for authorization enforcement."""

    PUBLIC = "PUBLIC"
    PRIVATE = "PRIVATE"
    CONFIDENTIAL = "CONFIDENTIAL"
    RESTRICTED = "RESTRICTED"


class MemorySource(StrEnum):
    """Provenance origins of memory items."""

    USER_INPUT = "USER_INPUT"
    PLANNER_OUTPUT = "PLANNER_OUTPUT"
    TOOL_OUTPUT = "TOOL_OUTPUT"
    SYSTEM_CONSOLIDATION = "SYSTEM_CONSOLIDATION"
    EXTERNAL_INGESTION = "EXTERNAL_INGESTION"


class VectorEmbedding(BaseModel):
    """Decoupled dense vector embedding representation supporting multi-model versioning."""

    vector: list[float] = Field(description="Dense floating-point embedding vector")
    model_name: str = Field(
        default="text-embedding-3-small", description="Name of vector embedding model"
    )
    dimension: int = Field(default=1536, ge=1, description="Embedding vector dimension")
    version: str = Field(default="v1", description="Embedding model version tag")

    @model_validator(mode="after")
    def validate_vector_dimension(self) -> "VectorEmbedding":
        """Ensures vector length matches declared dimension."""
        if len(self.vector) != self.dimension:
            raise ValueError(
                f"Embedding vector length ({len(self.vector)}) does not match declared dimension ({self.dimension})."
            )
        return self


class MemoryAccessContext(BaseModel):
    """Authorization and caller context used for memory privacy enforcement."""

    user_id: str | None = Field(default=None, description="Requesting user ID")
    session_id: str | None = Field(default=None, description="Requesting session ID")
    roles: set[str] = Field(
        default_factory=set, description="Security roles assigned to requesting principal"
    )
    is_system: bool = Field(
        default=False, description="True if caller is authorized system process"
    )

    def can_access(self, privacy: MemoryPrivacy, item_user_id: str | None) -> bool:
        """Evaluates whether the caller context is authorized to access a memory item."""
        if self.is_system:
            return True

        if privacy == MemoryPrivacy.PUBLIC:
            return True

        if privacy == MemoryPrivacy.PRIVATE:
            return self.user_id is not None and self.user_id == item_user_id

        if privacy == MemoryPrivacy.CONFIDENTIAL:
            is_owner = self.user_id is not None and self.user_id == item_user_id
            has_role = "confidential_access" in self.roles
            return is_owner or has_role

        if privacy == MemoryPrivacy.RESTRICTED:
            return "restricted_access" in self.roles

        return False


class MemoryMetadata(BaseModel):
    """Metadata attributes supporting provenance, privacy, confidence, and system authorization."""

    source: MemorySource = Field(
        default=MemorySource.USER_INPUT, description="Provenance source of memory item"
    )
    privacy: MemoryPrivacy = Field(
        default=MemoryPrivacy.PRIVATE, description="Privacy sensitivity classification"
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence score threshold [0.0, 1.0]"
    )
    provenance: str | None = Field(
        default=None, description="Detailed origin or audit trail statement"
    )
    is_encrypted: bool = Field(
        default=False, description="True if content payload is Fernet encrypted"
    )
    is_system_authorized: bool = Field(
        default=False, description="True if system granted GLOBAL memory write rights"
    )
    custom_attributes: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary custom metadata attributes"
    )


class MemoryItem(BaseModel):
    """Canonical domain model representing a single memory item across all tiers."""

    id: str = Field(
        default_factory=lambda: f"mem_{uuid.uuid4().hex[:12]}",
        description="Unique memory item identifier",
    )
    memory_type: MemoryType = Field(description="Memory tier classification")
    scope: MemoryScope = Field(
        default=MemoryScope.SESSION, description="Visibility and lifespan scope"
    )
    session_id: str | None = Field(
        default=None, description="Associated session ID for session-scoped items"
    )
    user_id: str | None = Field(
        default=None, description="Associated user ID for user-scoped items"
    )
    project_id: str | None = Field(
        default=None, description="Associated project ID for project-scoped items"
    )
    content: str = Field(description="Textual memory content payload")
    embeddings: list[VectorEmbedding] = Field(
        default_factory=list,
        description="List of multi-model/versioned vector embeddings",
    )
    importance: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Importance weighting score [0.0, 1.0]"
    )
    metadata: MemoryMetadata = Field(
        default_factory=MemoryMetadata, description="Memory metadata attributes"
    )
    is_active: bool = Field(
        default=True, description="False if item has been soft-deleted / tombstoned"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Last update timestamp"
    )
    expires_at: datetime | None = Field(
        default=None, description="Optional absolute expiration timestamp (TTL)"
    )
    ttl_seconds: int | None = Field(
        default=None, ge=1, description="Optional time-to-live window in seconds"
    )

    @field_validator("content")
    @classmethod
    def validate_content_not_empty(cls, v: str) -> str:
        """Ensures memory content payload is non-empty."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("Memory content payload cannot be empty or whitespace-only.")
        return stripped

    @model_validator(mode="after")
    def validate_scope_invariants(self) -> "MemoryItem":
        """Enforces mandatory scope identifier invariants and system authorization for GLOBAL scope."""
        if self.scope == MemoryScope.SESSION and not self.session_id:
            raise ValueError("MemoryScope.SESSION requires a valid non-null session_id.")
        if self.scope == MemoryScope.USER and not self.user_id:
            raise ValueError("MemoryScope.USER requires a valid non-null user_id.")
        if self.scope == MemoryScope.PROJECT and not self.project_id:
            raise ValueError("MemoryScope.PROJECT requires a valid non-null project_id.")
        if self.scope == MemoryScope.GLOBAL and not self.metadata.is_system_authorized:
            raise ValueError(
                "MemoryScope.GLOBAL scope items require explicit system authorization."
            )
        return self


class MemoryQuery(BaseModel):
    """Domain request query schema for hybrid memory retrieval."""

    query_text: str | None = Field(
        default=None, description="Semantic or lexical search query text"
    )
    query_embedding: VectorEmbedding | None = Field(
        default=None, description="Query vector embedding for similarity search"
    )
    memory_types: list[MemoryType] | None = Field(
        default=None, description="Target memory type classifications"
    )
    scope: MemoryScope | None = Field(default=None, description="Target visibility scope filter")
    session_id: str | None = Field(default=None, description="Filter by session identifier")
    user_id: str | None = Field(default=None, description="Filter by user identifier")
    project_id: str | None = Field(default=None, description="Filter by project identifier")
    limit: int = Field(default=10, ge=1, le=100, description="Maximum number of items to retrieve")
    min_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Minimum confidence filter threshold"
    )
    min_importance: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Minimum importance filter threshold"
    )
    include_inactive: bool = Field(
        default=False, description="True to include soft-deleted/tombstoned items"
    )
    filter_attributes: dict[str, Any] = Field(
        default_factory=dict, description="Metadata attribute exact-match filters"
    )


class MemorySearchResult(BaseModel):
    """Encapsulates results returned from a memory retrieval query."""

    items: list[MemoryItem] = Field(default_factory=list, description="Retrieved memory items")
    scores: list[float] = Field(
        default_factory=list, description="Corresponding similarity or relevance scores"
    )
    query: MemoryQuery = Field(description="Original search query parameters")
    total_found: int = Field(default=0, ge=0, description="Total matching items found")
    search_time_ms: float = Field(
        default=0.0, ge=0.0, description="Query execution latency in milliseconds"
    )


class MemoryWriteResult(BaseModel):
    """Result status object returned following a memory write operation."""

    memory_id: str = Field(description="Identifier of written memory item")
    success: bool = Field(description="True if write operation succeeded")
    memory_type: MemoryType = Field(description="Target memory tier classification")
    error_message: str | None = Field(
        default=None, description="Error message if write operation failed"
    )


class MemoryDeleteResult(BaseModel):
    """Result status object returned following a memory deletion operation."""

    memory_id: str = Field(description="Identifier of target deleted memory item")
    success: bool = Field(description="True if deletion operation succeeded")
    soft_deleted: bool = Field(default=True, description="True if item was tombstoned/soft-deleted")
    deleted_count: int = Field(default=1, ge=0, description="Count of deleted memory items")
    error_message: str | None = Field(default=None, description="Error message if deletion failed")
