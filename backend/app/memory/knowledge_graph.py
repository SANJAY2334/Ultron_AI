"""Relational Knowledge Graph Domain Models and Contracts.

Defines canonical EntityNode and EntityEdge domain models, explicit relationship enums,
bounded graph traversal schemas, and the IKnowledgeGraph abstract interface contract.
"""

import uuid
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.memory.models import MemoryAccessContext, MemoryPrivacy, MemorySource


class EntityType(StrEnum):
    """Canonical domain entity classifications."""

    USER = "USER"
    PROJECT = "PROJECT"
    COMPONENT = "COMPONENT"
    TECHNOLOGY = "TECHNOLOGY"
    CONFIGURATION = "CONFIGURATION"
    CONCEPT = "CONCEPT"
    AGENT = "AGENT"


class RelationshipType(StrEnum):
    """Explicit validated relationship semantics."""

    OWNS = "OWNS"
    CONTAINS = "CONTAINS"
    USES = "USES"
    PREFERS = "PREFERS"
    DEPENDS_ON = "DEPENDS_ON"
    RELATED_TO = "RELATED_TO"


class EntityNode(BaseModel):
    """Canonical domain representation of a Knowledge Graph entity node."""

    node_id: str = Field(
        default_factory=lambda: f"node_{uuid.uuid4().hex[:12]}",
        description="Unique node identifier",
    )
    entity_type: str = Field(description="Entity type classification")
    canonical_name: str = Field(description="Human-readable canonical entity name")
    properties: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary entity properties"
    )
    privacy: MemoryPrivacy = Field(
        default=MemoryPrivacy.PRIVATE, description="Privacy sensitivity classification"
    )
    user_id: str | None = Field(default=None, description="Associated owning user ID")
    project_id: str | None = Field(default=None, description="Associated project ID")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Last update timestamp"
    )
    is_active: bool = Field(
        default=True, description="False if node has been soft-deleted / tombstoned"
    )

    @field_validator("canonical_name")
    @classmethod
    def validate_name_not_empty(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Entity node canonical_name cannot be empty or whitespace-only.")
        return stripped


class EntityEdge(BaseModel):
    """Canonical domain representation of a directed relationship edge between two entity nodes."""

    edge_id: str = Field(
        default_factory=lambda: f"edge_{uuid.uuid4().hex[:12]}",
        description="Unique edge relationship identifier",
    )
    source_node_id: str = Field(description="Source entity node ID")
    target_node_id: str = Field(description="Target entity node ID")
    relationship_type: str = Field(description="Explicit relationship type")
    properties: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary relationship properties"
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Relationship confidence score [0.0, 1.0]"
    )
    source: MemorySource = Field(
        default=MemorySource.USER_INPUT, description="Provenance origin of relationship"
    )
    privacy: MemoryPrivacy = Field(
        default=MemoryPrivacy.PRIVATE, description="Privacy sensitivity of edge itself"
    )
    user_id: str | None = Field(default=None, description="Associated owning user ID")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Last update timestamp"
    )
    is_active: bool = Field(
        default=True, description="False if edge has been soft-deleted / tombstoned"
    )


class GraphTraversalResult(BaseModel):
    """Result payload returned following a bounded graph traversal query."""

    nodes: list[EntityNode] = Field(
        default_factory=list, description="Discovered accessible entity nodes"
    )
    edges: list[EntityEdge] = Field(
        default_factory=list, description="Discovered accessible relationship edges"
    )
    start_node_id: str = Field(description="Origin node ID of traversal")
    traversed_depth: int = Field(default=0, ge=0, description="Maximum depth reached")
    total_nodes_found: int = Field(default=0, ge=0, description="Count of nodes discovered")
    total_edges_found: int = Field(default=0, ge=0, description="Count of edges discovered")


class IKnowledgeGraph(ABC):
    """Abstract Interface for Relational Knowledge Graph Subsystem.

    Responsibility: Manages entity nodes, directed relationship edges, path privacy authorization,
    and bounded graph traversal.
    """

    @abstractmethod
    async def add_node(self, node: EntityNode, access_context: MemoryAccessContext) -> EntityNode:
        """Adds or updates an EntityNode in the Knowledge Graph enforcing authorization."""

    @abstractmethod
    async def get_node(
        self, node_id: str, access_context: MemoryAccessContext
    ) -> EntityNode | None:
        """Retrieves a single EntityNode by ID enforcing privacy access control."""

    @abstractmethod
    async def update_node(
        self, node: EntityNode, access_context: MemoryAccessContext
    ) -> EntityNode:
        """Updates an existing EntityNode payload."""

    @abstractmethod
    async def delete_node(
        self,
        node_id: str,
        access_context: MemoryAccessContext,
        soft_delete: bool = True,
    ) -> bool:
        """Deletes or tombstones an EntityNode and cascades edge deactivations."""

    @abstractmethod
    async def add_edge(self, edge: EntityEdge, access_context: MemoryAccessContext) -> EntityEdge:
        """Adds or updates a directed EntityEdge between existing source and target nodes."""

    @abstractmethod
    async def get_edge(
        self, edge_id: str, access_context: MemoryAccessContext
    ) -> EntityEdge | None:
        """Retrieves a single EntityEdge by ID enforcing relationship path privacy."""

    @abstractmethod
    async def delete_edge(
        self,
        edge_id: str,
        access_context: MemoryAccessContext,
        soft_delete: bool = True,
    ) -> bool:
        """Deletes or tombstones a single EntityEdge."""

    @abstractmethod
    async def query_neighbors(
        self,
        node_id: str,
        access_context: MemoryAccessContext,
        relationship_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Queries 1-hop connected neighbor nodes and edge relationships matching authorization."""

    @abstractmethod
    async def traverse(
        self,
        start_node_id: str,
        access_context: MemoryAccessContext,
        max_depth: int = 3,
        max_nodes: int = 50,
        max_edges: int = 100,
        relationship_types: list[str] | None = None,
    ) -> GraphTraversalResult:
        """Executes bounded graph traversal with cycle prevention and path privacy enforcement."""
