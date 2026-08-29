"""PostgreSQL Knowledge Graph Subsystem Adapter.

Provides concrete PostgreSQL implementation of IKnowledgeGraph using SQLAlchemy 2.x async ORM.
Enforces entity existence checks, relationship path privacy authorization, node deletion cascading,
relationship type validation, cycle detection, and bounded graph traversal.
"""

import logging
from collections import deque
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    String,
    or_,
    select,
)
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase

from app.database.session import get_async_session_factory
from app.memory.knowledge_graph import (
    EntityEdge,
    EntityNode,
    GraphTraversalResult,
    IKnowledgeGraph,
    RelationshipType,
)
from app.memory.models import (
    MemoryAccessContext,
    MemoryPrivacy,
    MemorySource,
)

logger = logging.getLogger(__name__)


class BaseGraphModel(DeclarativeBase):
    """Base declarative class for Knowledge Graph ORM models."""


class EntityNodeModel(BaseGraphModel):
    """SQLAlchemy 2.x ORM model representing persistent graph nodes in PostgreSQL."""

    __tablename__ = "ultron_graph_nodes"

    node_id = Column(String(64), primary_key=True, index=True)
    entity_type = Column(String(32), nullable=False, index=True)
    canonical_name = Column(String(256), nullable=False, index=True)
    properties = Column(JSON, nullable=False, default=dict)
    privacy = Column(String(32), nullable=False, default="PRIVATE")
    user_id = Column(String(128), nullable=True, index=True)
    project_id = Column(String(128), nullable=True, index=True)
    created_at = Column(DateTime(timezone=True), nullable=False)
    updated_at = Column(DateTime(timezone=True), nullable=False)
    is_active = Column(Boolean, nullable=False, default=True, index=True)


class EntityEdgeModel(BaseGraphModel):
    """SQLAlchemy 2.x ORM model representing persistent directed graph edges in PostgreSQL."""

    __tablename__ = "ultron_graph_edges"

    edge_id = Column(String(64), primary_key=True, index=True)
    source_node_id = Column(
        String(64),
        ForeignKey("ultron_graph_nodes.node_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    target_node_id = Column(
        String(64),
        ForeignKey("ultron_graph_nodes.node_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    relationship_type = Column(String(32), nullable=False, index=True)
    properties = Column(JSON, nullable=False, default=dict)
    confidence = Column(Float, nullable=False, default=1.0)
    source = Column(String(32), nullable=False, default="USER_INPUT")
    privacy = Column(String(32), nullable=False, default="PRIVATE")
    user_id = Column(String(128), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False)
    updated_at = Column(DateTime(timezone=True), nullable=False)
    is_active = Column(Boolean, nullable=False, default=True, index=True)

    __table_args__ = (
        Index(
            "ix_graph_edge_source_target",
            "source_node_id",
            "target_node_id",
            "relationship_type",
        ),
    )


class PostgresKnowledgeGraph(IKnowledgeGraph):
    """Concrete PostgreSQL implementation of IKnowledgeGraph.

    Architecture & Safety:
    - Relational Storage: Persistent `ultron_graph_nodes` & `ultron_graph_edges` tables.
    - Graph Integrity: Foreign key constraints and entity existence validation prior to edge creation.
    - Path Privacy Enforcement: Evaluates privacy settings across source node, edge, and target node.
    - Bounded Traversal: BFS traversal limited by `max_depth`, `max_nodes`, and `max_edges` with cycle detection.
    """

    def __init__(self, session_factory: Any = None) -> None:
        """Initializes PostgresKnowledgeGraph adapter."""
        self._session_factory = session_factory

    @property
    def session_factory(self) -> Any:
        """Lazy resolves async session factory if not injected."""
        if self._session_factory is None:
            self._session_factory = get_async_session_factory()
        return self._session_factory

    def _node_to_domain(self, rec: EntityNodeModel) -> EntityNode:
        """Converts node ORM record to domain EntityNode."""
        created_at_dt: datetime = rec.created_at  # type: ignore[assignment]
        if created_at_dt and created_at_dt.tzinfo is None:
            created_at_dt = created_at_dt.replace(tzinfo=UTC)

        updated_at_dt: datetime = rec.updated_at  # type: ignore[assignment]
        if updated_at_dt and updated_at_dt.tzinfo is None:
            updated_at_dt = updated_at_dt.replace(tzinfo=UTC)

        return EntityNode(
            node_id=str(rec.node_id),
            entity_type=str(rec.entity_type),
            canonical_name=str(rec.canonical_name),
            properties=dict(rec.properties or {}),
            privacy=MemoryPrivacy(str(rec.privacy)),
            user_id=str(rec.user_id) if rec.user_id else None,
            project_id=str(rec.project_id) if rec.project_id else None,
            created_at=created_at_dt,
            updated_at=updated_at_dt,
            is_active=bool(rec.is_active),
        )

    def _edge_to_domain(self, rec: EntityEdgeModel) -> EntityEdge:
        """Converts edge ORM record to domain EntityEdge."""
        created_at_dt: datetime = rec.created_at  # type: ignore[assignment]
        if created_at_dt and created_at_dt.tzinfo is None:
            created_at_dt = created_at_dt.replace(tzinfo=UTC)

        updated_at_dt: datetime = rec.updated_at  # type: ignore[assignment]
        if updated_at_dt and updated_at_dt.tzinfo is None:
            updated_at_dt = updated_at_dt.replace(tzinfo=UTC)

        return EntityEdge(
            edge_id=str(rec.edge_id),
            source_node_id=str(rec.source_node_id),
            target_node_id=str(rec.target_node_id),
            relationship_type=str(rec.relationship_type),
            properties=dict(rec.properties or {}),
            confidence=float(str(rec.confidence)),
            source=MemorySource(str(rec.source)),
            privacy=MemoryPrivacy(str(rec.privacy)),
            user_id=str(rec.user_id) if rec.user_id else None,
            created_at=created_at_dt,
            updated_at=updated_at_dt,
            is_active=bool(rec.is_active),
        )

    async def add_node(self, node: EntityNode, access_context: MemoryAccessContext) -> EntityNode:
        """Adds or updates an EntityNode in PostgreSQL enforcing authorization."""
        if access_context and not access_context.can_access(node.privacy, node.user_id):
            raise PermissionError("Access Denied: Caller context cannot create/update node.")

        now = datetime.now(UTC)
        record_data = {
            "node_id": node.node_id,
            "entity_type": node.entity_type,
            "canonical_name": node.canonical_name,
            "properties": node.properties,
            "privacy": node.privacy.value,
            "user_id": node.user_id,
            "project_id": node.project_id,
            "created_at": node.created_at,
            "updated_at": now,
            "is_active": node.is_active,
        }

        try:
            async with self.session_factory() as session:
                async with session.begin():
                    existing = await session.get(EntityNodeModel, node.node_id)
                    if existing:
                        for k, v in record_data.items():
                            setattr(existing, k, v)
                    else:
                        new_node = EntityNodeModel(**record_data)
                        session.add(new_node)
            return node
        except SQLAlchemyError as exc:
            logger.error(f"PostgreSQL add_node failed for '{node.node_id}': {exc}", exc_info=True)
            raise RuntimeError(f"Database error writing graph node: {exc}") from exc

    async def get_node(
        self, node_id: str, access_context: MemoryAccessContext
    ) -> EntityNode | None:
        """Retrieves a single EntityNode by ID enforcing privacy access control."""
        try:
            async with self.session_factory() as session:
                rec = await session.get(EntityNodeModel, node_id)
                if not rec or not rec.is_active:
                    return None

                node = self._node_to_domain(rec)
                if not access_context.can_access(node.privacy, node.user_id):
                    return None

                return node
        except SQLAlchemyError as exc:
            logger.error(f"PostgreSQL get_node failed for '{node_id}': {exc}", exc_info=True)
            return None

    async def update_node(
        self, node: EntityNode, access_context: MemoryAccessContext
    ) -> EntityNode:
        """Updates an existing EntityNode payload."""
        return await self.add_node(node, access_context)

    async def delete_node(
        self,
        node_id: str,
        access_context: MemoryAccessContext,
        soft_delete: bool = True,
    ) -> bool:
        """Deletes or tombstones an EntityNode and cascades edge deactivations."""
        try:
            async with self.session_factory() as session:
                async with session.begin():
                    rec = await session.get(EntityNodeModel, node_id)
                    if not rec:
                        return False

                    node = self._node_to_domain(rec)
                    if not access_context.can_access(node.privacy, node.user_id):
                        raise PermissionError("Access Denied: Cannot delete node.")

                    now = datetime.now(UTC)

                    # Cascade deactivation/deletion to all connected edges
                    stmt_edges = select(EntityEdgeModel).where(
                        or_(
                            EntityEdgeModel.source_node_id == node_id,
                            EntityEdgeModel.target_node_id == node_id,
                        )
                    )
                    edge_recs = (await session.execute(stmt_edges)).scalars().all()

                    for edge_rec in edge_recs:
                        if (
                            str(edge_rec.source_node_id) == node_id
                            or str(edge_rec.target_node_id) == node_id
                        ):
                            if soft_delete:
                                edge_rec.is_active = False
                                edge_rec.updated_at = now
                            else:
                                await session.delete(edge_rec)

                    if soft_delete:
                        rec.is_active = False
                        rec.updated_at = now
                    else:
                        await session.delete(rec)

            return True
        except PermissionError:
            raise
        except SQLAlchemyError as exc:
            logger.error(f"PostgreSQL delete_node failed for '{node_id}': {exc}", exc_info=True)
            return False

    async def add_edge(self, edge: EntityEdge, access_context: MemoryAccessContext) -> EntityEdge:
        """Adds or updates a directed EntityEdge between existing source and target nodes."""
        if access_context and not access_context.can_access(edge.privacy, edge.user_id):
            raise PermissionError("Access Denied: Cannot create edge.")

        # Validate relationship type semantics
        valid_types = {rt.value for rt in RelationshipType}
        if edge.relationship_type not in valid_types:
            # Allow custom string relationship types if non-empty, else raise
            if not edge.relationship_type or not edge.relationship_type.strip():
                raise ValueError("Relationship type cannot be empty.")

        try:
            async with self.session_factory() as session:
                async with session.begin():
                    # 1. Validate source node existence and access
                    source_rec = await session.get(EntityNodeModel, edge.source_node_id)
                    if not source_rec or not source_rec.is_active:
                        raise ValueError(
                            f"Source entity node '{edge.source_node_id}' does not exist or is inactive."
                        )
                    source_node = self._node_to_domain(source_rec)
                    if not access_context.can_access(source_node.privacy, source_node.user_id):
                        raise PermissionError(
                            f"Access Denied: Source node '{edge.source_node_id}' is inaccessible."
                        )

                    # 2. Validate target node existence and access
                    target_rec = await session.get(EntityNodeModel, edge.target_node_id)
                    if not target_rec or not target_rec.is_active:
                        raise ValueError(
                            f"Target entity node '{edge.target_node_id}' does not exist or is inactive."
                        )
                    target_node = self._node_to_domain(target_rec)
                    if not access_context.can_access(target_node.privacy, target_node.user_id):
                        raise PermissionError(
                            f"Access Denied: Target node '{edge.target_node_id}' is inaccessible."
                        )

                    # 3. Check for existing duplicate edge (same source, target, relationship_type)
                    stmt_dup = select(EntityEdgeModel).where(
                        EntityEdgeModel.source_node_id == edge.source_node_id,
                        EntityEdgeModel.target_node_id == edge.target_node_id,
                        EntityEdgeModel.relationship_type == edge.relationship_type,
                    )
                    existing_edge = (await session.execute(stmt_dup)).scalars().first()

                    now = datetime.now(UTC)
                    if existing_edge:
                        edge.edge_id = str(existing_edge.edge_id)
                        existing_edge.properties = edge.properties
                        existing_edge.confidence = edge.confidence
                        existing_edge.source = edge.source.value
                        existing_edge.privacy = edge.privacy.value
                        existing_edge.is_active = edge.is_active
                        existing_edge.updated_at = now
                    else:
                        new_edge = EntityEdgeModel(
                            edge_id=edge.edge_id,
                            source_node_id=edge.source_node_id,
                            target_node_id=edge.target_node_id,
                            relationship_type=edge.relationship_type,
                            properties=edge.properties,
                            confidence=edge.confidence,
                            source=edge.source.value,
                            privacy=edge.privacy.value,
                            user_id=edge.user_id,
                            created_at=edge.created_at,
                            updated_at=now,
                            is_active=edge.is_active,
                        )
                        session.add(new_edge)

            return edge
        except (ValueError, PermissionError):
            raise
        except SQLAlchemyError as exc:
            logger.error(f"PostgreSQL add_edge failed for '{edge.edge_id}': {exc}", exc_info=True)
            raise RuntimeError(f"Database error writing graph edge: {exc}") from exc

    async def get_edge(
        self, edge_id: str, access_context: MemoryAccessContext
    ) -> EntityEdge | None:
        """Retrieves a single EntityEdge by ID enforcing relationship path privacy."""
        try:
            async with self.session_factory() as session:
                rec = await session.get(EntityEdgeModel, edge_id)
                if not rec or not rec.is_active:
                    return None

                edge = self._edge_to_domain(rec)
                if not access_context.can_access(edge.privacy, edge.user_id):
                    return None

                # Verify path privacy for connected nodes
                source_node = await self.get_node(edge.source_node_id, access_context)
                target_node = await self.get_node(edge.target_node_id, access_context)
                if not source_node or not target_node:
                    return None

                return edge
        except SQLAlchemyError as exc:
            logger.error(f"PostgreSQL get_edge failed for '{edge_id}': {exc}", exc_info=True)
            return None

    async def delete_edge(
        self,
        edge_id: str,
        access_context: MemoryAccessContext,
        soft_delete: bool = True,
    ) -> bool:
        """Deletes or tombstones a single EntityEdge."""
        try:
            async with self.session_factory() as session:
                async with session.begin():
                    rec = await session.get(EntityEdgeModel, edge_id)
                    if not rec:
                        return False

                    edge = self._edge_to_domain(rec)
                    if not access_context.can_access(edge.privacy, edge.user_id):
                        raise PermissionError("Access Denied: Cannot delete edge.")

                    now = datetime.now(UTC)
                    if soft_delete:
                        rec.is_active = False
                        rec.updated_at = now
                    else:
                        await session.delete(rec)
            return True
        except PermissionError:
            raise
        except SQLAlchemyError as exc:
            logger.error(f"PostgreSQL delete_edge failed for '{edge_id}': {exc}", exc_info=True)
            return False

    async def query_neighbors(
        self,
        node_id: str,
        access_context: MemoryAccessContext,
        relationship_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Queries 1-hop connected neighbor nodes and edge relationships matching authorization."""
        node = await self.get_node(node_id, access_context)
        if not node:
            return []

        try:
            async with self.session_factory() as session:
                stmt = select(EntityEdgeModel).where(
                    EntityEdgeModel.is_active.is_(True),
                    or_(
                        EntityEdgeModel.source_node_id == node_id,
                        EntityEdgeModel.target_node_id == node_id,
                    ),
                )
                if relationship_type:
                    stmt = stmt.where(EntityEdgeModel.relationship_type == relationship_type)

                edge_recs = (await session.execute(stmt)).scalars().all()
                neighbors: list[dict[str, Any]] = []

                for edge_rec in edge_recs:
                    if not edge_rec.is_active:
                        continue

                    # Filter edges that are not actually connected to node_id
                    if (
                        str(edge_rec.source_node_id) != node_id
                        and str(edge_rec.target_node_id) != node_id
                    ):
                        continue

                    edge = self._edge_to_domain(edge_rec)
                    if not access_context.can_access(edge.privacy, edge.user_id):
                        continue

                    neighbor_id = (
                        edge.target_node_id
                        if edge.source_node_id == node_id
                        else edge.source_node_id
                    )
                    neighbor_node = await self.get_node(neighbor_id, access_context)
                    if not neighbor_node:
                        continue

                    neighbors.append(
                        {
                            "edge": edge,
                            "neighbor": neighbor_node,
                            "direction": (
                                "outgoing" if edge.source_node_id == node_id else "incoming"
                            ),
                        }
                    )

                return neighbors
        except SQLAlchemyError as exc:
            logger.error(f"PostgreSQL query_neighbors failed for '{node_id}': {exc}", exc_info=True)
            return []

    async def traverse(
        self,
        start_node_id: str,
        access_context: MemoryAccessContext,
        max_depth: int = 3,
        max_nodes: int = 50,
        max_edges: int = 100,
        relationship_types: list[str] | None = None,
    ) -> GraphTraversalResult:
        """Executes bounded BFS graph traversal with cycle prevention and path privacy enforcement."""
        start_node = await self.get_node(start_node_id, access_context)
        if not start_node:
            return GraphTraversalResult(
                nodes=[],
                edges=[],
                start_node_id=start_node_id,
                traversed_depth=0,
                total_nodes_found=0,
                total_edges_found=0,
            )

        visited_nodes: dict[str, EntityNode] = {start_node.node_id: start_node}
        visited_edges: dict[str, EntityEdge] = {}

        # Queue items: (current_node_id, current_depth)
        queue: deque[tuple[str, int]] = deque([(start_node.node_id, 0)])
        max_depth_reached = 0

        while queue:
            curr_id, curr_depth = queue.popleft()
            max_depth_reached = max(max_depth_reached, curr_depth)

            if curr_depth >= max_depth:
                continue

            if len(visited_nodes) >= max_nodes or len(visited_edges) >= max_edges:
                break

            neighbors = await self.query_neighbors(curr_id, access_context)
            for nbr_info in neighbors:
                edge: EntityEdge = nbr_info["edge"]
                nbr_node: EntityNode = nbr_info["neighbor"]

                if relationship_types and edge.relationship_type not in relationship_types:
                    continue

                if edge.edge_id not in visited_edges:
                    if len(visited_edges) < max_edges:
                        visited_edges[edge.edge_id] = edge

                if nbr_node.node_id not in visited_nodes:
                    if len(visited_nodes) < max_nodes:
                        visited_nodes[nbr_node.node_id] = nbr_node
                        queue.append((nbr_node.node_id, curr_depth + 1))

        nodes_list = list(visited_nodes.values())
        edges_list = list(visited_edges.values())

        return GraphTraversalResult(
            nodes=nodes_list,
            edges=edges_list,
            start_node_id=start_node_id,
            traversed_depth=max_depth_reached,
            total_nodes_found=len(nodes_list),
            total_edges_found=len(edges_list),
        )
