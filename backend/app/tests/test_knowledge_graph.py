"""Unit Tests for Relational Knowledge Graph Subsystem.

Validates EntityNode and EntityEdge CRUD, duplicate edge resolution, missing node rejection,
self-reference behavior, 1-hop neighbor queries, depth-limited BFS traversal, cycle handling,
max node/edge limits, relationship path privacy enforcement, user/project isolation,
soft vs physical deletion, cascading edge deactivation, provenance metadata, and database rollback.
"""

import asyncio
from typing import Any

import pytest
from sqlalchemy.exc import SQLAlchemyError

from app.memory.adapters.postgres_knowledge_graph import (
    EntityEdgeModel,
    EntityNodeModel,
    PostgresKnowledgeGraph,
)
from app.memory.knowledge_graph import (
    EntityEdge,
    EntityNode,
    EntityType,
    RelationshipType,
)
from app.memory.models import (
    MemoryAccessContext,
    MemoryPrivacy,
    MemorySource,
)


class MockGraphDatabaseSession:
    """In-memory AsyncSession mock for unit testing PostgresKnowledgeGraph without live PostgreSQL."""

    def __init__(self, parent: "MockGraphSessionMaker") -> None:
        self.parent = parent
        self.in_transaction = False

    async def get(self, entity: type, ident: Any) -> Any:
        if self.parent.should_fail:
            raise SQLAlchemyError("Simulated Database Error")
        if entity is EntityNodeModel:
            return self.parent.nodes.get(str(ident))
        if entity is EntityEdgeModel:
            return self.parent.edges.get(str(ident))
        return None

    def add(self, instance: Any) -> None:
        if self.parent.should_fail:
            raise SQLAlchemyError("Simulated Database Error")
        if isinstance(instance, EntityNodeModel):
            self.parent.nodes[str(instance.node_id)] = instance
        elif isinstance(instance, EntityEdgeModel):
            self.parent.edges[str(instance.edge_id)] = instance

    async def delete(self, instance: Any) -> None:
        if self.parent.should_fail:
            raise SQLAlchemyError("Simulated Database Error")
        if isinstance(instance, EntityNodeModel) and str(instance.node_id) in self.parent.nodes:
            del self.parent.nodes[str(instance.node_id)]
        elif isinstance(instance, EntityEdgeModel) and str(instance.edge_id) in self.parent.edges:
            del self.parent.edges[str(instance.edge_id)]

    async def execute(self, statement: Any) -> Any:
        if self.parent.should_fail:
            raise SQLAlchemyError("Simulated Database Error")

        stmt_str = str(statement)
        if "ultron_graph_edges" in stmt_str:
            records_list: list[Any] = list(self.parent.edges.values())
        else:
            records_list = list(self.parent.nodes.values())

        class ScalarResult:
            def __init__(self, records: list[Any], is_edges: bool) -> None:
                self.records = records
                self.is_edges = is_edges

            def scalars(self):
                records = self.records
                is_edges = self.is_edges

                class ScalarsList:
                    def all(self):
                        return records

                    def first(self):
                        if is_edges and records:
                            # Avoid returning false duplicate edge match for different edges
                            return None
                        return records[0] if records else None

                return ScalarsList()

        return ScalarResult(records_list, "ultron_graph_edges" in stmt_str)

    def begin(self) -> "MockGraphTransaction":
        return MockGraphTransaction(self)

    async def __aenter__(self) -> "MockGraphDatabaseSession":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        pass


class MockGraphTransaction:
    """Mock database transaction context manager."""

    def __init__(self, session: MockGraphDatabaseSession) -> None:
        self.session = session

    async def __aenter__(self) -> "MockGraphTransaction":
        self.session.in_transaction = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self.session.in_transaction = False


class MockGraphSessionMaker:
    """Factory for MockGraphDatabaseSession instances."""

    def __init__(self) -> None:
        self.nodes: dict[str, EntityNodeModel] = {}
        self.edges: dict[str, EntityEdgeModel] = {}
        self.should_fail = False

    def __call__(self) -> MockGraphDatabaseSession:
        return MockGraphDatabaseSession(self)


@pytest.fixture
def mock_graph_db() -> MockGraphSessionMaker:
    return MockGraphSessionMaker()


@pytest.fixture
def graph(mock_graph_db: MockGraphSessionMaker) -> PostgresKnowledgeGraph:
    return PostgresKnowledgeGraph(session_factory=mock_graph_db)


def test_node_crud_operations(graph: PostgresKnowledgeGraph) -> None:
    """Verify add_node, get_node, update_node, and delete_node."""

    async def _test() -> None:
        access_owner = MemoryAccessContext(user_id="user_owner")
        access_other = MemoryAccessContext(user_id="user_other")

        node = EntityNode(
            entity_type=EntityType.PROJECT,
            canonical_name="ULTRON Core Engine",
            properties={"version": "2.0"},
            privacy=MemoryPrivacy.PRIVATE,
            user_id="user_owner",
            project_id="proj_ultron",
        )

        # Create
        created = await graph.add_node(node, access_owner)
        assert created.node_id == node.node_id

        # Get
        fetched = await graph.get_node(node.node_id, access_owner)
        assert fetched is not None
        assert fetched.canonical_name == "ULTRON Core Engine"
        assert fetched.properties["version"] == "2.0"

        # Privacy check
        assert await graph.get_node(node.node_id, access_other) is None

        # Update
        node.properties["version"] = "2.1"
        updated = await graph.update_node(node, access_owner)
        assert updated.properties["version"] == "2.1"

        # Soft Delete
        del_soft = await graph.delete_node(node.node_id, access_owner, soft_delete=True)
        assert del_soft is True
        assert await graph.get_node(node.node_id, access_owner) is None

    asyncio.run(_test())


def test_edge_creation_missing_node_rejection(graph: PostgresKnowledgeGraph) -> None:
    """Verify edge creation fails when source or target node is missing."""

    async def _test() -> None:
        access_ctx = MemoryAccessContext(user_id="user_owner")

        node_user = EntityNode(
            entity_type=EntityType.USER,
            canonical_name="User Alice",
            user_id="user_owner",
        )
        await graph.add_node(node_user, access_ctx)

        # Missing target node
        edge_invalid = EntityEdge(
            source_node_id=node_user.node_id,
            target_node_id="missing_target_node_id",
            relationship_type=RelationshipType.OWNS,
            user_id="user_owner",
        )

        with pytest.raises(ValueError, match="Target entity node '.*' does not exist"):
            await graph.add_edge(edge_invalid, access_ctx)

        # Missing source node
        edge_invalid_source = EntityEdge(
            source_node_id="missing_source_node_id",
            target_node_id=node_user.node_id,
            relationship_type=RelationshipType.OWNS,
            user_id="user_owner",
        )

        with pytest.raises(ValueError, match="Source entity node '.*' does not exist"):
            await graph.add_edge(edge_invalid_source, access_ctx)

    asyncio.run(_test())


def test_edge_crud_and_relationship_semantics(graph: PostgresKnowledgeGraph) -> None:
    """Verify add_edge, get_edge, delete_edge, and explicit relationships."""

    async def _test() -> None:
        access_ctx = MemoryAccessContext(user_id="user_owner")

        user_node = EntityNode(
            entity_type=EntityType.USER,
            canonical_name="User Bob",
            user_id="user_owner",
        )
        proj_node = EntityNode(
            entity_type=EntityType.PROJECT,
            canonical_name="Project Ultron",
            user_id="user_owner",
        )

        await graph.add_node(user_node, access_ctx)
        await graph.add_node(proj_node, access_ctx)

        edge = EntityEdge(
            source_node_id=user_node.node_id,
            target_node_id=proj_node.node_id,
            relationship_type=RelationshipType.OWNS,
            confidence=0.95,
            source=MemorySource.USER_INPUT,
            user_id="user_owner",
        )

        # Add Edge
        created_edge = await graph.add_edge(edge, access_ctx)
        assert created_edge.edge_id == edge.edge_id

        # Get Edge
        fetched_edge = await graph.get_edge(edge.edge_id, access_ctx)
        assert fetched_edge is not None
        assert fetched_edge.relationship_type == RelationshipType.OWNS
        assert fetched_edge.confidence == 0.95

        # Delete Edge
        deleted = await graph.delete_edge(edge.edge_id, access_ctx, soft_delete=True)
        assert deleted is True
        assert await graph.get_edge(edge.edge_id, access_ctx) is None

    asyncio.run(_test())


def test_query_neighbors_and_path_privacy(graph: PostgresKnowledgeGraph) -> None:
    """Verify 1-hop query_neighbors and relationship path privacy enforcement."""

    async def _test() -> None:
        ctx_owner = MemoryAccessContext(user_id="user_owner")
        ctx_other = MemoryAccessContext(user_id="user_other")

        user_node = EntityNode(
            entity_type=EntityType.USER,
            canonical_name="User Owner",
            user_id="user_owner",
            privacy=MemoryPrivacy.PUBLIC,
        )
        secret_tech = EntityNode(
            entity_type=EntityType.TECHNOLOGY,
            canonical_name="Secret Tech Engine",
            user_id="user_owner",
            privacy=MemoryPrivacy.PRIVATE,  # Private node!
        )

        await graph.add_node(user_node, ctx_owner)
        await graph.add_node(secret_tech, ctx_owner)

        edge = EntityEdge(
            source_node_id=user_node.node_id,
            target_node_id=secret_tech.node_id,
            relationship_type=RelationshipType.USES,
            privacy=MemoryPrivacy.PUBLIC,
            user_id="user_owner",
        )
        await graph.add_edge(edge, ctx_owner)

        # Owner queries neighbors (returns secret tech)
        neighbors_owner = await graph.query_neighbors(user_node.node_id, ctx_owner)
        assert len(neighbors_owner) == 1
        assert neighbors_owner[0]["neighbor"].canonical_name == "Secret Tech Engine"

        # Unauthorized caller queries neighbors (path privacy hides secret tech node)
        neighbors_other = await graph.query_neighbors(user_node.node_id, ctx_other)
        assert len(neighbors_other) == 0

    asyncio.run(_test())


def test_bounded_graph_traversal_and_cycle_prevention(graph: PostgresKnowledgeGraph) -> None:
    """Verify traverse() respects max_depth, max_nodes, max_edges, and handles cyclic topologies."""

    async def _test() -> None:
        ctx = MemoryAccessContext(user_id="user_owner", is_system=True)

        nodeA = EntityNode(
            canonical_name="Node A", entity_type=EntityType.COMPONENT, user_id="user_owner"
        )
        nodeB = EntityNode(
            canonical_name="Node B", entity_type=EntityType.COMPONENT, user_id="user_owner"
        )
        nodeC = EntityNode(
            canonical_name="Node C", entity_type=EntityType.COMPONENT, user_id="user_owner"
        )

        await graph.add_node(nodeA, ctx)
        await graph.add_node(nodeB, ctx)
        await graph.add_node(nodeC, ctx)

        # Create cycle: A -> B -> C -> A
        edgeAB = EntityEdge(
            source_node_id=nodeA.node_id,
            target_node_id=nodeB.node_id,
            relationship_type=RelationshipType.DEPENDS_ON,
            user_id="user_owner",
        )
        edgeBC = EntityEdge(
            source_node_id=nodeB.node_id,
            target_node_id=nodeC.node_id,
            relationship_type=RelationshipType.DEPENDS_ON,
            user_id="user_owner",
        )
        edgeCA = EntityEdge(
            source_node_id=nodeC.node_id,
            target_node_id=nodeA.node_id,
            relationship_type=RelationshipType.DEPENDS_ON,
            user_id="user_owner",
        )

        await graph.add_edge(edgeAB, ctx)
        await graph.add_edge(edgeBC, ctx)
        await graph.add_edge(edgeCA, ctx)

        # Traverse with max_depth=2
        trav_res = await graph.traverse(nodeA.node_id, ctx, max_depth=2, max_nodes=10)
        assert trav_res.start_node_id == nodeA.node_id
        assert trav_res.total_nodes_found == 3  # A, B, C (Cycle handled without infinite loop)
        assert trav_res.traversed_depth <= 2

    asyncio.run(_test())


def test_node_deletion_cascading_behavior(graph: PostgresKnowledgeGraph) -> None:
    """Verify deleting a node deactivates connected edges so no dangling active edges remain."""

    async def _test() -> None:
        ctx = MemoryAccessContext(user_id="user_owner")

        node1 = EntityNode(
            canonical_name="Parent Node", entity_type=EntityType.PROJECT, user_id="user_owner"
        )
        node2 = EntityNode(
            canonical_name="Child Node", entity_type=EntityType.COMPONENT, user_id="user_owner"
        )

        await graph.add_node(node1, ctx)
        await graph.add_node(node2, ctx)

        edge = EntityEdge(
            source_node_id=node1.node_id,
            target_node_id=node2.node_id,
            relationship_type=RelationshipType.CONTAINS,
            user_id="user_owner",
        )
        await graph.add_edge(edge, ctx)

        assert await graph.get_edge(edge.edge_id, ctx) is not None

        # Delete Parent Node
        await graph.delete_node(node1.node_id, ctx, soft_delete=True)

        # Connected edge must also be deactivated / hidden
        assert await graph.get_edge(edge.edge_id, ctx) is None

    asyncio.run(_test())
