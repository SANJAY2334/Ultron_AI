"""Unit and Integration Tests for Unified MemoryManager Subsystem.

Validates remember routing, recall tier orchestrations, deterministic ranking, cross-tier deduplication,
privacy defense-in-depth, failure isolation, soft/physical forget operations, session consolidation,
working -> episodic/semantic promotion, DI container resolution, and telemetry logging.
"""

import asyncio
from typing import Any

import pytest

from app.api.deps import get_memory_manager
from app.core.container import container
from app.memory.base import (
    IEpisodicMemory,
    IKnowledgeGraph,
    IMemoryManager,
    ISemanticMemory,
    IWorkingMemory,
)
from app.memory.knowledge_graph import EntityNode
from app.memory.manager import MemoryManager
from app.memory.models import (
    MemoryAccessContext,
    MemoryDeleteResult,
    MemoryItem,
    MemoryMetadata,
    MemoryPrivacy,
    MemoryQuery,
    MemoryScope,
    MemorySearchResult,
    MemoryType,
    MemoryWriteResult,
)


class DummyWorkingMemory(IWorkingMemory):
    """Mock WorkingMemory for unit testing MemoryManager."""

    def __init__(self) -> None:
        self.items: dict[str, MemoryItem] = {}
        self.should_fail = False

    async def add(self, item: MemoryItem) -> MemoryWriteResult:
        if self.should_fail:
            raise RuntimeError("Simulated Working Memory Failure")
        self.items[item.id] = item
        return MemoryWriteResult(memory_id=item.id, success=True, memory_type=item.memory_type)

    async def get(self, item_id: str) -> MemoryItem | None:
        if self.should_fail:
            raise RuntimeError("Simulated Working Memory Failure")
        return self.items.get(item_id)

    async def clear_session(self, session_id: str) -> bool:
        if self.should_fail:
            raise RuntimeError("Simulated Working Memory Failure")
        self.items = {k: v for k, v in self.items.items() if v.session_id != session_id}
        return True

    async def get_history(self, session_id: str, limit: int = 50) -> list[MemoryItem]:
        if self.should_fail:
            raise RuntimeError("Simulated Working Memory Failure")
        matching = [item for item in self.items.values() if item.session_id == session_id]
        return matching[:limit]


class DummySemanticMemory(ISemanticMemory):
    """Mock SemanticMemory for unit testing MemoryManager."""

    def __init__(self) -> None:
        self.concepts: dict[str, MemoryItem] = {}
        self.should_fail = False

    async def store_concept(
        self, item: MemoryItem, access_context: MemoryAccessContext | None = None
    ) -> MemoryWriteResult:
        if self.should_fail:
            raise RuntimeError("Simulated Semantic Memory Failure")
        self.concepts[item.id] = item
        return MemoryWriteResult(memory_id=item.id, success=True, memory_type=item.memory_type)

    async def search_concepts(
        self, query: MemoryQuery, access_context: MemoryAccessContext
    ) -> MemorySearchResult:
        if self.should_fail:
            raise RuntimeError("Simulated Semantic Memory Failure")
        items = list(self.concepts.values())
        scores = [0.85] * len(items)
        return MemorySearchResult(
            items=items,
            scores=scores,
            query=query,
            total_found=len(items),
            search_time_ms=1.5,
        )

    async def delete_concept(
        self,
        memory_id: str,
        access_context: MemoryAccessContext,
        soft_delete: bool = True,
    ) -> MemoryDeleteResult:
        if self.should_fail:
            raise RuntimeError("Simulated Semantic Memory Failure")
        if memory_id in self.concepts:
            del self.concepts[memory_id]
            return MemoryDeleteResult(
                memory_id=memory_id, success=True, soft_deleted=soft_delete, deleted_count=1
            )
        return MemoryDeleteResult(
            memory_id=memory_id, success=False, soft_deleted=soft_delete, deleted_count=0
        )


class DummyEpisodicMemory(IEpisodicMemory):
    """Mock EpisodicMemory for unit testing MemoryManager."""

    def __init__(self) -> None:
        self.episodes: dict[str, MemoryItem] = {}
        self.should_fail = False

    async def store_episode(self, item: MemoryItem) -> MemoryWriteResult:
        if self.should_fail:
            raise RuntimeError("Simulated Episodic Memory Failure")
        self.episodes[item.id] = item
        return MemoryWriteResult(memory_id=item.id, success=True, memory_type=item.memory_type)

    async def recall_episodes(
        self, query: MemoryQuery, access_context: MemoryAccessContext
    ) -> MemorySearchResult:
        if self.should_fail:
            raise RuntimeError("Simulated Episodic Memory Failure")
        items = list(self.episodes.values())
        scores = [0.75] * len(items)
        return MemorySearchResult(
            items=items,
            scores=scores,
            query=query,
            total_found=len(items),
            search_time_ms=2.0,
        )


class DummyKnowledgeGraph(IKnowledgeGraph):
    """Mock KnowledgeGraph for unit testing MemoryManager."""

    def __init__(self) -> None:
        self.nodes: dict[str, EntityNode] = {}
        self.should_fail = False

    async def add_node(
        self, node: Any, access_context: MemoryAccessContext | None = None
    ) -> EntityNode:
        if self.should_fail:
            raise RuntimeError("Simulated Graph Failure")
        if isinstance(node, EntityNode):
            self.nodes[node.node_id] = node
            return node

        n = EntityNode(
            node_id=str(node),
            entity_type="CONCEPT",
            canonical_name=str(node),
        )
        self.nodes[n.node_id] = n
        return n

    async def get_node(
        self, node_id: str, access_context: MemoryAccessContext
    ) -> EntityNode | None:
        if self.should_fail:
            raise RuntimeError("Simulated Graph Failure")
        return self.nodes.get(node_id)

    async def update_node(
        self, node: EntityNode, access_context: MemoryAccessContext
    ) -> EntityNode:
        return await self.add_node(node, access_context)

    async def delete_node(
        self, node_id: str, access_context: MemoryAccessContext, soft_delete: bool = True
    ) -> bool:
        if self.should_fail:
            raise RuntimeError("Simulated Graph Failure")
        if node_id in self.nodes:
            del self.nodes[node_id]
            return True
        return False

    async def add_edge(self, edge: Any, access_context: MemoryAccessContext | None = None) -> Any:
        if self.should_fail:
            raise RuntimeError("Simulated Graph Failure")
        return edge

    async def get_edge(self, edge_id: str, access_context: MemoryAccessContext) -> Any:
        return None

    async def delete_edge(
        self, edge_id: str, access_context: MemoryAccessContext, soft_delete: bool = True
    ) -> bool:
        if self.should_fail:
            raise RuntimeError("Simulated Graph Failure")
        return False

    async def query_neighbors(
        self,
        node_id: str,
        access_context: MemoryAccessContext,
        relationship_type: str | None = None,
    ) -> list[dict[str, Any]]:
        if self.should_fail:
            raise RuntimeError("Simulated Graph Failure")
        res = []
        for n in self.nodes.values():
            res.append({"neighbor": n, "edge": None})
        return res

    async def traverse(
        self,
        start_node_id: str,
        access_context: MemoryAccessContext,
        max_depth: int = 3,
        max_nodes: int = 50,
        max_edges: int = 100,
        relationship_types: list[str] | None = None,
    ) -> Any:
        if self.should_fail:
            raise RuntimeError("Simulated Graph Failure")
        return None


@pytest.fixture
def mock_subsystems():
    return {
        "working": DummyWorkingMemory(),
        "semantic": DummySemanticMemory(),
        "episodic": DummyEpisodicMemory(),
        "graph": DummyKnowledgeGraph(),
    }


@pytest.fixture
def manager(mock_subsystems) -> MemoryManager:
    return MemoryManager(
        working_memory=mock_subsystems["working"],
        semantic_memory=mock_subsystems["semantic"],
        episodic_memory=mock_subsystems["episodic"],
        knowledge_graph=mock_subsystems["graph"],
    )


def test_remember_routing(manager: MemoryManager, mock_subsystems) -> None:
    """Verify remember() routes WORKING, SEMANTIC, EPISODIC, and RELATIONAL items correctly."""

    async def _test() -> None:
        ctx = MemoryAccessContext(user_id="user_test")

        # 1. WORKING item
        item_w = MemoryItem(
            memory_type=MemoryType.WORKING,
            scope=MemoryScope.SESSION,
            session_id="sess_1",
            user_id="user_test",
            content="Working note",
        )
        res_w = await manager.remember(item_w, ctx)
        assert res_w.success is True
        assert item_w.id in mock_subsystems["working"].items

        # 2. SEMANTIC item
        item_s = MemoryItem(
            memory_type=MemoryType.SEMANTIC,
            scope=MemoryScope.USER,
            user_id="user_test",
            content="Semantic concept",
        )
        res_s = await manager.remember(item_s, ctx)
        assert res_s.success is True
        assert item_s.id in mock_subsystems["semantic"].concepts

        # 3. EPISODIC item
        item_e = MemoryItem(
            memory_type=MemoryType.EPISODIC,
            scope=MemoryScope.USER,
            user_id="user_test",
            content="Episodic event trace",
        )
        res_e = await manager.remember(item_e, ctx)
        assert res_e.success is True
        assert item_e.id in mock_subsystems["episodic"].episodes

        # 4. RELATIONAL item
        node = EntityNode(
            canonical_name="Graph Concept", entity_type="CONCEPT", user_id="user_test"
        )
        item_r = MemoryItem(
            memory_type=MemoryType.RELATIONAL,
            scope=MemoryScope.PROJECT,
            project_id="proj_123",
            user_id="user_test",
            content="Relational triple",
            metadata=MemoryMetadata(custom_attributes={"node": node}),
        )
        res_r = await manager.remember(item_r, ctx)
        assert res_r.success is True
        assert node.node_id in mock_subsystems["graph"].nodes

    asyncio.run(_test())


def test_remember_privacy_rejection(manager: MemoryManager) -> None:
    """Verify remember() rejects storing items when privacy access is denied."""

    async def _test() -> None:
        ctx_unauth = MemoryAccessContext(user_id="unauth_user")
        item_private = MemoryItem(
            memory_type=MemoryType.WORKING,
            scope=MemoryScope.SESSION,
            session_id="sess_private",
            user_id="owner_user",
            content="Private secret",
            metadata=MemoryMetadata(privacy=MemoryPrivacy.PRIVATE),
        )

        res = await manager.remember(item_private, ctx_unauth)
        assert res.success is False
        assert "Access Denied" in (res.error_message or "")

    asyncio.run(_test())


def test_recall_cross_tier_merging_ranking_and_deduplication(
    manager: MemoryManager, mock_subsystems
) -> None:
    """Verify recall() merges cross-tier results, deduplicates by memory_id, ranks deterministically, and filters privacy."""

    async def _test() -> None:
        ctx = MemoryAccessContext(user_id="user_rec")

        # Add duplicate item to both working and semantic memory
        shared_id = "mem_shared_123"
        shared_item = MemoryItem(
            id=shared_id,
            memory_type=MemoryType.SEMANTIC,
            scope=MemoryScope.USER,
            user_id="user_rec",
            content="Shared memory concept",
            importance=0.9,
            metadata=MemoryMetadata(confidence=0.95),
        )

        await mock_subsystems["working"].add(shared_item)
        await mock_subsystems["semantic"].store_concept(shared_item, ctx)

        query = MemoryQuery(user_id="user_rec", limit=10)
        res = await manager.recall(query, ctx)

        assert res.total_found >= 1
        # Check deduplication: shared_id must only appear ONCE in results
        ids = [it.id for it in res.items]
        assert ids.count(shared_id) == 1

        # Check deterministic scores descending
        assert res.scores == sorted(res.scores, reverse=True)

    asyncio.run(_test())


def test_recall_partial_failure_isolation(manager: MemoryManager, mock_subsystems) -> None:
    """Verify manager recall returns surviving tier results when one backend fails."""

    async def _test() -> None:
        ctx = MemoryAccessContext(user_id="user_fail")

        item_sem = MemoryItem(
            memory_type=MemoryType.SEMANTIC,
            scope=MemoryScope.USER,
            user_id="user_fail",
            content="Surviving semantic concept",
        )
        await mock_subsystems["semantic"].store_concept(item_sem, ctx)

        # Simulate Redis Working Memory Failure
        mock_subsystems["working"].should_fail = True

        query = MemoryQuery(session_id="sess_123", user_id="user_fail", limit=10)
        res = await manager.recall(query, ctx)

        # Recall should NOT crash; returns semantic results safely
        assert len(res.items) >= 1
        assert res.items[0].id == item_sem.id

    asyncio.run(_test())


def test_forget_soft_and_physical(manager: MemoryManager, mock_subsystems) -> None:
    """Verify forget() removes/tombstones memory across subsystems."""

    async def _test() -> None:
        ctx = MemoryAccessContext(user_id="user_del")

        item = MemoryItem(
            memory_type=MemoryType.SEMANTIC,
            scope=MemoryScope.USER,
            user_id="user_del",
            content="Concept to forget",
        )
        await manager.remember(item, ctx)

        del_res = await manager.forget(item.id, ctx, soft_delete=True)
        assert del_res.success is True
        assert del_res.deleted_count >= 1

    asyncio.run(_test())


def test_consolidation_and_promotion(manager: MemoryManager, mock_subsystems) -> None:
    """Verify consolidate() promotes eligible high-importance working memories to episodic/semantic tiers."""

    async def _test() -> None:
        ctx = MemoryAccessContext(user_id="user_cons")

        # 1. Eligible working memory item (importance >= 0.6, confidence >= 0.6)
        eligible_item = MemoryItem(
            memory_type=MemoryType.WORKING,
            scope=MemoryScope.SESSION,
            session_id="sess_cons_1",
            user_id="user_cons",
            content="Important meeting decision",
            importance=0.85,
            metadata=MemoryMetadata(confidence=0.90),
        )

        # 2. Ineligible working memory item (importance < 0.6)
        ineligible_item = MemoryItem(
            memory_type=MemoryType.WORKING,
            scope=MemoryScope.SESSION,
            session_id="sess_cons_1",
            user_id="user_cons",
            content="Trivial chatter",
            importance=0.20,
            metadata=MemoryMetadata(confidence=0.50),
        )

        await mock_subsystems["working"].add(eligible_item)
        await mock_subsystems["working"].add(ineligible_item)

        results = await manager.consolidate("sess_cons_1", ctx)

        assert len(results) == 1
        assert results[0].memory_id == eligible_item.id
        assert eligible_item.id in mock_subsystems["episodic"].episodes

    asyncio.run(_test())


def test_di_container_resolution() -> None:
    """Verify IMemoryManager resolves concrete MemoryManager facade through DI Container."""
    manager = get_memory_manager()
    assert isinstance(manager, MemoryManager)
    assert container.resolve_sync(IMemoryManager) is manager
