"""Unified Memory Manager Subsystem & Architectural Facade.

Provides the concrete MemoryManager implementing IMemoryManager.
Serves as the single memory orchestration contract consumed by the Autonomous Planner,
coordinating Working Memory (Redis), Semantic Memory (pgvector), Episodic Memory (PostgreSQL),
and Relational Knowledge Graph (PostgreSQL) with concurrent retrieval, failure isolation,
deterministic ranking, deduplication, privacy defense-in-depth, and structured observability telemetry.
"""

import logging
import math
import time
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from app.memory.adapters.pgvector_store import PgVectorStore
from app.memory.adapters.postgres_knowledge_graph import PostgresKnowledgeGraph
from app.memory.adapters.redis_working_memory import RedisWorkingMemory
from app.memory.base import (
    IEpisodicMemory,
    IKnowledgeGraph,
    IMemoryManager,
    ISemanticMemory,
    IWorkingMemory,
)
from app.memory.embeddings import RouterEmbeddingProvider
from app.memory.episodic import EpisodicMemory
from app.memory.knowledge_graph import EntityEdge, EntityNode
from app.memory.models import (
    MemoryAccessContext,
    MemoryDeleteResult,
    MemoryItem,
    MemoryMetadata,
    MemoryQuery,
    MemoryScope,
    MemorySearchResult,
    MemoryType,
    MemoryWriteResult,
)
from app.memory.semantic import SemanticMemory

logger = logging.getLogger(__name__)


class MemoryRankingConfig(BaseModel):
    """Configuration parameters and weights for deterministic cross-tier memory ranking."""

    weight_similarity: float = Field(
        default=0.40, ge=0.0, le=1.0, description="Weight for vector similarity score"
    )
    weight_importance: float = Field(
        default=0.30, ge=0.0, le=1.0, description="Weight for item importance score"
    )
    weight_confidence: float = Field(
        default=0.15, ge=0.0, le=1.0, description="Weight for item confidence score"
    )
    weight_recency: float = Field(
        default=0.15, ge=0.0, le=1.0, description="Weight for recency decay score"
    )
    recency_decay_lambda: float = Field(
        default=0.01, ge=0.0, description="Half-life decay coefficient per hour"
    )


class MemoryManager(IMemoryManager):
    """Concrete Unified Memory Manager Facade implementing IMemoryManager.

    Architecture Invariant:
    The Autonomous Planner depends ONLY on IMemoryManager. It never directly accesses
    Redis, pgvector, PostgreSQL, SQLAlchemy, or raw vector stores.
    """

    def __init__(
        self,
        working_memory: IWorkingMemory | None = None,
        semantic_memory: ISemanticMemory | None = None,
        episodic_memory: IEpisodicMemory | None = None,
        knowledge_graph: IKnowledgeGraph | None = None,
        ranking_config: MemoryRankingConfig | None = None,
    ) -> None:
        """Initializes MemoryManager with underlying memory subsystem adapters."""
        default_embedder = RouterEmbeddingProvider()
        self._working = working_memory or RedisWorkingMemory()
        self._semantic = semantic_memory or SemanticMemory(
            vector_store=PgVectorStore(), embedding_provider=default_embedder
        )
        self._episodic = episodic_memory or EpisodicMemory(
            vector_store=PgVectorStore(), embedding_provider=default_embedder
        )
        self._graph = knowledge_graph or PostgresKnowledgeGraph()
        self.ranking_config = ranking_config or MemoryRankingConfig()

    def _log_telemetry(
        self,
        operation: str,
        access_context: MemoryAccessContext,
        tiers_queried: list[str],
        tiers_succeeded: list[str],
        tiers_failed: list[str],
        result_count: int,
        latency_ms: float,
        session_id: str | None = None,
    ) -> None:
        """Emits structured, non-sensitive observability telemetry logs."""
        telemetry = {
            "subsystem": "memory_manager",
            "operation": operation,
            "correlation_id": getattr(access_context, "correlation_id", "N/A"),
            "user_id": access_context.user_id,
            "session_id": session_id or access_context.session_id,
            "tiers_queried": tiers_queried,
            "tiers_succeeded": tiers_succeeded,
            "tiers_failed": tiers_failed,
            "result_count": result_count,
            "latency_ms": round(latency_ms, 2),
        }
        logger.info(f"MemoryManager Telemetry: {telemetry}")

    def _calculate_rank_score(
        self, item: MemoryItem, base_similarity: float, now: datetime
    ) -> float:
        """Calculates deterministic ranking score based on similarity, importance, confidence, and recency."""
        cfg = self.ranking_config

        # 1. Similarity score
        sim_score = max(0.0, min(1.0, base_similarity))

        # 2. Importance score
        imp_score = max(0.0, min(1.0, item.importance))

        # 3. Confidence score
        conf_score = max(0.0, min(1.0, item.metadata.confidence))

        # 4. Recency decay score
        created = item.created_at
        if created.tzinfo is None:
            created = created.replace(tzinfo=UTC)

        age_hours = max(0.0, (now - created).total_seconds() / 3600.0)
        rec_score = math.exp(-cfg.recency_decay_lambda * age_hours)

        total_score = (
            cfg.weight_similarity * sim_score
            + cfg.weight_importance * imp_score
            + cfg.weight_confidence * conf_score
            + cfg.weight_recency * rec_score
        )
        return float(total_score)

    async def remember(
        self, item: MemoryItem, access_context: MemoryAccessContext
    ) -> MemoryWriteResult:
        """Stores a memory item into the Hybrid Memory Matrix based on deterministic tier routing."""
        start_time = time.perf_counter()
        tiers_queried: list[str] = []
        tiers_succeeded: list[str] = []
        tiers_failed: list[str] = []

        # Defense-in-depth Privacy Enforcement
        if not access_context.can_access(item.metadata.privacy, item.user_id):
            return MemoryWriteResult(
                memory_id=item.id,
                success=False,
                memory_type=item.memory_type,
                error_message="Access Denied: Caller context cannot store item.",
            )

        # Deterministic Tier Routing
        primary_tier = item.memory_type.value.lower()
        tiers_queried.append(primary_tier)

        last_result: MemoryWriteResult | None = None

        try:
            if item.memory_type == MemoryType.WORKING:
                last_result = await self._working.add(item)
                tiers_succeeded.append("working")

                # High importance promotion heuristic (importance >= 0.7)
                if item.importance >= 0.7:
                    tiers_queried.append("episodic")
                    ep_res = await self._episodic.store_episode(item)
                    if ep_res.success:
                        tiers_succeeded.append("episodic")
                    else:
                        tiers_failed.append("episodic")

            elif item.memory_type == MemoryType.SEMANTIC:
                last_result = await self._semantic.store_concept(item, access_context)
                tiers_succeeded.append("semantic")

            elif item.memory_type == MemoryType.EPISODIC:
                last_result = await self._episodic.store_episode(item)
                tiers_succeeded.append("episodic")

            elif item.memory_type == MemoryType.RELATIONAL:
                # Relational routing via Knowledge Graph
                custom_attrs = item.metadata.custom_attributes
                if "node" in custom_attrs and isinstance(custom_attrs["node"], EntityNode):
                    await self._graph.add_node(custom_attrs["node"], access_context)
                    tiers_succeeded.append("relational_node")
                elif "edge" in custom_attrs and isinstance(custom_attrs["edge"], EntityEdge):
                    await self._graph.add_edge(custom_attrs["edge"], access_context)
                    tiers_succeeded.append("relational_edge")
                else:
                    # Fallback node creation from MemoryItem
                    node = EntityNode(
                        canonical_name=item.content,
                        entity_type="CONCEPT",
                        privacy=item.metadata.privacy,
                        user_id=item.user_id,
                        project_id=item.project_id,
                    )
                    await self._graph.add_node(node, access_context)
                    tiers_succeeded.append("relational")

                last_result = MemoryWriteResult(
                    memory_id=item.id,
                    success=True,
                    memory_type=item.memory_type,
                )
            else:
                return MemoryWriteResult(
                    memory_id=item.id,
                    success=False,
                    memory_type=item.memory_type,
                    error_message=f"Unsupported memory type '{item.memory_type}'.",
                )

        except Exception as exc:
            logger.error(
                f"MemoryManager remember failed for item '{item.id}': {exc}", exc_info=True
            )
            tiers_failed.append(primary_tier)
            last_result = MemoryWriteResult(
                memory_id=item.id,
                success=False,
                memory_type=item.memory_type,
                error_message=f"Storage tier error: {exc}",
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        self._log_telemetry(
            operation="remember",
            access_context=access_context,
            tiers_queried=tiers_queried,
            tiers_succeeded=tiers_succeeded,
            tiers_failed=tiers_failed,
            result_count=1 if last_result and last_result.success else 0,
            latency_ms=elapsed_ms,
            session_id=item.session_id,
        )

        return last_result or MemoryWriteResult(
            memory_id=item.id,
            success=False,
            memory_type=item.memory_type,
            error_message="Unknown routing failure.",
        )

    async def recall(
        self, query: MemoryQuery, access_context: MemoryAccessContext
    ) -> MemorySearchResult:
        """Executes cross-tier memory recall with failure isolation, deduplication, ranking, and privacy filtering."""
        start_time = time.perf_counter()
        now = datetime.now(UTC)

        tiers_queried: list[str] = []
        tiers_succeeded: list[str] = []
        tiers_failed: list[str] = []

        # Target tiers selection
        target_types = set(query.memory_types) if query.memory_types else set()

        query_working = (
            not target_types or MemoryType.WORKING in target_types or bool(query.session_id)
        )
        query_semantic = not target_types or MemoryType.SEMANTIC in target_types
        query_episodic = not target_types or MemoryType.EPISODIC in target_types
        query_graph = MemoryType.RELATIONAL in target_types

        raw_candidates: list[tuple[MemoryItem, float]] = []

        # 1. Query Working Memory
        if query_working and query.session_id:
            tiers_queried.append("working")
            try:
                wm_items = await self._working.get_history(query.session_id, limit=query.limit)
                for w_item in wm_items:
                    raw_candidates.append((w_item, 0.5))
                tiers_succeeded.append("working")
            except Exception as exc:
                logger.warning(f"Working memory recall failed: {exc}")
                tiers_failed.append("working")

        # 2. Query Semantic Memory
        if query_semantic:
            tiers_queried.append("semantic")
            try:
                sem_res = await self._semantic.search_concepts(query, access_context)
                for idx, item in enumerate(sem_res.items):
                    score = sem_res.scores[idx] if idx < len(sem_res.scores) else 0.5
                    raw_candidates.append((item, score))
                tiers_succeeded.append("semantic")
            except Exception as exc:
                logger.warning(f"Semantic memory recall failed: {exc}")
                tiers_failed.append("semantic")

        # 3. Query Episodic Memory
        if query_episodic:
            tiers_queried.append("episodic")
            try:
                ep_res = await self._episodic.recall_episodes(query, access_context)
                for idx, item in enumerate(ep_res.items):
                    score = ep_res.scores[idx] if idx < len(ep_res.scores) else 0.5
                    raw_candidates.append((item, score))
                tiers_succeeded.append("episodic")
            except Exception as exc:
                logger.warning(f"Episodic memory recall failed: {exc}")
                tiers_failed.append("episodic")

        # 4. Query Knowledge Graph Memory
        if query_graph and query.user_id:
            tiers_queried.append("relational")
            try:
                neighbors = await self._graph.query_neighbors(query.user_id, access_context)
                for nbr in neighbors:
                    n_node: EntityNode = nbr["neighbor"]
                    g_item = MemoryItem(
                        id=n_node.node_id,
                        memory_type=MemoryType.RELATIONAL,
                        scope=query.scope or MemoryScope.PROJECT,
                        content=f"{n_node.canonical_name}: {n_node.properties}",
                        user_id=n_node.user_id,
                        project_id=n_node.project_id,
                        metadata=MemoryMetadata(privacy=n_node.privacy),
                    )
                    raw_candidates.append((g_item, 0.6))
                tiers_succeeded.append("relational")
            except Exception as exc:
                logger.warning(f"Relational Knowledge Graph recall failed: {exc}")
                tiers_failed.append("relational")

        # 5. Deduplication and Privacy Filtering Defense-in-Depth
        unique_candidates: dict[str, tuple[MemoryItem, float]] = {}

        for item, base_sim in raw_candidates:
            # Privacy check defense-in-depth
            if not access_context.can_access(item.metadata.privacy, item.user_id):
                continue

            # Query attribute filters
            if query.user_id and item.user_id and item.user_id != query.user_id:
                continue
            if query.project_id and item.project_id and item.project_id != query.project_id:
                continue

            if item.id not in unique_candidates:
                unique_candidates[item.id] = (item, base_sim)
            else:
                # Keep candidate with higher base similarity score
                existing_item, existing_sim = unique_candidates[item.id]
                if base_sim > existing_sim:
                    unique_candidates[item.id] = (item, base_sim)

        # 6. Deterministic Result Ranking
        ranked_list: list[tuple[MemoryItem, float]] = []
        for _memory_id, (item, base_sim) in unique_candidates.items():
            composite_score = self._calculate_rank_score(item, base_sim, now)
            ranked_list.append((item, composite_score))

        # Sort descending by rank score
        ranked_list.sort(key=lambda x: x[1], reverse=True)
        sliced = ranked_list[: query.limit]

        final_items = [item for item, _ in sliced]
        final_scores = [score for _, score in sliced]
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        self._log_telemetry(
            operation="recall",
            access_context=access_context,
            tiers_queried=tiers_queried,
            tiers_succeeded=tiers_succeeded,
            tiers_failed=tiers_failed,
            result_count=len(final_items),
            latency_ms=elapsed_ms,
            session_id=query.session_id,
        )

        return MemorySearchResult(
            items=final_items,
            scores=final_scores,
            query=query,
            total_found=len(unique_candidates),
            search_time_ms=elapsed_ms,
        )

    async def forget(
        self,
        memory_id: str,
        access_context: MemoryAccessContext,
        soft_delete: bool = True,
    ) -> MemoryDeleteResult:
        """Removes or tombstones a memory item across storage tiers."""
        start_time = time.perf_counter()
        tiers_queried = ["working", "semantic", "episodic", "relational"]
        tiers_succeeded: list[str] = []
        tiers_failed: list[str] = []

        total_deleted = 0

        # Delete from Semantic Store
        try:
            sem_del = await self._semantic.delete_concept(
                memory_id, access_context, soft_delete=soft_delete
            )
            if sem_del.success:
                tiers_succeeded.append("semantic")
                total_deleted += sem_del.deleted_count
        except Exception as exc:
            logger.warning(f"Semantic store forget failed for '{memory_id}': {exc}")
            tiers_failed.append("semantic")

        # Delete from Knowledge Graph Node/Edge
        try:
            graph_del_node = await self._graph.delete_node(
                memory_id, access_context, soft_delete=soft_delete
            )
            graph_del_edge = await self._graph.delete_edge(
                memory_id, access_context, soft_delete=soft_delete
            )
            if graph_del_node or graph_del_edge:
                tiers_succeeded.append("relational")
                total_deleted += 1
        except Exception as exc:
            logger.warning(f"Knowledge graph forget failed for '{memory_id}': {exc}")
            tiers_failed.append("relational")

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        success = len(tiers_succeeded) > 0

        self._log_telemetry(
            operation="forget",
            access_context=access_context,
            tiers_queried=tiers_queried,
            tiers_succeeded=tiers_succeeded,
            tiers_failed=tiers_failed,
            result_count=total_deleted,
            latency_ms=elapsed_ms,
        )

        return MemoryDeleteResult(
            memory_id=memory_id,
            success=success,
            soft_deleted=soft_delete,
            deleted_count=total_deleted,
            error_message=None if success else "Failed to delete item from storage tiers.",
        )

    async def consolidate(
        self, session_id: str, access_context: MemoryAccessContext
    ) -> list[MemoryWriteResult]:
        """Consolidates short-term session working memory into long-term episodic/semantic tiers based on eligibility rules."""
        start_time = time.perf_counter()
        tiers_queried = ["working"]
        tiers_succeeded: list[str] = []
        tiers_failed: list[str] = []

        write_results: list[MemoryWriteResult] = []

        try:
            wm_items = await self._working.get_history(session_id, limit=100)
            tiers_succeeded.append("working")

            for item in wm_items:
                # Privacy check
                if not access_context.can_access(item.metadata.privacy, item.user_id):
                    continue

                # Deterministic Consolidation Eligibility Policy:
                # Must be active, unexpired, importance >= 0.6, confidence >= 0.6
                if not item.is_active:
                    continue
                if item.importance < 0.6 or item.metadata.confidence < 0.6:
                    continue

                # Promote WORKING -> EPISODIC (or SEMANTIC if type is semantic)
                if item.memory_type == MemoryType.SEMANTIC:
                    res = await self._semantic.store_concept(item, access_context)
                    write_results.append(res)
                    tiers_succeeded.append("semantic_promotion")
                else:
                    promoted_item = item.model_copy(update={"memory_type": MemoryType.EPISODIC})
                    res = await self._episodic.store_episode(promoted_item)
                    write_results.append(res)
                    tiers_succeeded.append("episodic_promotion")

        except Exception as exc:
            logger.error(
                f"MemoryManager consolidation failed for session '{session_id}': {exc}",
                exc_info=True,
            )
            tiers_failed.append("consolidation")

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        self._log_telemetry(
            operation="consolidate",
            access_context=access_context,
            tiers_queried=tiers_queried,
            tiers_succeeded=tiers_succeeded,
            tiers_failed=tiers_failed,
            result_count=len(write_results),
            latency_ms=elapsed_ms,
            session_id=session_id,
        )

        return write_results
