"""Unit Tests for EpisodicMemory Manager.

Validates episode storage preserving timestamps, session/user/project/planner attributes,
chronological recall sorting, session and user isolation, project filtering, deletion,
and MemoryAccessContext privacy enforcement.
"""

import asyncio
from datetime import UTC, datetime, timedelta

from app.memory.episodic import EpisodicMemory
from app.memory.models import (
    MemoryAccessContext,
    MemoryItem,
    MemoryMetadata,
    MemoryPrivacy,
    MemoryQuery,
    MemoryScope,
    MemorySource,
    MemoryType,
)
from app.memory.vector_store import InMemoryVectorStore


def test_episodic_memory_store_and_chronological_recall() -> None:
    """Verify episode storage and chronological recall sorting."""

    async def _test() -> None:
        vector_store = InMemoryVectorStore(expected_dimension=4)
        episodic = EpisodicMemory(vector_store=vector_store)

        t0 = datetime.now(UTC) - timedelta(minutes=10)
        t1 = datetime.now(UTC) - timedelta(minutes=5)
        t2 = datetime.now(UTC)

        episode1 = MemoryItem(
            memory_type=MemoryType.EPISODIC,
            scope=MemoryScope.SESSION,
            session_id="sess_ep_01",
            user_id="user_alpha",
            project_id="proj_100",
            content="First episode step",
            importance=0.8,
            created_at=t0,
            metadata=MemoryMetadata(
                source=MemorySource.USER_INPUT,
                privacy=MemoryPrivacy.PUBLIC,
                confidence=0.9,
                custom_attributes={"planner_id": "planner_v2"},
            ),
        )

        episode2 = MemoryItem(
            memory_type=MemoryType.EPISODIC,
            scope=MemoryScope.SESSION,
            session_id="sess_ep_01",
            user_id="user_alpha",
            project_id="proj_100",
            content="Second episode step",
            importance=0.9,
            created_at=t1,
            metadata=MemoryMetadata(
                source=MemorySource.PLANNER_OUTPUT,
                privacy=MemoryPrivacy.PUBLIC,
                confidence=0.95,
                custom_attributes={"planner_id": "planner_v2"},
            ),
        )

        episode3 = MemoryItem(
            memory_type=MemoryType.EPISODIC,
            scope=MemoryScope.SESSION,
            session_id="sess_ep_01",
            user_id="user_alpha",
            project_id="proj_100",
            content="Third episode step",
            importance=0.5,
            created_at=t2,
            metadata=MemoryMetadata(
                source=MemorySource.TOOL_OUTPUT,
                privacy=MemoryPrivacy.PUBLIC,
                confidence=1.0,
                custom_attributes={"planner_id": "planner_v2"},
            ),
        )

        # Store out of order
        await episodic.store_episode(episode2)
        await episodic.store_episode(episode1)
        await episodic.store_episode(episode3)

        access_ctx = MemoryAccessContext(user_id="user_alpha")
        query = MemoryQuery(session_id="sess_ep_01")

        recall_res = await episodic.recall_episodes(query, access_ctx)
        assert recall_res.total_found == 3
        # Chronologically sorted by created_at
        assert recall_res.items[0].content == "First episode step"
        assert recall_res.items[1].content == "Second episode step"
        assert recall_res.items[2].content == "Third episode step"

        # Verify preserved metadata attributes
        assert recall_res.items[0].metadata.source == MemorySource.USER_INPUT
        assert recall_res.items[0].metadata.custom_attributes.get("planner_id") == "planner_v2"
        assert recall_res.items[0].project_id == "proj_100"

    asyncio.run(_test())


def test_episodic_isolation_and_privacy_enforcement() -> None:
    """Verify session/user isolation, project filtering, and privacy filtering in Episodic Memory."""

    async def _test() -> None:
        vector_store = InMemoryVectorStore(expected_dimension=4)
        episodic = EpisodicMemory(vector_store=vector_store)

        ep_user1_projA = MemoryItem(
            memory_type=MemoryType.EPISODIC,
            scope=MemoryScope.PROJECT,
            user_id="user_1",
            project_id="proj_A",
            session_id="sess_1",
            content="User 1 Project A Episode",
            metadata=MemoryMetadata(privacy=MemoryPrivacy.PRIVATE),
        )

        ep_user2_projB = MemoryItem(
            memory_type=MemoryType.EPISODIC,
            scope=MemoryScope.PROJECT,
            user_id="user_2",
            project_id="proj_B",
            session_id="sess_2",
            content="User 2 Project B Episode",
            metadata=MemoryMetadata(privacy=MemoryPrivacy.PRIVATE),
        )

        await episodic.store_episode(ep_user1_projA)
        await episodic.store_episode(ep_user2_projB)

        ctx_user1 = MemoryAccessContext(user_id="user_1")
        ctx_user2 = MemoryAccessContext(user_id="user_2")

        # User 1 queries Project A
        res1 = await episodic.recall_episodes(MemoryQuery(project_id="proj_A"), ctx_user1)
        assert res1.total_found == 1
        assert res1.items[0].content == "User 1 Project A Episode"

        # User 2 cannot access User 1's Private Project A episode
        res_unauth = await episodic.recall_episodes(MemoryQuery(project_id="proj_A"), ctx_user2)
        assert res_unauth.total_found == 0

        # Deletion
        del_res = await episodic.delete_episode(ep_user1_projA.id, ctx_user1, soft_delete=True)
        assert del_res.success is True

        res_after_del = await episodic.recall_episodes(MemoryQuery(project_id="proj_A"), ctx_user1)
        assert res_after_del.total_found == 0

    asyncio.run(_test())
