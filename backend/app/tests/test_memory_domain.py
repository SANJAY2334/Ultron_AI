"""Unit Tests for Refined Hybrid Memory Matrix Domain Models and Interfaces.

Validates MemoryItem scope invariants, unauthorized GLOBAL memory rejection,
MemoryAccessContext privacy access rules, VectorEmbedding model/version metadata,
and IMemoryManager facade contracts.
"""

import asyncio

import pytest

from app.memory.base import (
    IMemoryManager,
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
    MemoryType,
    MemoryWriteResult,
    VectorEmbedding,
)


def test_memory_scope_invariants() -> None:
    """Verify scope identifier invariants (SESSION -> session_id, USER -> user_id, PROJECT -> project_id)."""
    # SESSION scope requires session_id
    with pytest.raises(ValueError, match="MemoryScope.SESSION requires a valid"):
        MemoryItem(
            memory_type=MemoryType.WORKING,
            scope=MemoryScope.SESSION,
            session_id=None,
            content="Test content",
        )

    # USER scope requires user_id
    with pytest.raises(ValueError, match="MemoryScope.USER requires a valid"):
        MemoryItem(
            memory_type=MemoryType.WORKING,
            scope=MemoryScope.USER,
            user_id=None,
            content="Test content",
        )

    # PROJECT scope requires project_id
    with pytest.raises(ValueError, match="MemoryScope.PROJECT requires a valid"):
        MemoryItem(
            memory_type=MemoryType.WORKING,
            scope=MemoryScope.PROJECT,
            project_id=None,
            content="Test content",
        )

    # Valid scope item
    valid_session_item = MemoryItem(
        memory_type=MemoryType.WORKING,
        scope=MemoryScope.SESSION,
        session_id="sess_valid_123",
        content="Valid session item content",
    )
    assert valid_session_item.session_id == "sess_valid_123"


def test_unauthorized_global_memory_rejected() -> None:
    """Verify GLOBAL memory creation without system authorization is rejected."""
    with pytest.raises(ValueError, match="MemoryScope.GLOBAL scope items require explicit"):
        MemoryItem(
            memory_type=MemoryType.SEMANTIC,
            scope=MemoryScope.GLOBAL,
            content="Attempting unauthorized global memory write",
        )

    # Authorized GLOBAL item
    authorized_item = MemoryItem(
        memory_type=MemoryType.SEMANTIC,
        scope=MemoryScope.GLOBAL,
        content="System-wide global architecture rule",
        metadata=MemoryMetadata(is_system_authorized=True),
    )
    assert authorized_item.scope == MemoryScope.GLOBAL
    assert authorized_item.metadata.is_system_authorized is True


def test_memory_access_context_privacy_enforcement() -> None:
    """Verify MemoryAccessContext privacy rules for PUBLIC, PRIVATE, CONFIDENTIAL, and RESTRICTED levels."""
    user_owner = MemoryAccessContext(user_id="user_owner_123", is_system=False)
    user_other = MemoryAccessContext(user_id="user_other_999", is_system=False)
    confidential_user = MemoryAccessContext(user_id="user_other_999", roles={"confidential_access"})
    restricted_user = MemoryAccessContext(user_id="user_other_999", roles={"restricted_access"})
    system_ctx = MemoryAccessContext(is_system=True)

    # PUBLIC
    assert user_other.can_access(MemoryPrivacy.PUBLIC, "user_owner_123") is True

    # PRIVATE
    assert user_owner.can_access(MemoryPrivacy.PRIVATE, "user_owner_123") is True
    assert user_other.can_access(MemoryPrivacy.PRIVATE, "user_owner_123") is False
    assert system_ctx.can_access(MemoryPrivacy.PRIVATE, "user_owner_123") is True

    # CONFIDENTIAL
    assert user_owner.can_access(MemoryPrivacy.CONFIDENTIAL, "user_owner_123") is True
    assert user_other.can_access(MemoryPrivacy.CONFIDENTIAL, "user_owner_123") is False
    assert confidential_user.can_access(MemoryPrivacy.CONFIDENTIAL, "user_owner_123") is True

    # RESTRICTED
    assert user_owner.can_access(MemoryPrivacy.RESTRICTED, "user_owner_123") is False
    assert restricted_user.can_access(MemoryPrivacy.RESTRICTED, "user_owner_123") is True
    assert system_ctx.can_access(MemoryPrivacy.RESTRICTED, "user_owner_123") is True


def test_vector_embedding_versioning_and_metadata() -> None:
    """Verify VectorEmbedding dimension validation and multi-model versioning support."""
    vec = VectorEmbedding(
        vector=[0.1, 0.2, 0.3, 0.4],
        model_name="text-embedding-3-small",
        dimension=4,
        version="v2.1",
    )
    assert vec.dimension == 4
    assert vec.version == "v2.1"

    # Dimension mismatch error
    with pytest.raises(
        ValueError, match="length \\(3\\) does not match declared dimension \\(4\\)"
    ):
        VectorEmbedding(vector=[0.1, 0.2, 0.3], dimension=4)

    # Multi-embedding memory item
    item = MemoryItem(
        memory_type=MemoryType.SEMANTIC,
        scope=MemoryScope.USER,
        user_id="user_123",
        content="Embedding test content",
        embeddings=[vec],
    )
    assert len(item.embeddings) == 1
    assert item.embeddings[0].version == "v2.1"


class MockMemoryManager(IMemoryManager):
    """Dummy concrete implementation of IMemoryManager for testing interface adherence."""

    async def remember(
        self, item: MemoryItem, access_context: MemoryAccessContext
    ) -> MemoryWriteResult:
        if not access_context.can_access(item.metadata.privacy, item.user_id):
            return MemoryWriteResult(
                memory_id=item.id,
                success=False,
                memory_type=item.memory_type,
                error_message="Access Denied",
            )
        return MemoryWriteResult(memory_id=item.id, success=True, memory_type=item.memory_type)

    async def recall(
        self, query: MemoryQuery, access_context: MemoryAccessContext
    ) -> MemorySearchResult:
        return MemorySearchResult(
            items=[], scores=[], query=query, total_found=0, search_time_ms=0.5
        )

    async def forget(
        self,
        memory_id: str,
        access_context: MemoryAccessContext,
        soft_delete: bool = True,
    ) -> MemoryDeleteResult:
        return MemoryDeleteResult(
            memory_id=memory_id,
            success=True,
            soft_deleted=soft_delete,
            deleted_count=1,
        )

    async def consolidate(
        self, session_id: str, access_context: MemoryAccessContext
    ) -> list[MemoryWriteResult]:
        return []


def test_memory_manager_interface_adherence() -> None:
    """Verify concrete subclass adheres cleanly to IMemoryManager interface contract with access context."""

    async def _test() -> None:
        manager: IMemoryManager = MockMemoryManager()
        owner_ctx = MemoryAccessContext(user_id="user_123")
        unauth_ctx = MemoryAccessContext(user_id="user_999")

        item = MemoryItem(
            memory_type=MemoryType.WORKING,
            scope=MemoryScope.USER,
            user_id="user_123",
            content="Test memory content",
            metadata=MemoryMetadata(privacy=MemoryPrivacy.PRIVATE),
        )

        # Unauthorized write attempt
        write_denied = await manager.remember(item, unauth_ctx)
        assert write_denied.success is False
        assert write_denied.error_message == "Access Denied"

        # Authorized write attempt
        write_ok = await manager.remember(item, owner_ctx)
        assert write_ok.success is True

        # Search recall
        query = MemoryQuery(query_text="Test")
        search_res = await manager.recall(query, owner_ctx)
        assert search_res.total_found == 0

        # Deletion (soft delete vs hard delete)
        soft_del = await manager.forget(item.id, owner_ctx, soft_delete=True)
        assert soft_del.success is True
        assert soft_del.soft_deleted is True

        hard_del = await manager.forget(item.id, owner_ctx, soft_delete=False)
        assert hard_del.success is True
        assert hard_del.soft_deleted is False

        consolidated = await manager.consolidate("sess_123", owner_ctx)
        assert consolidated == []

    asyncio.run(_test())
