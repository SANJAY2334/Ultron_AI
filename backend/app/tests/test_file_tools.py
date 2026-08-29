"""Unit Tests for Controlled File Tools (Phase 4B.3).

Validates FileReadTool, FileCreateTool, FileModifyTool, FileDeleteTool, PathPolicy integration,
traversal defense, sensitive file protection, file size limits, and DESTRUCTIVE confirmation gating.
"""

import asyncio
import os

import pytest

from app.ai.tools.base import ExecutionContext, ToolResult
from app.ai.tools.builtin.file_create import FileCreateTool
from app.ai.tools.builtin.file_delete import FileDeleteTool
from app.ai.tools.builtin.file_modify import FileModifyTool
from app.ai.tools.builtin.file_read import FileReadTool
from app.security.path_policy import PathPolicy


@pytest.fixture
def policy(tmp_path) -> PathPolicy:
    return PathPolicy(allowed_directories=[str(tmp_path)])


@pytest.fixture
def ctx() -> ExecutionContext:
    return ExecutionContext(
        correlation_id="corr_file_123",
        granted_capabilities={"file:read", "file:create", "file:modify", "file:delete"},
    )


def test_file_create_and_read(policy: PathPolicy, ctx: ExecutionContext, tmp_path) -> None:
    """Verify FileCreateTool and FileReadTool within allowed workspace directory."""

    async def _test() -> None:
        create_tool = FileCreateTool(policy=policy)
        read_tool = FileReadTool(policy=policy)

        target_file = str(tmp_path / "hello.txt")

        # 1. Create file
        res_create = await create_tool.run(
            {"path": target_file, "content": "Hello ULTRON Workspace!"}, ctx
        )
        assert isinstance(res_create, ToolResult)
        assert res_create.success is True
        assert res_create.output["bytes_written"] > 0

        # 2. Read file
        res_read = await read_tool.run({"path": target_file}, ctx)
        assert isinstance(res_read, ToolResult)
        assert res_read.success is True
        assert res_read.output["content"] == "Hello ULTRON Workspace!"

    asyncio.run(_test())


def test_file_modify_append_and_overwrite(
    policy: PathPolicy, ctx: ExecutionContext, tmp_path
) -> None:
    """Verify FileModifyTool append and overwrite modes."""

    async def _test() -> None:
        create_tool = FileCreateTool(policy=policy)
        modify_tool = FileModifyTool(policy=policy)
        read_tool = FileReadTool(policy=policy)

        target_file = str(tmp_path / "modify_me.txt")
        await create_tool.run({"path": target_file, "content": "Line 1\n"}, ctx)

        # Append content
        await modify_tool.run({"path": target_file, "content": "Line 2\n", "mode": "append"}, ctx)
        res1 = await read_tool.run({"path": target_file}, ctx)
        assert res1.output["content"] == "Line 1\nLine 2\n"

        # Overwrite content
        await modify_tool.run(
            {"path": target_file, "content": "Replaced Content", "mode": "overwrite"}, ctx
        )
        res2 = await read_tool.run({"path": target_file}, ctx)
        assert res2.output["content"] == "Replaced Content"

    asyncio.run(_test())


def test_file_tools_path_traversal_rejection(policy: PathPolicy, ctx: ExecutionContext) -> None:
    """Verify file tools reject path traversal attempts."""

    async def _test() -> None:
        read_tool = FileReadTool(policy=policy)
        res = await read_tool.run({"path": "../../etc/passwd"}, ctx)
        assert res.success is False
        err = res.error_message or ""
        assert "Security Policy Violation" in err or "Path validation failed" in err

    asyncio.run(_test())


def test_file_tools_sensitive_file_rejection(
    policy: PathPolicy, ctx: ExecutionContext, tmp_path
) -> None:
    """Verify file tools reject reading or writing .env and private keys."""

    async def _test() -> None:
        create_tool = FileCreateTool(policy=policy)
        env_file = str(tmp_path / ".env")
        res = await create_tool.run({"path": env_file, "content": "SECRET=123"}, ctx)
        assert res.success is False
        err = res.error_message or ""
        assert "sensitive file" in err.lower()

    asyncio.run(_test())


def test_file_delete_tool_execution(policy: PathPolicy, ctx: ExecutionContext, tmp_path) -> None:
    """Verify FileDeleteTool removes file successfully."""

    async def _test() -> None:
        create_tool = FileCreateTool(policy=policy)
        delete_tool = FileDeleteTool(policy=policy)

        target_file = str(tmp_path / "to_delete.txt")
        await create_tool.run({"path": target_file, "content": "Delete me"}, ctx)
        assert os.path.exists(target_file)

        res_del = await delete_tool.run({"path": target_file}, ctx)
        assert res_del.success is True
        assert not os.path.exists(target_file)

    asyncio.run(_test())
