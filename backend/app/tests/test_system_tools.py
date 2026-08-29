"""Unit and Integration Tests for System Awareness Tools (Phase 4A).

Validates GetSystemInfoTool, GetSystemResourcesTool, GetProcessInfoTool, ToolRegistry discovery,
ToolExecutor pipeline execution, PolicyEngine capability authorization, capability denial,
read-only non-destructive guarantees, secrets redaction, and structured ToolResult output contracts.
"""

import asyncio
import os

import pytest

from app.ai.tools.base import ExecutionContext, ToolResult
from app.ai.tools.builtin import register_builtin_system_tools
from app.ai.tools.builtin.process_info import GetProcessInfoTool
from app.ai.tools.builtin.system_info import GetSystemInfoTool
from app.ai.tools.builtin.system_resources import GetSystemResourcesTool
from app.ai.tools.executor import ToolExecutor
from app.ai.tools.registry import ToolRegistry
from app.security.policy import PolicyEngine


@pytest.fixture
def registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(GetSystemInfoTool())
    reg.register(GetSystemResourcesTool())
    reg.register(GetProcessInfoTool())
    return reg


@pytest.fixture
def policy_engine() -> PolicyEngine:
    return PolicyEngine()


@pytest.fixture
def executor(registry: ToolRegistry, policy_engine: PolicyEngine) -> ToolExecutor:
    return ToolExecutor(registry=registry, policy=policy_engine)


def test_get_system_info_tool_execution(registry: ToolRegistry) -> None:
    """Verify GetSystemInfoTool returns structured, non-sensitive OS and runtime metadata."""

    async def _test() -> None:
        tool = registry.find("get_system_info")
        assert tool is not None

        ctx = ExecutionContext(
            correlation_id="corr_sys_info_123",
            granted_capabilities={"system:read"},
        )
        result = await tool.run({}, ctx)

        assert isinstance(result, ToolResult)
        assert result.success is True
        assert result.tool_name == "get_system_info"
        assert isinstance(result.output, dict)

        data = result.output
        assert "operating_system" in data
        assert "os_version" in data
        assert "architecture" in data
        assert "python_version" in data
        assert data["application_name"] == "ULTRON Autonomous Engine"
        assert data["application_version"] == "2.0.0"

        # Verify secrets redaction (no environment variables exposed)
        for key in os.environ:
            if len(key) > 4 and key in data:
                pytest.fail(f"Environment variable '{key}' was leaked in system info output!")

    asyncio.run(_test())


def test_get_system_resources_tool_execution(registry: ToolRegistry) -> None:
    """Verify GetSystemResourcesTool inspects CPU, RAM, and Disk without executing shell commands."""

    async def _test() -> None:
        tool = registry.find("get_system_resources")
        assert tool is not None

        ctx = ExecutionContext(
            correlation_id="corr_sys_res_456",
            granted_capabilities={"system:read"},
        )
        result = await tool.run({}, ctx)

        assert isinstance(result, ToolResult)
        assert result.success is True
        assert isinstance(result.output, dict)

        output = result.output
        assert "cpu" in output
        assert "memory" in output
        assert "disk" in output

        cpu = output["cpu"]
        assert cpu["logical_count"] >= 1
        assert cpu["physical_count"] >= 1
        assert 0.0 <= cpu["utilization_percent"] <= 100.0

        mem = output["memory"]
        assert mem["total_bytes"] > 0
        assert mem["available_bytes"] > 0
        assert 0.0 <= mem["percentage"] <= 100.0

        disk = output["disk"]
        assert disk["total_bytes"] > 0
        assert disk["free_bytes"] > 0

    asyncio.run(_test())


def test_get_process_info_tool_execution(registry: ToolRegistry) -> None:
    """Verify GetProcessInfoTool lists active processes and supports PID filtering."""

    async def _test() -> None:
        tool = registry.find("get_process_info")
        assert tool is not None

        ctx = ExecutionContext(
            correlation_id="corr_proc_789",
            granted_capabilities={"system:process_read"},
        )

        # 1. Process listing
        res_list = await tool.run({"limit": 5}, ctx)
        assert res_list.success is True
        data_list = res_list.output
        assert "processes" in data_list
        assert len(data_list["processes"]) <= 5
        assert data_list["total_count"] == len(data_list["processes"])

        if data_list["processes"]:
            first_proc = data_list["processes"][0]
            assert "pid" in first_proc
            assert "name" in first_proc
            assert "status" in first_proc

            # 2. Inspect specific PID
            pid = first_proc["pid"]
            res_pid = await tool.run({"pid": pid}, ctx)
            assert res_pid.success is True
            assert res_pid.output["total_count"] == 1
            assert res_pid.output["processes"][0]["pid"] == pid

    asyncio.run(_test())


def test_tool_executor_pipeline_authorization(
    executor: ToolExecutor, registry: ToolRegistry
) -> None:
    """Verify ToolExecutor enforces PolicyEngine authorization and executes tools cleanly."""

    async def _test() -> None:
        ctx_authorized = ExecutionContext(
            user_id="user_admin",
            correlation_id="corr_exec_001",
            granted_capabilities={"system:read", "system:process_read"},
        )

        # Authorized execution
        res_info = await executor.execute("get_system_info", {}, ctx_authorized)
        assert res_info.success is True
        assert res_info.output["operating_system"] is not None

        res_proc = await executor.execute("get_process_info", {"limit": 2}, ctx_authorized)
        assert res_proc.success is True

        # Unauthorized execution (missing capability)
        ctx_unauthorized = ExecutionContext(
            user_id="user_guest",
            correlation_id="corr_exec_002",
            granted_capabilities=set(),  # No capabilities!
        )

        res_denied = await executor.execute("get_system_info", {}, ctx_unauthorized)
        assert res_denied.success is False
        assert (
            "DENIED" in (res_denied.error_message or "").upper()
            or "CAPABILITY" in (res_denied.error_message or "").upper()
        )

    asyncio.run(_test())


def test_tool_non_destructive_guarantees(registry: ToolRegistry) -> None:
    """Verify system tools strictly declare non-destructive metadata and required capabilities."""
    for tool_name in ["get_system_info", "get_system_resources", "get_process_info"]:
        tool = registry.find(tool_name)
        assert tool is not None
        meta = tool.metadata
        assert meta.destructive is False
        assert meta.confirmation_required is False
        assert len(meta.required_capabilities) >= 1
        assert meta.category == "system"


def test_register_builtin_system_tools() -> None:
    """Verify register_builtin_system_tools helper populates global tool_registry."""
    test_reg = ToolRegistry()
    register_builtin_system_tools(test_reg)
    assert test_reg.find("get_system_info") is not None
    assert test_reg.find("get_system_resources") is not None
    assert test_reg.find("get_process_info") is not None
