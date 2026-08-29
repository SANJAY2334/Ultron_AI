"""Unit Tests for ToolRegistry and ToolExecutor Pipeline.

Validates tool registration, discovery, categories, policy-gated execution,
timeout enforcement, error handling, and telemetry collection.
"""

import asyncio
from typing import Any

from app.ai.tools.base import BaseTool, ExecutionContext, ToolMetadata, ToolResult
from app.ai.tools.executor import ToolExecutor
from app.ai.tools.registry import ToolRegistry


class SimpleAdderTool(BaseTool):
    """Adder tool for testing."""

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="adder",
            description="Adds numbers.",
            version="1.0.0",
            category="math",
            required_capabilities=["math:add"],
        )

    async def run(self, arguments: dict[str, Any], context: ExecutionContext) -> ToolResult:
        return ToolResult(
            tool_name="adder",
            success=True,
            output=arguments.get("x", 0) + arguments.get("y", 0),
        )


class SlowTool(BaseTool):
    """Tool that sleeps longer than its timeout."""

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="slow_tool",
            description="Sleeps forever.",
            timeout_seconds=0.1,
            required_capabilities=["system:sleep"],
        )

    async def run(self, arguments: dict[str, Any], context: ExecutionContext) -> ToolResult:
        await asyncio.sleep(0.5)
        return ToolResult(tool_name="slow_tool", success=True, output="Done")


def test_tool_registry_operations() -> None:
    """Verify ToolRegistry register, find, discover, categories, capabilities, health, unregister, reload."""
    reg = ToolRegistry()
    adder = SimpleAdderTool()
    reg.register(adder)

    assert reg.find("adder") is adder
    assert len(reg.discover()) == 1
    assert reg.categories() == ["math"]
    assert reg.capabilities() == ["math:add"]
    assert reg.health()["adder"] is True

    reg.unregister("adder")
    assert reg.find("adder") is None

    reg.register(adder)
    reg.reload()
    assert len(reg.discover()) == 0


def test_tool_executor_authorized_execution() -> None:
    """Verify ToolExecutor executes authorized tool and returns result + telemetry."""

    async def _test() -> None:
        reg = ToolRegistry()
        reg.register(SimpleAdderTool())
        executor = ToolExecutor(registry=reg)

        ctx = ExecutionContext(
            user_id="user_test",
            planner_id="planner_v1",
            correlation_id="corr_123",
            granted_capabilities={"math:add"},
        )

        res = await executor.execute("adder", {"x": 5, "y": 7}, ctx)
        assert res.success is True
        assert res.output == 12
        assert "telemetry" in res.metadata
        telemetry = res.metadata["telemetry"]
        assert telemetry["tool_name"] == "adder"
        assert telemetry["success"] is True
        assert telemetry["planner_id"] == "planner_v1"

    asyncio.run(_test())


def test_tool_executor_policy_denial() -> None:
    """Verify ToolExecutor rejects execution when policy engine denies access."""

    async def _test() -> None:
        reg = ToolRegistry()
        reg.register(SimpleAdderTool())
        executor = ToolExecutor(registry=reg)

        # Context lacks 'math:add' capability
        ctx = ExecutionContext(granted_capabilities=set())

        res = await executor.execute("adder", {"x": 5, "y": 7}, ctx)
        assert res.success is False
        assert res.error_message is not None and "Access Denied" in res.error_message
        assert res.metadata["policy_decision"] == "DENY"

    asyncio.run(_test())


def test_tool_executor_timeout_handling() -> None:
    """Verify ToolExecutor enforces tool timeout_seconds limit."""

    async def _test() -> None:
        reg = ToolRegistry()
        reg.register(SlowTool())
        executor = ToolExecutor(registry=reg)

        ctx = ExecutionContext(granted_capabilities={"system:sleep"})
        res = await executor.execute("slow_tool", {}, ctx)

        assert res.success is False
        assert res.error_message is not None and "timed out" in res.error_message.lower()
        telemetry = res.metadata["telemetry"]
        assert telemetry["exception_type"] == "TimeoutError"

    asyncio.run(_test())
