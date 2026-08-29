"""Unit Tests for Base Tool Interface and Capability Models.

Validates Capability, ExecutionContext, expanded ToolMetadata, ToolResult,
and concrete BaseTool subclass implementation contracts.
"""

import asyncio
from typing import Any

from app.ai.tools.base import (
    BaseTool,
    Capability,
    ExecutionContext,
    ToolMetadata,
    ToolResult,
)


class DummyCalculatorTool(BaseTool):
    """Concrete dummy tool for interface testing."""

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="calculator",
            description="Adds two numbers.",
            version="1.1.0",
            category="math",
            timeout_seconds=5.0,
            required_capabilities=["math:basic"],
            input_schema={
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["a", "b"],
            },
        )

    async def run(self, arguments: dict[str, Any], context: ExecutionContext) -> ToolResult:
        a = arguments.get("a", 0)
        b = arguments.get("b", 0)
        return ToolResult(
            tool_name=self.metadata.name,
            success=True,
            output={"result": a + b},
            execution_time_ms=1.5,
        )


def test_capability_model() -> None:
    """Verify Capability model fields."""
    cap = Capability(
        name="file:write",
        description="Allows writing to local filesystem.",
        category="filesystem",
        risk_level="high",
    )
    assert cap.name == "file:write"
    assert cap.risk_level == "high"


def test_execution_context_model() -> None:
    """Verify ExecutionContext fields and capability set."""
    ctx = ExecutionContext(
        user_id="user_123",
        correlation_id="corr_999",
        granted_capabilities={"file:write", "math:basic"},
    )
    assert ctx.user_id == "user_123"
    assert "file:write" in ctx.granted_capabilities


def test_tool_metadata_expanded_fields() -> None:
    """Verify expanded ToolMetadata fields."""
    meta = ToolMetadata(
        name="system_reboot",
        description="Reboots system kernel.",
        destructive=True,
        confirmation_required=True,
        required_capabilities=["system:admin"],
    )
    assert meta.destructive is True
    assert meta.confirmation_required is True
    assert meta.required_capabilities == ["system:admin"]


def test_concrete_tool_execution() -> None:
    """Verify concrete BaseTool execution via run()."""

    async def _test() -> None:
        tool = DummyCalculatorTool()
        assert tool.metadata.name == "calculator"
        assert tool.metadata.version == "1.1.0"

        ctx = ExecutionContext(granted_capabilities={"math:basic"})
        res = await tool.run({"a": 10, "b": 20}, ctx)

        assert res.tool_name == "calculator"
        assert res.success is True
        assert res.output["result"] == 30

    asyncio.run(_test())
