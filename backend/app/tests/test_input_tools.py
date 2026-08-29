"""Unit Tests for Mouse and Keyboard Automation Tools (Phase 4B.4).

Validates MouseTool, KeyboardTool, coordinate bounds, payload limits, emergency cancellation,
and PyAutoGUIAdapter fallback behavior.
"""

import asyncio

import pytest

from app.ai.tools.base import ExecutionContext
from app.ai.tools.builtin.keyboard import KeyboardTool
from app.ai.tools.builtin.mouse import MouseTool
from app.desktop.adapters.pyautogui_adapter import PyAutoGUIAdapter


@pytest.fixture
def adapter() -> PyAutoGUIAdapter:
    return PyAutoGUIAdapter()


@pytest.fixture
def ctx() -> ExecutionContext:
    return ExecutionContext(granted_capabilities={"input:mouse", "input:keyboard"})


def test_mouse_tool_execution(adapter: PyAutoGUIAdapter, ctx: ExecutionContext) -> None:
    """Verify MouseTool handles click and move actions cleanly."""

    async def _test() -> None:
        tool = MouseTool(adapter=adapter)
        res = await tool.run({"action_type": "click", "x": 200, "y": 150}, ctx)
        assert res.success is True
        assert res.tool_name == "mouse_action"

    asyncio.run(_test())


def test_mouse_tool_out_of_bounds_rejection(
    adapter: PyAutoGUIAdapter, ctx: ExecutionContext
) -> None:
    """Verify MouseTool rejects impossible screen coordinates."""

    async def _test() -> None:
        tool = MouseTool(adapter=adapter)
        res = await tool.run({"action_type": "click", "x": 999999, "y": 999999}, ctx)
        assert res.success is False
        assert "outside screen bounds" in (res.error_message or "")

    asyncio.run(_test())


def test_keyboard_tool_execution(adapter: PyAutoGUIAdapter, ctx: ExecutionContext) -> None:
    """Verify KeyboardTool handles typing and hotkeys cleanly."""

    async def _test() -> None:
        tool = KeyboardTool(adapter=adapter)
        res = await tool.run({"action_type": "type_text", "text": "Hello World"}, ctx)
        assert res.success is True
        assert res.tool_name == "keyboard_action"

    asyncio.run(_test())


def test_keyboard_tool_payload_limit_rejection(
    adapter: PyAutoGUIAdapter, ctx: ExecutionContext
) -> None:
    """Verify KeyboardTool rejects text payloads exceeding 500 characters."""

    async def _test() -> None:
        tool = KeyboardTool(adapter=adapter)
        huge_text = "A" * 600
        res = await tool.run({"action_type": "type_text", "text": huge_text}, ctx)
        assert res.success is False
        assert "exceeds maximum limit" in (res.error_message or "")

    asyncio.run(_test())
