"""Unit Tests for Application and Window Tools (Phase 4B.2).

Validates GetWindowInfoTool, ApplicationOpenTool, ApplicationCloseTool, target validation,
critical process protection, and error sanitization.
"""

import asyncio

import pytest

from app.ai.tools.base import ExecutionContext
from app.ai.tools.builtin.application_close import ApplicationCloseTool
from app.ai.tools.builtin.application_open import ApplicationOpenTool
from app.ai.tools.builtin.window_info import GetWindowInfoTool
from app.desktop.base import IDesktopAdapter
from app.desktop.models import (
    ActionResult,
    KeyboardAction,
    MouseAction,
    ProcessOperation,
    WindowInfo,
)


class MockDesktopAdapter(IDesktopAdapter):
    """Mock desktop adapter for testing window and application tools."""

    def __init__(self) -> None:
        self.windows_list = [
            WindowInfo(
                window_id="win_1",
                title="Calculator",
                process_name="calc.exe",
                pid=101,
                is_active=True,
            ),
            WindowInfo(
                window_id="win_2",
                title="Notepad",
                process_name="notepad.exe",
                pid=202,
                is_active=False,
            ),
        ]
        self.opened_apps: list[str] = []
        self.closed_apps: list[str | int] = []

    async def get_windows(self) -> list[WindowInfo]:
        return self.windows_list

    async def open_application(
        self, app_path: str, args: list[str] | None = None
    ) -> ProcessOperation:
        self.opened_apps.append(app_path)
        return ProcessOperation(
            operation_type="open",
            app_name=app_path,
            app_path=app_path,
            pid=999,
            arguments=args or [],
        )

    async def close_application(self, pid: int | None = None, app_name: str | None = None) -> bool:
        if pid:
            self.closed_apps.append(pid)
        elif app_name:
            self.closed_apps.append(app_name)
        return True

    async def execute_mouse(self, action: MouseAction) -> ActionResult:
        return ActionResult(action_id="m_1", success=True)

    async def execute_keyboard(self, action: KeyboardAction) -> ActionResult:
        return ActionResult(action_id="k_1", success=True)


@pytest.fixture
def mock_adapter() -> MockDesktopAdapter:
    return MockDesktopAdapter()


@pytest.fixture
def ctx() -> ExecutionContext:
    return ExecutionContext(
        granted_capabilities={"desktop:window_read", "desktop:window_open", "desktop:window_close"}
    )


def test_get_window_info_tool(mock_adapter: MockDesktopAdapter, ctx: ExecutionContext) -> None:
    """Verify GetWindowInfoTool returns active window listing."""

    async def _test() -> None:
        tool = GetWindowInfoTool(adapter=mock_adapter)
        res = await tool.run({}, ctx)
        assert res.success is True
        assert res.output["total_count"] == 2
        assert res.output["windows"][0]["title"] == "Calculator"

    asyncio.run(_test())


def test_application_open_tool_whitelisted(
    mock_adapter: MockDesktopAdapter, ctx: ExecutionContext
) -> None:
    """Verify ApplicationOpenTool permits launching whitelisted application."""

    async def _test() -> None:
        tool = ApplicationOpenTool(adapter=mock_adapter)
        res = await tool.run({"app_path": "calc.exe"}, ctx)
        assert res.success is True
        assert res.output["pid"] == 999
        assert "calc.exe" in mock_adapter.opened_apps

    asyncio.run(_test())


def test_application_open_tool_unauthorized_rejection(
    mock_adapter: MockDesktopAdapter, ctx: ExecutionContext
) -> None:
    """Verify ApplicationOpenTool rejects arbitrary non-whitelisted commands."""

    async def _test() -> None:
        tool = ApplicationOpenTool(adapter=mock_adapter)
        res = await tool.run({"app_path": "malicious_script.bat"}, ctx)
        assert res.success is False
        assert "Application Open Denied" in (res.error_message or "")

    asyncio.run(_test())


def test_application_close_critical_process_protection(
    mock_adapter: MockDesktopAdapter, ctx: ExecutionContext
) -> None:
    """Verify ApplicationCloseTool rejects closing critical system processes."""

    async def _test() -> None:
        tool = ApplicationCloseTool(adapter=mock_adapter)
        res = await tool.run({"app_name": "svchost.exe"}, ctx)
        assert res.success is False
        assert "Cannot terminate critical system process" in (res.error_message or "")

    asyncio.run(_test())
