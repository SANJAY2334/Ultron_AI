"""Controlled Application Closing Tool.

Closes a specific application window or PID cleanly via IDesktopAdapter.
Capability: 'desktop:window_close'
Classification: MUTATING / DESTRUCTIVE
Confirmation Required: True
"""

import time
from datetime import UTC, datetime
from typing import Any

from app.ai.tools.base import BaseTool, ExecutionContext, ToolMetadata, ToolResult
from app.desktop.adapters.pyautogui_adapter import PyAutoGUIAdapter
from app.desktop.base import IDesktopAdapter

# System critical processes that can NEVER be closed by ULTRON
CRITICAL_SYSTEM_PROCESSES = {
    "system",
    "smss.exe",
    "csrss.exe",
    "wininit.exe",
    "services.exe",
    "lsass.exe",
    "svchost.exe",
    "systemd",
    "init",
    "launchd",
}


class ApplicationCloseTool(BaseTool):
    """Tool for closing specific user desktop applications safely.

    Security Boundary:
    - Capability Required: 'desktop:window_close'
    - Destructive Gating: Marked confirmation_required = True.
    - Critical Process Protection: Never permits terminating core OS system processes.
    """

    def __init__(self, adapter: IDesktopAdapter | None = None) -> None:
        """Initializes ApplicationCloseTool.

        Args:
            adapter: Optional IDesktopAdapter instance (defaults to PyAutoGUIAdapter).
        """
        self._adapter = adapter or PyAutoGUIAdapter()

    @property
    def metadata(self) -> ToolMetadata:
        """Returns ToolMetadata declaration for ApplicationCloseTool."""
        return ToolMetadata(
            name="close_application",
            description="Closes a specific user application by PID or application name.",
            version="1.0.0",
            category="desktop",
            timeout_seconds=10.0,
            supports_streaming=False,
            destructive=True,
            confirmation_required=True,
            required_capabilities=["desktop:window_close"],
            input_schema={
                "type": "object",
                "properties": {
                    "pid": {
                        "type": ["integer", "null"],
                        "description": "Optional specific Process ID to close.",
                        "default": None,
                    },
                    "app_name": {
                        "type": ["string", "null"],
                        "description": "Optional application name to close.",
                        "default": None,
                    },
                },
                "additionalProperties": False,
            },
            output_schema={
                "type": "object",
                "properties": {
                    "closed": {"type": "boolean"},
                    "target": {"type": "string"},
                },
            },
        )

    async def run(self, arguments: dict[str, Any], context: ExecutionContext) -> ToolResult:
        """Executes application close."""
        start_time = time.perf_counter()
        pid = arguments.get("pid")
        app_name = arguments.get("app_name")

        if pid is None and not app_name:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return ToolResult(
                tool_name=self.metadata.name,
                success=False,
                output=None,
                error_message="Application Close Error: Must provide either 'pid' or 'app_name'.",
                execution_time_ms=elapsed_ms,
                metadata={"timestamp": datetime.now(UTC).isoformat()},
            )

        if app_name and app_name.lower().strip() in CRITICAL_SYSTEM_PROCESSES:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return ToolResult(
                tool_name=self.metadata.name,
                success=False,
                output=None,
                error_message=f"Application Close Denied: Cannot terminate critical system process '{app_name}'.",
                execution_time_ms=elapsed_ms,
                metadata={"timestamp": datetime.now(UTC).isoformat()},
            )

        try:
            success = await self._adapter.close_application(pid=pid, app_name=app_name)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            return ToolResult(
                tool_name=self.metadata.name,
                success=success,
                output={
                    "closed": success,
                    "target": f"PID={pid}" if pid else f"Name={app_name}",
                },
                execution_time_ms=elapsed_ms,
                metadata={
                    "timestamp": datetime.now(UTC).isoformat(),
                    "correlation_id": context.correlation_id,
                },
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return ToolResult(
                tool_name=self.metadata.name,
                success=False,
                output=None,
                error_message=f"Failed to close application safely: {exc}",
                execution_time_ms=elapsed_ms,
                metadata={"timestamp": datetime.now(UTC).isoformat()},
            )
