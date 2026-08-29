"""Read-Only Window Information Tool.

Provides safe, read-only inspection of active desktop application windows using IDesktopAdapter.
Capability: 'desktop:window_read'
Classification: READ_ONLY
"""

import time
from datetime import UTC, datetime
from typing import Any

from app.ai.tools.base import BaseTool, ExecutionContext, ToolMetadata, ToolResult
from app.desktop.adapters.pyautogui_adapter import PyAutoGUIAdapter
from app.desktop.base import IDesktopAdapter


class GetWindowInfoTool(BaseTool):
    """Read-only tool providing active desktop window information.

    Security Boundary:
    - Capability Required: 'desktop:window_read'
    - Read-Only: True
    """

    def __init__(self, adapter: IDesktopAdapter | None = None) -> None:
        """Initializes GetWindowInfoTool.

        Args:
            adapter: Optional IDesktopAdapter instance (defaults to PyAutoGUIAdapter).
        """
        self._adapter = adapter or PyAutoGUIAdapter()

    @property
    def metadata(self) -> ToolMetadata:
        """Returns ToolMetadata declaration for GetWindowInfoTool."""
        return ToolMetadata(
            name="get_window_info",
            description="Retrieves a list of active desktop application windows and titles.",
            version="1.0.0",
            category="desktop",
            timeout_seconds=5.0,
            supports_streaming=False,
            destructive=False,
            confirmation_required=False,
            required_capabilities=["desktop:window_read"],
            input_schema={
                "type": "object",
                "properties": {
                    "filter_title": {
                        "type": ["string", "null"],
                        "description": "Optional window title filter string.",
                        "default": None,
                    }
                },
                "additionalProperties": False,
            },
            output_schema={
                "type": "object",
                "properties": {
                    "windows": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "window_id": {"type": "string"},
                                "title": {"type": "string"},
                                "process_name": {"type": "string"},
                                "pid": {"type": "integer"},
                                "is_active": {"type": "boolean"},
                            },
                        },
                    },
                    "total_count": {"type": "integer"},
                },
            },
        )

    async def run(self, arguments: dict[str, Any], context: ExecutionContext) -> ToolResult:
        """Executes window inspection."""
        start_time = time.perf_counter()
        filter_title = arguments.get("filter_title")
        if filter_title:
            filter_title = filter_title.lower().strip()

        try:
            windows = await self._adapter.get_windows()
            windows_data = []
            for win in windows:
                if filter_title and filter_title not in win.title.lower():
                    continue
                windows_data.append(win.model_dump())

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            return ToolResult(
                tool_name=self.metadata.name,
                success=True,
                output={
                    "windows": windows_data,
                    "total_count": len(windows_data),
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
                error_message=f"Failed to inspect desktop windows: {exc}",
                execution_time_ms=elapsed_ms,
                metadata={
                    "timestamp": datetime.now(UTC).isoformat(),
                    "correlation_id": context.correlation_id,
                },
            )
