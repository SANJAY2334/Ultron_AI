"""Controlled Mouse Automation Tool.

Executes bounded mouse operations (click, move, drag, scroll) via IDesktopAdapter.
Capability: 'input:mouse'
Classification: MUTATING
"""

import time
from datetime import UTC, datetime
from typing import Any

from app.ai.tools.base import BaseTool, ExecutionContext, ToolMetadata, ToolResult
from app.desktop.adapters.pyautogui_adapter import PyAutoGUIAdapter
from app.desktop.base import IDesktopAdapter
from app.desktop.models import MouseAction


class MouseTool(BaseTool):
    """Tool for controlled mouse interactions on the desktop.

    Security Boundary:
    - Capability Required: 'input:mouse'
    - Bounds Enforcement: Validates non-negative screen coordinates and bounded click counts.
    """

    def __init__(self, adapter: IDesktopAdapter | None = None) -> None:
        """Initializes MouseTool.

        Args:
            adapter: Optional IDesktopAdapter instance (defaults to PyAutoGUIAdapter).
        """
        self._adapter = adapter or PyAutoGUIAdapter()

    @property
    def metadata(self) -> ToolMetadata:
        """Returns ToolMetadata declaration for MouseTool."""
        return ToolMetadata(
            name="mouse_action",
            description="Executes a bounded mouse movement or click operation.",
            version="1.0.0",
            category="input",
            timeout_seconds=5.0,
            supports_streaming=False,
            destructive=False,
            confirmation_required=False,
            required_capabilities=["input:mouse"],
            input_schema={
                "type": "object",
                "properties": {
                    "action_type": {
                        "type": "string",
                        "enum": ["move", "click", "double_click", "right_click", "drag", "scroll"],
                        "description": "Mouse operation type.",
                    },
                    "x": {"type": "integer", "minimum": 0, "description": "Target X coordinate."},
                    "y": {"type": "integer", "minimum": 0, "description": "Target Y coordinate."},
                    "button": {
                        "type": "string",
                        "enum": ["left", "right", "middle"],
                        "default": "left",
                    },
                    "clicks": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "default": 1,
                    },
                },
                "required": ["action_type", "x", "y"],
                "additionalProperties": False,
            },
            output_schema={
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "action_id": {"type": "string"},
                },
            },
        )

    async def run(self, arguments: dict[str, Any], context: ExecutionContext) -> ToolResult:
        """Executes mouse operation."""
        start_time = time.perf_counter()

        try:
            action = MouseAction(
                action_type=arguments.get("action_type", "click"),
                x=arguments.get("x", 0),
                y=arguments.get("y", 0),
                button=arguments.get("button", "left"),
                clicks=arguments.get("clicks", 1),
            )

            res = await self._adapter.execute_mouse(action)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            return ToolResult(
                tool_name=self.metadata.name,
                success=res.success,
                output=res.output,
                error_message=res.error_message,
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
                error_message=f"Failed to execute mouse action: {exc}",
                execution_time_ms=elapsed_ms,
                metadata={"timestamp": datetime.now(UTC).isoformat()},
            )
