"""Controlled Keyboard Automation Tool.

Executes bounded keyboard operations (type_text, press_key, hotkey) via IDesktopAdapter.
Capability: 'input:keyboard'
Classification: MUTATING
"""

import time
from datetime import UTC, datetime
from typing import Any

from app.ai.tools.base import BaseTool, ExecutionContext, ToolMetadata, ToolResult
from app.desktop.adapters.pyautogui_adapter import PyAutoGUIAdapter
from app.desktop.base import IDesktopAdapter
from app.desktop.models import KeyboardAction


class KeyboardTool(BaseTool):
    """Tool for controlled keyboard interactions on the desktop.

    Security Boundary:
    - Capability Required: 'input:keyboard'
    - Bounds Enforcement: Enforces maximum text payload lengths (500 chars) and key counts (10 keys).
    """

    def __init__(self, adapter: IDesktopAdapter | None = None) -> None:
        """Initializes KeyboardTool.

        Args:
            adapter: Optional IDesktopAdapter instance (defaults to PyAutoGUIAdapter).
        """
        self._adapter = adapter or PyAutoGUIAdapter()

    @property
    def metadata(self) -> ToolMetadata:
        """Returns ToolMetadata declaration for KeyboardTool."""
        return ToolMetadata(
            name="keyboard_action",
            description="Executes a bounded keyboard typing or hotkey operation.",
            version="1.0.0",
            category="input",
            timeout_seconds=5.0,
            supports_streaming=False,
            destructive=False,
            confirmation_required=False,
            required_capabilities=["input:keyboard"],
            input_schema={
                "type": "object",
                "properties": {
                    "action_type": {
                        "type": "string",
                        "enum": ["type_text", "press_key", "hotkey"],
                        "description": "Keyboard operation type.",
                    },
                    "text": {
                        "type": ["string", "null"],
                        "maxLength": 500,
                        "description": "Text payload for typing.",
                        "default": None,
                    },
                    "keys": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 10,
                        "description": "Key names or hotkey key sequence.",
                        "default": [],
                    },
                },
                "required": ["action_type"],
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
        """Executes keyboard operation."""
        start_time = time.perf_counter()

        try:
            action = KeyboardAction(
                action_type=arguments.get("action_type", "type_text"),
                text=arguments.get("text"),
                keys=arguments.get("keys", []),
            )

            res = await self._adapter.execute_keyboard(action)
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
                error_message=f"Failed to execute keyboard action: {exc}",
                execution_time_ms=elapsed_ms,
                metadata={"timestamp": datetime.now(UTC).isoformat()},
            )
