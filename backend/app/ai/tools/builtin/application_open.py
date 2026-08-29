"""Safe Application Opening Tool.

Launches validated desktop applications via IDesktopAdapter without arbitrary shell commands.
Capability: 'desktop:window_open'
Classification: MUTATING
"""

import os
import time
from datetime import UTC, datetime
from typing import Any

from app.ai.tools.base import BaseTool, ExecutionContext, ToolMetadata, ToolResult
from app.desktop.adapters.pyautogui_adapter import PyAutoGUIAdapter
from app.desktop.base import IDesktopAdapter

# Whitelist of standard desktop application names allowed for launching
SAFE_APP_WHITELIST = {
    "calc.exe",
    "calc",
    "notepad.exe",
    "notepad",
    "explorer.exe",
    "explorer",
    "mspaint.exe",
    "mspaint",
    "code.exe",
    "code",
}


class ApplicationOpenTool(BaseTool):
    """Tool for launching validated desktop applications safely.

    Security Boundary:
    - Capability Required: 'desktop:window_open'
    - Application Validation: Only launches whitelisted app names or verified absolute executable paths.
    - No Shell Commands: Launches process directly via OS process API.
    """

    def __init__(self, adapter: IDesktopAdapter | None = None) -> None:
        """Initializes ApplicationOpenTool.

        Args:
            adapter: Optional IDesktopAdapter instance (defaults to PyAutoGUIAdapter).
        """
        self._adapter = adapter or PyAutoGUIAdapter()

    @property
    def metadata(self) -> ToolMetadata:
        """Returns ToolMetadata declaration for ApplicationOpenTool."""
        return ToolMetadata(
            name="open_application",
            description="Opens a validated desktop application safely.",
            version="1.0.0",
            category="desktop",
            timeout_seconds=10.0,
            supports_streaming=False,
            destructive=False,
            confirmation_required=False,
            required_capabilities=["desktop:window_open"],
            input_schema={
                "type": "object",
                "properties": {
                    "app_path": {
                        "type": "string",
                        "description": "Validated application executable name or absolute binary path.",
                    },
                    "arguments": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional command line argument list.",
                        "default": [],
                    },
                },
                "required": ["app_path"],
                "additionalProperties": False,
            },
            output_schema={
                "type": "object",
                "properties": {
                    "app_name": {"type": "string"},
                    "pid": {"type": ["integer", "null"]},
                    "status": {"type": "string"},
                },
            },
        )

    def _validate_app_target(self, app_path: str) -> bool:
        """Validates that application target is whitelisted or is an existing absolute executable file."""
        clean_app = os.path.basename(app_path).lower().strip()
        if clean_app in SAFE_APP_WHITELIST or app_path.lower().strip() in SAFE_APP_WHITELIST:
            return True

        if os.path.isabs(app_path) and os.path.isfile(app_path) and os.access(app_path, os.X_OK):
            return True

        return False

    async def run(self, arguments: dict[str, Any], context: ExecutionContext) -> ToolResult:
        """Executes application launch."""
        start_time = time.perf_counter()
        app_path = arguments.get("app_path", "").strip()
        args = arguments.get("arguments", [])

        if not app_path or not self._validate_app_target(app_path):
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return ToolResult(
                tool_name=self.metadata.name,
                success=False,
                output=None,
                error_message=(
                    f"Application Open Denied: Target '{app_path}' is not in allowed "
                    f"application whitelist and is not a valid executable file."
                ),
                execution_time_ms=elapsed_ms,
                metadata={"timestamp": datetime.now(UTC).isoformat()},
            )

        try:
            op_result = await self._adapter.open_application(app_path, args)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            return ToolResult(
                tool_name=self.metadata.name,
                success=True,
                output=op_result.model_dump(),
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
                error_message=f"Failed to open application safely: {exc}",
                execution_time_ms=elapsed_ms,
                metadata={"timestamp": datetime.now(UTC).isoformat()},
            )
