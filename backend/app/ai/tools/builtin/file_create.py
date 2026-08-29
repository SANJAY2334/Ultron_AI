"""Controlled File Creation Tool.

Creates new text files in allowed workspace directories governed by PathPolicy.
Capability: 'file:create'
Classification: MUTATING
"""

import os
import time
from datetime import UTC, datetime
from typing import Any

from app.ai.tools.base import BaseTool, ExecutionContext, ToolMetadata, ToolResult
from app.security.path_policy import PathPolicy, path_policy


class FileCreateTool(BaseTool):
    """Tool for creating new files safely in workspace directories.

    Security Boundary:
    - Capability Required: 'file:create'
    - Traversal & Secrets Defense: Enforced via PathPolicy.
    """

    def __init__(self, policy: PathPolicy = path_policy) -> None:
        """Initializes FileCreateTool.

        Args:
            policy: PathPolicy instance.
        """
        self._policy = policy

    @property
    def metadata(self) -> ToolMetadata:
        """Returns ToolMetadata declaration for FileCreateTool."""
        return ToolMetadata(
            name="create_file",
            description="Creates a new text file with content in an allowed workspace directory.",
            version="1.0.0",
            category="file",
            timeout_seconds=5.0,
            supports_streaming=False,
            destructive=False,
            confirmation_required=False,
            required_capabilities=["file:create"],
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Target file path to create.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Text content to write to file.",
                        "default": "",
                    },
                    "overwrite": {
                        "type": "boolean",
                        "description": "If True, permits overwriting an existing file.",
                        "default": False,
                    },
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            output_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "bytes_written": {"type": "integer"},
                },
            },
        )

    async def run(self, arguments: dict[str, Any], context: ExecutionContext) -> ToolResult:
        """Executes file creation."""
        start_time = time.perf_counter()
        target_path = arguments.get("path", "")
        content = arguments.get("content", "")
        overwrite = arguments.get("overwrite", False)

        val_res = self._policy.validate_path(target_path, check_exists=False)
        if not val_res.is_valid:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return ToolResult(
                tool_name=self.metadata.name,
                success=False,
                output=None,
                error_message=val_res.reason,
                execution_time_ms=elapsed_ms,
                metadata={"timestamp": datetime.now(UTC).isoformat()},
            )

        canon_path = val_res.canonical_path
        assert canon_path is not None

        if os.path.exists(canon_path) and not overwrite:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return ToolResult(
                tool_name=self.metadata.name,
                success=False,
                output=None,
                error_message=f"File Creation Denied: File '{canon_path}' already exists and overwrite flag is False.",
                execution_time_ms=elapsed_ms,
                metadata={"timestamp": datetime.now(UTC).isoformat()},
            )

        try:
            os.makedirs(os.path.dirname(canon_path), exist_ok=True)
            with open(canon_path, "w", encoding="utf-8") as f:
                f.write(content)

            bytes_written = len(content.encode("utf-8"))
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            return ToolResult(
                tool_name=self.metadata.name,
                success=True,
                output={
                    "path": canon_path,
                    "bytes_written": bytes_written,
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
                error_message=f"Failed to create file: {exc}",
                execution_time_ms=elapsed_ms,
                metadata={"timestamp": datetime.now(UTC).isoformat()},
            )
