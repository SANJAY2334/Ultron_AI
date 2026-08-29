"""Controlled File Modification Tool.

Modifies existing text files in allowed workspace directories governed by PathPolicy.
Capability: 'file:modify'
Classification: MUTATING
"""

import time
from datetime import UTC, datetime
from typing import Any

from app.ai.tools.base import BaseTool, ExecutionContext, ToolMetadata, ToolResult
from app.security.path_policy import PathPolicy, path_policy


class FileModifyTool(BaseTool):
    """Tool for modifying existing text files safely.

    Security Boundary:
    - Capability Required: 'file:modify'
    - Traversal & Secrets Defense: Enforced via PathPolicy.
    """

    def __init__(self, policy: PathPolicy = path_policy) -> None:
        """Initializes FileModifyTool.

        Args:
            policy: PathPolicy instance.
        """
        self._policy = policy

    @property
    def metadata(self) -> ToolMetadata:
        """Returns ToolMetadata declaration for FileModifyTool."""
        return ToolMetadata(
            name="modify_file",
            description="Appends or replaces content in an existing workspace file.",
            version="1.0.0",
            category="file",
            timeout_seconds=5.0,
            supports_streaming=False,
            destructive=False,
            confirmation_required=False,
            required_capabilities=["file:modify"],
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Target workspace file path to modify.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content string to write or append.",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["append", "overwrite"],
                        "default": "append",
                        "description": "Modification mode ('append' or 'overwrite').",
                    },
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
            output_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "bytes_modified": {"type": "integer"},
                    "mode": {"type": "string"},
                },
            },
        )

    async def run(self, arguments: dict[str, Any], context: ExecutionContext) -> ToolResult:
        """Executes file modification."""
        start_time = time.perf_counter()
        target_path = arguments.get("path", "")
        content = arguments.get("content", "")
        mode = arguments.get("mode", "append")

        val_res = self._policy.validate_path(target_path, check_exists=True)
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

        try:
            write_mode = "a" if mode == "append" else "w"
            with open(canon_path, write_mode, encoding="utf-8") as f:
                f.write(content)

            bytes_modified = len(content.encode("utf-8"))
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            return ToolResult(
                tool_name=self.metadata.name,
                success=True,
                output={
                    "path": canon_path,
                    "bytes_modified": bytes_modified,
                    "mode": mode,
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
                error_message=f"Failed to modify file: {exc}",
                execution_time_ms=elapsed_ms,
                metadata={"timestamp": datetime.now(UTC).isoformat()},
            )
