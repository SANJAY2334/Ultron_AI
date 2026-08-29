"""Controlled File Reading Tool.

Reads text content from safe workspace files governed by PathPolicy.
Capability: 'file:read'
Classification: READ_ONLY
"""

import time
from datetime import UTC, datetime
from typing import Any

from app.ai.tools.base import BaseTool, ExecutionContext, ToolMetadata, ToolResult
from app.security.path_policy import PathPolicy, path_policy


class FileReadTool(BaseTool):
    """Tool for reading text file contents safely.

    Security Boundary:
    - Capability Required: 'file:read'
    - Traversal & Secrets Defense: Enforced via PathPolicy.
    - Size Bound: Rejects files exceeding size limits.
    """

    def __init__(self, policy: PathPolicy = path_policy) -> None:
        """Initializes FileReadTool.

        Args:
            policy: PathPolicy instance.
        """
        self._policy = policy

    @property
    def metadata(self) -> ToolMetadata:
        """Returns ToolMetadata declaration for FileReadTool."""
        return ToolMetadata(
            name="read_file",
            description="Reads the text contents of a specified workspace file.",
            version="1.0.0",
            category="file",
            timeout_seconds=5.0,
            supports_streaming=False,
            destructive=False,
            confirmation_required=False,
            required_capabilities=["file:read"],
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Target workspace file path.",
                    },
                    "max_bytes": {
                        "type": ["integer", "null"],
                        "description": "Maximum bytes to read.",
                        "default": None,
                    },
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            output_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "bytes_read": {"type": "integer"},
                },
            },
        )

    async def run(self, arguments: dict[str, Any], context: ExecutionContext) -> ToolResult:
        """Executes file read."""
        start_time = time.perf_counter()
        target_path = arguments.get("path", "")
        max_bytes = arguments.get("max_bytes")

        val_res = self._policy.validate_path(target_path, check_exists=True, max_bytes=max_bytes)
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
            with open(canon_path, encoding="utf-8", errors="replace") as f:
                if max_bytes:
                    content = f.read(max_bytes)
                else:
                    content = f.read()

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            return ToolResult(
                tool_name=self.metadata.name,
                success=True,
                output={
                    "path": canon_path,
                    "content": content,
                    "bytes_read": len(content.encode("utf-8")),
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
                error_message=f"Failed to read file: {exc}",
                execution_time_ms=elapsed_ms,
                metadata={"timestamp": datetime.now(UTC).isoformat()},
            )
