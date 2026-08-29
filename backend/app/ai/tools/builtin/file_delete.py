"""Controlled File Deletion Tool.

Deletes specified workspace files under strict PathPolicy and DESTRUCTIVE confirmation gating.
Capability: 'file:delete'
Classification: DESTRUCTIVE
Confirmation Required: True
"""

import os
import time
from datetime import UTC, datetime
from typing import Any

from app.ai.tools.base import BaseTool, ExecutionContext, ToolMetadata, ToolResult
from app.security.path_policy import PathPolicy, path_policy


class FileDeleteTool(BaseTool):
    """Tool for deleting workspace files safely under destructive policy gating.

    Security Boundary:
    - Capability Required: 'file:delete'
    - Destructive Gating: Marked destructive=True, confirmation_required=True.
    - Traversal & Secrets Defense: Enforced via PathPolicy.
    """

    def __init__(self, policy: PathPolicy = path_policy) -> None:
        """Initializes FileDeleteTool.

        Args:
            policy: PathPolicy instance.
        """
        self._policy = policy

    @property
    def metadata(self) -> ToolMetadata:
        """Returns ToolMetadata declaration for FileDeleteTool."""
        return ToolMetadata(
            name="delete_file",
            description="Deletes a specified workspace file. Marked DESTRUCTIVE and requires explicit confirmation.",
            version="1.0.0",
            category="file",
            timeout_seconds=5.0,
            supports_streaming=False,
            destructive=True,
            confirmation_required=True,
            required_capabilities=["file:delete"],
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Target workspace file path to delete.",
                    }
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            output_schema={
                "type": "object",
                "properties": {
                    "deleted": {"type": "boolean"},
                    "path": {"type": "string"},
                },
            },
        )

    async def run(self, arguments: dict[str, Any], context: ExecutionContext) -> ToolResult:
        """Executes file deletion."""
        start_time = time.perf_counter()
        target_path = arguments.get("path", "")

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
            if os.path.isfile(canon_path):
                os.remove(canon_path)
            elif os.path.isdir(canon_path):
                os.rmdir(canon_path)

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            return ToolResult(
                tool_name=self.metadata.name,
                success=True,
                output={
                    "deleted": True,
                    "path": canon_path,
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
                error_message=f"Failed to delete file: {exc}",
                execution_time_ms=elapsed_ms,
                metadata={"timestamp": datetime.now(UTC).isoformat()},
            )
