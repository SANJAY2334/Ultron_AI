"""Read-Only System Information Tool.

Provides safe, non-sensitive operating system and runtime environment metadata.
Exposes OS name, OS version, architecture, CPU processor details, Python runtime version,
and application metadata while strictly redacting environment variables, secrets, and credentials.
"""

import platform
import sys
import time
from datetime import UTC, datetime
from typing import Any

from app.ai.tools.base import BaseTool, ExecutionContext, ToolMetadata, ToolResult


class GetSystemInfoTool(BaseTool):
    """Read-only tool providing operating system and runtime metadata.

    Security Boundary:
    - Capability Required: 'system:read'
    - Read-Only: True
    - Secrets Redaction: Environment variables, API keys, JWTs, and credentials are strictly excluded.
    """

    @property
    def metadata(self) -> ToolMetadata:
        """Returns the ToolMetadata declaration for GetSystemInfoTool."""
        return ToolMetadata(
            name="get_system_info",
            description="Retrieves safe, non-sensitive operating system and application runtime metadata.",
            version="1.0.0",
            category="system",
            timeout_seconds=5.0,
            supports_streaming=False,
            destructive=False,
            confirmation_required=False,
            required_capabilities=["system:read"],
            input_schema={
                "type": "object",
                "properties": {
                    "include_hostname": {
                        "type": "boolean",
                        "description": "Optional flag to include host network node name.",
                        "default": True,
                    }
                },
                "additionalProperties": False,
            },
            output_schema={
                "type": "object",
                "properties": {
                    "operating_system": {"type": "string"},
                    "os_version": {"type": "string"},
                    "os_release": {"type": "string"},
                    "architecture": {"type": "string"},
                    "processor": {"type": "string"},
                    "python_version": {"type": "string"},
                    "application_name": {"type": "string"},
                    "application_version": {"type": "string"},
                },
            },
        )

    async def run(self, arguments: dict[str, Any], context: ExecutionContext) -> ToolResult:
        """Executes system information retrieval."""
        start_time = time.perf_counter()

        include_hostname = arguments.get("include_hostname", True)

        try:
            os_name = platform.system()
            os_ver = platform.version()
            os_rel = platform.release()
            arch = platform.machine() or platform.architecture()[0]
            proc = platform.processor() or "Unknown Processor"
            py_ver = sys.version.split()[0]

            info_data = {
                "operating_system": os_name,
                "os_version": os_ver,
                "os_release": os_rel,
                "architecture": arch,
                "processor": proc,
                "python_version": py_ver,
                "application_name": "ULTRON Autonomous Engine",
                "application_version": "2.0.0",
            }

            if include_hostname:
                info_data["hostname"] = platform.node()

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            return ToolResult(
                tool_name=self.metadata.name,
                success=True,
                output=info_data,
                execution_time_ms=elapsed_ms,
                metadata={
                    "timestamp": datetime.now(UTC).isoformat(),
                    "tool_version": self.metadata.version,
                    "correlation_id": context.correlation_id,
                },
            )
        except Exception:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return ToolResult(
                tool_name=self.metadata.name,
                success=False,
                output=None,
                error_message="Failed to retrieve system information safely.",
                execution_time_ms=elapsed_ms,
                metadata={
                    "timestamp": datetime.now(UTC).isoformat(),
                    "tool_version": self.metadata.version,
                    "correlation_id": context.correlation_id,
                },
            )
