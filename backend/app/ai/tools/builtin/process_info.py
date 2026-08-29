"""Read-Only Process Information Tool.

Provides safe, read-only process listing and status inspection using psutil bindings.
Exposes PID, process name, CPU usage %, memory usage (bytes), and status.
Strictly non-destructive: Does NOT support process termination, suspension, or creation.
"""

import time
from datetime import UTC, datetime
from typing import Any

from app.ai.tools.base import BaseTool, ExecutionContext, ToolMetadata, ToolResult

try:
    import psutil  # type: ignore[import-untyped, import-not-found]

    HAS_PSUTIL = True
except ImportError:
    psutil = None  # type: ignore[assignment]
    HAS_PSUTIL = False


class GetProcessInfoTool(BaseTool):
    """Read-only tool for inspecting active operating system processes.

    Security Boundary:
    - Capability Required: 'system:process_read'
    - Read-Only: True
    - Non-Destructive: Process kill, suspend, and spawn commands are prohibited.
    """

    @property
    def metadata(self) -> ToolMetadata:
        """Returns the ToolMetadata declaration for GetProcessInfoTool."""
        return ToolMetadata(
            name="get_process_info",
            description="Retrieves a list of running operating system processes or detailed status for a specific PID.",
            version="1.0.0",
            category="system",
            timeout_seconds=10.0,
            supports_streaming=False,
            destructive=False,
            confirmation_required=False,
            required_capabilities=["system:process_read"],
            input_schema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of process items to return.",
                        "default": 50,
                    },
                    "pid": {
                        "type": ["integer", "null"],
                        "description": "Optional specific Process ID to inspect.",
                        "default": None,
                    },
                    "filter_name": {
                        "type": ["string", "null"],
                        "description": "Optional process name search filter string.",
                        "default": None,
                    },
                },
                "additionalProperties": False,
            },
            output_schema={
                "type": "object",
                "properties": {
                    "processes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "pid": {"type": "integer"},
                                "name": {"type": "string"},
                                "status": {"type": "string"},
                                "cpu_percent": {"type": "number"},
                                "memory_bytes": {"type": "integer"},
                                "created_at": {"type": "string"},
                            },
                        },
                    },
                    "total_count": {"type": "integer"},
                },
            },
        )

    async def run(self, arguments: dict[str, Any], context: ExecutionContext) -> ToolResult:
        """Executes process listing and inspection."""
        start_time = time.perf_counter()

        limit = arguments.get("limit", 50)
        target_pid = arguments.get("pid")
        filter_name = arguments.get("filter_name")

        if filter_name:
            filter_name = filter_name.lower().strip()

        processes_data: list[dict[str, Any]] = []

        try:
            if not HAS_PSUTIL:
                return ToolResult(
                    tool_name=self.metadata.name,
                    success=False,
                    output=None,
                    error_message="Process inspection library (psutil) unavailable.",
                    execution_time_ms=(time.perf_counter() - start_time) * 1000.0,
                    metadata={"timestamp": datetime.now(UTC).isoformat()},
                )

            # Case A: Specific PID query
            if target_pid is not None:
                try:
                    proc = psutil.Process(target_pid)
                    with proc.oneshot():
                        mem_info = proc.memory_info()
                        created_ts = datetime.fromtimestamp(proc.create_time(), tz=UTC).isoformat()
                        processes_data.append(
                            {
                                "pid": proc.pid,
                                "name": proc.name(),
                                "status": proc.status(),
                                "cpu_percent": round(proc.cpu_percent(interval=0.01), 1),
                                "memory_bytes": mem_info.rss,
                                "created_at": created_ts,
                            }
                        )
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    pass
            else:
                # Case B: Process list iteration
                for proc in psutil.process_iter(["pid", "name", "status", "memory_info"]):
                    if len(processes_data) >= limit:
                        break
                    try:
                        p_info = proc.info
                        p_name = str(p_info.get("name") or "Unknown")

                        if filter_name and filter_name not in p_name.lower():
                            continue

                        mem_bytes = p_info["memory_info"].rss if p_info.get("memory_info") else 0

                        processes_data.append(
                            {
                                "pid": p_info["pid"],
                                "name": p_name,
                                "status": str(p_info.get("status") or "unknown"),
                                "cpu_percent": 0.0,
                                "memory_bytes": mem_bytes,
                                "created_at": datetime.now(UTC).isoformat(),
                            }
                        )
                    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                        continue

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            return ToolResult(
                tool_name=self.metadata.name,
                success=True,
                output={
                    "processes": processes_data,
                    "total_count": len(processes_data),
                },
                execution_time_ms=elapsed_ms,
                metadata={
                    "timestamp": datetime.now(UTC).isoformat(),
                    "tool_version": self.metadata.version,
                    "correlation_id": context.correlation_id,
                },
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return ToolResult(
                tool_name=self.metadata.name,
                success=False,
                output=None,
                error_message=f"Failed to inspect system processes safely: {exc}",
                execution_time_ms=elapsed_ms,
                metadata={
                    "timestamp": datetime.now(UTC).isoformat(),
                    "tool_version": self.metadata.version,
                    "correlation_id": context.correlation_id,
                },
            )
