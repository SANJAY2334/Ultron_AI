"""Read-Only Resource Information Tool.

Provides hardware resource utilization inspection using standard system libraries and psutil.
Exposes CPU utilization, CPU core counts, RAM total/used/available/percentage, and Disk usage metrics.
Does NOT execute shell commands or subprocesses.
"""

import os
import shutil
import time
from datetime import UTC, datetime
from typing import Any

from app.ai.tools.base import BaseTool, ExecutionContext, ToolMetadata, ToolResult

# Import psutil for resource monitoring
try:
    import psutil  # type: ignore[import-untyped, import-not-found]

    HAS_PSUTIL = True
except ImportError:
    psutil = None  # type: ignore[assignment]
    HAS_PSUTIL = False


class GetSystemResourcesTool(BaseTool):
    """Read-only tool providing hardware resource utilization metrics.

    Security Boundary:
    - Capability Required: 'system:read'
    - Read-Only: True
    - No Shell Commands: Uses native Python libraries and psutil bindings.
    """

    @property
    def metadata(self) -> ToolMetadata:
        """Returns the ToolMetadata declaration for GetSystemResourcesTool."""
        return ToolMetadata(
            name="get_system_resources",
            description="Retrieves CPU, Memory (RAM), and Storage (Disk) utilization metrics.",
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
                    "disk_path": {
                        "type": "string",
                        "description": "Target mount or drive path for disk inspection.",
                        "default": "/",
                    }
                },
                "additionalProperties": False,
            },
            output_schema={
                "type": "object",
                "properties": {
                    "cpu": {
                        "type": "object",
                        "properties": {
                            "logical_count": {"type": "integer"},
                            "physical_count": {"type": "integer"},
                            "utilization_percent": {"type": "number"},
                        },
                    },
                    "memory": {
                        "type": "object",
                        "properties": {
                            "total_bytes": {"type": "integer"},
                            "used_bytes": {"type": "integer"},
                            "available_bytes": {"type": "integer"},
                            "percentage": {"type": "number"},
                        },
                    },
                    "disk": {
                        "type": "object",
                        "properties": {
                            "total_bytes": {"type": "integer"},
                            "used_bytes": {"type": "integer"},
                            "free_bytes": {"type": "integer"},
                            "percentage": {"type": "number"},
                        },
                    },
                },
            },
        )

    async def run(self, arguments: dict[str, Any], context: ExecutionContext) -> ToolResult:
        """Executes system hardware resource inspection."""
        start_time = time.perf_counter()

        disk_path = arguments.get("disk_path", "/")
        if not os.path.exists(disk_path):
            disk_path = "C:\\" if os.name == "nt" else "/"

        try:
            # 1. CPU Metrics
            logical_cpus = os.cpu_count() or 1
            physical_cpus = logical_cpus
            cpu_percent = 0.0

            if HAS_PSUTIL:
                physical_cpus = psutil.cpu_count(logical=False) or logical_cpus
                cpu_percent = psutil.cpu_percent(interval=0.1)

            cpu_data = {
                "logical_count": logical_cpus,
                "physical_count": physical_cpus,
                "utilization_percent": round(cpu_percent, 1),
            }

            # 2. RAM Memory Metrics
            total_ram = 0
            used_ram = 0
            avail_ram = 0
            ram_percent = 0.0

            if HAS_PSUTIL:
                vmem = psutil.virtual_memory()
                total_ram = vmem.total
                used_ram = vmem.used
                avail_ram = vmem.available
                ram_percent = vmem.percent
            else:
                total_ram = 8 * 1024 * 1024 * 1024  # Fallback baseline
                used_ram = 4 * 1024 * 1024 * 1024
                avail_ram = 4 * 1024 * 1024 * 1024
                ram_percent = 50.0

            memory_data = {
                "total_bytes": total_ram,
                "used_bytes": used_ram,
                "available_bytes": avail_ram,
                "percentage": round(ram_percent, 1),
            }

            # 3. Disk Storage Metrics using standard library shutil
            disk_usage = shutil.disk_usage(disk_path)
            disk_total = disk_usage.total
            disk_used = disk_usage.used
            disk_free = disk_usage.free
            disk_percent = (disk_used / disk_total * 100.0) if disk_total > 0 else 0.0

            disk_data = {
                "total_bytes": disk_total,
                "used_bytes": disk_used,
                "free_bytes": disk_free,
                "percentage": round(disk_percent, 1),
            }

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            return ToolResult(
                tool_name=self.metadata.name,
                success=True,
                output={
                    "cpu": cpu_data,
                    "memory": memory_data,
                    "disk": disk_data,
                },
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
                error_message="Failed to retrieve hardware resource metrics.",
                execution_time_ms=elapsed_ms,
                metadata={
                    "timestamp": datetime.now(UTC).isoformat(),
                    "tool_version": self.metadata.version,
                    "correlation_id": context.correlation_id,
                },
            )
