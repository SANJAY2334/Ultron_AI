"""Base Tool Interface and Capability Schemas.

Defines the contract for all tools in ULTRON. Incorporates Capability permissions,
ExecutionContext context passing, expanded ToolMetadata, and canonical ToolResult output.
"""

from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import BaseModel, Field


class Capability(BaseModel):
    """System Capability definition required for gated tools and actions."""

    name: str = Field(description="Unique capability string identifier (e.g. 'file:write')")
    description: str = Field(description="Human-readable capability description")
    category: str = Field(default="general", description="Capability functional domain category")
    risk_level: Literal["low", "medium", "high", "critical"] = Field(
        default="low", description="Associated risk classification"
    )


class ExecutionContext(BaseModel):
    """Contextual metadata passed to every tool execution pipeline."""

    user_id: str | None = Field(default=None, description="Requesting user identifier")
    session_id: str | None = Field(default=None, description="Active session ID")
    correlation_id: str | None = Field(default=None, description="Tracing correlation ID")
    planner_id: str | None = Field(default=None, description="Invoking planner engine ID")
    granted_capabilities: set[str] = Field(
        default_factory=set, description="Set of capability permissions granted to context"
    )
    environment: dict[str, Any] = Field(
        default_factory=dict, description="Execution environment variables and state flags"
    )


class ToolMetadata(BaseModel):
    """Complete metadata and functional contract definition for an ULTRON tool."""

    name: str = Field(description="Unique tool identifier name")
    description: str = Field(description="Clear function description for LLM tool selection")
    version: str = Field(default="1.0.0", description="Semantic version string")
    category: str = Field(
        default="general", description="Tool category (e.g., system, memory, web)"
    )
    timeout_seconds: float = Field(
        default=30.0, ge=0.1, description="Execution timeout limit in seconds"
    )
    supports_streaming: bool = Field(
        default=False, description="True if tool supports streaming outputs"
    )
    destructive: bool = Field(
        default=False, description="True if tool performs destructive/irreversible side effects"
    )
    confirmation_required: bool = Field(
        default=False,
        description="True if tool requires explicit user confirmation prior to execution",
    )
    required_capabilities: list[str] = Field(
        default_factory=list, description="List of required Capability names"
    )
    input_schema: dict[str, Any] = Field(
        default_factory=dict, description="JSON Schema specification for tool input arguments"
    )
    output_schema: dict[str, Any] = Field(
        default_factory=dict, description="JSON Schema specification for tool output results"
    )


class ToolResult(BaseModel):
    """Canonical result object returned by tool execution."""

    tool_name: str = Field(description="Name of executed tool")
    success: bool = Field(description="True if execution completed without error")
    output: Any = Field(default=None, description="Execution result payload")
    error_message: str | None = Field(default=None, description="Error details if execution failed")
    execution_time_ms: float = Field(
        default=0.0, ge=0.0, description="Execution duration in milliseconds"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional execution telemetry and audit flags"
    )


class BaseTool(ABC):
    """Abstract Base Class for all ULTRON tools.

    Tools declare required capabilities and metadata, relying on PolicyEngine and ToolExecutor
    for permission enforcement and safety gating.
    """

    @property
    @abstractmethod
    def metadata(self) -> ToolMetadata:
        """Returns the complete metadata contract for this tool."""

    @abstractmethod
    async def run(self, arguments: dict[str, Any], context: ExecutionContext) -> ToolResult:
        """Executes the tool logic with given arguments and execution context.

        Args:
            arguments: Parsed argument dictionary matching input_schema.
            context: Active ExecutionContext carrying permissions and session state.

        Returns:
            ToolResult: Canonical result object.
        """
