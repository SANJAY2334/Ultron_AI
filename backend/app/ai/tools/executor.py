"""Tool Execution Pipeline & Observability Subsystem.

Implements the complete tool safety and execution pipeline:
Planner -> Policy Engine -> Input Validation -> Audit Logging -> Execution -> Telemetry -> Response.
"""

import asyncio
import logging
import time
from typing import Any

from pydantic import BaseModel

from app.ai.tools.base import ExecutionContext, ToolResult
from app.ai.tools.registry import ToolRegistry, tool_registry
from app.security.policy import PolicyEngine, policy_engine

logger = logging.getLogger(__name__)


class ToolExecutionTelemetry(BaseModel):
    """Telemetry telemetry record captured during every tool execution."""

    tool_name: str
    success: bool
    latency_ms: float
    exception_type: str | None = None
    planner_id: str | None = None
    correlation_id: str | None = None
    user_id: str | None = None


class ToolExecutor:
    """Executes tools through a strict safety, policy, audit, and telemetry pipeline."""

    def __init__(
        self,
        registry: ToolRegistry = tool_registry,
        policy: PolicyEngine = policy_engine,
    ) -> None:
        """Initializes the ToolExecutor.

        Args:
            registry: ToolRegistry instance.
            policy: PolicyEngine instance.
        """
        self._registry = registry
        self._policy = policy

    async def execute(
        self, tool_name: str, arguments: dict[str, Any], context: ExecutionContext
    ) -> ToolResult:
        """Executes a named tool through the 6-stage security and observability pipeline.

        Args:
            tool_name: Name of registered tool to execute.
            arguments: Input argument dictionary.
            context: Active ExecutionContext carrying permissions and session state.

        Returns:
            ToolResult: Execution output or policy denial error.
        """
        start_time = time.perf_counter()
        exception_type: str | None = None

        # Stage 1: Tool Discovery
        tool = self._registry.find(tool_name)
        if not tool:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error_message=f"Tool '{tool_name}' not found in registry.",
                execution_time_ms=elapsed_ms,
            )

        meta = tool.metadata

        # Stage 2: Policy Engine Authorization Gating
        decision = self._policy.evaluate_tool_execution(meta, context)
        if decision.decision != "ALLOW":
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            logger.warning(
                f"Policy Engine rejected tool '{tool_name}' execution (user: {context.user_id}): {decision.reason}"
            )
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error_message=decision.reason,
                execution_time_ms=elapsed_ms,
                metadata={"policy_decision": decision.decision},
            )

        # Stage 3: Input Validation Check
        if not isinstance(arguments, dict):
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error_message="Invalid arguments payload: expected JSON object dictionary.",
                execution_time_ms=elapsed_ms,
            )

        # Stage 4: Audit Logging
        logger.info(
            f"Executing tool '{tool_name}' [user_id={context.user_id}, planner_id={context.planner_id}, corr_id={context.correlation_id}]"
        )

        # Stage 5: Async Tool Execution with Timeout
        success = False
        output: Any = None
        error_msg: str | None = None

        try:
            result = await asyncio.wait_for(
                tool.run(arguments, context), timeout=meta.timeout_seconds
            )
            success = result.success
            output = result.output
            error_msg = result.error_message
        except TimeoutError:
            success = False
            exception_type = "TimeoutError"
            error_msg = (
                f"Tool '{tool_name}' execution timed out after {meta.timeout_seconds} seconds."
            )
            logger.error(error_msg)
        except Exception as exc:
            success = False
            exception_type = exc.__class__.__name__
            error_msg = f"Tool '{tool_name}' raised execution exception: {exc}"
            logger.error(error_msg, exc_info=True)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Stage 6: Telemetry Collection
        telemetry = ToolExecutionTelemetry(
            tool_name=tool_name,
            success=success,
            latency_ms=elapsed_ms,
            exception_type=exception_type,
            planner_id=context.planner_id,
            correlation_id=context.correlation_id,
            user_id=context.user_id,
        )
        logger.info(
            f"Tool '{tool_name}' execution finished in {elapsed_ms:.2f}ms (success={success}, telemetry={telemetry.model_dump()})"
        )

        return ToolResult(
            tool_name=tool_name,
            success=success,
            output=output,
            error_message=error_msg,
            execution_time_ms=elapsed_ms,
            metadata={"telemetry": telemetry.model_dump()},
        )


# Global Tool Executor singleton
tool_executor = ToolExecutor()
