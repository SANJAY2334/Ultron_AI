"""Framework-Agnostic Abstract Base Planner and AgentState Schemas.

Defines the multi-stage lifecycle contract (analyze, reason, plan, execute, recover, summarize),
PlannerStatus lifecycle state machine, and serializable AgentState domain model.
"""

import uuid
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from app.ai.models import Message, ToolCall, ToolResultReference
from app.ai.router import AIProviderRouter, ai_router
from app.ai.tools.base import ExecutionContext
from app.ai.tools.executor import ToolExecutor, tool_executor
from app.ai.tools.registry import ToolRegistry, tool_registry
from app.security.policy import PolicyEngine, policy_engine


class PlannerStatus(StrEnum):
    """Planner Execution Lifecycle Status values."""

    IDLE = "IDLE"
    ANALYZING = "ANALYZING"
    REASONING = "REASONING"
    PLANNING = "PLANNING"
    EXECUTING = "EXECUTING"
    RECOVERING = "RECOVERING"
    SUMMARIZING = "SUMMARIZING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class AgentError(BaseModel):
    """Sanitized, strongly-typed error representation stored in AgentState.

    Ensures raw stack tracebacks never leak into AgentState or LLM context.
    """

    error_code: str = Field(
        description="Structured error code (e.g. POLICY_DENIED, TOOL_TIMEOUT, EXECUTION_FAILED)"
    )
    safe_message: str = Field(
        description="Sanitized, human-readable error message safe for LLM context"
    )
    retryable: bool = Field(
        default=True, description="True if error is eligible for recovery retry attempts"
    )
    source: str = Field(description="Origin source of error (e.g. tool name or provider name)")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Error creation timestamp"
    )


class AgentState(BaseModel):
    """Canonical, serializable state object passed through planner lifecycle stages."""

    session_id: str | None = Field(default=None, description="Active session ID")
    planner_id: str = Field(
        default_factory=lambda: f"planner_{uuid.uuid4().hex[:12]}",
        description="Unique planner run ID",
    )
    correlation_id: str | None = Field(default=None, description="Tracing correlation ID")
    status: PlannerStatus = Field(
        default=PlannerStatus.IDLE, description="Current planner lifecycle status"
    )
    messages: list[Message] = Field(
        default_factory=list, description="Ordered conversation message history"
    )
    analysis_result: str | None = Field(default=None, description="Output from analyze() stage")
    reasoning_result: str | None = Field(default=None, description="Output from reason() stage")
    current_plan: list[str] = Field(
        default_factory=list, description="Step-by-step action plan steps"
    )
    planned_tool_calls: list[ToolCall] = Field(
        default_factory=list, description="Target tool calls to execute"
    )
    active_tool_executions: list[str] = Field(
        default_factory=list, description="IDs of tools currently executing"
    )
    tool_outputs: list[ToolResultReference] = Field(
        default_factory=list, description="Outputs of executed tools"
    )
    errors: list[AgentError] = Field(
        default_factory=list, description="List of sanitized, strongly-typed AgentError records"
    )
    recovery_attempts: int = Field(
        default=0, ge=0, description="Count of recovery attempt iterations"
    )
    max_recovery_attempts: int = Field(
        default=3, ge=1, description="Maximum recovery retry threshold limit"
    )
    final_response: str | None = Field(
        default=None, description="Final synthesized user-facing response"
    )
    context: ExecutionContext = Field(
        default_factory=ExecutionContext, description="Execution permissions and session context"
    )
    vision_context: Any | None = Field(
        default=None, description="Sanitized visual perception context for prompt augmentation"
    )


class InvalidStateTransitionError(Exception):
    """Raised when an illegal planner lifecycle state transition is attempted."""


class BasePlanner(ABC):
    """Framework-agnostic abstract base class for multi-stage AI Autonomous Planners.

    Enforces distinct async pipeline stages:
    analyze() -> reason() -> plan() -> execute() -> (recover() if errors) -> summarize().
    """

    def __init__(
        self,
        router: AIProviderRouter = ai_router,
        registry: ToolRegistry = tool_registry,
        executor: ToolExecutor = tool_executor,
        policy: PolicyEngine = policy_engine,
    ) -> None:
        """Initializes BasePlanner with core AI subsystem dependencies."""
        self.router = router
        self.registry = registry
        self.executor = executor
        self.policy = policy

    def validate_transition(
        self, current_status: PlannerStatus, target_status: PlannerStatus
    ) -> None:
        """Validates whether transitioning from current_status to target_status is permitted.

        Args:
            current_status: Current state.
            target_status: Target transition state.

        Raises:
            InvalidStateTransitionError: If transition is illegal.
        """
        allowed: dict[PlannerStatus, set[PlannerStatus]] = {
            PlannerStatus.IDLE: {PlannerStatus.ANALYZING, PlannerStatus.FAILED},
            PlannerStatus.ANALYZING: {
                PlannerStatus.REASONING,
                PlannerStatus.FAILED,
            },
            PlannerStatus.REASONING: {PlannerStatus.PLANNING, PlannerStatus.FAILED},
            PlannerStatus.PLANNING: {
                PlannerStatus.EXECUTING,
                PlannerStatus.SUMMARIZING,
                PlannerStatus.FAILED,
            },
            PlannerStatus.EXECUTING: {
                PlannerStatus.SUMMARIZING,
                PlannerStatus.RECOVERING,
                PlannerStatus.FAILED,
            },
            PlannerStatus.RECOVERING: {
                PlannerStatus.EXECUTING,
                PlannerStatus.SUMMARIZING,
                PlannerStatus.FAILED,
            },
            PlannerStatus.SUMMARIZING: {
                PlannerStatus.COMPLETED,
                PlannerStatus.FAILED,
            },
            PlannerStatus.COMPLETED: set(),
            PlannerStatus.FAILED: set(),
        }

        if target_status not in allowed.get(current_status, set()):
            raise InvalidStateTransitionError(
                f"Illegal PlannerStatus transition from '{current_status.value}' to '{target_status.value}'."
            )

    @abstractmethod
    async def analyze(self, state: AgentState) -> AgentState:
        """Stage 1: Analyzes user intent, input constraints, and conversation context."""

    @abstractmethod
    async def reason(self, state: AgentState) -> AgentState:
        """Stage 2: Reasons over domain knowledge, goals, and required tools."""

    @abstractmethod
    async def plan(self, state: AgentState) -> AgentState:
        """Stage 3: Constructs structured step-by-step plan and planned tool calls."""

    @abstractmethod
    async def execute(self, state: AgentState) -> AgentState:
        """Stage 4: Executes planned tool calls through ToolExecutor and PolicyEngine."""

    @abstractmethod
    async def recover(self, state: AgentState) -> AgentState:
        """Stage 5: Handles errors, tool failures, or exceptions and attempts recovery."""

    @abstractmethod
    async def summarize(self, state: AgentState) -> AgentState:
        """Stage 6: Synthesizes final user-facing response from outputs and history."""

    async def run_pipeline(self, initial_state: AgentState) -> AgentState:
        """Orchestrates the complete multi-stage planner lifecycle pipeline.

        Args:
            initial_state: AgentState in IDLE status.

        Returns:
            AgentState: Final state in COMPLETED or FAILED status.
        """
        state = initial_state
        try:
            # Stage 1: Analyze
            self.validate_transition(state.status, PlannerStatus.ANALYZING)
            state.status = PlannerStatus.ANALYZING
            state = await self.analyze(state)

            # Stage 2: Reason
            self.validate_transition(state.status, PlannerStatus.REASONING)
            state.status = PlannerStatus.REASONING
            state = await self.reason(state)

            # Stage 3: Plan
            self.validate_transition(state.status, PlannerStatus.PLANNING)
            state.status = PlannerStatus.PLANNING
            state = await self.plan(state)

            # Stage 4: Execute
            self.validate_transition(state.status, PlannerStatus.EXECUTING)
            state.status = PlannerStatus.EXECUTING
            state = await self.execute(state)

            # Stage 5: Recovery loop if retryable errors occurred
            retryable_errors = [e for e in state.errors if e.retryable]
            if retryable_errors and state.recovery_attempts < state.max_recovery_attempts:
                self.validate_transition(state.status, PlannerStatus.RECOVERING)
                state.status = PlannerStatus.RECOVERING
                state = await self.recover(state)

            # Stage 6: Summarize
            self.validate_transition(state.status, PlannerStatus.SUMMARIZING)
            state.status = PlannerStatus.SUMMARIZING
            state = await self.summarize(state)

            self.validate_transition(state.status, PlannerStatus.COMPLETED)
            state.status = PlannerStatus.COMPLETED
            return state
        except Exception as exc:
            state.errors.append(
                AgentError(
                    error_code="PIPELINE_ERROR",
                    safe_message=f"Pipeline exception occurred: {exc}",
                    retryable=False,
                    source="run_pipeline",
                )
            )
            state.status = PlannerStatus.FAILED
            return state
