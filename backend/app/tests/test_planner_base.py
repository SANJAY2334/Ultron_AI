"""Unit Tests for BasePlanner Interface and AgentState Schemas.

Validates AgentState creation, JSON serialization/deserialization, PlannerStatus lifecycle transitions,
invalid transition rejection, and pipeline stage execution.
"""

import asyncio

import pytest

from app.ai.models import Message, ToolCall, ToolResultReference
from app.ai.planner.base import (
    AgentError,
    AgentState,
    BasePlanner,
    InvalidStateTransitionError,
    PlannerStatus,
)
from app.ai.tools.base import ExecutionContext


class DummyPlanner(BasePlanner):
    """Concrete dummy planner for testing multi-stage pipeline execution."""

    async def analyze(self, state: AgentState) -> AgentState:
        state.analysis_result = "Intent: Math addition query"
        return state

    async def reason(self, state: AgentState) -> AgentState:
        state.reasoning_result = "Need to call adder tool"
        return state

    async def plan(self, state: AgentState) -> AgentState:
        state.current_plan = ["1. Invoke adder tool", "2. Return result"]
        state.planned_tool_calls = [ToolCall(function_name="adder", arguments={"x": 3, "y": 4})]
        return state

    async def execute(self, state: AgentState) -> AgentState:
        state.tool_outputs.append(
            ToolResultReference(
                tool_call_id=state.planned_tool_calls[0].id,
                tool_name="adder",
                status="success",
                output=7,
            )
        )
        return state

    async def recover(self, state: AgentState) -> AgentState:
        state.recovery_attempts += 1
        state.errors.clear()
        return state

    async def summarize(self, state: AgentState) -> AgentState:
        state.final_response = "The sum of 3 and 4 is 7."
        return state


class FaultyPlanner(DummyPlanner):
    """Planner that injects an error during execution."""

    async def execute(self, state: AgentState) -> AgentState:
        state.errors.append(
            AgentError(
                error_code="EXECUTION_FAILED",
                safe_message="Execution error: tool connection reset",
                retryable=True,
                source="adder",
            )
        )
        return state


def test_agent_state_creation_and_defaults() -> None:
    """Verify AgentState creation and default properties."""
    state = AgentState(session_id="sess_123", correlation_id="corr_456")
    assert state.planner_id.startswith("planner_")
    assert state.status == PlannerStatus.IDLE
    assert state.messages == []
    assert state.recovery_attempts == 0
    assert state.max_recovery_attempts == 3


def test_agent_state_serialization_deserialization() -> None:
    """Verify AgentState JSON serialization and deserialization."""
    state = AgentState(
        session_id="sess_abc",
        messages=[Message(role="user", content="Add 5 and 5")],
        context=ExecutionContext(user_id="user_test", granted_capabilities={"math:add"}),
    )

    json_str = state.model_dump_json()
    assert "sess_abc" in json_str

    deserialized = AgentState.model_validate_json(json_str)
    assert deserialized.session_id == "sess_abc"
    assert len(deserialized.messages) == 1
    assert deserialized.messages[0].content == "Add 5 and 5"
    assert deserialized.context.user_id == "user_test"
    assert "math:add" in deserialized.context.granted_capabilities


def test_planner_state_transition_validation() -> None:
    """Verify valid and invalid PlannerStatus state transitions."""
    planner = DummyPlanner()

    # Valid transitions
    planner.validate_transition(PlannerStatus.IDLE, PlannerStatus.ANALYZING)
    planner.validate_transition(PlannerStatus.ANALYZING, PlannerStatus.REASONING)
    planner.validate_transition(PlannerStatus.EXECUTING, PlannerStatus.RECOVERING)

    # Invalid transitions
    with pytest.raises(InvalidStateTransitionError):
        planner.validate_transition(PlannerStatus.IDLE, PlannerStatus.EXECUTING)

    with pytest.raises(InvalidStateTransitionError):
        planner.validate_transition(PlannerStatus.COMPLETED, PlannerStatus.ANALYZING)


def test_concrete_planner_pipeline_execution() -> None:
    """Verify successful execution of full 6-stage pipeline."""

    async def _test() -> None:
        planner = DummyPlanner()
        initial_state = AgentState(messages=[Message(role="user", content="Add 3 and 4")])

        final_state = await planner.run_pipeline(initial_state)

        assert final_state.status == PlannerStatus.COMPLETED
        assert final_state.analysis_result == "Intent: Math addition query"
        assert final_state.reasoning_result == "Need to call adder tool"
        assert len(final_state.current_plan) == 2
        assert len(final_state.tool_outputs) == 1
        assert final_state.tool_outputs[0].output == 7
        assert final_state.final_response == "The sum of 3 and 4 is 7."

    asyncio.run(_test())


def test_concrete_planner_recovery_flow() -> None:
    """Verify recovery stage is triggered when retryable errors occur in execute stage."""

    async def _test() -> None:
        planner = FaultyPlanner()
        initial_state = AgentState(messages=[Message(role="user", content="Trigger recovery test")])

        final_state = await planner.run_pipeline(initial_state)

        assert final_state.status == PlannerStatus.COMPLETED
        assert final_state.recovery_attempts == 1
        assert final_state.final_response == "The sum of 3 and 4 is 7."

    asyncio.run(_test())
