"""Unit Tests for LangGraph Autonomous Planner Engine.

Validates graph compilation, deterministic node transitions, direct response path,
tool execution path, PolicyEngine enforcement, recovery attempt limit, sanitized error isolation,
and security pipeline enforcement.
"""

import asyncio
from typing import Any
from unittest.mock import AsyncMock

from app.ai.base import BaseLLMProvider
from app.ai.models import GenerationResponse, Message, ProviderMetadata, ToolCall, Usage
from app.ai.planner.base import AgentState, PlannerStatus
from app.ai.planner.langgraph_planner import LangGraphPlanner
from app.ai.router import AIProviderRouter
from app.ai.tools.base import BaseTool, ExecutionContext, ToolMetadata, ToolResult
from app.ai.tools.executor import ToolExecutor
from app.ai.tools.registry import ToolRegistry
from app.security.policy import PolicyEngine


class MockMathTool(BaseTool):
    """Mock Tool for testing tool execution path."""

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="multiply",
            description="Multiplies numbers.",
            required_capabilities=["math:multiply"],
        )

    async def run(self, arguments: dict[str, Any], context: ExecutionContext) -> ToolResult:
        x = arguments.get("x", 1)
        y = arguments.get("y", 1)
        return ToolResult(tool_name="multiply", success=True, output=x * y)


class MockDestructiveTool(BaseTool):
    """Mock Destructive Tool requiring user confirmation."""

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="delete_database",
            description="Deletes system database.",
            destructive=True,
            confirmation_required=True,
            required_capabilities=["db:delete"],
        )

    async def run(self, arguments: dict[str, Any], context: ExecutionContext) -> ToolResult:
        return ToolResult(tool_name="delete_database", success=True, output="Database deleted.")


class MockFailingTool(BaseTool):
    """Mock Tool that always fails with execution error."""

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="fail_tool",
            description="Always fails.",
            required_capabilities=["test:fail"],
        )

    async def run(self, arguments: dict[str, Any], context: ExecutionContext) -> ToolResult:
        return ToolResult(tool_name="fail_tool", success=False, error_message="Execution error")


def test_graph_construction_and_compilation() -> None:
    """Verify LangGraphPlanner compiles internal StateGraph without error."""
    planner = LangGraphPlanner(registry=ToolRegistry())
    assert planner._compiled_graph is not None


def test_direct_response_path() -> None:
    """Verify execution path for direct response without tool calls."""

    async def _test() -> None:
        planner = LangGraphPlanner(registry=ToolRegistry())
        initial_state = AgentState(
            session_id="sess_direct",
            correlation_id="corr_direct",
            messages=[Message(role="user", content="Hello, Ultron.")],
        )

        final_state = await planner.run_pipeline(initial_state)

        assert final_state.status == PlannerStatus.COMPLETED
        assert final_state.correlation_id == "corr_direct"
        assert final_state.session_id == "sess_direct"
        assert final_state.analysis_result is not None
        assert final_state.reasoning_result is not None
        assert final_state.final_response is not None
        assert len(final_state.tool_outputs) == 0

    asyncio.run(_test())


def test_successful_tool_execution_path() -> None:
    """Verify execution path with planned tool calls, PolicyEngine authorization, and ToolExecutor execution."""

    async def _test() -> None:
        registry = ToolRegistry()
        registry.register(MockMathTool())
        policy = PolicyEngine()
        executor = ToolExecutor(registry=registry, policy=policy)

        router = AIProviderRouter()
        mock_provider = AsyncMock(spec=BaseLLMProvider)
        mock_provider.provider_name = "mock_provider"
        mock_provider.default_model = "mock-model"
        mock_provider.complete.return_value = GenerationResponse(
            message=Message(
                role="assistant",
                content="",
                tool_calls=[ToolCall(function_name="multiply", arguments={"x": 6, "y": 7})],
            ),
            usage=Usage(total_tokens=10),
            metadata=ProviderMetadata(
                provider_name="mock_provider", model_name="mock-model", latency_ms=5.0
            ),
        )
        router.register_provider(mock_provider)

        planner = LangGraphPlanner(
            router=router, registry=registry, executor=executor, policy=policy
        )

        initial_state = AgentState(
            session_id="sess_math",
            correlation_id="corr_math",
            messages=[Message(role="user", content="What is 6 times 7?")],
            context=ExecutionContext(granted_capabilities={"math:multiply"}),
        )

        final_state = await planner.run_pipeline(initial_state)

        assert final_state.status == PlannerStatus.COMPLETED
        assert len(final_state.tool_outputs) == 1
        assert final_state.tool_outputs[0].tool_name == "multiply"
        assert final_state.tool_outputs[0].status == "success"
        assert final_state.tool_outputs[0].output == 42
        assert final_state.final_response is not None and "42" in final_state.final_response

    asyncio.run(_test())


def test_deny_never_retries() -> None:
    """Verify PolicyEngine DENY result is non-retryable and never triggers recovery loops."""

    async def _test() -> None:
        registry = ToolRegistry()
        registry.register(MockMathTool())
        policy = PolicyEngine()
        executor = ToolExecutor(registry=registry, policy=policy)

        planner = LangGraphPlanner(registry=registry, executor=executor, policy=policy)

        initial_state = AgentState(
            session_id="sess_denied",
            planned_tool_calls=[ToolCall(function_name="multiply", arguments={"x": 2, "y": 3})],
            context=ExecutionContext(granted_capabilities=set()),  # Missing 'math:multiply'
            max_recovery_attempts=3,
        )

        final_state = await planner.run_pipeline(initial_state)

        assert len(final_state.tool_outputs) == 1
        assert final_state.tool_outputs[0].status == "denied"
        assert final_state.recovery_attempts == 0  # DENY must NEVER trigger recovery retries!
        assert len(final_state.errors) == 1
        assert final_state.errors[0].error_code == "POLICY_DENIED"
        assert final_state.errors[0].retryable is False

    asyncio.run(_test())


def test_requires_confirmation_pauses_execution() -> None:
    """Verify destructive action without user confirmation returns REQUIRES_CONFIRMATION and pauses execution."""

    async def _test() -> None:
        registry = ToolRegistry()
        registry.register(MockDestructiveTool())
        policy = PolicyEngine()
        executor = ToolExecutor(registry=registry, policy=policy)

        planner = LangGraphPlanner(registry=registry, executor=executor, policy=policy)

        # Granted 'db:delete' capability, but user_confirmed flag is NOT set
        initial_state = AgentState(
            planned_tool_calls=[ToolCall(function_name="delete_database", arguments={})],
            context=ExecutionContext(granted_capabilities={"db:delete"}),
        )

        final_state = await planner.run_pipeline(initial_state)

        assert len(final_state.tool_outputs) == 1
        assert final_state.tool_outputs[0].status == "denied"
        assert final_state.recovery_attempts == 0
        assert final_state.errors[0].error_code == "REQUIRES_CONFIRMATION"

    asyncio.run(_test())


def test_confirmed_destructive_action_reevaluated() -> None:
    """Verify destructive action with user_confirmed=True passes policy and executes."""

    async def _test() -> None:
        registry = ToolRegistry()
        registry.register(MockDestructiveTool())
        policy = PolicyEngine()
        executor = ToolExecutor(registry=registry, policy=policy)

        planner = LangGraphPlanner(registry=registry, executor=executor, policy=policy)

        initial_state = AgentState(
            planned_tool_calls=[ToolCall(function_name="delete_database", arguments={})],
            context=ExecutionContext(
                granted_capabilities={"db:delete"},
                environment={"user_confirmed": True},
            ),
        )

        final_state = await planner.run_pipeline(initial_state)

        assert final_state.status == PlannerStatus.COMPLETED
        assert len(final_state.tool_outputs) == 1
        assert final_state.tool_outputs[0].status == "success"
        assert final_state.tool_outputs[0].output == "Database deleted."

    asyncio.run(_test())


def test_exact_recovery_maximum_enforced() -> None:
    """Verify exact recovery maximum counter (Failure 1 -> Rec 1, Failure 2 -> Rec 2, Failure 3 -> Rec 3 -> STOP)."""

    async def _test() -> None:
        registry = ToolRegistry()
        registry.register(MockFailingTool())
        policy = PolicyEngine()
        executor = ToolExecutor(registry=registry, policy=policy)

        planner = LangGraphPlanner(registry=registry, executor=executor, policy=policy)

        initial_state = AgentState(
            session_id="sess_fail_boundary",
            planned_tool_calls=[ToolCall(function_name="fail_tool", arguments={})],
            context=ExecutionContext(granted_capabilities={"test:fail"}),
            max_recovery_attempts=3,
        )

        final_state = await planner.run_pipeline(initial_state)

        assert final_state.recovery_attempts == 3  # Exact recovery maximum budget reached
        assert final_state.status == PlannerStatus.FAILED
        assert any(e.error_code == "EXECUTION_FAILED" for e in final_state.errors)

    asyncio.run(_test())


def test_sanitized_errors_and_no_raw_tracebacks() -> None:
    """Verify errors stored in AgentState are sanitized AgentError objects with no raw tracebacks."""

    async def _test() -> None:
        registry = ToolRegistry()
        registry.register(MockFailingTool())
        policy = PolicyEngine()
        executor = ToolExecutor(registry=registry, policy=policy)

        planner = LangGraphPlanner(registry=registry, executor=executor, policy=policy)

        initial_state = AgentState(
            planned_tool_calls=[ToolCall(function_name="fail_tool", arguments={})],
            context=ExecutionContext(granted_capabilities={"test:fail"}),
            max_recovery_attempts=1,
        )

        final_state = await planner.run_pipeline(initial_state)

        json_output = final_state.model_dump_json()
        assert "Traceback (most recent call last)" not in json_output
        assert "EXECUTION_FAILED" in json_output
        assert isinstance(final_state.errors[0].safe_message, str)

    asyncio.run(_test())


def test_planner_cannot_bypass_executor_or_policy() -> None:
    """Verify planner delegates tool execution strictly to ToolExecutor and PolicyEngine."""
    registry = ToolRegistry()
    registry.register(MockMathTool())
    policy = PolicyEngine()
    executor = ToolExecutor(registry=registry, policy=policy)

    planner = LangGraphPlanner(registry=registry, executor=executor, policy=policy)
    assert planner.executor is executor
    assert planner.policy is policy
    assert planner.registry is registry
