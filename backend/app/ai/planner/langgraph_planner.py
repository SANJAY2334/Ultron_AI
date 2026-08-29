"""LangGraph Autonomous Planner Engine Implementation.

Implements concrete LangGraphPlanner executing deterministic state graph topology and bounded state transitions
with non-deterministic LLM decision generation. Preserves AgentState boundary isolation, PolicyEngine safety,
and ToolExecutor authorization pipelines.
"""

import logging
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from app.ai.models import GenerationRequest, Message, ToolResultReference
from app.ai.planner.base import AgentError, AgentState, BasePlanner, PlannerStatus
from app.ai.router import AIProviderRouter, ai_router
from app.ai.tools.executor import ToolExecutor, tool_executor
from app.ai.tools.registry import ToolRegistry, tool_registry
from app.security.policy import PolicyEngine, policy_engine

logger = logging.getLogger(__name__)


class PlannerGraphState(TypedDict):
    """Internal TypedDict schema for LangGraph state machine node operations."""

    agent_state_json: str


class LangGraphPlanner(BasePlanner):
    """Concrete Autonomous Planner Engine using LangGraph state machine graph orchestration.

    Architecture Note: Provides deterministic graph topology and bounded state transitions
    with non-deterministic LLM decision generation.
    """

    def __init__(
        self,
        router: AIProviderRouter = ai_router,
        registry: ToolRegistry = tool_registry,
        executor: ToolExecutor = tool_executor,
        policy: PolicyEngine = policy_engine,
    ) -> None:
        """Initializes LangGraphPlanner and compiles the underlying LangGraph state graph."""
        super().__init__(router=router, registry=registry, executor=executor, policy=policy)
        self._compiled_graph = self._build_and_compile_graph()

    def _build_and_compile_graph(self) -> Any:
        """Constructs and compiles the deterministic LangGraph state machine graph."""
        builder = StateGraph(PlannerGraphState)

        # Register Node Functions
        builder.add_node("analyze_node", self._node_analyze)  # type: ignore[call-overload]
        builder.add_node("reason_node", self._node_reason)  # type: ignore[call-overload]
        builder.add_node("plan_node", self._node_plan)  # type: ignore[call-overload]
        builder.add_node("execute_node", self._node_execute)  # type: ignore[call-overload]
        builder.add_node("recover_node", self._node_recover)  # type: ignore[call-overload]
        builder.add_node("summarize_node", self._node_summarize)  # type: ignore[call-overload]

        # Static Edges
        builder.add_edge(START, "analyze_node")
        builder.add_edge("analyze_node", "reason_node")
        builder.add_edge("reason_node", "plan_node")

        # Conditional Edges
        builder.add_conditional_edges(
            "plan_node",
            self._route_after_plan,
            {"execute": "execute_node", "summarize": "summarize_node"},
        )
        builder.add_conditional_edges(
            "execute_node",
            self._route_after_execute,
            {
                "summarize": "summarize_node",
                "recover": "recover_node",
            },
        )
        builder.add_conditional_edges(
            "recover_node",
            self._route_after_recover,
            {"execute": "execute_node", "summarize": "summarize_node"},
        )
        builder.add_edge("summarize_node", END)

        return builder.compile()

    # --- Node Implementations ---

    async def _node_analyze(self, state_dict: PlannerGraphState) -> PlannerGraphState:
        """LangGraph Node: Analyze user intent and request parameters."""
        state = AgentState.model_validate_json(state_dict["agent_state_json"])
        self.validate_transition(state.status, PlannerStatus.ANALYZING)
        state.status = PlannerStatus.ANALYZING

        state = await self.analyze(state)
        logger.info(f"[{state.planner_id}] LangGraph Node 'analyze_node' completed.")
        return {"agent_state_json": state.model_dump_json()}

    async def _node_reason(self, state_dict: PlannerGraphState) -> PlannerGraphState:
        """LangGraph Node: Reason over query, domain knowledge, and strategies."""
        state = AgentState.model_validate_json(state_dict["agent_state_json"])
        self.validate_transition(state.status, PlannerStatus.REASONING)
        state.status = PlannerStatus.REASONING

        state = await self.reason(state)
        logger.info(f"[{state.planner_id}] LangGraph Node 'reason_node' completed.")
        return {"agent_state_json": state.model_dump_json()}

    async def _node_plan(self, state_dict: PlannerGraphState) -> PlannerGraphState:
        """LangGraph Node: Construct step-by-step action plan and identify target tool calls."""
        state = AgentState.model_validate_json(state_dict["agent_state_json"])
        self.validate_transition(state.status, PlannerStatus.PLANNING)
        state.status = PlannerStatus.PLANNING

        state = await self.plan(state)
        logger.info(
            f"[{state.planner_id}] LangGraph Node 'plan_node' completed (tool_calls={len(state.planned_tool_calls)})."
        )
        return {"agent_state_json": state.model_dump_json()}

    async def _node_execute(self, state_dict: PlannerGraphState) -> PlannerGraphState:
        """LangGraph Node: Safely execute planned tool calls via ToolExecutor & PolicyEngine."""
        state = AgentState.model_validate_json(state_dict["agent_state_json"])
        self.validate_transition(state.status, PlannerStatus.EXECUTING)
        state.status = PlannerStatus.EXECUTING

        state = await self.execute(state)
        logger.info(
            f"[{state.planner_id}] LangGraph Node 'execute_node' completed (outputs={len(state.tool_outputs)}, errors={len(state.errors)})."
        )
        return {"agent_state_json": state.model_dump_json()}

    async def _node_recover(self, state_dict: PlannerGraphState) -> PlannerGraphState:
        """LangGraph Node: Controlled recovery handling for transient execution errors."""
        state = AgentState.model_validate_json(state_dict["agent_state_json"])
        self.validate_transition(state.status, PlannerStatus.RECOVERING)
        state.status = PlannerStatus.RECOVERING

        state = await self.recover(state)
        logger.warning(
            f"[{state.planner_id}] LangGraph Node 'recover_node' completed (attempt {state.recovery_attempts}/{state.max_recovery_attempts})."
        )
        return {"agent_state_json": state.model_dump_json()}

    async def _node_summarize(self, state_dict: PlannerGraphState) -> PlannerGraphState:
        """LangGraph Node: Synthesize final user-facing response from tool outputs and context."""
        state = AgentState.model_validate_json(state_dict["agent_state_json"])
        self.validate_transition(state.status, PlannerStatus.SUMMARIZING)
        state.status = PlannerStatus.SUMMARIZING

        state = await self.summarize(state)
        if state.errors and state.recovery_attempts >= state.max_recovery_attempts:
            self.validate_transition(state.status, PlannerStatus.FAILED)
            state.status = PlannerStatus.FAILED
        else:
            self.validate_transition(state.status, PlannerStatus.COMPLETED)
            state.status = PlannerStatus.COMPLETED

        logger.info(
            f"[{state.planner_id}] LangGraph Node 'summarize_node' completed (status={state.status.value})."
        )
        return {"agent_state_json": state.model_dump_json()}

    # --- Conditional Routing Functions ---

    def _route_after_plan(self, state_dict: PlannerGraphState) -> str:
        """Routes to 'execute_node' if tool calls exist, else skips to 'summarize_node'."""
        state = AgentState.model_validate_json(state_dict["agent_state_json"])
        if state.planned_tool_calls:
            return "execute"
        return "summarize"

    def _route_after_execute(self, state_dict: PlannerGraphState) -> str:
        """Routes to 'recover_node' if retryable errors occurred and retry budget remains, else 'summarize_node'."""
        state = AgentState.model_validate_json(state_dict["agent_state_json"])
        retryable_errors = [e for e in state.errors if e.retryable]
        if retryable_errors and state.recovery_attempts < state.max_recovery_attempts:
            return "recover"
        return "summarize"

    def _route_after_recover(self, state_dict: PlannerGraphState) -> str:
        """Routes back to 'execute_node' if recovery budget remains, else skips to 'summarize_node'."""
        state = AgentState.model_validate_json(state_dict["agent_state_json"])
        if state.recovery_attempts <= state.max_recovery_attempts:
            return "execute"
        return "summarize"

    # --- Abstract Base Class Stage Implementations ---

    async def analyze(self, state: AgentState) -> AgentState:
        """Analyzes query intent."""
        if not state.analysis_result and state.messages:
            last_msg = state.messages[-1].content
            state.analysis_result = f"Analyzed user query: {last_msg}"
        return state

    async def reason(self, state: AgentState) -> AgentState:
        """Reasons over available tools and formulation strategy."""
        if not state.reasoning_result:
            state.reasoning_result = f"Reasoned strategy for analysis: {state.analysis_result}"
        return state

    async def plan(self, state: AgentState) -> AgentState:
        """Constructs action plan using registered tool definitions via AIProviderRouter."""
        if not state.current_plan:
            tool_schemas = self.registry.get_openai_tool_schemas()
            if tool_schemas and state.messages:
                # Query AI Provider Router for tool selection
                try:
                    req = GenerationRequest(
                        messages=state.messages,
                        tools=tool_schemas,
                        correlation_id=state.correlation_id,
                    )
                    res = await self.router.complete(req)
                    if res.message.tool_calls:
                        state.planned_tool_calls = res.message.tool_calls
                        state.current_plan = [
                            f"Execute tool: {tc.function_name}" for tc in res.message.tool_calls
                        ]
                except Exception as exc:
                    logger.warning(f"Planner LLM router tool selection query failed: {exc}")
            if not state.current_plan:
                state.current_plan = ["Generate direct response without tool calls."]
        return state

    async def execute(self, state: AgentState) -> AgentState:
        """Executes planned tool calls safely through ToolExecutor and PolicyEngine."""
        state.active_tool_executions.clear()

        # Only skip tool calls that previously succeeded
        successful_tool_call_ids = {
            ref.tool_call_id for ref in state.tool_outputs if ref.status == "success"
        }

        for tc in state.planned_tool_calls:
            if tc.id in successful_tool_call_ids:
                continue

            state.active_tool_executions.append(tc.id)
            result = await self.executor.execute(tc.function_name, tc.arguments, state.context)

            policy_decision = result.metadata.get("policy_decision")
            status_val: Literal["success", "error", "denied"]

            if policy_decision == "DENY":
                status_val = "denied"
                err = AgentError(
                    error_code="POLICY_DENIED",
                    safe_message=result.error_message or "Access Denied by Policy Engine.",
                    retryable=False,
                    source=tc.function_name,
                )
                state.errors.append(err)
            elif policy_decision == "REQUIRES_CONFIRMATION":
                status_val = "denied"
                err = AgentError(
                    error_code="REQUIRES_CONFIRMATION",
                    safe_message=result.error_message or "Explicit user confirmation required.",
                    retryable=False,
                    source=tc.function_name,
                )
                state.errors.append(err)
            elif not result.success:
                status_val = "error"
                err = AgentError(
                    error_code="EXECUTION_FAILED",
                    safe_message=result.error_message or "Tool execution failed.",
                    retryable=True,
                    source=tc.function_name,
                )
                state.errors.append(err)
            else:
                status_val = "success"

            ref = ToolResultReference(
                tool_call_id=tc.id,
                tool_name=tc.function_name,
                status=status_val,
                output=result.output if result.success else result.error_message,
                execution_time_ms=result.execution_time_ms,
            )
            state.tool_outputs.append(ref)

        state.active_tool_executions.clear()
        return state

    async def recover(self, state: AgentState) -> AgentState:
        """Executes controlled recovery strategy."""
        state.recovery_attempts += 1
        logger.warning(
            f"Executing recovery attempt #{state.recovery_attempts}/{state.max_recovery_attempts}"
        )
        if state.recovery_attempts <= state.max_recovery_attempts:
            # Clear retryable errors so re-execution can retry
            state.errors = [e for e in state.errors if not e.retryable]
        return state

    async def summarize(self, state: AgentState) -> AgentState:
        """Synthesizes final response via AIProviderRouter."""
        if not state.final_response:
            if state.tool_outputs:
                outputs_summary = "; ".join(
                    f"{t.tool_name}: {t.output}" for t in state.tool_outputs
                )
                state.final_response = (
                    f"Synthesized answer based on tool outputs: {outputs_summary}"
                )
            elif state.messages:
                state.final_response = f"Response to '{state.messages[-1].content}'."
            else:
                state.final_response = "ULTRON Autonomous Planner execution finished."

            # Append assistant message to conversation history
            state.messages.append(Message(role="assistant", content=state.final_response))

        return state

    async def run_pipeline(self, initial_state: AgentState) -> AgentState:
        """Executes the complete Autonomous Planner via compiled LangGraph state graph.

        Args:
            initial_state: Canonical AgentState object in IDLE status.

        Returns:
            AgentState: Final updated AgentState object.
        """
        input_dict: PlannerGraphState = {"agent_state_json": initial_state.model_dump_json()}
        output_dict = await self._compiled_graph.ainvoke(input_dict)
        final_state = AgentState.model_validate_json(output_dict["agent_state_json"])
        return final_state
