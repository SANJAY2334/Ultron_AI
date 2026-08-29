"""ULTRON Policy Engine Subsystem.

Evaluates security policies, capability authorizations, destructive action safety,
and user confirmation requirements between the Planner and Tool Executor.
"""

from typing import Literal

from pydantic import BaseModel, Field

from app.ai.tools.base import ExecutionContext, ToolMetadata


class PolicyDecision(BaseModel):
    """Policy evaluation decision returned by PolicyEngine."""

    decision: Literal["ALLOW", "DENY", "REQUIRES_CONFIRMATION"] = Field(
        description="Policy decision outcome"
    )
    reason: str = Field(description="Explanation of policy decision")
    missing_capabilities: list[str] = Field(
        default_factory=list, description="Capabilities missing from execution context"
    )


class PolicyEngine:
    """Zero Trust Policy Engine enforcing permission authorization and safety gating."""

    def evaluate_tool_execution(
        self, tool_metadata: ToolMetadata, context: ExecutionContext
    ) -> PolicyDecision:
        """Evaluates whether a tool execution is authorized under given context.

        Args:
            tool_metadata: Complete metadata contract of target tool.
            context: Active ExecutionContext carrying granted capabilities and environment flags.

        Returns:
            PolicyDecision: Policy outcome (ALLOW, DENY, REQUIRES_CONFIRMATION).
        """
        # Step 1: Capability permission validation
        required_caps = set(tool_metadata.required_capabilities)
        granted_caps = context.granted_capabilities
        missing_caps = list(required_caps - granted_caps)

        if missing_caps:
            return PolicyDecision(
                decision="DENY",
                reason=(
                    f"Access Denied: Tool '{tool_metadata.name}' requires capabilities "
                    f"{missing_caps} which are not granted in ExecutionContext."
                ),
                missing_capabilities=missing_caps,
            )

        # Step 2: Destructive / Confirmation safety gating
        if tool_metadata.confirmation_required or tool_metadata.destructive:
            user_confirmed = context.environment.get("user_confirmed", False)
            if not user_confirmed:
                return PolicyDecision(
                    decision="REQUIRES_CONFIRMATION",
                    reason=(
                        f"Confirmation Required: Tool '{tool_metadata.name}' is marked "
                        f"destructive/confirmation_required and requires explicit user approval."
                    ),
                )

        # Step 3: Default ALLOW
        return PolicyDecision(
            decision="ALLOW",
            reason=f"Execution of tool '{tool_metadata.name}' authorized by PolicyEngine.",
        )


# Global Policy Engine singleton
policy_engine = PolicyEngine()
