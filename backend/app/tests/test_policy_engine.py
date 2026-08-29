"""Unit Tests for ULTRON Policy Engine Subsystem.

Validates capability authorization checks, missing permission rejection,
destructive action safety gating, and user confirmation evaluation.
"""

from app.ai.tools.base import ExecutionContext, ToolMetadata
from app.security.policy import PolicyEngine


def test_policy_engine_allow_authorized_tool() -> None:
    """Verify tool execution is ALLOWED when all required capabilities are granted."""
    pe = PolicyEngine()
    meta = ToolMetadata(
        name="read_file",
        description="Reads file.",
        required_capabilities=["file:read"],
    )
    ctx = ExecutionContext(granted_capabilities={"file:read", "other:cap"})
    decision = pe.evaluate_tool_execution(meta, ctx)
    assert decision.decision == "ALLOW"
    assert decision.missing_capabilities == []


def test_policy_engine_deny_missing_capability() -> None:
    """Verify tool execution is DENIED when required capabilities are missing."""
    pe = PolicyEngine()
    meta = ToolMetadata(
        name="delete_database",
        description="Deletes database.",
        required_capabilities=["db:admin", "system:sudo"],
    )
    ctx = ExecutionContext(granted_capabilities={"file:read"})
    decision = pe.evaluate_tool_execution(meta, ctx)
    assert decision.decision == "DENY"
    assert "db:admin" in decision.missing_capabilities
    assert "system:sudo" in decision.missing_capabilities


def test_policy_engine_requires_confirmation_unconfirmed() -> None:
    """Verify destructive tool without user_confirmed flag returns REQUIRES_CONFIRMATION."""
    pe = PolicyEngine()
    meta = ToolMetadata(
        name="format_disk",
        description="Formats disk.",
        destructive=True,
        required_capabilities=["disk:format"],
    )
    ctx = ExecutionContext(granted_capabilities={"disk:format"})
    decision = pe.evaluate_tool_execution(meta, ctx)
    assert decision.decision == "REQUIRES_CONFIRMATION"


def test_policy_engine_allow_confirmed_destructive_tool() -> None:
    """Verify destructive tool with environment user_confirmed=True is ALLOWED."""
    pe = PolicyEngine()
    meta = ToolMetadata(
        name="format_disk",
        description="Formats disk.",
        destructive=True,
        required_capabilities=["disk:format"],
    )
    ctx = ExecutionContext(
        granted_capabilities={"disk:format"},
        environment={"user_confirmed": True},
    )
    decision = pe.evaluate_tool_execution(meta, ctx)
    assert decision.decision == "ALLOW"
