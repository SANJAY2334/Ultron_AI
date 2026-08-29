"""Unit Tests for Safety Interlock and Path Policy (Phase 4B.5).

Validates SafetyInterlock decision logic (ALLOW, DENY, REQUIRES_CONFIRMATION),
capability permission checks, path traversal protection, restricted file protection,
and classification rules (READ_ONLY, NON_DESTRUCTIVE, MUTATING, DESTRUCTIVE).
"""

import pytest

from app.ai.tools.base import ExecutionContext
from app.desktop.models import ActionClassification, DesktopAction
from app.security.path_policy import PathPolicy
from app.security.safety import SafetyInterlock


@pytest.fixture
def path_policy_obj(tmp_path) -> PathPolicy:
    return PathPolicy(allowed_directories=[str(tmp_path)])


@pytest.fixture
def safety(path_policy_obj: PathPolicy) -> SafetyInterlock:
    return SafetyInterlock(path_pol=path_policy_obj)


def test_path_policy_traversal_defense(path_policy_obj: PathPolicy, tmp_path) -> None:
    """Verify PathPolicy blocks path traversal outside allowed directory."""
    traversal_path = str(tmp_path / ".." / ".." / "etc" / "passwd")
    res = path_policy_obj.validate_path(traversal_path)
    assert res.is_valid is False
    assert "OUTSIDE_ALLOWED_DIRECTORY" in res.violations


def test_path_policy_restricted_secrets_defense(path_policy_obj: PathPolicy, tmp_path) -> None:
    """Verify PathPolicy blocks access to .env, credential files, and private keys."""
    env_file = str(tmp_path / ".env")
    res_env = path_policy_obj.validate_path(env_file)
    assert res_env.is_valid is False
    assert "RESTRICTED_SENSITIVE_FILE" in res_env.violations

    key_file = str(tmp_path / "id_rsa")
    res_key = path_policy_obj.validate_path(key_file)
    assert res_key.is_valid is False
    assert "RESTRICTED_SENSITIVE_FILE" in res_key.violations


def test_safety_interlock_read_only_allowed(safety: SafetyInterlock) -> None:
    """Verify READ_ONLY action is automatically allowed when capability is granted."""
    action = DesktopAction(
        action_id="act_read_1",
        action_type="window.inspect",
        classification=ActionClassification.READ_ONLY,
        capability="desktop:window_read",
        target="Desktop Windows",
    )
    ctx = ExecutionContext(granted_capabilities={"desktop:window_read"})
    dec = safety.evaluate(action, ctx)
    assert dec.decision == "ALLOW"


def test_safety_interlock_missing_capability_denied(safety: SafetyInterlock) -> None:
    """Verify SafetyInterlock returns DENY if required capability is missing."""
    action = DesktopAction(
        action_id="act_read_2",
        action_type="window.inspect",
        classification=ActionClassification.READ_ONLY,
        capability="desktop:window_read",
        target="Desktop Windows",
    )
    ctx = ExecutionContext(granted_capabilities=set())  # No capabilities
    dec = safety.evaluate(action, ctx)
    assert dec.decision == "DENY"
    assert "desktop:window_read" in dec.missing_capabilities


def test_safety_interlock_destructive_requires_confirmation(
    safety: SafetyInterlock, tmp_path
) -> None:
    """Verify DESTRUCTIVE action requires explicit user confirmation."""
    target_file = str(tmp_path / "data.txt")
    with open(target_file, "w") as f:
        f.write("sample content")

    action = DesktopAction(
        action_id="act_del_1",
        action_type="file.delete",
        classification=ActionClassification.DESTRUCTIVE,
        capability="file:delete",
        target=target_file,
    )
    ctx_unconfirmed = ExecutionContext(granted_capabilities={"file:delete"})

    # Without confirmation
    dec1 = safety.evaluate(action, ctx_unconfirmed)
    assert dec1.decision == "REQUIRES_CONFIRMATION"

    # With explicit confirmation
    ctx_confirmed = ExecutionContext(
        granted_capabilities={"file:delete"},
        environment={"user_confirmed": True},
    )
    dec2 = safety.evaluate(action, ctx_confirmed)
    assert dec2.decision == "ALLOW"
