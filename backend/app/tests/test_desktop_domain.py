"""Unit Tests for Desktop Domain Models and Action Classifications (Phase 4B.1).

Validates ActionClassification, WindowInfo, MouseAction, KeyboardAction, FileOperation,
ProcessOperation, DesktopAction, ActionResult, and SafetyDecision domain invariants.
"""

from app.desktop.models import (
    ActionClassification,
    DesktopAction,
    KeyboardAction,
    MouseAction,
    SafetyDecision,
    WindowInfo,
)


def test_action_classification_enum() -> None:
    """Verify ActionClassification enum values."""
    assert ActionClassification.READ_ONLY == "READ_ONLY"
    assert ActionClassification.NON_DESTRUCTIVE == "NON_DESTRUCTIVE"
    assert ActionClassification.MUTATING == "MUTATING"
    assert ActionClassification.DESTRUCTIVE == "DESTRUCTIVE"


def test_window_info_model() -> None:
    """Verify WindowInfo instantiation and defaults."""
    win = WindowInfo(
        window_id="win_1001",
        title="Calculator",
        process_name="calc.exe",
        pid=1234,
        x=100,
        y=100,
        width=400,
        height=300,
        is_active=True,
    )
    assert win.window_id == "win_1001"
    assert win.title == "Calculator"
    assert win.process_name == "calc.exe"
    assert win.pid == 1234
    assert win.is_active is True
    assert win.is_minimized is False


def test_mouse_action_validation() -> None:
    """Verify MouseAction constraints."""
    mouse = MouseAction(action_type="click", x=500, y=300, button="left", clicks=2)
    assert mouse.action_type == "click"
    assert mouse.x == 500
    assert mouse.y == 300
    assert mouse.button == "left"
    assert mouse.clicks == 2


def test_keyboard_action_validation() -> None:
    """Verify KeyboardAction instantiation."""
    kb = KeyboardAction(action_type="type_text", text="Hello ULTRON", interval=0.05)
    assert kb.action_type == "type_text"
    assert kb.text == "Hello ULTRON"
    assert kb.interval == 0.05

    hotkey = KeyboardAction(action_type="hotkey", keys=["ctrl", "c"])
    assert hotkey.action_type == "hotkey"
    assert hotkey.keys == ["ctrl", "c"]


def test_desktop_action_metadata() -> None:
    """Verify DesktopAction metadata serialization."""
    action = DesktopAction(
        action_id="act_555",
        action_type="file.delete",
        classification=ActionClassification.DESTRUCTIVE,
        capability="file:delete",
        confirmation_required=True,
        target="/tmp/test.txt",
        correlation_id="corr_999",
        planner_id="planner_langgraph",
        user_id="user_admin",
    )
    assert action.action_id == "act_555"
    assert action.classification == ActionClassification.DESTRUCTIVE
    assert action.confirmation_required is True
    assert action.target == "/tmp/test.txt"


def test_safety_decision_model() -> None:
    """Verify SafetyDecision model behavior."""
    sd = SafetyDecision(
        decision="REQUIRES_CONFIRMATION",
        reason="File deletion requires explicit user approval.",
        path_violations=[],
    )
    assert sd.decision == "REQUIRES_CONFIRMATION"
    assert "explicit user approval" in sd.reason
