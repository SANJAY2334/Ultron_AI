"""Desktop Automation Domain Models & Action Classifications.

Defines framework-agnostic models for windows, inputs, file operations, process operations,
desktop action metadata, and action safety classifications without PyAutoGUI or OS dependencies.
"""

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


class ActionClassification(StrEnum):
    """Action safety classifications for PolicyEngine and SafetyInterlock evaluation."""

    READ_ONLY = "READ_ONLY"
    NON_DESTRUCTIVE = "NON_DESTRUCTIVE"
    MUTATING = "MUTATING"
    DESTRUCTIVE = "DESTRUCTIVE"


class WindowInfo(BaseModel):
    """Model representing an operating system window."""

    window_id: str = Field(description="Unique window handle or identifier")
    title: str = Field(description="Window title string")
    process_name: str = Field(description="Associated process name")
    pid: int = Field(description="Associated Process ID")
    x: int = Field(default=0, description="Window top-left X coordinate")
    y: int = Field(default=0, description="Window top-left Y coordinate")
    width: int = Field(default=800, description="Window width in pixels")
    height: int = Field(default=600, description="Window height in pixels")
    is_active: bool = Field(default=False, description="True if window currently has focus")
    is_minimized: bool = Field(default=False, description="True if window is minimized")
    is_maximized: bool = Field(default=False, description="True if window is maximized")


class MouseAction(BaseModel):
    """Model representing a mouse input operation."""

    action_type: Literal["move", "click", "double_click", "right_click", "drag", "scroll"] = Field(
        description="Type of mouse action"
    )
    x: int = Field(ge=0, description="Target X coordinate")
    y: int = Field(ge=0, description="Target Y coordinate")
    button: Literal["left", "right", "middle"] = Field(default="left", description="Mouse button")
    clicks: int = Field(default=1, ge=1, le=10, description="Number of click iterations")
    duration: float = Field(default=0.1, ge=0.0, le=5.0, description="Movement duration in seconds")


class KeyboardAction(BaseModel):
    """Model representing a keyboard input operation."""

    action_type: Literal["type_text", "press_key", "hotkey"] = Field(
        description="Type of keyboard action"
    )
    text: str | None = Field(default=None, description="Text payload for typing")
    keys: list[str] = Field(default_factory=list, description="Keys or hotkey sequence")
    interval: float = Field(default=0.01, ge=0.0, le=1.0, description="Inter-key typing delay")


class FileOperation(BaseModel):
    """Model representing a filesystem operation."""

    operation_type: Literal["read", "create", "modify", "delete"] = Field(
        description="Type of filesystem operation"
    )
    path: str = Field(description="Target file or directory path")
    content: str | None = Field(default=None, description="File text content for write/modify")
    bytes_size: int | None = Field(default=None, description="File size in bytes")


class ProcessOperation(BaseModel):
    """Model representing a process control operation."""

    operation_type: Literal["list", "inspect", "open", "close"] = Field(
        description="Type of process operation"
    )
    app_name: str | None = Field(default=None, description="Application identifier or name")
    app_path: str | None = Field(default=None, description="Validated executable binary path")
    pid: int | None = Field(default=None, description="Target Process ID")
    arguments: list[str] = Field(default_factory=list, description="Command line arguments")


class DesktopAction(BaseModel):
    """Canonical desktop action payload carrying safety metadata and correlation context."""

    action_id: str = Field(description="Unique action identifier")
    action_type: str = Field(description="Action functional identifier (e.g. 'window.close')")
    classification: ActionClassification = Field(
        description="Safety classification (READ_ONLY, NON_DESTRUCTIVE, MUTATING, DESTRUCTIVE)"
    )
    capability: str = Field(description="Required capability string (e.g. 'desktop:window_open')")
    confirmation_required: bool = Field(
        default=False, description="True if action requires user confirmation"
    )
    target: str = Field(description="Action target description or path")
    timeout_seconds: float = Field(default=10.0, ge=0.1, description="Execution timeout limit")
    correlation_id: str | None = Field(default=None, description="Tracing correlation ID")
    planner_id: str | None = Field(default=None, description="Planner engine ID")
    user_id: str | None = Field(default=None, description="Requesting user ID")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Custom action metadata")


class ActionResult(BaseModel):
    """Result object returned by DesktopAdapter execution."""

    action_id: str = Field(description="Target action identifier")
    success: bool = Field(description="True if action executed cleanly")
    output: Any = Field(default=None, description="Structured execution payload")
    error_message: str | None = Field(default=None, description="Sanitized error description")
    execution_time_ms: float = Field(default=0.0, ge=0.0, description="Duration in milliseconds")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Audit telemetry")


class SafetyDecision(BaseModel):
    """Safety evaluation outcome from SafetyInterlock."""

    decision: Literal["ALLOW", "DENY", "REQUIRES_CONFIRMATION"] = Field(
        description="Safety decision outcome"
    )
    reason: str = Field(description="Detailed explanation of safety decision")
    missing_capabilities: list[str] = Field(
        default_factory=list, description="Missing capability permissions"
    )
    path_violations: list[str] = Field(
        default_factory=list, description="Path policy security violations if any"
    )
