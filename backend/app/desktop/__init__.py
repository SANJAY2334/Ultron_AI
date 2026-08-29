"""Desktop Automation Subsystem for ULTRON.

Provides domain models, action safety classifications, desktop adapter interface contracts,
and PyAutoGUI desktop interaction adapters.
"""

from app.desktop.base import IDesktopAdapter
from app.desktop.models import (
    ActionClassification,
    ActionResult,
    DesktopAction,
    FileOperation,
    KeyboardAction,
    MouseAction,
    ProcessOperation,
    SafetyDecision,
    WindowInfo,
)

__all__ = [
    "ActionClassification",
    "WindowInfo",
    "MouseAction",
    "KeyboardAction",
    "FileOperation",
    "ProcessOperation",
    "DesktopAction",
    "ActionResult",
    "SafetyDecision",
    "IDesktopAdapter",
]
