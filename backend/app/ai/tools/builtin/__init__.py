"""Built-in System and Desktop Automation Tools for ULTRON.

Provides read-only system awareness, safe window/application control, controlled file operations,
and bounded mouse/keyboard input automation tools.
"""

from app.ai.tools.builtin.application_close import ApplicationCloseTool
from app.ai.tools.builtin.application_open import ApplicationOpenTool
from app.ai.tools.builtin.file_create import FileCreateTool
from app.ai.tools.builtin.file_delete import FileDeleteTool
from app.ai.tools.builtin.file_modify import FileModifyTool
from app.ai.tools.builtin.file_read import FileReadTool
from app.ai.tools.builtin.keyboard import KeyboardTool
from app.ai.tools.builtin.mouse import MouseTool
from app.ai.tools.builtin.process_info import GetProcessInfoTool
from app.ai.tools.builtin.system_info import GetSystemInfoTool
from app.ai.tools.builtin.system_resources import GetSystemResourcesTool
from app.ai.tools.builtin.window_info import GetWindowInfoTool
from app.ai.tools.registry import ToolRegistry, tool_registry

__all__ = [
    "GetSystemInfoTool",
    "GetSystemResourcesTool",
    "GetProcessInfoTool",
    "GetWindowInfoTool",
    "ApplicationOpenTool",
    "ApplicationCloseTool",
    "FileReadTool",
    "FileCreateTool",
    "FileModifyTool",
    "FileDeleteTool",
    "MouseTool",
    "KeyboardTool",
    "register_builtin_system_tools",
]


def register_builtin_system_tools(registry: ToolRegistry | None = None) -> None:
    """Registers all built-in system and desktop tools into the specified ToolRegistry (or global singleton)."""
    target_registry = registry or tool_registry

    tools = [
        GetSystemInfoTool(),
        GetSystemResourcesTool(),
        GetProcessInfoTool(),
        GetWindowInfoTool(),
        ApplicationOpenTool(),
        ApplicationCloseTool(),
        FileReadTool(),
        FileCreateTool(),
        FileModifyTool(),
        FileDeleteTool(),
        MouseTool(),
        KeyboardTool(),
    ]

    for tool in tools:
        try:
            target_registry.register(tool)
        except Exception:
            pass  # Already registered
