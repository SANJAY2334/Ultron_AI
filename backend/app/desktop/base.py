"""Abstract Desktop Adapter Interface.

Defines the framework-agnostic IDesktopAdapter contract.
Isolates PyAutoGUI and OS-specific window/input bindings behind an abstract interface contract.
"""

from abc import ABC, abstractmethod

from app.desktop.models import (
    ActionResult,
    KeyboardAction,
    MouseAction,
    ProcessOperation,
    WindowInfo,
)


class IDesktopAdapter(ABC):
    """Abstract interface for desktop environment automation adapters."""

    @abstractmethod
    async def get_windows(self) -> list[WindowInfo]:
        """Retrieves active desktop window information.

        Returns:
            list[WindowInfo]: List of active desktop windows.
        """

    @abstractmethod
    async def open_application(
        self, app_path: str, args: list[str] | None = None
    ) -> ProcessOperation:
        """Launches a validated desktop application.

        Args:
            app_path: Validated executable binary path.
            args: Optional command line argument strings.

        Returns:
            ProcessOperation: Result process details.
        """

    @abstractmethod
    async def close_application(self, pid: int | None = None, app_name: str | None = None) -> bool:
        """Closes a specific application by PID or validated app name.

        Args:
            pid: Optional Process ID.
            app_name: Optional validated application name string.

        Returns:
            bool: True if process closed successfully.
        """

    @abstractmethod
    async def execute_mouse(self, action: MouseAction) -> ActionResult:
        """Executes a bounded mouse operation.

        Args:
            action: Validated MouseAction domain payload.

        Returns:
            ActionResult: Execution result object.
        """

    @abstractmethod
    async def execute_keyboard(self, action: KeyboardAction) -> ActionResult:
        """Executes a bounded keyboard operation.

        Args:
            action: Validated KeyboardAction domain payload.

        Returns:
            ActionResult: Execution result object.
        """
