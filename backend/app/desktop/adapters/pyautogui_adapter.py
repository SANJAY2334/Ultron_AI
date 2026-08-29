"""PyAutoGUI Desktop Interaction Adapter.

Concrete implementation of IDesktopAdapter wrapping PyAutoGUI and OS process APIs.
Enforces coordinate bounds, failsafe interlocks, typing length bounds, and headless fallback.
"""

import logging
import os
import subprocess
import time

from app.desktop.base import IDesktopAdapter
from app.desktop.models import (
    ActionResult,
    KeyboardAction,
    MouseAction,
    ProcessOperation,
    WindowInfo,
)

logger = logging.getLogger(__name__)

# Import pyautogui with graceful fallback for headless environments
try:
    import pyautogui  # type: ignore[import-not-found, import-untyped]

    pyautogui.FAILSAFE = True
    HAS_PYAUTOGUI = True
except Exception:
    pyautogui = None  # type: ignore[assignment]
    HAS_PYAUTOGUI = False

# Import psutil for process management
try:
    import psutil  # type: ignore[import-not-found, import-untyped]

    HAS_PSUTIL = True
except ImportError:
    psutil = None  # type: ignore[assignment]
    HAS_PSUTIL = False


class PyAutoGUIAdapter(IDesktopAdapter):
    """Adapter encapsulating PyAutoGUI and OS process interactions."""

    def __init__(self) -> None:
        """Initializes PyAutoGUIAdapter."""
        self._has_gui = HAS_PYAUTOGUI

    def _get_screen_size(self) -> tuple[int, int]:
        """Returns active screen dimensions (width, height)."""
        if self._has_gui and pyautogui is not None:
            try:
                return pyautogui.size()
            except Exception:
                pass
        return (1920, 1080)

    async def get_windows(self) -> list[WindowInfo]:
        """Retrieves active desktop window information using psutil/OS API."""
        windows: list[WindowInfo] = []

        if HAS_PSUTIL:
            for proc in psutil.process_iter(["pid", "name", "status"]):
                try:
                    p_info = proc.info
                    p_name = str(p_info.get("name") or "")
                    if p_name and p_name.lower().endswith((".exe", "app")):
                        windows.append(
                            WindowInfo(
                                window_id=f"win_{p_info['pid']}",
                                title=f"{p_name} (PID: {p_info['pid']})",
                                process_name=p_name,
                                pid=p_info["pid"],
                                is_active=False,
                            )
                        )
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue

        return windows[:50]

    async def open_application(
        self, app_path: str, args: list[str] | None = None
    ) -> ProcessOperation:
        """Launches a desktop application via subprocess without shell."""
        cmd = [app_path] + (args or [])
        proc = subprocess.Popen(cmd)

        return ProcessOperation(
            operation_type="open",
            app_name=os.path.basename(app_path),
            app_path=app_path,
            pid=proc.pid,
            arguments=args or [],
        )

    async def close_application(self, pid: int | None = None, app_name: str | None = None) -> bool:
        """Closes a specific application by PID or app_name using psutil."""
        if not HAS_PSUTIL:
            return False

        closed_any = False

        if pid is not None:
            try:
                p = psutil.Process(pid)
                p.terminate()
                closed_any = True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        elif app_name:
            clean_name = app_name.lower().strip()
            for proc in psutil.process_iter(["pid", "name"]):
                try:
                    if proc.info["name"] and proc.info["name"].lower() == clean_name:
                        proc.terminate()
                        closed_any = True
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

        return closed_any

    async def execute_mouse(self, action: MouseAction) -> ActionResult:
        """Executes a bounded mouse operation."""
        start_time = time.perf_counter()
        screen_w, screen_h = self._get_screen_size()

        # Bounds validation
        if not (0 <= action.x <= screen_w and 0 <= action.y <= screen_h):
            return ActionResult(
                action_id=f"mouse_{int(time.time())}",
                success=False,
                error_message=f"Mouse Error: Coordinates ({action.x}, {action.y}) are outside screen bounds ({screen_w}x{screen_h}).",
            )

        if not self._has_gui or pyautogui is None:
            # Fallback simulated mouse execution for headless environments
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return ActionResult(
                action_id=f"mouse_{int(time.time())}",
                success=True,
                output={
                    "mode": "simulated",
                    "action_type": action.action_type,
                    "target_x": action.x,
                    "target_y": action.y,
                },
                execution_time_ms=elapsed_ms,
            )

        try:
            if action.action_type == "move":
                pyautogui.moveTo(action.x, action.y, duration=action.duration)
            elif action.action_type == "click":
                pyautogui.click(
                    x=action.x,
                    y=action.y,
                    clicks=action.clicks,
                    button=action.button,
                    duration=action.duration,
                )
            elif action.action_type == "right_click":
                pyautogui.rightClick(x=action.x, y=action.y)
            elif action.action_type == "double_click":
                pyautogui.doubleClick(x=action.x, y=action.y)

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return ActionResult(
                action_id=f"mouse_{int(time.time())}",
                success=True,
                output={"action": action.action_type, "x": action.x, "y": action.y},
                execution_time_ms=elapsed_ms,
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return ActionResult(
                action_id=f"mouse_{int(time.time())}",
                success=False,
                error_message=f"Mouse execution exception: {exc}",
                execution_time_ms=elapsed_ms,
            )

    async def execute_keyboard(self, action: KeyboardAction) -> ActionResult:
        """Executes a bounded keyboard operation."""
        start_time = time.perf_counter()

        # Input bounding
        if action.text and len(action.text) > 500:
            return ActionResult(
                action_id=f"kb_{int(time.time())}",
                success=False,
                error_message="Keyboard Error: Typing text payload exceeds maximum limit of 500 characters.",
            )

        if len(action.keys) > 10:
            return ActionResult(
                action_id=f"kb_{int(time.time())}",
                success=False,
                error_message="Keyboard Error: Key sequence exceeds maximum limit of 10 keys.",
            )

        if not self._has_gui or pyautogui is None:
            # Fallback simulated keyboard execution
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return ActionResult(
                action_id=f"kb_{int(time.time())}",
                success=True,
                output={
                    "mode": "simulated",
                    "action_type": action.action_type,
                    "text_length": len(action.text) if action.text else 0,
                    "keys": action.keys,
                },
                execution_time_ms=elapsed_ms,
            )

        try:
            if action.action_type == "type_text" and action.text:
                pyautogui.write(action.text, interval=action.interval)
            elif action.action_type == "press_key" and action.keys:
                for k in action.keys:
                    pyautogui.press(k)
            elif action.action_type == "hotkey" and action.keys:
                pyautogui.hotkey(*action.keys)

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return ActionResult(
                action_id=f"kb_{int(time.time())}",
                success=True,
                output={"action": action.action_type},
                execution_time_ms=elapsed_ms,
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return ActionResult(
                action_id=f"kb_{int(time.time())}",
                success=False,
                error_message=f"Keyboard execution exception: {exc}",
                execution_time_ms=elapsed_ms,
            )
