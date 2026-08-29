"""ULTRON Phase 1 -> 4G.4 Comprehensive Acceptance Test Runner."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend")))

import asyncio
from app.ai.tools.base import ExecutionContext
from app.ai.tools.registry import ToolRegistry
from app.ai.tools.executor import ToolExecutor
from app.ai.tools.builtin.system_info import GetSystemInfoTool
from app.ai.tools.builtin.file_read import FileReadTool
from app.ai.tools.builtin.file_delete import FileDeleteTool
from app.desktop.models import ActionClassification, DesktopAction
from app.security.policy import PolicyEngine
from app.security.safety import SafetyInterlock
from app.security.path_policy import PathPolicy
from app.memory.vector_store import InMemoryVectorStore
from app.memory.working import get_working_memory


async def run_full_acceptance_audit():
    print("================== ULTRON ACCEPTANCE TEST EXECUTION ==================")

    # Setup temporary test file in workspace root
    test_target_file = os.path.abspath("acceptance_test_file.tmp")
    with open(test_target_file, "w") as f:
        f.write("temporary_test_content")

    try:
        # 1. TOOL REGISTRY & BUILTIN TOOLS
        print("\n--- [H] Tool Execution Verification ---")
        sys_tool = GetSystemInfoTool()
        file_read_tool = FileReadTool()
        file_del_tool = FileDeleteTool()

        reg = ToolRegistry()
        reg.register(sys_tool)
        reg.register(file_read_tool)
        reg.register(file_del_tool)

        ctx_read_only = ExecutionContext(
            session_id="test_sess",
            user_id="test_user",
            granted_capabilities=["system:read", "file:read"],
            user_confirmed=False
        )

        policy = PolicyEngine()
        path_pol = PathPolicy(allowed_directories=[os.path.abspath(".")])
        safety = SafetyInterlock(path_pol=path_pol)
        executor = ToolExecutor(registry=reg, policy=policy)

        # Executing system:read tool
        res_sys = await executor.execute(sys_tool.metadata.name, {}, context=ctx_read_only)
        print(f"System Info Tool Execution: Success={res_sys.success}")
        assert res_sys.success is True

        # 2. SAFETY INTERLOCK & POLICY ENGINE
        print("\n--- [I] Safety Interlock Verification ---")
        # Attempt destructive file delete without permission
        res_del_unauth = await executor.execute(file_del_tool.metadata.name, {"path": test_target_file}, context=ctx_read_only)
        print(f"Destructive Action (No Capability): Success={res_del_unauth.success}, Error={res_del_unauth.error_message}")
        assert res_del_unauth.success is False

        # Attempt destructive file delete with file:delete capability but user_confirmed=False
        ctx_del_unconf = ExecutionContext(
            session_id="test_sess",
            user_id="test_user",
            granted_capabilities=["file:delete"],
            user_confirmed=False,
            environment={"user_confirmed": False}
        )
        res_del_unconf = await executor.execute(file_del_tool.metadata.name, {"path": test_target_file}, context=ctx_del_unconf)
        print(f"Destructive Action (Unconfirmed): Success={res_del_unconf.success}, Error={res_del_unconf.error_message}")
        assert res_del_unconf.success is False

        # Direct SafetyInterlock evaluation
        dest_action = DesktopAction(
            action_id="act_001",
            action_type="delete_file",
            capability="file:delete",
            classification=ActionClassification.DESTRUCTIVE,
            confirmation_required=True,
            target=test_target_file,
            parameters={"path": test_target_file}
        )
        safety_dec_unconf = safety.evaluate(dest_action, ctx_del_unconf)
        print(f"Safety Interlock Evaluation (Unconfirmed): {safety_dec_unconf.decision}")
        assert safety_dec_unconf.decision == "REQUIRES_CONFIRMATION"

        # Confirmed execution context
        ctx_del_conf = ExecutionContext(
            session_id="test_sess",
            user_id="test_user",
            granted_capabilities=["file:delete"],
            user_confirmed=True,
            environment={"user_confirmed": True}
        )
        safety_dec_conf = safety.evaluate(dest_action, ctx_del_conf)
        print(f"Safety Interlock Evaluation (Confirmed): {safety_dec_conf.decision}")
        assert safety_dec_conf.decision == "ALLOW"

        # 3. MEMORY PIPELINE VERIFICATION
        print("\n--- [F] Memory Pipeline Verification ---")
        vs = InMemoryVectorStore()
        print("Vector store provider instantiated:", type(vs).__name__)

        wm = get_working_memory()
        print("Working memory provider resolved:", type(wm).__name__)

        # 4. ZERO-TRUST INVARIANT TESTS
        print("\n--- [J] Zero-Trust Invariant Verification ---")
        # Invariant: DesktopAction without granted capability cannot bypass PolicyEngine
        unauth_action = DesktopAction(
            action_id="act_002",
            action_type="shell_exec",
            capability="shell:exec",
            classification=ActionClassification.DESTRUCTIVE,
            target="powershell.exe",
            parameters={"cmd": "whoami"}
        )
        safety_dec_unauth = safety.evaluate(unauth_action, ctx_del_conf)
        print(f"Zero-Trust Unauthorized Action (Capability check): {safety_dec_unauth.decision}")
        assert safety_dec_unauth.decision == "DENY"

        print("\n================== ALL ACCEPTANCE SUITES VERIFIED ==================")

    finally:
        if os.path.exists(test_target_file):
            try:
                os.remove(test_target_file)
            except Exception:
                pass

if __name__ == "__main__":
    asyncio.run(run_full_acceptance_audit())
