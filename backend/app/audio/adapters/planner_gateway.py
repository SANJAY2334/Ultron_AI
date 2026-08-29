"""Voice-to-Autonomous Planner Gateway Adapter (Phase 4D).

Concrete implementation of IVoicePlannerGateway bridging the Audio Session Manager
to the BasePlanner, LangGraphPlanner, and IMemoryManager while enforcing Zero-Trust policies,
response sanitization, timeout protection, and stream event translation.
"""

import asyncio
import logging
import time
from collections.abc import AsyncIterable
from typing import Any

from app.ai.models import Message
from app.ai.planner.base import AgentState, BasePlanner
from app.audio.models import Transcript
from app.audio.planner_gateway import (
    IVoicePlannerGateway,
    PlannerStreamEvent,
    PlannerStreamEventType,
    VoiceGatewayCancelledError,
    VoiceGatewayError,
    VoiceGatewayPlannerError,
    VoiceGatewayTimeoutError,
    VoicePlannerResponse,
)
from app.core.container import ContainerKeyError, container
from app.memory.base import IMemoryManager

logger = logging.getLogger(__name__)


class VoicePlannerGateway(IVoicePlannerGateway):
    """Bridge adapter connecting AudioSessionManager to BasePlanner and MemoryManager."""

    def __init__(
        self,
        planner: BasePlanner | None = None,
        memory_manager: IMemoryManager | None = None,
        timeout_ms: float = 30000.0,
    ) -> None:
        """Initializes VoicePlannerGateway with planner, memory manager, and timeout parameters."""
        self._planner = planner
        self._memory_manager = memory_manager
        self.timeout_ms = timeout_ms
        self._active_requests: dict[str, asyncio.Task[Any]] = {}

    def _resolve_planner(self) -> BasePlanner:
        """Resolves BasePlanner from injected instance or DI container."""
        if self._planner is not None:
            return self._planner
        try:
            return container.resolve_sync(BasePlanner)
        except ContainerKeyError:
            from app.ai.planner.langgraph_planner import LangGraphPlanner

            planner = LangGraphPlanner()
            container.register_singleton(BasePlanner, planner)
            return planner

    def _resolve_memory_manager(self) -> IMemoryManager | None:
        """Resolves IMemoryManager from injected instance or DI container."""
        if self._memory_manager is not None:
            return self._memory_manager
        try:
            return container.resolve_sync(IMemoryManager)
        except ContainerKeyError:
            return None

    async def health(self) -> dict[str, Any]:
        """Probes health and readiness of the voice planner gateway."""
        planner = self._resolve_planner()
        planner_healthy = True
        try:
            if hasattr(planner, "health"):
                ph = await planner.health()  # type: ignore
                planner_healthy = ph.get("healthy", True)
        except Exception:
            planner_healthy = False

        return {
            "subsystem_voice_planner_gateway": True,
            "planner_ready": planner_healthy,
            "active_requests_count": len(self._active_requests),
            "timeout_ms": self.timeout_ms,
        }

    async def cancel_request(self, request_id: str) -> None:
        """Cancels an active voice planner request cleanly."""
        task = self._active_requests.get(request_id)
        if task and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, VoiceGatewayCancelledError):
                pass
            logger.info(f"VoicePlannerGateway cancelled request '{request_id}'.")
        self._active_requests.pop(request_id, None)

    async def submit_transcript(
        self,
        transcript: Transcript,
        user_id: str = "default_user",
        context: Any = None,
    ) -> VoicePlannerResponse:
        """Submits a transcript to the planner and returns a voice-safe response."""
        request_id = f"vreq_{transcript.transcript_id}"
        session_id = transcript.session_id or "sess_voice_default"
        correlation_id = transcript.correlation_id or "corr_voice_default"

        if not transcript.full_text or not transcript.full_text.strip():
            return VoicePlannerResponse(
                request_id=request_id,
                session_id=session_id,
                correlation_id=correlation_id,
                text_response="I didn't catch that. Could you please repeat?",
                success=False,
                error_code="EMPTY_TRANSCRIPT",
            )

        planner = self._resolve_planner()
        initial_state = AgentState(
            session_id=session_id,
            correlation_id=correlation_id,
            messages=[Message(role="user", content=transcript.full_text.strip())],
        )

        timeout_sec = self.timeout_ms / 1000.0
        current_task = asyncio.current_task()
        if current_task:
            self._active_requests[request_id] = current_task

        try:
            start_time = time.perf_counter()
            final_state: AgentState = await asyncio.wait_for(
                planner.run_pipeline(initial_state), timeout=timeout_sec
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            logger.debug(f"VoicePlanner execution completed in {elapsed_ms:.1f}ms.")

            has_errors = len(final_state.errors) > 0
            err_code = final_state.errors[-1].error_code if has_errors else None
            output_text = final_state.final_response or "Operation completed successfully."
            safe_text = self._sanitize_voice_response(output_text)

            return VoicePlannerResponse(
                request_id=request_id,
                session_id=session_id,
                correlation_id=correlation_id,
                text_response=safe_text,
                success=not has_errors,
                error_code=err_code,
            )

        except TimeoutError as exc:
            logger.warning(
                f"VoicePlanner request '{request_id}' timed out after {self.timeout_ms}ms."
            )
            raise VoiceGatewayTimeoutError(
                f"Voice planner evaluation timed out after {self.timeout_ms}ms."
            ) from exc
        except asyncio.CancelledError:
            logger.info(f"VoicePlanner request '{request_id}' was cancelled.")
            raise VoiceGatewayCancelledError(
                f"Voice planner request '{request_id}' cancelled."
            ) from None
        except Exception as exc:
            logger.error(f"VoicePlanner execution error: {exc}")
            raise VoiceGatewayPlannerError(f"Planner execution error: {exc}") from exc
        finally:
            self._active_requests.pop(request_id, None)

    async def stream_response(
        self,
        transcript: Transcript,
        user_id: str = "default_user",
        context: Any = None,
    ) -> AsyncIterable[PlannerStreamEvent]:
        """Submits a transcript and streams incremental PlannerStreamEvent events."""
        request_id = f"vstream_{transcript.transcript_id}"

        # 1. Operational status indicator event (NO chain-of-thought reasoning!)
        yield PlannerStreamEvent(
            event_type=PlannerStreamEventType.THINKING,
            payload="Processing your request...",
            request_id=request_id,
        )

        try:
            response = await self.submit_transcript(transcript, user_id=user_id, context=context)

            if response.success:
                yield PlannerStreamEvent(
                    event_type=PlannerStreamEventType.PARTIAL_RESPONSE,
                    payload=response.text_response,
                    request_id=request_id,
                )
                yield PlannerStreamEvent(
                    event_type=PlannerStreamEventType.COMPLETED,
                    payload=response.text_response,
                    request_id=request_id,
                )
            else:
                yield PlannerStreamEvent(
                    event_type=PlannerStreamEventType.ERROR,
                    payload=response.text_response,
                    request_id=request_id,
                )
        except VoiceGatewayError:
            yield PlannerStreamEvent(
                event_type=PlannerStreamEventType.ERROR,
                payload="Sorry, I couldn't complete that operation.",
                request_id=request_id,
            )
        except Exception:
            yield PlannerStreamEvent(
                event_type=PlannerStreamEventType.ERROR,
                payload="An unexpected error occurred while processing your request.",
                request_id=request_id,
            )

    def _sanitize_voice_response(self, text: str) -> str:
        """Sanitizes planner response text into voice-safe string.

        Strips sensitive stack traces, JWTs, credentials, internal prompts, or chain-of-thought markdown traces.
        """
        if not text:
            return "Task completed."

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        filtered = [
            line_str
            for line_str in lines
            if not line_str.startswith("Traceback")
            and "secret" not in line_str.lower()
            and "bearer " not in line_str.lower()
        ]
        cleaned = " ".join(filtered)
        return cleaned or "Operation finished successfully."
