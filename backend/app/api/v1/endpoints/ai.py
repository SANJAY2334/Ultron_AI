"""AI Autonomous Planner REST & SSE Streaming API Endpoints.

Provides secure HTTP REST and Server-Sent Events (SSE) streaming endpoints for interaction
with ULTRON's Autonomous Planner subsystem. Adheres to Clean Architecture and Security rules.
"""

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from app.ai.models import Message, ToolResultReference, Usage
from app.ai.planner.base import AgentState, BasePlanner, PlannerStatus
from app.ai.tools.base import ExecutionContext
from app.api.deps import get_current_user, get_planner

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ai", tags=["AI Autonomous Planner"])


class ChatRequest(BaseModel):
    """Payload schema for synchronous and streaming AI Chat requests."""

    message: str = Field(
        min_length=1,
        max_length=16000,
        description="User message prompt text",
    )
    session_id: str | None = Field(
        default=None,
        pattern=r"^[a-zA-Z0-9_-]{1,64}$",
        description="Optional session identifier (alphanumeric, max 64 chars)",
    )
    correlation_id: str | None = Field(
        default=None,
        pattern=r"^[a-zA-Z0-9_-]{1,64}$",
        description="Optional correlation tracing identifier",
    )

    @field_validator("message")
    @classmethod
    def validate_message_not_whitespace(cls, v: str) -> str:
        """Ensures prompt message is not empty or whitespace-only."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("Message prompt cannot be empty or whitespace-only.")
        return stripped


class ChatResponse(BaseModel):
    """Response schema for synchronous AI Chat requests."""

    response: str = Field(description="Synthesized AI assistant response message")
    planner_status: PlannerStatus = Field(description="Final planner lifecycle status")
    session_id: str = Field(description="Session identifier")
    correlation_id: str = Field(description="Correlation tracing identifier")
    planner_id: str = Field(description="Unique planner run identifier")
    usage: Usage | None = Field(default=None, description="Token usage metadata")
    tool_outputs: list[ToolResultReference] = Field(
        default_factory=list, description="Outputs from executed tools"
    )


def _format_sse_event(event_type: str, data: dict) -> str:
    """Formats event type and JSON payload into standard SSE wire string."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


@router.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Synchronous AI Autonomous Planner Chat Endpoint",
    description="Submits a message prompt to the Autonomous Planner engine and returns synthesized response.",
)
async def chat_endpoint(
    req: ChatRequest,
    x_correlation_id: str | None = Header(default=None, alias="X-Correlation-ID"),
    current_user: str = Depends(get_current_user),
    planner: BasePlanner = Depends(get_planner),
) -> ChatResponse:
    """Executes a synchronous multi-stage autonomous planner chat interaction."""
    start_time = time.perf_counter()

    correlation_id = req.correlation_id or x_correlation_id or f"corr_{uuid.uuid4().hex[:12]}"
    session_id = req.session_id or f"sess_{uuid.uuid4().hex[:12]}"

    logger.info(
        f"[{correlation_id}] Processing REST chat request for user '{current_user}' (session: {session_id})."
    )

    initial_state = AgentState(
        session_id=session_id,
        correlation_id=correlation_id,
        messages=[Message(role="user", content=req.message)],
        context=ExecutionContext(
            user_id=current_user,
            session_id=session_id,
            correlation_id=correlation_id,
        ),
    )

    try:
        final_state = await planner.run_pipeline(initial_state)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        logger.info(
            f"[{correlation_id}] REST chat completed in {latency_ms:.2f}ms with status {final_state.status.value}."
        )

        return ChatResponse(
            response=final_state.final_response or "Planner execution completed.",
            planner_status=final_state.status,
            session_id=final_state.session_id or session_id,
            correlation_id=final_state.correlation_id or correlation_id,
            planner_id=final_state.planner_id,
            tool_outputs=final_state.tool_outputs,
        )
    except Exception as exc:
        logger.error(
            f"[{correlation_id}] Autonomous Planner pipeline failed: {exc}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="AI Planner service encountered an operational error.",
        ) from None


@router.post(
    "/chat/stream",
    status_code=status.HTTP_200_OK,
    summary="Real SSE Streaming AI Autonomous Planner Chat Endpoint",
    description="Streams structured Server-Sent Events (SSE) during Autonomous Planner execution.",
)
async def chat_stream_endpoint(
    req: ChatRequest,
    request: Request,
    x_correlation_id: str | None = Header(default=None, alias="X-Correlation-ID"),
    current_user: str = Depends(get_current_user),
    planner: BasePlanner = Depends(get_planner),
) -> StreamingResponse:
    """Streams Server-Sent Events (SSE) representing Autonomous Planner execution lifecycle."""
    correlation_id = req.correlation_id or x_correlation_id or f"corr_{uuid.uuid4().hex[:12]}"
    session_id = req.session_id or f"sess_{uuid.uuid4().hex[:12]}"

    initial_state = AgentState(
        session_id=session_id,
        correlation_id=correlation_id,
        messages=[Message(role="user", content=req.message)],
        context=ExecutionContext(
            user_id=current_user,
            session_id=session_id,
            correlation_id=correlation_id,
        ),
    )

    async def event_generator() -> AsyncGenerator[str, None]:
        start_time = time.perf_counter()
        planner_id = initial_state.planner_id

        try:
            # Emit 1: start event
            yield _format_sse_event(
                "start",
                {
                    "planner_id": planner_id,
                    "correlation_id": correlation_id,
                    "session_id": session_id,
                },
            )

            # Check client disconnect
            if await request.is_disconnected():
                logger.warning(f"[{correlation_id}] Client disconnected before pipeline start.")
                return

            # Execute pipeline
            final_state = await planner.run_pipeline(initial_state)

            # Emit 2: tool_status events
            for output_ref in final_state.tool_outputs:
                if await request.is_disconnected():
                    logger.warning(
                        f"[{correlation_id}] Client disconnected during tool status emit."
                    )
                    return
                yield _format_sse_event(
                    "tool_status",
                    {
                        "tool_name": output_ref.tool_name,
                        "status": output_ref.status,
                        "execution_time_ms": output_ref.execution_time_ms,
                    },
                )

            # Emit 3: token/chunk event (simulated streaming response chunk)
            if final_state.final_response:
                yield _format_sse_event("token", {"chunk": final_state.final_response})

            # Emit 4: completion event
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            yield _format_sse_event(
                "completion",
                {
                    "response": final_state.final_response or "",
                    "planner_status": final_state.status.value,
                    "planner_id": planner_id,
                    "correlation_id": correlation_id,
                    "session_id": session_id,
                    "latency_ms": round(latency_ms, 2),
                },
            )

        except asyncio.CancelledError:
            logger.warning(f"[{correlation_id}] SSE stream cancelled due to client disconnect.")
            yield _format_sse_event(
                "error",
                {
                    "error_code": "CLIENT_DISCONNECTED",
                    "message": "Stream cancelled by client disconnect.",
                },
            )
        except Exception as exc:
            logger.error(f"[{correlation_id}] SSE streaming endpoint error: {exc}", exc_info=True)
            yield _format_sse_event(
                "error",
                {
                    "error_code": "PLANNER_STREAM_ERROR",
                    "message": "AI Planner streaming encountered an internal error.",
                },
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
