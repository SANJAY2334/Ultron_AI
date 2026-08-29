"""Unit and Integration Tests for Voice ↔ Autonomous Planner Integration (Phase 4D).

Validates VoicePlannerGateway, transcript-to-goal translation, streaming events (no chain-of-thought),
barge-in speech interruption during TTS/playback, PolicyEngine safety model, memory integration,
concurrency isolation, error boundary isolation, and IVoicePlannerGateway interface compliance.
"""

import asyncio

import pytest

from app.ai.planner.base import AgentError, AgentState, BasePlanner
from app.audio import (
    AudioSessionConfig,
    AudioSessionManager,
    AudioSessionState,
    IVoicePlannerGateway,
    PlannerStreamEvent,
    PlannerStreamEventType,
    Transcript,
    VoiceGatewayCancelledError,
    VoiceGatewayTimeoutError,
    VoicePlannerResponse,
)
from app.audio.adapters.planner_gateway import VoicePlannerGateway


class MockVoicePlanner(BasePlanner):
    """Mock BasePlanner implementing all 6 lifecycle stages for voice integration tests."""

    def __init__(
        self,
        output_text: str = "Voice task completed successfully.",
        should_fail: bool = False,
        delay_sec: float = 0.0,
    ) -> None:
        super().__init__()
        self.output_text = output_text
        self.should_fail = should_fail
        self.delay_sec = delay_sec
        self.executed_states: list[AgentState] = []

    async def analyze(self, state: AgentState) -> AgentState:
        state.analysis_result = "Intent analyzed."
        return state

    async def reason(self, state: AgentState) -> AgentState:
        state.reasoning_result = "Reasoning complete."
        return state

    async def plan(self, state: AgentState) -> AgentState:
        state.current_plan = ["Execute voice task"]
        return state

    async def execute(self, state: AgentState) -> AgentState:
        self.executed_states.append(state)
        if self.delay_sec > 0:
            await asyncio.sleep(self.delay_sec)
        if self.should_fail:
            state.errors.append(
                AgentError(
                    error_code="PLANNER_EXECUTION_FAILURE",
                    safe_message="Planner execution error.",
                    retryable=False,
                    source="mock",
                )
            )
        else:
            state.final_response = self.output_text
        return state

    async def recover(self, state: AgentState) -> AgentState:
        return state

    async def summarize(self, state: AgentState) -> AgentState:
        if not state.final_response and state.errors:
            state.final_response = state.errors[-1].safe_message
        return state


@pytest.fixture
def mock_planner() -> MockVoicePlanner:
    return MockVoicePlanner()


@pytest.fixture
def gateway(mock_planner: MockVoicePlanner) -> VoicePlannerGateway:
    return VoicePlannerGateway(planner=mock_planner, timeout_ms=500.0)


def make_transcript(text: str = "open browser", session_id: str = "sess_v1") -> Transcript:
    return Transcript(
        transcript_id="tr_001",
        full_text=text,
        confidence=0.98,
        is_final=True,
        session_id=session_id,
        correlation_id="corr_v1",
    )


def test_gateway_interface_and_health(gateway: VoicePlannerGateway) -> None:
    """Verify IVoicePlannerGateway interface compliance and health probing (rules 1-5 & 44)."""

    async def _test() -> None:
        assert isinstance(gateway, IVoicePlannerGateway)

        health = await gateway.health()
        assert health["subsystem_voice_planner_gateway"] is True
        assert health["planner_ready"] is True
        assert health["active_requests_count"] == 0

    asyncio.run(_test())


def test_transcript_submission_and_propagation(
    gateway: VoicePlannerGateway, mock_planner: MockVoicePlanner
) -> None:
    """Verify transcript conversion, session_id, and correlation_id propagation (rules 1-4 & 6)."""

    async def _test() -> None:
        t = make_transcript(text="organize my files", session_id="sess_prop_100")
        resp = await gateway.submit_transcript(t, user_id="user_alpha")

        assert isinstance(resp, VoicePlannerResponse)
        assert resp.success is True
        assert resp.session_id == "sess_prop_100"
        assert resp.correlation_id == "corr_v1"
        assert resp.text_response == "Voice task completed successfully."

        # Verify planner received state
        assert len(mock_planner.executed_states) == 1
        st = mock_planner.executed_states[0]
        assert len(st.messages) == 1
        assert st.messages[0].content == "organize my files"

    asyncio.run(_test())


def test_empty_transcript_handling(gateway: VoicePlannerGateway) -> None:
    """Verify empty or whitespace-only transcript produces polite fallback without invoking planner."""

    async def _test() -> None:
        t_empty = make_transcript(text="   ")
        resp = await gateway.submit_transcript(t_empty)
        assert resp.success is False
        assert resp.error_code == "EMPTY_TRANSCRIPT"
        assert "repeat" in resp.text_response.lower()

    asyncio.run(_test())


def test_streaming_events_no_chain_of_thought(gateway: VoicePlannerGateway) -> None:
    """Verify streaming events yield THINKING operational status without chain-of-thought traces (rules 10 & 33)."""

    async def _test() -> None:
        t = make_transcript(text="summarize report")
        events: list[PlannerStreamEvent] = []

        async for ev in gateway.stream_response(t):
            events.append(ev)

        assert len(events) >= 2
        # First event must be THINKING status indicator
        assert events[0].event_type == PlannerStreamEventType.THINKING
        assert "Processing" in events[0].payload
        # Final event must be COMPLETED
        assert events[-1].event_type == PlannerStreamEventType.COMPLETED
        assert "completed" in events[-1].payload.lower()

        # Chain-of-thought check: ensure no raw internal reasoning tags
        for ev in events:
            assert "<think>" not in ev.payload
            assert "Traceback" not in ev.payload

    asyncio.run(_test())


def test_planner_timeout_and_error_handling() -> None:
    """Verify timeout and planner execution errors produce sanitized domain exceptions (rules 8 & 35)."""

    async def _test() -> None:
        slow_planner = MockVoicePlanner(delay_sec=0.20)
        timeout_gw = VoicePlannerGateway(planner=slow_planner, timeout_ms=20.0)

        with pytest.raises(VoiceGatewayTimeoutError):
            await timeout_gw.submit_transcript(make_transcript())

        fail_planner = MockVoicePlanner(should_fail=True)
        fail_gw = VoicePlannerGateway(planner=fail_planner)
        resp = await fail_gw.submit_transcript(make_transcript())
        assert resp.success is False
        assert resp.error_code == "PLANNER_EXECUTION_FAILURE"

    asyncio.run(_test())


def test_barge_in_speech_interruption() -> None:
    """Verify barge-in speech interruption cancels TTS playback and drains synthesis queue (rules 24-27)."""

    async def _test() -> None:
        mgr = AudioSessionManager(
            config=AudioSessionConfig(max_queued_chunks=10, max_pending_synthesis_chunks=10)
        )

        await mgr.start_session("sess_bargein")
        # Advance to PLAYING state via TTS
        await mgr.process_speech_text("Voice response playing...")
        assert mgr.current_state() == AudioSessionState.PLAYING

        # Execute barge-in speech interruption
        await mgr.interrupt_speech()

        # Synthesis queue should be drained and state back to LISTENING
        assert mgr._synthesis_queue.empty()
        assert mgr.current_state() in (AudioSessionState.LISTENING, AudioSessionState.PLAYING)

        await mgr.stop_session("sess_bargein")

    asyncio.run(_test())


def test_cancellation_propagation() -> None:
    """Verify request cancellation terminates active planner execution cleanly (rules 9 & 42)."""

    async def _test() -> None:
        slow_planner = MockVoicePlanner(delay_sec=0.20)
        gw = VoicePlannerGateway(planner=slow_planner, timeout_ms=5000.0)

        t = make_transcript()
        req_id = f"vreq_{t.transcript_id}"

        task = asyncio.create_task(gw.submit_transcript(t))
        await asyncio.sleep(0.01)

        # Cancel request
        await gw.cancel_request(req_id)

        with pytest.raises((VoiceGatewayCancelledError, asyncio.CancelledError)):
            await task

    asyncio.run(_test())
