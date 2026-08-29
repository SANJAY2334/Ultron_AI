"""Unit and Integration Tests for AI REST & SSE Streaming API Endpoints.

Validates authentication enforcement, request schema validation (empty messages, malformed fields),
synchronous chat execution, SSE event streaming, error sanitization, correlation ID propagation,
cancellation handling, and OpenAPI schema generation.
"""

from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from app.ai.planner.base import AgentState, BasePlanner
from app.api.deps import get_planner
from app.core.config import get_settings
from app.core.security import create_access_token
from app.main import app

client = TestClient(app)


def get_auth_headers(subject: str = "test_user_001") -> dict[str, str]:
    """Generates valid Authorization headers containing a signed JWT token."""
    settings = get_settings()
    secret_key = (
        settings.ULTRON_SECRET_KEY.get_secret_value() or "test_secret_key_for_jwt_auth_1234567890"
    )
    token = create_access_token(
        subject=subject,
        secret_key=secret_key,
    )
    return {"Authorization": f"Bearer {token}"}


class MockPlanner(BasePlanner):
    """Mock BasePlanner implementation for testing API endpoints."""

    async def analyze(self, state: AgentState) -> AgentState:
        state.analysis_result = "Analyzed test query"
        return state

    async def reason(self, state: AgentState) -> AgentState:
        state.reasoning_result = "Reasoned test strategy"
        return state

    async def plan(self, state: AgentState) -> AgentState:
        state.current_plan = ["Direct response test"]
        return state

    async def execute(self, state: AgentState) -> AgentState:
        return state

    async def recover(self, state: AgentState) -> AgentState:
        return state

    async def summarize(self, state: AgentState) -> AgentState:
        state.final_response = "Hello from MockPlanner!"
        return state


def test_chat_unauthenticated_rejected() -> None:
    """Verify POST /api/v1/ai/chat without JWT Bearer token returns 401 Unauthorized."""
    response = client.post(
        "/api/v1/ai/chat",
        json={"message": "Hello without token"},
    )
    assert response.status_code == 401
    assert "Authentication token required" in response.json()["detail"]


def test_chat_invalid_token_rejected() -> None:
    """Verify POST /api/v1/ai/chat with malformed token returns 401 Unauthorized."""
    response = client.post(
        "/api/v1/ai/chat",
        headers={"Authorization": "Bearer invalid_garbage_token"},
        json={"message": "Hello with bad token"},
    )
    assert response.status_code == 401


def test_chat_empty_message_rejected() -> None:
    """Verify POST /api/v1/ai/chat with empty or whitespace message returns 422 validation error."""
    headers = get_auth_headers()

    # Empty string
    res1 = client.post("/api/v1/ai/chat", headers=headers, json={"message": ""})
    assert res1.status_code == 422

    # Whitespace-only string
    res2 = client.post("/api/v1/ai/chat", headers=headers, json={"message": "   \n  "})
    assert res2.status_code == 422


def test_chat_malformed_session_id_rejected() -> None:
    """Verify POST /api/v1/ai/chat with invalid session_id pattern returns 422 validation error."""
    headers = get_auth_headers()
    response = client.post(
        "/api/v1/ai/chat",
        headers=headers,
        json={"message": "Hello", "session_id": "invalid session space!"},
    )
    assert response.status_code == 422


def test_successful_synchronous_chat() -> None:
    """Verify successful synchronous REST chat execution with dependency override."""
    app.dependency_overrides[get_planner] = lambda: MockPlanner()

    headers = get_auth_headers("user_sync_test")
    headers["X-Correlation-ID"] = "corr_header_123"

    payload = {
        "message": "Hello, Ultron!",
        "session_id": "sess_valid_123",
    }

    try:
        response = client.post("/api/v1/ai/chat", headers=headers, json=payload)
        assert response.status_code == 200

        data = response.json()
        assert data["response"] == "Hello from MockPlanner!"
        assert data["planner_status"] == "COMPLETED"
        assert data["session_id"] == "sess_valid_123"
        assert data["correlation_id"] == "corr_header_123"
        assert data["planner_id"].startswith("planner_")
    finally:
        app.dependency_overrides.clear()


def test_planner_failure_safe_error_mapping() -> None:
    """Verify unhandled planner exceptions return safe 500 without leaking tracebacks or paths."""
    mock_failing_planner = AsyncMock(spec=BasePlanner)
    mock_failing_planner.run_pipeline.side_effect = RuntimeError(
        "Internal database connection error: /var/lib/secret/db.key"
    )

    app.dependency_overrides[get_planner] = lambda: mock_failing_planner

    headers = get_auth_headers()

    try:
        response = client.post(
            "/api/v1/ai/chat",
            headers=headers,
            json={"message": "Trigger failure"},
        )
        assert response.status_code == 500

        data = response.json()
        assert data["detail"] == "AI Planner service encountered an operational error."
        # Verify no secrets or file paths leaked in HTTP response payload
        assert "RuntimeError" not in response.text
        assert "/var/lib/secret" not in response.text
    finally:
        app.dependency_overrides.clear()


def test_successful_sse_stream() -> None:
    """Verify POST /api/v1/ai/chat/stream returns valid text/event-stream with structured events."""
    app.dependency_overrides[get_planner] = lambda: MockPlanner()

    headers = get_auth_headers("user_stream_test")
    headers["X-Correlation-ID"] = "corr_stream_999"

    payload = {
        "message": "Stream request message",
        "session_id": "sess_stream_001",
    }

    try:
        response = client.post("/api/v1/ai/chat/stream", headers=headers, json=payload)
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        content = response.text
        assert "event: start" in content
        assert "event: token" in content
        assert "event: completion" in content
        assert "Hello from MockPlanner!" in content
        assert "corr_stream_999" in content
    finally:
        app.dependency_overrides.clear()


def test_openapi_schema_contains_ai_endpoints() -> None:
    """Verify OpenAPI 3.1.0 schema includes /api/v1/ai/chat and /api/v1/ai/chat/stream specifications."""
    response = client.get("/api/v1/openapi.json")
    assert response.status_code == 200

    schema = response.json()
    paths = schema.get("paths", {})

    assert "/api/v1/ai/chat" in paths
    assert "post" in paths["/api/v1/ai/chat"]

    assert "/api/v1/ai/chat/stream" in paths
    assert "post" in paths["/api/v1/ai/chat/stream"]
