"""Unit & Integration Tests for System API Endpoints.

Validates /api/v1/system/health, /api/v1/system/status, correlation ID headers,
and OpenAPI schema generation.
"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_system_health_endpoint() -> None:
    """Verify HTTP 200 response and payload structure from /api/v1/system/health."""
    response = client.get("/api/v1/system/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "kernel_state" in data
    assert "components" in data
    assert "database" in data["components"]
    assert "redis" in data["components"]


def test_system_status_endpoint() -> None:
    """Verify HTTP 200 response and payload structure from /api/v1/system/status."""
    response = client.get("/api/v1/system/status")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "ultron-backend"
    assert data["version"] == "0.1.0"
    assert data["environment"] == "development"
    assert data["api_prefix"] == "/api/v1"


def test_correlation_id_header_in_response() -> None:
    """Verify X-Correlation-ID header is propagated in API responses."""
    custom_cid = "corr_custom_req_9999"
    response = client.get(
        "/api/v1/system/status",
        headers={"X-Correlation-ID": custom_cid},
    )
    assert response.status_code == 200
    assert response.headers.get("X-Correlation-ID") == custom_cid


def test_openapi_schema_generated() -> None:
    """Verify OpenAPI JSON schema endpoint generates valid API documentation."""
    response = client.get("/api/v1/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert schema["info"]["title"] == "PROJECT ULTRON - AI Operating System"
    assert "/api/v1/system/health" in schema["paths"]
    assert "/api/v1/system/status" in schema["paths"]
