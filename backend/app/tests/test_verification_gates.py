"""Engineering Validation Gate Tests.

Validates FastAPI application startup context, OpenAPI 3.0 schema structure,
and live REST API endpoint smoke test contracts.
"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_gate_fastapi_startup_verification() -> None:
    """Gate 8: Verify FastAPI application initializes app instance and lifespan."""
    assert app.title == "PROJECT ULTRON - AI Operating System"
    assert app.version == "0.1.0"


def test_gate_openapi_schema_structure_validation() -> None:
    """Gate 9: Validate structure and completeness of OpenAPI 3.1.0 JSON schema."""
    response = client.get("/api/v1/openapi.json")
    assert response.status_code == 200, "OpenAPI schema endpoint must return HTTP 200"

    schema = response.json()
    assert "openapi" in schema, "Schema must include 'openapi' version field"
    assert "info" in schema, "Schema must include 'info' section"
    assert schema["info"]["title"] == "PROJECT ULTRON - AI Operating System"
    assert "paths" in schema, "Schema must contain API 'paths' map"

    # Verify registered paths
    assert "/api/v1/system/health" in schema["paths"]
    assert "/api/v1/system/status" in schema["paths"]


def test_gate_api_smoke_test_contracts() -> None:
    """Gate 10: Smoke test REST API endpoint response structures and status codes."""
    # Smoke Test 1: Health endpoint contract
    res_health = client.get("/api/v1/system/health")
    assert res_health.status_code == 200
    health_payload = res_health.json()
    assert health_payload["status"] in ("healthy", "degraded")
    assert "kernel_state" in health_payload

    # Smoke Test 2: Status endpoint contract
    res_status = client.get("/api/v1/system/status")
    assert res_status.status_code == 200
    status_payload = res_status.json()
    assert status_payload["version"] == "0.1.0"
    assert status_payload["environment"] == "development"

    # Smoke Test 3: Correlation ID propagation
    cid = "gate_smoke_cid_1001"
    res_cid = client.get("/api/v1/system/status", headers={"X-Correlation-ID": cid})
    assert res_cid.headers.get("X-Correlation-ID") == cid
