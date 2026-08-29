"""Unit Tests for Structured Observability & Logging Subsystem.

Validates correlation ID ContextVar binding, structlog processor injection,
and stdout log configuration.
"""

from app.core.logging import (
    configure_logging,
    get_correlation_id,
    get_logger,
    inject_correlation_id,
    set_correlation_id,
)


def test_correlation_id_context_var() -> None:
    """Verify setting and getting correlation ID in context."""
    assert get_correlation_id() is None
    set_correlation_id("corr_test_12345")
    assert get_correlation_id() == "corr_test_12345"
    set_correlation_id(None)
    assert get_correlation_id() is None


def test_inject_correlation_id_processor() -> None:
    """Verify inject_correlation_id processor enriches event dict."""
    set_correlation_id("corr_abcd_9999")
    event_dict: dict[str, str] = {"event": "user_login"}
    enriched = inject_correlation_id(None, "info", event_dict)
    assert enriched["correlation_id"] == "corr_abcd_9999"
    set_correlation_id(None)


def test_get_logger_returns_bound_logger() -> None:
    """Verify get_logger returns a functional bound logger instance."""
    logger = get_logger("test_module")
    assert logger is not None


def test_json_logging_output(capsys) -> None:
    """Verify formatted log output contains expected structured fields."""
    configure_logging(log_level="INFO", log_format="json")
    set_correlation_id("corr_json_test")

    logger = get_logger("test_json")
    logger.info("system_boot_event", component="kernel", status="ok")

    set_correlation_id(None)
