"""Structured Observability & Logging Subsystem.

Provides enterprise-grade, structured logging with correlation ID contextual tracing
and standard library interception using Structlog and Python's logging module.
"""

import logging
import sys
from contextvars import ContextVar
from typing import Any

import structlog
from structlog.types import EventDict, Processor

# Thread-safe & async-safe ContextVar for correlation ID tracing
correlation_id_ctx: ContextVar[str | None] = ContextVar("correlation_id", default=None)


def get_correlation_id() -> str | None:
    """Retrieves the current async context correlation ID.

    Returns:
        str | None: Correlation ID string if set, otherwise None.
    """
    return correlation_id_ctx.get()


def set_correlation_id(correlation_id: str | None) -> None:
    """Sets the correlation ID for the current async execution context.

    Args:
        correlation_id: Unique correlation ID string to bind.
    """
    correlation_id_ctx.set(correlation_id)


def inject_correlation_id(_: Any, __: str, event_dict: EventDict) -> EventDict:
    """Structlog processor that automatically injects correlation_id into log records.

    Args:
        _: Logger instance (unused).
        __: Method name (unused).
        event_dict: Active log event dictionary.

    Returns:
        EventDict: Enriched event dictionary containing correlation_id if bound.
    """
    cid = get_correlation_id()
    if cid:
        event_dict["correlation_id"] = cid
    return event_dict


def configure_logging(log_level: str = "INFO", log_format: str = "json") -> None:
    """Configures global structured logging for structlog and stdlib logging.

    Args:
        log_level: Severity threshold (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_format: Output log format ('json' for production, 'console' for dev).
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        inject_correlation_id,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if log_format.lower() == "console":
        renderer: Processor = structlog.dev.ConsoleRenderer(colors=True)
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.filter_by_level,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(numeric_level)

    # Silence overly noisy third-party loggers
    for noisy_logger in ("uvicorn.access", "asyncio"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Obtains a bound structlog logger instance.

    Args:
        name: Name of the logger category or module.

    Returns:
        structlog.stdlib.BoundLogger: Configured logger instance.
    """
    return structlog.get_logger(name)
