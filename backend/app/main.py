"""ULTRON Kernel Gateway & FastAPI Application Entrypoint.

Initializes FastAPI application lifecycle, CORS policy, correlation ID tracing middleware,
and mounts versioned REST routes.
"""

import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from app import __version__
from app.api.v1.router import api_router
from app.core.config import get_settings
from app.core.kernel import kernel
from app.core.logging import configure_logging, get_logger, set_correlation_id
from app.core.redis import close_redis_connection
from app.database.session import close_db_connection

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager handling boot and graceful shutdown."""
    settings = get_settings()
    configure_logging(log_level=settings.LOG_LEVEL, log_format=settings.LOG_FORMAT)

    logger.info("Starting ULTRON Kernel Gateway...", version=__version__)
    await kernel.boot()

    yield

    logger.info("Shutting down ULTRON Kernel Gateway...")
    await kernel.shutdown()
    await close_db_connection()
    await close_redis_connection()


def create_application() -> FastAPI:
    """FastAPI Application Factory.

    Returns:
        FastAPI: Configured web application instance.
    """
    settings = get_settings()

    app = FastAPI(
        title="PROJECT ULTRON - AI Operating System",
        description="Next Generation Personal Artificial Intelligence Operating System Kernel & Backend APIs",
        version=__version__,
        openapi_url=f"{settings.ULTRON_API_PREFIX}/openapi.json",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # CORS Policy Middleware
    if settings.ULTRON_CORS_ORIGINS:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.ULTRON_CORS_ORIGINS,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Correlation ID Tracing Middleware
    class CorrelationIdMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next) -> Response:
            correlation_id = request.headers.get("X-Correlation-ID") or str(uuid.uuid4())
            set_correlation_id(correlation_id)

            response: Response = await call_next(request)
            response.headers["X-Correlation-ID"] = correlation_id
            return response

    app.add_middleware(CorrelationIdMiddleware)

    # Mount Master API Router
    app.include_router(api_router, prefix=settings.ULTRON_API_PREFIX)

    return app


app = create_application()

if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.ULTRON_HOST,
        port=settings.ULTRON_PORT,
        reload=settings.ULTRON_DEBUG and settings.is_development,
    )
