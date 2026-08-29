"""System Diagnostics & Health Check Endpoints.

Provides REST API routes for health status checks, database/cache connectivity probes,
and system operational telemetry.
"""

from typing import Any

from fastapi import APIRouter, Depends, status

from app import __version__
from app.api.deps import get_current_settings
from app.core.config import Settings
from app.core.kernel import kernel
from app.core.redis import check_redis_health
from app.database.session import check_db_health

router = APIRouter(prefix="/system", tags=["System Diagnostics"])


@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="System Health Check",
    description="Probes PostgreSQL, Redis, and Kernel state to return overall system operational status.",
)
async def get_system_health() -> dict[str, Any]:
    """Probes system sub-components and returns consolidated health status."""
    db_ok = await check_db_health()
    redis_ok = await check_redis_health()

    is_healthy = db_ok and redis_ok

    return {
        "status": "healthy" if is_healthy else "degraded",
        "kernel_state": kernel.state.name,
        "components": {
            "database": "connected" if db_ok else "disconnected",
            "redis": "connected" if redis_ok else "disconnected",
        },
    }


@router.get(
    "/status",
    status_code=status.HTTP_200_OK,
    summary="System Telemetry Status",
    description="Returns application version, active environment, debug state, and kernel metrics.",
)
async def get_system_status(
    settings: Settings = Depends(get_current_settings),
) -> dict[str, Any]:
    """Returns application status and telemetry metadata."""
    return {
        "name": "ultron-backend",
        "version": __version__,
        "environment": settings.ULTRON_ENV,
        "debug": settings.ULTRON_DEBUG,
        "kernel_state": kernel.state.name,
        "api_prefix": settings.ULTRON_API_PREFIX,
    }
