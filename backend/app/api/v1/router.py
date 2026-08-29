"""ULTRON API Version 1 Master Router.

Consolidates all versioned endpoints under /api/v1 prefix.
"""

from fastapi import APIRouter

from app.api.v1.endpoints import ai, system

api_router = APIRouter()
api_router.include_router(system.router)
api_router.include_router(ai.router)
