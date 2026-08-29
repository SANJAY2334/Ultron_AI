"""Unit Tests for Database & Redis Connection Pool Managers.

Validates engine instantiation, session generation, health check error handling,
and connection teardown logic.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.redis import check_redis_health, close_redis_connection
from app.database.session import check_db_health, close_db_connection, get_async_engine


def test_async_engine_initialization() -> None:
    """Verify async database engine instantiation."""
    engine = get_async_engine()
    assert engine is not None
    assert "postgresql+asyncpg://" in str(engine.url)


def test_db_health_check_handles_connection_error() -> None:
    """Verify database health check returns False when database connection fails."""

    async def _test() -> None:
        with patch("app.database.session.get_async_session_factory") as mock_factory:
            mock_session = AsyncMock()
            mock_session.execute.side_effect = Exception("DB Connection Refused")
            mock_factory.return_value.return_value.__aenter__.return_value = mock_session

            healthy = await check_db_health()
            assert healthy is False

    asyncio.run(_test())


def test_db_health_check_success() -> None:
    """Verify database health check returns True when ping query succeeds."""

    async def _test() -> None:
        with patch("app.database.session.get_async_session_factory") as mock_factory:
            mock_result = MagicMock()
            mock_result.scalar.return_value = 1
            mock_session = AsyncMock()
            mock_session.execute = AsyncMock(return_value=mock_result)
            mock_factory.return_value.return_value.__aenter__.return_value = mock_session

            healthy = await check_db_health()
            assert healthy is True

    asyncio.run(_test())


def test_redis_health_check_handles_connection_error() -> None:
    """Verify Redis health check returns False when Redis connection fails."""

    async def _test() -> None:
        with patch("app.core.redis.get_redis_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.ping.side_effect = Exception("Redis Connection Refused")
            mock_get_client.return_value = mock_client

            healthy = await check_redis_health()
            assert healthy is False

    asyncio.run(_test())


def test_connection_teardown() -> None:
    """Verify database and Redis connection teardown executes without errors."""

    async def _test() -> None:
        await close_db_connection()
        await close_redis_connection()

    asyncio.run(_test())
