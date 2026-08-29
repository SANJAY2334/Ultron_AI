"""Async Database Session Management Module.

Provides asynchronous PostgreSQL connection pool initialization, async SQLAlchemy 2.0 session
generators, and database health check probes.
"""

import logging
from collections.abc import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# Global async engine and session factory placeholders
_async_engine: AsyncEngine | None = None
_async_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_async_engine() -> AsyncEngine:
    """Retrieves or initializes the global async SQLAlchemy engine singleton.

    Returns:
        AsyncEngine: Configured async database engine instance.
    """
    global _async_engine
    if _async_engine is None:
        settings = get_settings()
        _async_engine = create_async_engine(
            settings.DATABASE_URL,
            echo=settings.ULTRON_DEBUG and settings.is_development,
            future=True,
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=20,
        )
        logger.info("Async PostgreSQL engine initialized.")
    return _async_engine


def get_async_session_factory() -> async_sessionmaker[AsyncSession]:
    """Retrieves or initializes the async sessionmaker factory singleton.

    Returns:
        async_sessionmaker[AsyncSession]: Session factory.
    """
    global _async_session_factory
    if _async_session_factory is None:
        engine = get_async_engine()
        _async_session_factory = async_sessionmaker(
            bind=engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
    return _async_session_factory


async def get_async_db_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI Dependency Injection generator yielding an async database session.

    Yields:
        AsyncSession: Open database session within context manager.
    """
    session_factory = get_async_session_factory()
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def check_db_health() -> bool:
    """Executes a diagnostic ping query (SELECT 1) to verify database connectivity.

    Returns:
        bool: True if database is responsive, False otherwise.
    """
    try:
        session_factory = get_async_session_factory()
        async with session_factory() as session:
            result = await session.execute(text("SELECT 1"))
            return result.scalar() == 1
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False


async def close_db_connection() -> None:
    """Disposes of the async engine and closes active database connection pools."""
    global _async_engine, _async_session_factory
    if _async_engine is not None:
        await _async_engine.dispose()
        _async_engine = None
        _async_session_factory = None
        logger.info("Database connection pools disposed cleanly.")
