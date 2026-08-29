"""Async Redis Connection Pool Manager.

Provides async Redis client instances for LRU Working Memory cache and Event Bus pub/sub.
"""

import logging

import redis.asyncio as aioredis

from app.core.config import get_settings

logger = logging.getLogger(__name__)

_redis_client: aioredis.Redis | None = None


def get_redis_client() -> aioredis.Redis:
    """Retrieves or initializes the global async Redis client singleton.

    Returns:
        aioredis.Redis: Active async Redis client connection pool.
    """
    global _redis_client
    if _redis_client is None:
        settings = get_settings()
        _redis_client = aioredis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
        )
        logger.info("Async Redis connection pool initialized.")
    return _redis_client


async def check_redis_health() -> bool:
    """Executes a PING command to verify Redis server connectivity.

    Returns:
        bool: True if Redis responds with PONG, False otherwise.
    """
    try:
        client = get_redis_client()
        pong = await client.ping()
        return bool(pong)
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        return False


async def close_redis_connection() -> None:
    """Closes the async Redis client connection pool cleanly."""
    global _redis_client
    if _redis_client is not None:
        await _redis_client.aclose()
        _redis_client = None
        logger.info("Redis connection pool closed cleanly.")
