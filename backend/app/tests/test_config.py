"""Unit Tests for Core Configuration Engine.

Validates Pydantic settings loading, SecretStr masking, DSN builders and validation,
invalid configuration rejection, and production environment secret enforcement.
"""

import pytest
from pydantic import SecretStr, ValidationError

from app.core.config import (
    Settings,
    build_postgres_dsn,
    build_redis_dsn,
    get_settings,
)


def test_default_settings() -> None:
    """Verify default settings values when initialized without overrides."""
    settings = Settings()
    assert settings.ULTRON_ENV == "development"
    assert settings.ULTRON_DEBUG is True
    assert settings.ULTRON_API_PREFIX == "/api/v1"
    assert settings.ULTRON_HOST == "0.0.0.0"
    assert settings.ULTRON_PORT == 8000
    assert settings.LOG_LEVEL == "INFO"
    assert settings.LOG_FORMAT == "json"


def test_secret_str_masking_and_retrieval() -> None:
    """Verify SecretStr masks representation and returns raw value on demand."""
    settings = Settings(
        ULTRON_SECRET_KEY=SecretStr("super_secret_jwt_key"),
        OPENAI_API_KEY=SecretStr("sk-test-key"),
    )
    # Representation should be masked
    assert "super_secret_jwt_key" not in str(settings.ULTRON_SECRET_KEY)
    assert "sk-test-key" not in str(settings.OPENAI_API_KEY)

    # Secret retrieval via get_secret_value
    assert settings.ULTRON_SECRET_KEY.get_secret_value() == "super_secret_jwt_key"
    assert settings.OPENAI_API_KEY.get_secret_value() == "sk-test-key"


def test_decoupled_dsn_builders() -> None:
    """Verify standalone DSN builder functions."""
    postgres_dsn = build_postgres_dsn("user", "pass", "host", 5432, "db")
    assert postgres_dsn == "postgresql+asyncpg://user:pass@host:5432/db"

    redis_dsn = build_redis_dsn("redishost", 6379, "redispass", 2)
    assert redis_dsn == "redis://:redispass@redishost:6379/2"


def test_custom_settings_override() -> None:
    """Verify that settings can be overridden via explicit parameters."""
    settings = Settings(
        ULTRON_ENV="staging",
        ULTRON_DEBUG=False,
        ULTRON_PORT=9000,
        LOG_LEVEL="ERROR",
    )
    assert settings.ULTRON_ENV == "staging"
    assert settings.ULTRON_DEBUG is False
    assert settings.ULTRON_PORT == 9000
    assert settings.LOG_LEVEL == "ERROR"


def test_malformed_database_url_rejection() -> None:
    """Verify that invalid DATABASE_URL scheme raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        Settings(DATABASE_URL="mysql://user:pass@localhost/db")
    assert "DATABASE_URL must start with 'postgresql+asyncpg://'" in str(exc_info.value)


def test_malformed_redis_url_rejection() -> None:
    """Verify that invalid REDIS_URL scheme raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        Settings(REDIS_URL="http://localhost:6379")
    assert "REDIS_URL must start with 'redis://'" in str(exc_info.value)


def test_production_validation_failure_on_missing_secrets() -> None:
    """Verify production startup fails when mandatory secrets are empty."""
    with pytest.raises(ValidationError) as exc_info:
        Settings(
            ULTRON_ENV="production",
            ULTRON_SECRET_KEY=SecretStr(""),
            ULTRON_MEMORY_ENCRYPTION_KEY=SecretStr(""),
            POSTGRES_PASSWORD=SecretStr(""),
        )
    assert "Production environment requires the following non-empty secrets" in str(exc_info.value)


def test_production_validation_success_with_secrets() -> None:
    """Verify production settings instantiate cleanly when required secrets exist."""
    settings = Settings(
        ULTRON_ENV="production",
        ULTRON_SECRET_KEY=SecretStr("prod_secret_jwt_key_32_bytes_long"),
        ULTRON_MEMORY_ENCRYPTION_KEY=SecretStr("prod_fernet_32_byte_url_safe_key_value="),
        POSTGRES_PASSWORD=SecretStr("prod_secure_password"),
    )
    assert settings.ULTRON_ENV == "production"
    assert settings.POSTGRES_PASSWORD.get_secret_value() == "prod_secure_password"


def test_environment_properties() -> None:
    """Verify is_development and is_testing properties."""
    dev_settings = Settings(ULTRON_ENV="development")
    assert dev_settings.is_development is True
    assert dev_settings.is_testing is False

    test_settings = Settings(ULTRON_ENV="testing")
    assert test_settings.is_development is False
    assert test_settings.is_testing is True


def test_get_settings_caching() -> None:
    """Verify get_settings returns a cached lru_cache singleton."""
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
