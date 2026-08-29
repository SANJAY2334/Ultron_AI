"""Core Configuration Module.

Provides strongly-typed, validated application settings using Pydantic v2 BaseSettings.
All configuration is loaded exclusively from environment variables or .env files,
strictly enforcing Zero Hardcoded Secrets (Engineering Principle #12 & #13).
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def build_postgres_dsn(
    user: str = "ultron_admin",
    password: str = "",  # nosec B107
    server: str = "localhost",
    port: int = 5432,
    db: str = "ultron_db",
) -> str:
    """Constructs a standard async PostgreSQL connection DSN.

    Args:
        user: PostgreSQL username.
        password: PostgreSQL password.
        server: PostgreSQL hostname.
        port: PostgreSQL port.
        db: Database name.

    Returns:
        str: Assembled postgresql+asyncpg DSN.
    """
    auth = f"{user}:{password}@" if password else f"{user}@"
    return f"postgresql+asyncpg://{auth}{server}:{port}/{db}"


def build_redis_dsn(
    host: str = "localhost",
    port: int = 6379,
    password: str = "",  # nosec B107
    db: int = 0,
) -> str:
    """Constructs a standard Redis connection DSN.

    Args:
        host: Redis hostname.
        port: Redis port.
        password: Redis password.
        db: Redis database index.

    Returns:
        str: Assembled redis DSN.
    """
    auth = f":{password}@" if password else ""
    return f"redis://{auth}{host}:{port}/{db}"


class Settings(BaseSettings):
    """ULTRON System Master Settings Schema."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # Core Application Settings
    ULTRON_ENV: Literal["development", "testing", "staging", "production"] = Field(
        default="development", description="Application execution environment"
    )
    ULTRON_DEBUG: bool = Field(default=True, description="Global debug flag")
    ULTRON_SECRET_KEY: SecretStr = Field(
        default=SecretStr(""), description="Secret key for cryptographic operations"
    )
    ULTRON_API_PREFIX: str = Field(default="/api/v1", description="Global versioned API prefix")
    ULTRON_HOST: str = Field(
        default="0.0.0.0",
        description="ASGI server binding host",  # nosec B104
    )
    ULTRON_PORT: int = Field(default=8000, description="ASGI server binding port")
    ULTRON_CORS_ORIGINS: list[str] = Field(
        default=["http://localhost:5173", "http://localhost:3000"],
        description="Allowed origins for CORS middleware",
    )

    # Cryptography & Security
    ULTRON_MEMORY_ENCRYPTION_KEY: SecretStr = Field(
        default=SecretStr(""), description="Fernet 32-byte URL-safe base64 encryption key"
    )
    ULTRON_JWT_ALGORITHM: str = Field(default="HS256", description="JWT signing algorithm")
    ULTRON_ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        default=15, description="Access token expiration window in minutes"
    )
    ULTRON_REFRESH_TOKEN_EXPIRE_DAYS: int = Field(
        default=7, description="Refresh token expiration window in days"
    )

    # Database Settings (PostgreSQL)
    POSTGRES_SERVER: str = Field(default="localhost", description="PostgreSQL host")
    POSTGRES_PORT: int = Field(default=5432, description="PostgreSQL port")
    POSTGRES_USER: str = Field(default="ultron_admin", description="PostgreSQL user")
    POSTGRES_PASSWORD: SecretStr = Field(default=SecretStr(""), description="PostgreSQL password")
    POSTGRES_DB: str = Field(default="ultron_db", description="PostgreSQL database name")
    DATABASE_URL: str = Field(
        default="", description="Async PostgreSQL DSN (postgresql+asyncpg://...)"
    )

    # Cache & Event Bus Settings (Redis)
    REDIS_HOST: str = Field(default="localhost", description="Redis host")
    REDIS_PORT: int = Field(default=6379, description="Redis port")
    REDIS_PASSWORD: SecretStr = Field(default=SecretStr(""), description="Redis password")
    REDIS_DB: int = Field(default=0, description="Redis database index")
    REDIS_URL: str = Field(default="", description="Redis connection DSN")

    # Vector Storage Settings (ChromaDB)
    CHROMADB_HOST: str = Field(default="localhost", description="ChromaDB host")
    CHROMADB_PORT: int = Field(default=8000, description="ChromaDB port")
    CHROMADB_PERSIST_DIRECTORY: str = Field(
        default="./chroma_data", description="Local path for ChromaDB storage"
    )

    # AI Model Provider Settings
    OPENAI_API_KEY: SecretStr = Field(default=SecretStr(""), description="OpenAI API key")
    ANTHROPIC_API_KEY: SecretStr = Field(
        default=SecretStr(""), description="Anthropic Claude API key"
    )
    DEFAULT_AI_MODEL: str = Field(default="gpt-4o", description="Default primary AI model")
    FALLBACK_AI_MODEL: str = Field(
        default="claude-3-5-sonnet-20241022", description="Fallback AI model"
    )

    # Audio Engine Subsystem Settings (Phase 4E.3)
    AUDIO_VAD_PROVIDER: str = Field(
        default="local", description="Voice Activity Detection engine provider"
    )
    AUDIO_WAKE_WORD_PROVIDER: str = Field(
        default="local", description="Wake word detection engine provider"
    )
    AUDIO_STT_PROVIDER: str = Field(
        default="whisper_api", description="Speech-to-Text transcription provider"
    )
    AUDIO_TTS_PROVIDER: str = Field(
        default="openai_tts", description="Text-to-Speech synthesis provider"
    )
    AUDIO_LANGUAGE: str = Field(default="en-IN", description="Default audio language code")
    AUDIO_SAMPLE_RATE: int = Field(default=16000, description="Target audio sample rate in Hz")
    AUDIO_CHANNELS: int = Field(default=1, description="Target audio channel count")
    AUDIO_WAKE_WORD: str = Field(
        default="ultron", description="Primary system wake word trigger phrase"
    )
    AUDIO_WAKE_WORD_THRESHOLD: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Wake word detection confidence threshold"
    )

    # Vision Intelligence Subsystem Settings (Phase 4F.1)
    VISION_ENABLED: bool = Field(
        default=False, description="True if vision perception subsystem is enabled"
    )
    VISION_PROVIDER: str = Field(default="local", description="Vision engine provider identifier")
    VISION_CAMERA_DEVICE: str = Field(
        default="default", description="Target video capture device identifier"
    )
    VISION_MAX_FPS: int = Field(
        default=15, gt=0, le=60, description="Maximum capture frame rate in FPS"
    )
    VISION_MAX_WIDTH: int = Field(
        default=1280, gt=0, description="Maximum capture frame width in pixels"
    )
    VISION_MAX_HEIGHT: int = Field(
        default=720, gt=0, description="Maximum capture frame height in pixels"
    )
    VISION_MAX_BUFFERED_FRAMES: int = Field(
        default=10, gt=0, description="Maximum frame queue buffer depth"
    )
    VISION_PROCESSING_TIMEOUT_MS: float = Field(
        default=100.0, gt=0.0, description="Maximum frame processing timeout in ms"
    )
    VISION_PRIVACY: str = Field(
        default="ephemeral", description="Default vision privacy classification"
    )
    VISION_OBJECT_DETECTOR_PROVIDER: str = Field(
        default="mock", description="Object detector provider identifier (mock, onnx)"
    )
    VISION_OBJECT_MODEL_PATH: str = Field(
        default="", description="Configured path to ONNX object detector model file"
    )
    VISION_OBJECT_CONFIDENCE_THRESHOLD: float = Field(
        default=0.50, ge=0.0, le=1.0, description="Object detection confidence threshold"
    )
    VISION_OBJECT_IOU_THRESHOLD: float = Field(
        default=0.45, ge=0.0, le=1.0, description="Object detection IoU NMS threshold"
    )
    VISION_OBJECT_MAX_DETECTIONS: int = Field(
        default=50, gt=0, le=200, description="Maximum allowed detections per frame"
    )
    VISION_OBJECT_TIMEOUT_MS: float = Field(
        default=100.0, gt=0.0, description="Maximum object inference timeout in ms"
    )
    VISION_OBJECT_DEVICE: str = Field(
        default="cpu", description="Execution hardware target (cpu, cuda)"
    )
    VISION_TRACKING_ENABLED: bool = Field(
        default=True, description="True if visual object tracking is enabled"
    )
    VISION_TRACKING_PROVIDER: str = Field(
        default="baseline", description="Vision tracker provider identifier"
    )
    VISION_TRACKING_IOU_THRESHOLD: float = Field(
        default=0.30, ge=0.0, le=1.0, description="Tracking association IoU threshold"
    )
    VISION_TRACKING_MAX_DISTANCE: float = Field(
        default=100.0, gt=0.0, description="Maximum centroid association distance in pixels"
    )
    VISION_TRACKING_MAX_MISSING_FRAMES: int = Field(
        default=10, ge=1, le=100, description="Maximum missing frames before track termination"
    )
    VISION_TRACKING_MAX_TRACKS: int = Field(
        default=50, gt=0, le=200, description="Maximum simultaneous active tracks"
    )
    VISION_TRACKING_TIMEOUT_MS: float = Field(
        default=50.0, gt=0.0, description="Maximum tracking processing timeout in ms"
    )
    VISION_TRACKING_MOVEMENT_THRESHOLD: float = Field(
        default=5.0, ge=0.0, description="Movement detection threshold in pixels"
    )
    VISION_SCENE_ANALYSIS_ENABLED: bool = Field(
        default=True, description="True if visual scene understanding is enabled"
    )
    VISION_SCENE_MAX_PREVIOUS_OBSERVATIONS: int = Field(
        default=2, ge=1, le=10, description="Maximum bounded observation history depth"
    )
    VISION_SCENE_CROWDED_THRESHOLD: int = Field(
        default=10, ge=1, description="Object count threshold for crowded scene state"
    )
    VISION_SCENE_LOW_MOTION_THRESHOLD: float = Field(
        default=0.10, ge=0.0, le=1.0, description="Low motion level threshold ratio"
    )
    VISION_SCENE_HIGH_MOTION_THRESHOLD: float = Field(
        default=0.50, ge=0.0, le=1.0, description="High motion level threshold ratio"
    )
    VISION_SCENE_CHANGE_THRESHOLD: float = Field(
        default=0.30, ge=0.0, le=1.0, description="Scene change detection sensitivity ratio"
    )
    VISION_SCENE_TIMEOUT_MS: float = Field(
        default=50.0, gt=0.0, description="Maximum scene analysis timeout in ms"
    )
    VISION_PLANNER_CONTEXT_ENABLED: bool = Field(
        default=True, description="True if vision planner context integration is enabled"
    )
    VISION_PLANNER_MAX_OBJECTS: int = Field(
        default=20, ge=1, le=50, description="Maximum objects included in vision planner context"
    )
    VISION_PLANNER_MAX_TRACKS: int = Field(
        default=20, ge=1, le=50, description="Maximum tracks included in vision planner context"
    )
    VISION_PLANNER_MAX_EVENTS: int = Field(
        default=20, ge=1, le=50, description="Maximum events included in vision planner context"
    )
    VISION_PLANNER_MAX_CONTEXT_CHARS: int = Field(
        default=8000, ge=100, le=20000, description="Maximum vision context prompt length in chars"
    )
    VISION_PLANNER_CONTEXT_TIMEOUT_MS: float = Field(
        default=20.0, gt=0.0, description="Maximum vision context build timeout in ms"
    )
    LOCAL_LLM_BASE_URL: str = Field(
        default="http://localhost:11434/v1", description="Local vLLM/Ollama API endpoint"
    )

    # Observability & Logging
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Log severity threshold"
    )
    LOG_FORMAT: Literal["json", "console"] = Field(
        default="json", description="Structured log formatting"
    )
    OPENTELEMETRY_ENDPOINT: str = Field(default="", description="OpenTelemetry collector endpoint")

    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def assemble_db_connection(cls, v: str | None, info) -> str:
        """Assembles and validates default async PostgreSQL DSN."""
        if isinstance(v, str) and v.strip():
            if not (v.startswith("postgresql+asyncpg://") or v.startswith("postgresql://")):
                raise ValueError(
                    "DATABASE_URL must start with 'postgresql+asyncpg://' or 'postgresql://'"
                )
            return v

        user = info.data.get("POSTGRES_USER", "ultron_admin")
        pwd_secret = info.data.get("POSTGRES_PASSWORD", SecretStr(""))
        password = (
            pwd_secret.get_secret_value()
            if isinstance(pwd_secret, SecretStr)
            else str(pwd_secret or "")
        )
        server = info.data.get("POSTGRES_SERVER", "localhost")
        port = info.data.get("POSTGRES_PORT", 5432)
        db = info.data.get("POSTGRES_DB", "ultron_db")

        return build_postgres_dsn(user=user, password=password, server=server, port=port, db=db)

    @field_validator("REDIS_URL", mode="before")
    @classmethod
    def assemble_redis_connection(cls, v: str | None, info) -> str:
        """Assembles and validates default Redis connection DSN."""
        if isinstance(v, str) and v.strip():
            if not (v.startswith("redis://") or v.startswith("rediss://")):
                raise ValueError("REDIS_URL must start with 'redis://' or 'rediss://'")
            return v

        host = info.data.get("REDIS_HOST", "localhost")
        port = info.data.get("REDIS_PORT", 6379)
        db = info.data.get("REDIS_DB", 0)
        pwd_secret = info.data.get("REDIS_PASSWORD", SecretStr(""))
        password = (
            pwd_secret.get_secret_value()
            if isinstance(pwd_secret, SecretStr)
            else str(pwd_secret or "")
        )

        return build_redis_dsn(host=host, port=port, password=password, db=db)

    @model_validator(mode="after")
    def validate_production_secrets(self) -> "Settings":
        """Enforces mandatory secrets when running in production environment."""
        if self.ULTRON_ENV == "production":
            missing: list[str] = []
            if not self.ULTRON_SECRET_KEY.get_secret_value().strip():
                missing.append("ULTRON_SECRET_KEY")
            if not self.ULTRON_MEMORY_ENCRYPTION_KEY.get_secret_value().strip():
                missing.append("ULTRON_MEMORY_ENCRYPTION_KEY")
            if not self.POSTGRES_PASSWORD.get_secret_value().strip():
                missing.append("POSTGRES_PASSWORD")

            if missing:
                raise ValueError(
                    f"Production environment requires the following non-empty secrets: {', '.join(missing)}"
                )
        return self

    @property
    def is_development(self) -> bool:
        """Returns True if the application is running in development mode."""
        return self.ULTRON_ENV == "development"

    @property
    def is_testing(self) -> bool:
        """Returns True if the application is running in testing mode."""
        return self.ULTRON_ENV == "testing"


@lru_cache
def get_settings() -> Settings:
    """Retrieves and caches the global application settings instance."""
    return Settings()
