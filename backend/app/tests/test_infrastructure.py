"""Infrastructure & Dependency Blueprint Verification Tests.

Validates that all core third-party dependencies, project packaging specs,
and environment blueprint templates conform to the system requirements.
"""

import importlib
from pathlib import Path

from app import __version__


def test_package_version() -> None:
    """Verify application version matches semantic version 0.1.0."""
    assert __version__ == "0.1.0"


def test_core_dependencies_importable() -> None:
    """Verify all critical framework dependencies can be imported successfully.

    Guarantees that FastAPI, Pydantic, SQLAlchemy, Structlog, Cryptography, and PyJWT
    are properly installed and present in the runtime environment.
    """
    required_modules = [
        "fastapi",
        "pydantic",
        "pydantic_settings",
        "sqlalchemy",
        "structlog",
        "cryptography",
        "jwt",
        "passlib",
    ]
    for module_name in required_modules:
        mod = importlib.import_module(module_name)
        assert mod is not None, f"Failed to import required module: {module_name}"


def test_env_example_blueprint_exists() -> None:
    """Verify .env.example blueprint exists and contains required keys.

    Ensures zero hardcoded secret compliance by verifying that key configuration
    placeholders are present in the blueprint template.
    """
    root_dir = Path(__file__).parent.parent.parent
    env_example_path = root_dir / ".env.example"

    assert env_example_path.exists(), ".env.example file must exist in backend root"

    content = env_example_path.read_text(encoding="utf-8")

    required_keys = [
        "ULTRON_ENV",
        "ULTRON_SECRET_KEY",
        "ULTRON_MEMORY_ENCRYPTION_KEY",
        "DATABASE_URL",
        "REDIS_URL",
        "OPENAI_API_KEY",
    ]

    for key in required_keys:
        assert key in content, f"Required configuration key missing from .env.example: {key}"


def test_pyproject_configuration_exists() -> None:
    """Verify pyproject.toml exists and configures python version constraint."""
    root_dir = Path(__file__).parent.parent.parent
    pyproject_path = root_dir / "pyproject.toml"

    assert pyproject_path.exists(), "pyproject.toml file must exist in backend root"

    content = pyproject_path.read_text(encoding="utf-8")
    assert 'requires-python = ">=3.12,<3.14"' in content
    assert 'name = "ultron-backend"' in content
    assert 'version = "0.1.0"' in content
