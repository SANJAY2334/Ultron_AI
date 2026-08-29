"""Cryptographic & Security Subsystem.

Provides JWT token management, security-hardened password hashing (Argon2/bcrypt),
and AES-256 Fernet payload encryption for sensitive memory items.
"""

from datetime import UTC, datetime, timedelta
from typing import Any

import jwt
from cryptography.fernet import Fernet
from passlib.context import CryptContext  # type: ignore[import-untyped]

# Password hashing context (Argon2 primary, bcrypt fallback)
pwd_context = CryptContext(schemes=["argon2", "bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    """Hashes a raw password string using Argon2/bcrypt.

    Args:
        password: Raw plain-text password.

    Returns:
        str: Cryptographically secure hashed password string.
    """
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain-text password against a stored hashed password.

    Args:
        plain_password: Candidate plain-text password.
        hashed_password: Stored target hashed password.

    Returns:
        bool: True if match is verified, False otherwise.
    """
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(
    subject: str | Any,
    secret_key: str,
    algorithm: str = "HS256",
    expires_delta: timedelta | None = None,
    extra_claims: dict[str, Any] | None = None,
) -> str:
    """Generates a signed JWT access token.

    Args:
        subject: Subject claim ('sub'), typically user ID or username.
        secret_key: Secret key used to sign the token.
        algorithm: Signing algorithm (default 'HS256').
        expires_delta: Optional custom lifespan (default 15 minutes).
        extra_claims: Optional dictionary of additional claims (roles, permissions).

    Returns:
        str: Encoded, signed JWT string.
    """
    now = datetime.now(UTC)
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=15)

    payload: dict[str, Any] = {
        "sub": str(subject),
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
        "type": "access",
    }
    if extra_claims:
        payload.update(extra_claims)

    return jwt.encode(payload, secret_key, algorithm=algorithm)


def create_refresh_token(
    subject: str | Any,
    secret_key: str,
    algorithm: str = "HS256",
    expires_delta: timedelta | None = None,
) -> str:
    """Generates a signed JWT refresh token.

    Args:
        subject: Subject claim ('sub'), typically user ID or username.
        secret_key: Secret key used to sign the token.
        algorithm: Signing algorithm (default 'HS256').
        expires_delta: Optional custom lifespan (default 7 days).

    Returns:
        str: Encoded, signed JWT refresh token string.
    """
    now = datetime.now(UTC)
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(days=7)

    payload: dict[str, Any] = {
        "sub": str(subject),
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
        "type": "refresh",
    }

    return jwt.encode(payload, secret_key, algorithm=algorithm)


def decode_jwt_token(
    token: str, secret_key: str, algorithms: list[str] | None = None
) -> dict[str, Any]:
    """Decodes and validates a signed JWT token.

    Args:
        token: JWT string to decode.
        secret_key: Secret key used for signature verification.
        algorithms: Allowed signing algorithms (default ['HS256']).

    Returns:
        dict[str, Any]: Decoded payload dictionary.

    Raises:
        jwt.ExpiredSignatureError: If token has expired.
        jwt.PyJWTError: If signature is invalid or payload is malformed.
    """
    if algorithms is None:
        algorithms = ["HS256"]
    return jwt.decode(token, secret_key, algorithms=algorithms)


def generate_fernet_key() -> str:
    """Generates a new random 32-byte URL-safe base64 key for Fernet encryption.

    Returns:
        str: URL-safe base64 Fernet key.
    """
    return Fernet.generate_key().decode("utf-8")


def encrypt_payload(payload: str, fernet_key: str) -> str:
    """Encrypts a string payload using AES-256 Fernet symmetric encryption.

    Args:
        payload: Plain-text string payload to encrypt.
        fernet_key: 32-byte URL-safe base64 Fernet key.

    Returns:
        str: Encrypted ciphertext string (base64 encoded).
    """
    fernet = Fernet(fernet_key.encode("utf-8") if isinstance(fernet_key, str) else fernet_key)
    return fernet.encrypt(payload.encode("utf-8")).decode("utf-8")


def decrypt_payload(ciphertext: str, fernet_key: str) -> str:
    """Decrypts a Fernet-encrypted ciphertext string back to plain-text.

    Args:
        ciphertext: Encrypted ciphertext string.
        fernet_key: 32-byte URL-safe base64 Fernet key.

    Returns:
        str: Decrypted plain-text string payload.
    """
    fernet = Fernet(fernet_key.encode("utf-8") if isinstance(fernet_key, str) else fernet_key)
    return fernet.decrypt(ciphertext.encode("utf-8")).decode("utf-8")
