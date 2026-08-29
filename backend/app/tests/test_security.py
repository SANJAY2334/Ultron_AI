"""Unit Tests for Cryptographic & Security Subsystem.

Validates password hashing/verification, JWT token creation and decoding,
expired token rejection, and AES-256 Fernet payload encryption.
"""

from datetime import timedelta

import jwt
import pytest

from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_jwt_token,
    decrypt_payload,
    encrypt_payload,
    generate_fernet_key,
    hash_password,
    verify_password,
)


def test_password_hashing_and_verification() -> None:
    """Verify password hashing and verification."""
    password = "UltronSecurePassword123!"
    hashed = hash_password(password)

    assert hashed != password
    assert verify_password(password, hashed) is True
    assert verify_password("WrongPassword!", hashed) is False


def test_jwt_access_and_refresh_token_lifecycle() -> None:
    """Verify JWT access and refresh token creation and decoding."""
    secret_key = "test_super_secret_jwt_key_32_bytes"

    # Access Token
    access_token = create_access_token(
        subject="user_123",
        secret_key=secret_key,
        extra_claims={"role": "Admin"},
    )
    assert access_token is not None

    decoded_access = decode_jwt_token(access_token, secret_key)
    assert decoded_access["sub"] == "user_123"
    assert decoded_access["role"] == "Admin"
    assert decoded_access["type"] == "access"

    # Refresh Token
    refresh_token = create_refresh_token(subject="user_123", secret_key=secret_key)
    decoded_refresh = decode_jwt_token(refresh_token, secret_key)
    assert decoded_refresh["sub"] == "user_123"
    assert decoded_refresh["type"] == "refresh"


def test_jwt_expired_token_rejection() -> None:
    """Verify that expired JWT tokens raise ExpiredSignatureError."""
    secret_key = "test_super_secret_jwt_key_32_bytes"
    expired_token = create_access_token(
        subject="user_expired",
        secret_key=secret_key,
        expires_delta=timedelta(seconds=-10),
    )

    with pytest.raises(jwt.ExpiredSignatureError):
        decode_jwt_token(expired_token, secret_key)


def test_fernet_payload_encryption_and_decryption() -> None:
    """Verify AES-256 Fernet payload encryption and decryption round-trip."""
    fernet_key = generate_fernet_key()
    original_payload = "Sensitive User Preference: User prefers pitch black dark HUD theme."

    ciphertext = encrypt_payload(original_payload, fernet_key)
    assert ciphertext != original_payload

    decrypted = decrypt_payload(ciphertext, fernet_key)
    assert decrypted == original_payload
