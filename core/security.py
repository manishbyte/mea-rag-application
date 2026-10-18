
"""Password hashing and JWT access-token utilities."""

from datetime import datetime, timedelta, timezone
from uuid import UUID

import jwt
from pwdlib import PasswordHash

from core.config import get_settings

_password_hash = PasswordHash.recommended()


class TokenError(ValueError):
    """Raised when an access token is invalid or expired."""


def hash_password(password: str) -> str:
    """Hash a password using the recommended Argon2 configuration."""

    if len(password) < 8:
        raise ValueError("Password must contain at least 8 characters.")

    if len(password) > 128:
        raise ValueError("Password must not exceed 128 characters.")

    return _password_hash.hash(password)


def verify_password(password: str, hashed_password: str) -> bool:
    """Verify a plaintext password against its stored hash."""

    try:
        return _password_hash.verify(password, hashed_password)
    except (ValueError, TypeError):
        return False


def _get_jwt_secret() -> str:
    settings = get_settings()
    secret = settings.jwt_secret_key

    if secret is None or not secret.get_secret_value().strip():
        raise RuntimeError(
            "JWT_SECRET_KEY is missing. Configure a strong secret "
            "in your environment before starting the API."
        )

    value = secret.get_secret_value()

    if settings.app_env == "production" and len(value) < 32:
        raise RuntimeError(
            "JWT_SECRET_KEY must be at least 32 characters in production."
        )

    return value


def create_access_token(user_id: UUID | str) -> str:
    """Create a signed JWT with a subject and expiration timestamp."""

    settings = get_settings()
    now = datetime.now(timezone.utc)

    payload = {
        "sub": str(user_id),
        "iat": now,
        "exp": now + timedelta(
            minutes=settings.access_token_expire_minutes
        ),
    }

    return jwt.encode(
        payload,
        _get_jwt_secret(),
        algorithm=settings.jwt_algorithm,
    )


def decode_access_token(token: str) -> UUID:
    """Validate a JWT and return its user UUID."""

    settings = get_settings()

    try:
        payload = jwt.decode(
            token,
            _get_jwt_secret(),
            algorithms=[settings.jwt_algorithm],
            options={"require": ["sub", "exp", "iat"]},
        )
        return UUID(payload["sub"])
    except (
        jwt.InvalidTokenError,
        ValueError,
        TypeError,
        KeyError,
    ) as exc:
        raise TokenError("Invalid or expired access token.") from exc
