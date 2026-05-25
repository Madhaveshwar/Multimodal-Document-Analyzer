"""Security helpers for password hashing and session tokens."""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
from typing import Tuple

_HASH_NAME = "sha256"
_HASH_ITERATIONS = 260_000
_SALT_BYTES = 16
_DERIVED_KEY_BYTES = 32


def hash_password(password: str) -> str:
    """Return a PBKDF2 password hash with embedded salt and iterations."""
    if not password:
        raise ValueError("Password cannot be empty.")

    salt = secrets.token_bytes(_SALT_BYTES)
    derived_key = hashlib.pbkdf2_hmac(
        _HASH_NAME,
        password.encode("utf-8"),
        salt,
        _HASH_ITERATIONS,
        dklen=_DERIVED_KEY_BYTES,
    )
    return "pbkdf2_sha256${iterations}${salt}${hash_value}".format(
        iterations=_HASH_ITERATIONS,
        salt=base64.urlsafe_b64encode(salt).decode("ascii"),
        hash_value=base64.urlsafe_b64encode(derived_key).decode("ascii"),
    )


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a PBKDF2 password hash created by hash_password()."""
    try:
        algorithm, iterations_text, salt_text, hash_text = password_hash.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        iterations = int(iterations_text)
        salt = base64.urlsafe_b64decode(salt_text.encode("ascii"))
        expected_hash = base64.urlsafe_b64decode(hash_text.encode("ascii"))
    except Exception:
        return False

    candidate_hash = hashlib.pbkdf2_hmac(
        _HASH_NAME,
        password.encode("utf-8"),
        salt,
        iterations,
        dklen=len(expected_hash),
    )
    return hmac.compare_digest(candidate_hash, expected_hash)


def generate_session_token() -> str:
    """Generate a secure opaque session token."""
    return secrets.token_urlsafe(32)
