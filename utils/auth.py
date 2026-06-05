"""
API Key validation module for proxy authentication.
Supports keys from: SQLite database, environment variables, and settings/keys.yaml.
"""
import os
from fastapi import Request, HTTPException, status

from utils.key_store import validate_key as db_validate_key, has_active_keys as db_has_active_keys
from utils.key_config import validate_static_key, get_static_keys


def get_configured_api_keys() -> list[str]:
    """Get the configured API keys from environment variables.
    Supports multiple keys separated by commas."""
    raw = os.getenv("API_KEY", "").strip()
    if not raw:
        return []
    return [k.strip() for k in raw.split(",") if k.strip()]


def is_auth_enabled() -> bool:
    """Check if API key authentication is enabled (DB, env, or yaml)."""
    return bool(get_configured_api_keys()) or db_has_active_keys() or bool(get_static_keys())


async def validate_api_key(request: Request) -> str:
    """
    Validate API key from request headers.

    Checks: DB active keys -> yaml static keys -> env keys.
    Returns the matched key if auth is enabled, or empty string if auth is disabled.
    """
    env_keys = get_configured_api_keys()
    db_enabled = db_has_active_keys()
    static_keys = get_static_keys()

    if not env_keys and not db_enabled and not static_keys:
        return ""

    api_key = None

    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        api_key = auth_header.split("Bearer ", 1)[1].strip()

    if not api_key:
        api_key = request.headers.get("x-api-key", "").strip()

    if api_key:
        if db_validate_key(api_key) is not None:
            return api_key
        if validate_static_key(api_key) is not None:
            return api_key
        for key in env_keys:
            if api_key == key:
                return api_key

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API key",
        headers={"WWW-Authenticate": "Bearer"},
    )
