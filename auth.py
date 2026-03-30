"""
API Key validation module for proxy authentication.
"""
import os
from fastapi import Request, HTTPException, status


def get_configured_api_keys() -> list[str]:
    """Get the configured API keys from environment variables.
    Supports multiple keys separated by commas."""
    raw = os.getenv("API_KEY", "").strip()
    if not raw:
        return []
    return [k.strip() for k in raw.split(",") if k.strip()]


def is_auth_enabled() -> bool:
    """Check if API key authentication is enabled."""
    return bool(get_configured_api_keys())


async def validate_api_key(request: Request) -> str:
    """
    Validate API key from request headers.

    Checks for API key in:
    1. Authorization header (Bearer token)
    2. x-api-key header

    Returns the matched key (masked as 'key_{index+1}') if auth is enabled,
    or empty string if auth is disabled.
    Raises HTTPException if validation fails and auth is enabled.
    """
    configured_keys = get_configured_api_keys()

    # If no API key is configured, skip validation
    if not configured_keys:
        return ""

    # Extract API key from headers
    api_key = None

    # Check Authorization header (Bearer token)
    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        api_key = auth_header.split("Bearer ", 1)[1].strip()

    # Check x-api-key header
    if not api_key:
        api_key = request.headers.get("x-api-key", "").strip()

    # Validate the API key against all configured keys
    if api_key:
        for key in configured_keys:
            if api_key == key:
                return api_key

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API key",
        headers={"WWW-Authenticate": "Bearer"},
    )
