"""
API Key validation module for proxy authentication.
"""
import os
from fastapi import Request, HTTPException, status


def get_configured_api_key() -> str:
    """Get the configured API key from environment variables."""
    return os.getenv("API_KEY", "").strip()


def is_auth_enabled() -> bool:
    """Check if API key authentication is enabled."""
    return bool(get_configured_api_key())


async def validate_api_key(request: Request) -> None:
    """
    Validate API key from request headers.

    Checks for API key in:
    1. Authorization header (Bearer token)
    2. x-api-key header

    Raises HTTPException if validation fails and auth is enabled.
    """
    configured_key = get_configured_api_key()

    # If no API key is configured, skip validation
    if not configured_key:
        return

    # Extract API key from headers
    api_key = None

    # Check Authorization header (Bearer token)
    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        api_key = auth_header.split("Bearer ", 1)[1].strip()

    # Check x-api-key header
    if not api_key:
        api_key = request.headers.get("x-api-key", "").strip()

    # Validate the API key
    if not api_key or api_key != configured_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
