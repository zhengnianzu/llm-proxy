"""
API Key validation module for proxy authentication.
Supports keys from: SQLite database, environment variables, and settings/keys.yaml.
"""
import os
from typing import Optional
from fastapi import Request, HTTPException, status

from utils.key_store import validate_key as db_validate_key, has_active_keys as db_has_active_keys, get_key_id_by_value
from utils.key_config import validate_static_key, get_static_keys
from utils.channel_store import resolve_channel_for_key, get_default_channel, has_channel_bindings


def get_configured_api_keys() -> list[str]:
    """Get the configured API keys from environment variables.
    Supports multiple keys separated by commas."""
    raw = os.getenv("API_KEY", "").strip()
    if not raw:
        return []
    return [k.strip() for k in raw.split(",") if k.strip()]


def _is_key_access_enabled() -> bool:
    raw = os.getenv("ENABLE_KEY_ACCESS", "true").strip().lower()
    return raw not in ("false", "0", "off", "no")


def is_auth_enabled() -> bool:
    """Check if API key authentication is enabled.
    Controlled by ENABLE_KEY_ACCESS env (default true).
    When true, auth is enforced regardless of whether keys exist."""
    return _is_key_access_enabled()


def get_auth_status() -> dict:
    """Return current auth status for UI display."""
    if not _is_key_access_enabled():
        return {"enabled": False, "reason": "ENABLE_KEY_ACCESS=false"}
    env = bool(get_configured_api_keys())
    db = db_has_active_keys()
    yaml_keys = bool(get_static_keys())
    if env or db or yaml_keys:
        sources = [s for s, v in [("env", env), ("db", db), ("yaml", yaml_keys)] if v]
        return {"enabled": True, "reason": f"来源: {', '.join(sources)}"}
    return {"enabled": True, "reason": "未配置任何 Key，所有请求将被拒绝"}


async def validate_api_key(request: Request) -> str:
    """
    Validate API key from request headers.

    Checks: DB active keys -> yaml static keys -> env keys.
    Returns the matched key if auth is enabled, or empty string if auth is disabled.
    """
    if not is_auth_enabled():
        return ""

    env_keys = get_configured_api_keys()
    db_enabled = db_has_active_keys()
    static_keys = get_static_keys()

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


def resolve_upstream_channel(api_key: str) -> Optional[dict]:
    """根据已验证的 api_key 解析要使用的上游渠道。
    优先级：DB key 绑定的渠道 > 默认渠道 > None（回退到 .env）。
    返回的 dict 可能含 _fallback 字段标识回退原因。"""
    if api_key:
        key_id = get_key_id_by_value(api_key)
        if key_id is not None:
            ch = resolve_channel_for_key(key_id)
            if ch:
                return ch
            if has_channel_bindings(key_id):
                fallback = get_default_channel()
                if fallback:
                    fallback["_fallback"] = "bound_channels_offline"
                    return fallback
                # 无默认渠道，回退到 .env，但标记原因
                return {"_fallback": "bound_channels_offline"}
    return get_default_channel()
