"""
Key 管理 Web 路由。
- /keys, /keys/login, /keys/logout — 独立登录（密码来自 key_state.yaml）
- /api/keys/* — 仅 key session 有效时可访问
- /invite, /api/invite — 公开（邀请码验证）
"""

import hmac
import os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from utils.key_store import add_key, list_keys, disable_key, enable_key, delete_key, mask_key
from utils.auth import get_configured_api_keys, get_auth_status
from utils.key_config import load_key_state
from utils.channel_store import get_channels_for_key_display, set_key_channels, get_channel_ids_by_invite_code, get_default_channel_id, get_all_channel_invite_codes


def _is_key_authenticated(request: Request) -> bool:
    return bool(request.session.get("key_authenticated"))


def _is_internal_request(request: Request) -> bool:
    """只允许页面内 AJAX 调用，浏览器直接访问拒绝。"""
    return request.headers.get("x-requested-with") == "XMLHttpRequest"


def _require_key_api(request: Request):
    """API 接口鉴权：必须是内部 AJAX + key session 有效。"""
    if not _is_internal_request(request):
        return JSONResponse({"detail": "Not found"}, status_code=404)
    state = load_key_state()
    if not state.get("password"):
        return None
    if _is_key_authenticated(request):
        return None
    return JSONResponse({"detail": "Key management login required"}, status_code=401)


def register_key_routes(app: FastAPI, templates: Jinja2Templates):

    @app.post("/keys")
    async def keys_login(request: Request):
        if not _is_internal_request(request):
            return JSONResponse({"detail": "Not found"}, status_code=404)
        state = load_key_state()
        body = await request.json()
        user = body.get("user", "")
        password = body.get("password", "")
        expected_user = state.get("user", "")
        expected_pwd = state.get("password", "")
        if not expected_pwd:
            return JSONResponse({"detail": "No password configured"}, status_code=403)
        user_ok = (not expected_user) or hmac.compare_digest(user, expected_user)
        pwd_ok = hmac.compare_digest(password, expected_pwd)
        if not (user_ok and pwd_ok):
            return JSONResponse({"detail": "Invalid credentials"}, status_code=403)
        request.session["key_authenticated"] = True
        return JSONResponse({"success": True})

    @app.get("/keys/logout")
    def keys_logout(request: Request):
        request.session.pop("key_authenticated", None)
        return RedirectResponse(url="/keys", status_code=303)

    @app.get("/keys")
    def keys_page(request: Request):
        state = load_key_state()
        if state.get("password") and not _is_key_authenticated(request):
            return templates.TemplateResponse(request, "keys_login.html")
        return templates.TemplateResponse(request, "keys.html")

    @app.get("/api/keys")
    def api_keys_list(request: Request):
        denied = _require_key_api(request)
        if denied:
            return denied
        state = load_key_state()
        db_keys = list_keys()
        for k in db_keys:
            k["channels"] = get_channels_for_key_display(k["id"])
        env_keys = get_configured_api_keys()
        env_list = [{"key": mask_key(k), "source": "env"} for k in env_keys]
        yaml_keys = [{"name": k.get("name", ""), "key": mask_key(k.get("value", "")), "source": "yaml"} for k in state.get("keys", [])]
        return JSONResponse({"db_keys": db_keys, "env_keys": env_list, "yaml_keys": yaml_keys, "auth_status": get_auth_status()})

    @app.post("/api/keys")
    async def api_keys_create(request: Request):
        denied = _require_key_api(request)
        if denied:
            return denied
        state = load_key_state()
        body = await request.json()
        name = body.get("name", "").strip()
        channel_ids = body.get("channel_ids", [])
        result = add_key(name, key_len=state.get("key_len", 24))
        if channel_ids:
            set_key_channels(result["id"], channel_ids)
        return JSONResponse(result)

    @app.post("/api/keys/{key_id}/disable")
    def api_keys_disable(request: Request, key_id: int):
        denied = _require_key_api(request)
        if denied:
            return denied
        return JSONResponse({"success": disable_key(key_id)})

    @app.post("/api/keys/{key_id}/enable")
    def api_keys_enable(request: Request, key_id: int):
        denied = _require_key_api(request)
        if denied:
            return denied
        return JSONResponse({"success": enable_key(key_id)})

    @app.delete("/api/keys/{key_id}")
    def api_keys_delete(request: Request, key_id: int):
        denied = _require_key_api(request)
        if denied:
            return denied
        return JSONResponse({"success": delete_key(key_id)})

    @app.post("/api/keys/auth-toggle")
    async def api_keys_auth_toggle(request: Request):
        denied = _require_key_api(request)
        if denied:
            return denied
        body = await request.json()
        enabled = body.get("enabled", True)
        os.environ["ENABLE_KEY_ACCESS"] = "true" if enabled else "false"
        return JSONResponse({"success": True, "auth_status": get_auth_status()})

    def _bind_channels_for_invite(key_id: int, code: str):
        """invite 创建 key 后自动绑定渠道：按 invite_code 匹配，多个则随机选一个；无匹配则绑定默认渠道。"""
        ch_ids = get_channel_ids_by_invite_code(code)
        if ch_ids:
            import random
            set_key_channels(key_id, [random.choice(ch_ids)])
            return
        default_id = get_default_channel_id()
        if default_id is not None:
            set_key_channels(key_id, [default_id])

    @app.get("/invite")
    def invite_page(request: Request):
        state = load_key_state()
        yaml_codes = state.get("invite_codes", [])
        channel_codes = get_all_channel_invite_codes()
        if not yaml_codes and not channel_codes:
            return JSONResponse({"detail": "Invite feature is disabled"}, status_code=404)
        return templates.TemplateResponse(request, "invite.html")

    @app.post("/api/invite")
    async def api_invite(request: Request):
        if not _is_internal_request(request):
            return JSONResponse({"detail": "Not found"}, status_code=404)
        state = load_key_state()
        yaml_codes = state.get("invite_codes", [])
        channel_codes = get_all_channel_invite_codes()
        if not yaml_codes and not channel_codes:
            return JSONResponse({"detail": "Invite feature is disabled"}, status_code=404)
        body = await request.json()
        code = body.get("invite_code", "").strip()
        name = body.get("name", "").strip()
        matched = any(hmac.compare_digest(code, c) for c in yaml_codes) or code in channel_codes
        if not matched:
            return JSONResponse({"detail": "Invalid invite code"}, status_code=403)
        result = add_key(name or "invite", key_len=state.get("key_len", 24), invite_code=code)
        _bind_channels_for_invite(result["id"], code)
        return JSONResponse(result)

    @app.get("/api/invite")
    def api_invite_get(invite_code: str = "", name: str = ""):
        code = invite_code.strip()
        name = name.strip()
        if not code:
            return JSONResponse({"detail": "invite_code is required"}, status_code=400)
        state = load_key_state()
        yaml_codes = state.get("invite_codes", [])
        channel_codes = get_all_channel_invite_codes()
        if not yaml_codes and not channel_codes:
            return JSONResponse({"detail": "Invite feature is disabled"}, status_code=404)
        matched = any(hmac.compare_digest(code, c) for c in yaml_codes) or code in channel_codes
        if not matched:
            return JSONResponse({"detail": "Invalid invite code"}, status_code=403)
        result = add_key(name or "invite", key_len=state.get("key_len", 24), invite_code=code)
        _bind_channels_for_invite(result["id"], code)
        return JSONResponse({"api_key": result["key"], "name": result["name"], "created_at": result["created_at"]})
