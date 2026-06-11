"""
Key 管理 Web 路由。
- /keys, /keys/login, /keys/logout — 独立登录（密码来自 key_state.yaml）
- /api/keys/* — 仅 key session 有效时可访问
- /invite, /api/invite — 公开（邀请码验证）
"""

import hmac
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from utils.key_store import add_key, list_keys, disable_key, enable_key, delete_key, mask_key
from utils.auth import get_configured_api_keys, get_auth_status
from utils.key_config import load_key_state


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
            return templates.TemplateResponse("keys_login.html", {"request": request})
        return templates.TemplateResponse("keys.html", {"request": request})

    @app.get("/api/keys")
    def api_keys_list(request: Request):
        denied = _require_key_api(request)
        if denied:
            return denied
        state = load_key_state()
        db_keys = list_keys()
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
        result = add_key(name, key_len=state.get("key_len", 24))
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

    @app.get("/invite")
    def invite_page(request: Request):
        state = load_key_state()
        if not state.get("invite_codes"):
            return JSONResponse({"detail": "Invite feature is disabled"}, status_code=404)
        return templates.TemplateResponse("invite.html", {"request": request})

    @app.post("/api/invite")
    async def api_invite(request: Request):
        if not _is_internal_request(request):
            return JSONResponse({"detail": "Not found"}, status_code=404)
        state = load_key_state()
        valid_codes = state.get("invite_codes", [])
        if not valid_codes:
            return JSONResponse({"detail": "Invite feature is disabled"}, status_code=404)
        body = await request.json()
        code = body.get("invite_code", "").strip()
        name = body.get("name", "").strip()
        matched = any(hmac.compare_digest(code, c) for c in valid_codes)
        if not matched:
            return JSONResponse({"detail": "Invalid invite code"}, status_code=403)
        result = add_key(name or "invite", key_len=state.get("key_len", 24), invite_code=code)
        return JSONResponse(result)
