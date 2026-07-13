"""
渠道管理 Web 路由。
- /channels — 渠道管理界面（需 admin 登录，由 MonitorAuthMiddleware 控制）
- /api/channels/* — 渠道 CRUD
- /api/keys/{key_id}/channels — Key-Channel 绑定
"""

import os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates

from utils.channel_store import (
    add_channel, list_channels, toggle_channel_alive, delete_channel,
    set_key_channels, get_key_channels, set_default_channel,
    update_channel_invite_code, update_channel_type, get_channel_raw,
)
from utils.connection_test import run_test


def _is_internal_request(request: Request) -> bool:
    return request.headers.get("x-requested-with") == "XMLHttpRequest"


def _require_key_api(request: Request):
    if not _is_internal_request(request):
        return JSONResponse({"detail": "Not found"}, status_code=404)
    return None


def register_channel_routes(app: FastAPI, templates: Jinja2Templates):

    @app.get("/channels")
    def channels_page(request: Request):
        return templates.TemplateResponse(request, "channels.html", context={"active_page": "channels", "user_role": request.session.get("monitor_role", "user"), "user_name": request.session.get("monitor_user", ""), "user_permissions": [p.strip() for p in (request.session.get("monitor_permissions") or "").split(",") if p.strip()]})

    @app.get("/api/channels")
    def api_channels_list(request: Request):
        denied = _require_key_api(request)
        if denied:
            return denied
        return JSONResponse({"channels": list_channels()})

    @app.post("/api/channels")
    async def api_channels_create(request: Request):
        denied = _require_key_api(request)
        if denied:
            return denied
        body = await request.json()
        name = body.get("name", "").strip()
        upstream_url = body.get("upstream_url", "").strip()
        upstream_key = body.get("upstream_key", "").strip()
        invite_code = body.get("invite_code", "").strip()
        channel_type = body.get("channel_type", "").strip()
        if not upstream_url or not upstream_key:
            return JSONResponse({"detail": "upstream_url and upstream_key are required"}, status_code=400)
        result = add_channel(name, upstream_url, upstream_key, invite_code=invite_code, channel_type=channel_type)
        return JSONResponse(result)

    @app.post("/api/channels/test")
    async def api_channel_test(request: Request):
        denied = _require_key_api(request)
        if denied:
            return denied
        body = await request.json()
        channel_id = body.get("channel_id")
        api_key = body.get("api_key", "")
        model = body.get("model", "claude-sonnet-4-6")
        method = body.get("method", "anthropic")
        message = body.get("message", "")

        if channel_id:
            ch = get_channel_raw(channel_id)
            if not ch:
                return JSONResponse({"ok": False, "error": "渠道不存在"}, status_code=404)
            base_url = ch["upstream_url"].rstrip("/").removesuffix("/v1")
            result = run_test(base_url, method, model, ch["upstream_key"], message=message)
        elif api_key:
            port = os.getenv("PROXY_PORT", "4000")
            result = run_test(f"http://127.0.0.1:{port}", method, model, api_key, message=message)
        else:
            return JSONResponse({"ok": False, "error": "需要 channel_id 或 api_key"}, status_code=400)

        return JSONResponse(result)

    @app.post("/api/channels/{channel_id}/toggle")
    async def api_channels_toggle(request: Request, channel_id: int):
        denied = _require_key_api(request)
        if denied:
            return denied
        body = await request.json()
        alive = body.get("alive", True)
        return JSONResponse({"success": toggle_channel_alive(channel_id, alive)})

    @app.delete("/api/channels/{channel_id}")
    def api_channels_delete(request: Request, channel_id: int):
        denied = _require_key_api(request)
        if denied:
            return denied
        return JSONResponse({"success": delete_channel(channel_id)})

    @app.post("/api/channels/{channel_id}/set-default")
    def api_channels_set_default(request: Request, channel_id: int):
        denied = _require_key_api(request)
        if denied:
            return denied
        return JSONResponse({"success": set_default_channel(channel_id)})

    @app.post("/api/channels/{channel_id}/invite-code")
    async def api_channels_update_invite_code(request: Request, channel_id: int):
        denied = _require_key_api(request)
        if denied:
            return denied
        body = await request.json()
        invite_code = body.get("invite_code", "").strip()
        return JSONResponse({"success": update_channel_invite_code(channel_id, invite_code)})

    @app.post("/api/channels/{channel_id}/channel-type")
    async def api_channels_update_type(request: Request, channel_id: int):
        denied = _require_key_api(request)
        if denied:
            return denied
        body = await request.json()
        channel_type = body.get("channel_type", "").strip()
        return JSONResponse({"success": update_channel_type(channel_id, channel_type)})

    @app.get("/api/keys/{key_id}/channels")
    def api_key_channels(request: Request, key_id: int):
        denied = _require_key_api(request)
        if denied:
            return denied
        return JSONResponse({"channels": get_key_channels(key_id)})

    @app.post("/api/keys/{key_id}/channels")
    async def api_key_channels_set(request: Request, key_id: int):
        denied = _require_key_api(request)
        if denied:
            return denied
        body = await request.json()
        channel_ids = body.get("channel_ids", [])
        set_key_channels(key_id, channel_ids)
        return JSONResponse({"success": True})
