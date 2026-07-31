"""
用户管理 Web 路由（admin-only，由 MonitorAuthMiddleware 控制）。
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates

from utils.user_store import list_users, create_user, update_user, delete_user


def _is_internal(request: Request) -> bool:
    return request.headers.get("x-requested-with") == "XMLHttpRequest"


def register_user_routes(app: FastAPI, templates: Jinja2Templates, context_builder=None):

    @app.get("/users")
    def users_page(request: Request):
        if context_builder is not None:
            context = context_builder(request, "users")
        else:
            perms_raw = request.session.get("monitor_permissions") or ""
            perms_list = [p.strip() for p in perms_raw.split(",") if p.strip()]
            context = {
                "active_page": "users",
                "user_role": request.session.get("monitor_role", "user"),
                "user_name": request.session.get("monitor_user", ""),
                "user_permissions": perms_list,
            }
        return templates.TemplateResponse(request, "users.html", context=context)

    @app.get("/api/users")
    def api_users_list(request: Request):
        if not _is_internal(request):
            return JSONResponse({"detail": "Not found"}, status_code=404)
        return JSONResponse({"users": list_users()})

    @app.post("/api/users")
    async def api_users_create(request: Request):
        if not _is_internal(request):
            return JSONResponse({"detail": "Not found"}, status_code=404)
        body = await request.json()
        username = (body.get("username") or "").strip()
        password = body.get("password") or ""
        role = body.get("role", "user")
        permissions = body.get("permissions", "")

        if not username or not password:
            return JSONResponse({"detail": "用户名和密码不能为空"}, status_code=400)
        if len(username) < 2 or len(username) > 32:
            return JSONResponse({"detail": "用户名长度应在 2-32 个字符之间"}, status_code=400)
        if len(password) < 6:
            return JSONResponse({"detail": "密码长度至少 6 个字符"}, status_code=400)
        if role not in ("admin", "user"):
            role = "user"

        user = create_user(username, password, role=role, permissions=permissions)
        if user is None:
            return JSONResponse({"detail": "该用户名已被注册"}, status_code=409)
        return JSONResponse({"ok": True, "user": {"id": user["id"], "username": user["username"], "role": user["role"], "permissions": user.get("permissions", "")}})

    @app.put("/api/users/{user_id}")
    async def api_users_update(request: Request, user_id: int):
        if not _is_internal(request):
            return JSONResponse({"detail": "Not found"}, status_code=404)
        body = await request.json()
        role = body.get("role")
        permissions = body.get("permissions")
        ok = update_user(user_id, role=role, permissions=permissions)
        if not ok:
            return JSONResponse({"detail": "用户不存在"}, status_code=404)
        return JSONResponse({"ok": True})

    @app.delete("/api/users/{user_id}")
    async def api_users_delete(request: Request, user_id: int):
        if not _is_internal(request):
            return JSONResponse({"detail": "Not found"}, status_code=404)
        users = list_users()
        target = next((u for u in users if u["id"] == user_id), None)
        if not target:
            return JSONResponse({"detail": "用户不存在"}, status_code=404)
        current_user = request.session.get("monitor_user", "")
        if target["username"] == current_user:
            return JSONResponse({"detail": "不能删除当前登录的用户"}, status_code=400)
        delete_user(user_id)
        return JSONResponse({"ok": True})
