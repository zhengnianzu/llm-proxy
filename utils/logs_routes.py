"""
utils/logs_routes.py — 日志管理页面路由

提供:
  GET  /logs-admin                 — 日志管理页面
  GET  /api/logs-admin/list        — 列出活跃 base + 历史路径（含大小/格式/请求数）
  POST /api/logs-admin/add         — 新增历史路径
  POST /api/logs-admin/remove      — 移除历史路径（不删磁盘文件）

「都是到 env-key 这个层级」：历史路径应指向 env-key 层目录，
或包含 index.jsonl 叶子（含可变深度，如 new-api 的 details）。
统计/预览会自动合并这些路径。
"""

import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates

from utils.logs_config import (
    get_active_base,
    get_history_paths,
    add_history_path,
    remove_history_path,
    dir_size,
    human_size,
)
from utils.log_scan import detect_format, iter_index_dirs


def _require_ajax(request: Request):
    if request.headers.get("x-requested-with", "").lower() != "xmlhttprequest":
        return JSONResponse({"detail": "Forbidden"}, status_code=403)
    return None


def _is_admin(request: Request) -> bool:
    role = request.session.get("monitor_role", "admin")
    return role == "admin"


def _describe(path: str, active: bool, active_env_dir: str = "") -> dict:
    exists = os.path.isdir(path)
    fmt = detect_format(path) if exists else "missing"
    leaf_count = 0
    if exists:
        try:
            leaf_count = sum(1 for _ in iter_index_dirs(Path(path)))
        except Exception:
            leaf_count = 0
    return {
        "path": path,
        "active": active,
        "exists": exists,
        "format": fmt,
        "leaf_count": leaf_count,
        "status": "活跃" if active else "历史",
    }


def register_logs_routes(app: FastAPI, templates: Jinja2Templates, active_env_dir: str = "") -> None:

    def _ctx(request: Request) -> dict:
        return {
            "active_page": "logs_admin",
            "user_role": request.session.get("monitor_role", "admin"),
            "user_name": request.session.get("monitor_user", ""),
            "user_permissions": [
                p.strip()
                for p in (request.session.get("monitor_permissions") or "").split(",")
                if p.strip()
            ],
        }

    @app.get("/logs-admin")
    def logs_admin_page(request: Request):
        return templates.TemplateResponse(request, "logs_admin.html", context=_ctx(request))

    @app.get("/api/logs-admin/list")
    def logs_admin_list(request: Request, with_size: bool = False):
        denied = _require_ajax(request)
        if denied:
            return denied

        active_base = get_active_base()
        rows = []

        # 活跃：进程实际写入的 env_dir
        active_row = _describe(active_env_dir, active=True)
        active_row["base"] = active_base
        rows.append(active_row)

        # 历史
        for p in get_history_paths():
            # 跳过与活跃 env_dir 相同的路径
            if os.path.normpath(p) == os.path.normpath(active_env_dir):
                continue
            rows.append(_describe(p, active=False))

        if with_size:
            for r in rows:
                if r["exists"]:
                    nbytes = dir_size(r["path"])
                    r["size_bytes"] = nbytes
                    r["size"] = human_size(nbytes)
                else:
                    r["size_bytes"] = 0
                    r["size"] = "-"

        return JSONResponse({
            "active_base": active_base,
            "active_env_dir": active_env_dir,
            "rows": rows,
        })

    @app.post("/api/logs-admin/add")
    async def logs_admin_add(request: Request):
        denied = _require_ajax(request)
        if denied:
            return denied
        if not _is_admin(request):
            return JSONResponse({"ok": False, "msg": "仅管理员可操作"}, status_code=403)

        body = await request.json()
        path = (body.get("path") or "").strip()
        ok, msg = add_history_path(path)
        return JSONResponse({"ok": ok, "msg": msg})

    @app.post("/api/logs-admin/remove")
    async def logs_admin_remove(request: Request):
        denied = _require_ajax(request)
        if denied:
            return denied
        if not _is_admin(request):
            return JSONResponse({"ok": False, "msg": "仅管理员可操作"}, status_code=403)

        body = await request.json()
        path = (body.get("path") or "").strip()
        ok, msg = remove_history_path(path)
        return JSONResponse({"ok": ok, "msg": msg})

    @app.post("/api/logs-admin/backfill")
    async def logs_admin_backfill(request: Request):
        """启动 new-api 叶子 index.db 后台构建（ingest+补 meta+聚合 sessions；历史预览/导出前置）。"""
        denied = _require_ajax(request)
        if denied:
            return denied
        if not _is_admin(request):
            return JSONResponse({"ok": False, "msg": "仅管理员可操作"}, status_code=403)

        body = await request.json()
        path = (body.get("path") or "").strip()
        workers = body.get("workers")
        force = bool(body.get("force", False))
        if not os.path.isdir(path):
            return JSONResponse({"ok": False, "msg": f"目录不存在: {path}"}, status_code=400)
        from utils.newapi_backfill import start_backfill
        st = start_backfill(path, workers=int(workers) if workers else None, force=force)
        return JSONResponse({"ok": True, "status": st})

    @app.get("/api/logs-admin/backfill-status")
    def logs_admin_backfill_status(request: Request, path: str = ""):
        denied = _require_ajax(request)
        if denied:
            return denied
        from utils.newapi_backfill import get_backfill_status
        return JSONResponse(get_backfill_status(path))
