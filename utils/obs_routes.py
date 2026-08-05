"""
utils/obs_routes.py — OBS 管理页面路由

提供:
  GET /obs                              — OBS 管理页面
  GET /api/obs/browse/{path}?refresh=1  — 列出 OBS 目录内容（默认走缓存，refresh 强制刷新）
  GET /api/obs/cat/{path}               — 下载并返回 OBS 文件内容（≤10MB）
"""

import os
import subprocess
import tempfile
import threading
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates

from utils.obs_utils import OBSUTIL_BIN, obsutil_ls


# 目录列表缓存：path -> items（进程内存，服务重启即失效）
_browse_cache: dict = {}
_browse_cache_lock = threading.Lock()


def _require_ajax(request: Request):
    if request.headers.get("x-requested-with", "").lower() != "xmlhttprequest":
        return JSONResponse({"detail": "Forbidden"}, status_code=403)
    return None


def register_obs_routes(app: FastAPI, templates: Jinja2Templates, context_builder=None) -> None:

    def _ctx(request: Request) -> dict:
        if context_builder is not None:
            return context_builder(request, "obs")
        return {
            "active_page": "obs",
            "user_role": request.session.get("monitor_role", "user"),
            "user_name": request.session.get("monitor_user", ""),
            "user_permissions": [
                p.strip()
                for p in (request.session.get("monitor_permissions") or "").split(",")
                if p.strip()
            ],
        }

    @app.get("/obs")
    def obs_page(request: Request):
        return templates.TemplateResponse(request, "obs.html", context=_ctx(request))

    @app.get("/api/obs/browse/{obs_path:path}")
    def obs_browse(request: Request, obs_path: str):
        denied = _require_ajax(request)
        if denied:
            return denied
        if not obs_path.startswith("obs://"):
            obs_path = "obs://" + obs_path
        refresh = request.query_params.get("refresh", "") in ("1", "true", "yes")

        if not refresh:
            with _browse_cache_lock:
                cached = _browse_cache.get(obs_path)
            if cached is not None:
                return JSONResponse({"path": obs_path, "items": cached, "cached": True})

        try:
            items = obsutil_ls(obs_path)
            with _browse_cache_lock:
                _browse_cache[obs_path] = items
            return JSONResponse({"path": obs_path, "items": items, "cached": False})
        except Exception as e:
            return JSONResponse({"detail": str(e)}, status_code=500)

    @app.get("/api/obs/cat/{obs_path:path}")
    def obs_cat(request: Request, obs_path: str):
        denied = _require_ajax(request)
        if denied:
            return denied
        if not obs_path.startswith("obs://"):
            obs_path = "obs://" + obs_path
        if not os.path.isfile(OBSUTIL_BIN):
            return JSONResponse({"detail": "obsutil 未找到"}, status_code=500)

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as f:
                tmp_path = f.name
            r = subprocess.run(
                [OBSUTIL_BIN, "cp", obs_path, tmp_path, "-f"],
                capture_output=True, text=True, timeout=30,
            )
            if r.returncode != 0:
                detail = (r.stderr or r.stdout or "下载失败").strip()[-500:]
                return JSONResponse({"detail": detail}, status_code=500)
            size = os.path.getsize(tmp_path)
            if size > 10 * 1024 * 1024:
                return JSONResponse(
                    {"detail": f"文件过大（{size // 1024 // 1024}MB），不支持预览"},
                    status_code=413,
                )
            content = Path(tmp_path).read_text(encoding="utf-8", errors="replace")
            return JSONResponse({"content": content, "size": size})
        except subprocess.TimeoutExpired:
            return JSONResponse({"detail": "下载超时（30s）"}, status_code=500)
        except Exception as e:
            return JSONResponse({"detail": str(e)}, status_code=500)
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
