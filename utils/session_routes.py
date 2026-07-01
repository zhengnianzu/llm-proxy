"""
utils/session_routes.py — Session 会话统计 API

使用 stats_index 增量索引，避免每次全量扫描 .session_cache.jsonl。
索引持久化在 {env_dir}/.stats_index.json，只重新读变化的目录。
"""

import os
import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from utils.log_paths import get_service_log_dir
from utils.stats_index import (
    refresh_index,
    build_stats_from_index,
    QUALIFIED_THRESHOLD_DEFAULT,
)

_CACHE_TTL = 30

_mem_cache: dict = {}
_mem_cache_ts: float = 0


def _build_stats_json(env_dir: Path, threshold: int = QUALIFIED_THRESHOLD_DEFAULT) -> dict:
    """兼容接口：供 export_routes 等外部调用。"""
    index = refresh_index(env_dir, threshold)
    return build_stats_from_index(index, threshold)


def register_session_routes(app: FastAPI, logs_dir: str) -> None:
    env_dir = Path(logs_dir).parent
    cache_path = os.path.join(get_service_log_dir(), ".sessions_status.json")

    from fastapi.templating import Jinja2Templates
    _templates = Jinja2Templates(directory="templates")

    @app.get("/sessions")
    async def sessions_page(request: Request):
        return _templates.TemplateResponse(request, "sessions.html", context={
            "active_page": "sessions",
            "user_role": request.session.get("monitor_role", "user"),
            "user_name": request.session.get("monitor_user", ""),
            "user_permissions": [p.strip() for p in (request.session.get("monitor_permissions") or "").split(",") if p.strip()],
        })

    @app.get("/sessions/stats")
    def sessions_stats(threshold: int = QUALIFIED_THRESHOLD_DEFAULT, refresh: bool = False):
        global _mem_cache, _mem_cache_ts

        if not Path(env_dir).is_dir():
            return JSONResponse({"error": f"directory not found: {env_dir}"}, status_code=404)

        now = time.time()
        if not refresh and _mem_cache and (now - _mem_cache_ts) < _CACHE_TTL:
            if _mem_cache.get("_threshold") == threshold:
                return JSONResponse(_mem_cache)

        index = refresh_index(env_dir, threshold, force=refresh)
        stats = build_stats_from_index(index, threshold)

        stats["_dir"] = str(env_dir)
        stats["_threshold"] = threshold
        stats["_updated_at"] = now

        _mem_cache = stats
        _mem_cache_ts = now

        return JSONResponse(stats)
