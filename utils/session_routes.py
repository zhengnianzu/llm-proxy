"""
utils/session_routes.py — Session 会话统计 API

提供 /sessions/stats 接口，扫描当前实例 logs_dir 上一级 env-key 目录下所有
时间戳子目录的 .session_cache.jsonl，返回统计结果并缓存到 logs/.sessions_status.json。
"""

import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse

from utils.log_paths import get_service_log_dir

QUALIFIED_THRESHOLD = 5
_CACHE_TTL = 30


# ---------------------------------------------------------------------------
# 核心统计逻辑
# ---------------------------------------------------------------------------

def _load_sessions(base_dir: Path) -> list:
    sessions = []
    for sub in sorted(base_dir.iterdir()):
        if not sub.is_dir():
            continue
        cache_file = sub / ".session_cache.jsonl"
        if not cache_file.is_file():
            continue
        mtime_name = sub.name
        with open(cache_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("_meta"):
                    continue
                obj["_mtime_dir"] = mtime_name
                sessions.append(obj)
    return sessions


def _extract_date(ts: str) -> str:
    return ts.split("_")[0] if ts else "unknown"


def _build_stats_json(base_dir: Path, threshold: int = QUALIFIED_THRESHOLD) -> Dict[str, Any]:
    sessions = _load_sessions(base_dir)
    total = len(sessions)
    qualified = sum(1 for s in sessions if len(s.get("trace_list", [])) >= threshold)

    table: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: {"total": 0, "qualified": 0}))
    mtime_table: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: {"total": 0, "qualified": 0}))
    all_dates: set = set()
    all_mtimes: set = set()

    for s in sessions:
        key = s.get("api_key", "") or "(empty)"
        date = _extract_date(s.get("first_ts", ""))
        mt = s.get("_mtime_dir", "")
        q = len(s.get("trace_list", [])) >= threshold
        all_dates.add(date)
        table[key][date]["total"] += 1
        if q:
            table[key][date]["qualified"] += 1
        if mt:
            all_mtimes.add(mt)
            mtime_table[key][mt]["total"] += 1
            if q:
                mtime_table[key][mt]["qualified"] += 1

    dates = sorted(all_dates)
    keys = sorted(table.keys())

    rows = []
    for key in keys:
        row_total = 0
        row_qualified = 0
        cells = {}
        for d in dates:
            cell = table[key][d]
            cells[d] = {"total": cell["total"], "qualified": cell["qualified"]}
            row_total += cell["total"]
            row_qualified += cell["qualified"]
        mtime_cells = {}
        for mt in sorted(all_mtimes):
            mc = mtime_table[key][mt]
            if mc["total"] > 0:
                mtime_cells[mt] = {"total": mc["total"], "qualified": mc["qualified"]}
        rows.append({
            "api_key": key,
            "cells": cells,
            "mtime_cells": mtime_cells,
            "row_total": row_total,
            "row_qualified": row_qualified,
        })

    return {
        "total_sessions": total,
        "qualified_sessions": qualified,
        "threshold": threshold,
        "dates": dates,
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# 缓存
# ---------------------------------------------------------------------------

def _load_cache(cache_path: str, dir_key: str, threshold: int) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(cache_path):
        return None
    try:
        mtime = os.path.getmtime(cache_path)
        if time.time() - mtime > _CACHE_TTL:
            return None
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("_dir") == dir_key and data.get("_threshold") == threshold:
            return data
        return None
    except (OSError, json.JSONDecodeError):
        return None


def _save_cache(cache_path: str, data: Dict[str, Any]) -> None:
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        tmp = cache_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, cache_path)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# 路由注册
# ---------------------------------------------------------------------------

def register_session_routes(app: FastAPI, logs_dir: str) -> None:
    env_dir = str(Path(logs_dir).parent)
    cache_path = os.path.join(get_service_log_dir(), ".sessions_status.json")

    @app.get("/sessions")
    async def sessions_page():
        return FileResponse(path="templates/sessions.html")

    @app.get("/sessions/stats")
    def sessions_stats(threshold: int = QUALIFIED_THRESHOLD, refresh: bool = False):
        if not Path(env_dir).is_dir():
            return JSONResponse({"error": f"directory not found: {env_dir}"}, status_code=404)

        if not refresh:
            cache = _load_cache(cache_path, env_dir, threshold)
            if cache:
                return JSONResponse(cache)

        stats = _build_stats_json(Path(env_dir), threshold)

        stats["_dir"] = env_dir
        stats["_threshold"] = threshold
        stats["_updated_at"] = time.time()
        _save_cache(cache_path, stats)

        return JSONResponse(stats)
