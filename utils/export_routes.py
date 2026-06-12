"""
utils/export_routes.py — Session 导出 Web 路由

与 keys 管理界面共享认证（key_state.yaml 密码），
通过 /keys/export 进入导出页面。
"""

import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from utils.export_store import (
    append_log,
    create_record,
    get_record,
    list_records,
    list_records_by_key,
    update_status,
)
from utils.export_sync import export_session_index, sync_session_index
from utils.key_config import load_key_state
from utils.key_store import list_keys, mask_key
from utils.log_paths import get_service_log_dir
from utils.obs_utils import load_obs_base, obsutil_ls
from utils.session_routes import _build_stats_json, _load_sessions

logger = logging.getLogger(__name__)


def _load_sync_config() -> dict:
    """读取 sync_config 配置文件，获取 workers/interval 等参数。"""
    import yaml
    from utils.obs_utils import PROJECT_ROOT

    cli_state_path = PROJECT_ROOT / ".cli_state.yaml"
    if not cli_state_path.is_file():
        return {}
    try:
        with open(cli_state_path, "r", encoding="utf-8") as f:
            state = yaml.safe_load(f) or {}
        sync_cfg = state.get("sync_config", "")
        if not sync_cfg:
            return {}
        p = Path(sync_cfg)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if not p.is_file():
            return {}
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _is_key_authenticated(request: Request) -> bool:
    return bool(request.session.get("key_authenticated"))


def _require_key_api(request: Request):
    if request.headers.get("x-requested-with") != "XMLHttpRequest":
        return JSONResponse({"detail": "Not found"}, status_code=404)
    state = load_key_state()
    if not state.get("password"):
        return None
    if _is_key_authenticated(request):
        return None
    return JSONResponse({"detail": "Key management login required"}, status_code=401)


def _date_to_mtime_map(env_dir: Path) -> dict:
    """建立 session 日期 (2026-06-01) 到 mtime 目录名 (26060115) 的映射。

    一个日期可能跨多个 mtime 目录（如 26060100 和 26060115 都包含 2026-06-01 的请求）。
    返回 {date_str: [mtime_dir_name, ...]}
    """
    mapping = {}
    for sub in sorted(env_dir.iterdir()):
        if not sub.is_dir():
            continue
        cache_file = sub / ".session_cache.jsonl"
        if not cache_file.is_file():
            continue
        dates_in_dir = set()
        try:
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
                    ts = obj.get("first_ts", "")
                    if ts:
                        date_str = ts.split("_")[0]
                        dates_in_dir.add(date_str)
        except OSError:
            continue
        for d in dates_in_dir:
            mapping.setdefault(d, [])
            if sub.name not in mapping[d]:
                mapping[d].append(sub.name)
    return mapping


def _key_slot(api_key: str) -> str:
    if not api_key:
        return "all"
    return "key-" + api_key[-4:]


def register_export_routes(app: FastAPI, logs_dir: str) -> None:
    env_dir = Path(logs_dir).parent
    env_key_name = env_dir.name
    templates = Jinja2Templates(directory="templates")

    @app.get("/keys/export")
    def export_page(request: Request):
        state = load_key_state()
        if state.get("password") and not _is_key_authenticated(request):
            return templates.TemplateResponse("keys_login.html", {"request": request})
        return FileResponse(path="templates/export.html")

    @app.get("/api/export/config")
    def export_config(request: Request):
        denied = _require_key_api(request)
        if denied:
            return denied
        port = os.getenv("PROXY_PORT", "")
        mtimes = sorted(
            [d.name for d in env_dir.iterdir() if d.is_dir()],
            reverse=True,
        ) if env_dir.is_dir() else []
        sync_cfg = _load_sync_config()
        return JSONResponse({
            "port": port,
            "env_key": env_key_name,
            "env_dir": str(env_dir),
            "current_logs_dir": logs_dir,
            "obs_base": load_obs_base(),
            "workers": sync_cfg.get("workers", 4),
            "interval": sync_cfg.get("interval", 600),
            "upload_script": sync_cfg.get("upload_script", ""),
            "mtimes": mtimes,
        })

    @app.get("/api/export/keys")
    def export_keys(request: Request, threshold: int = 5):
        denied = _require_key_api(request)
        if denied:
            return denied
        if not env_dir.is_dir():
            return JSONResponse({"keys": [], "mtimes": []})

        stats = _build_stats_json(env_dir, threshold)
        date_mtime = _date_to_mtime_map(env_dir)

        db_keys = {}
        db_key_created = {}
        for k in list_keys():
            db_keys[k["key"]] = k["name"]
            db_key_created[k["key"]] = k.get("created_at", "")

        keys_result = []
        for row in stats.get("rows", []):
            api_key = row["api_key"]
            slot = _key_slot(api_key if api_key != "(empty)" else "")
            mtime_dist = {}
            for date_str, cell in row.get("cells", {}).items():
                if cell["total"] == 0:
                    continue
                for mt in date_mtime.get(date_str, []):
                    if mt not in mtime_dist:
                        mtime_dist[mt] = {"total": 0, "qualified": 0}
                    mtime_dist[mt]["total"] += cell["total"]
                    mtime_dist[mt]["qualified"] += cell["qualified"]

            matched_name = ""
            created_at = ""
            for masked, name in db_keys.items():
                if api_key != "(empty)" and masked.endswith(api_key[-4:]):
                    matched_name = name
                    created_at = db_key_created.get(masked, "")
                    break

            records = list_records_by_key(slot, limit=10)

            keys_result.append({
                "api_key": api_key,
                "key_name": matched_name,
                "key_slot": slot,
                "created_at": created_at,
                "total_sessions": row["row_total"],
                "qualified_sessions": row["row_qualified"],
                "mtime_distribution": mtime_dist,
                "records": records,
            })

        keys_result.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        mtimes = sorted(
            [d.name for d in env_dir.iterdir() if d.is_dir()],
            reverse=True,
        )

        return JSONResponse({"keys": keys_result, "mtimes": mtimes})

    @app.post("/api/export/run")
    async def export_run(request: Request):
        denied = _require_key_api(request)
        if denied:
            return denied
        body = await request.json()
        api_key = body.get("api_key", "")
        if api_key == "(empty)":
            api_key = ""
        mtime_dirs = body.get("mtime_dirs", [])
        obs_prefix = body.get("obs_prefix", "").strip().rstrip("/")
        force = body.get("force", False)

        if not mtime_dirs:
            return JSONResponse({"detail": "mtime_dirs is required"}, status_code=400)

        for mt in mtime_dirs:
            if not (env_dir / mt).is_dir():
                return JSONResponse({"detail": f"目录不存在: {mt}"}, status_code=400)

        slot = _key_slot(api_key)
        now_tag = datetime.now().strftime("%y%m%d%H%M%S")
        local_base = env_dir.parent / "logs_session" / env_key_name / slot / f"ex-{now_tag}"
        obs_dst = f"{obs_prefix}/session/{env_key_name}/{slot}/ex-{now_tag}/" if obs_prefix else ""

        record_id = create_record(
            api_key=api_key,
            key_slot=slot,
            mtime_dirs=json.dumps(mtime_dirs),
            obs_dst=obs_dst,
            local_copy_dir=str(local_base),
        )

        def _do_export():
            _log = lambda msg: append_log(record_id, msg)
            update_status(record_id, "running")
            _log(f"开始导出: key_slot={slot}, mtime_dirs={mtime_dirs}")
            if obs_dst:
                _log(f"OBS 目标: {obs_dst}")
            _log(f"本地目录: {local_base}")
            total_sessions = 0
            total_uploaded = 0
            total_skipped = 0
            errors = []

            for mt in mtime_dirs:
                mt_logs_dir = str(env_dir / mt)
                _log(f"[{mt}] 生成 session_index...")
                try:
                    exp_result = export_session_index(mt_logs_dir, force=force)
                    _log(f"[{mt}] session_index: {exp_result.get('total_sessions', 0)} sessions" +
                         (" (已是最新，跳过)" if exp_result.get("skipped") else ""))

                    _log(f"[{mt}] 开始同步文件" + (f" (按 key 过滤: ...{api_key[-8:]})" if api_key else " (全量)"))
                    sync_result = sync_session_index(
                        mt_logs_dir,
                        obs_dst=obs_dst,
                        key=api_key or None,
                        local_copy_dir=str(local_base),
                        force=force,
                    )
                    matched = sync_result.get("matched_sessions", 0)
                    uploaded = sync_result.get("uploaded", 0)
                    skipped = sync_result.get("skipped", 0)
                    total_sessions += matched
                    total_uploaded += uploaded
                    total_skipped += skipped
                    _log(f"[{mt}] 匹配 {matched} sessions, 上传 {uploaded} files, 跳过 {skipped}")
                    if sync_result.get("failed", 0) > 0:
                        _log(f"[{mt}] 上传失败!")
                        errors.append(f"{mt}: upload failed")
                except Exception as e:
                    _log(f"[{mt}] 错误: {e}")
                    errors.append(f"{mt}: {e}")
                    logger.exception("export failed for %s", mt)

            _log(f"汇总: {total_sessions} sessions, {total_uploaded} files uploaded, {total_skipped} skipped")
            if errors:
                _log(f"导出失败: {'; '.join(errors)}")
                update_status(
                    record_id, "failed",
                    error_message="; ".join(errors),
                    total_sessions=total_sessions,
                    files_uploaded=total_uploaded,
                    files_skipped=total_skipped,
                )
            else:
                _log("导出完成!")
                update_status(
                    record_id, "success",
                    total_sessions=total_sessions,
                    files_uploaded=total_uploaded,
                    files_skipped=total_skipped,
                )

        t = threading.Thread(target=_do_export, daemon=True)
        t.start()

        return JSONResponse({"record_id": record_id, "status": "running"})

    @app.get("/api/export/status/{record_id}")
    def export_status(request: Request, record_id: int):
        denied = _require_key_api(request)
        if denied:
            return denied
        rec = get_record(record_id)
        if not rec:
            return JSONResponse({"detail": "Not found"}, status_code=404)
        return JSONResponse(rec)

    @app.get("/api/export/records")
    def export_records_list(request: Request, key_slot: str = ""):
        denied = _require_key_api(request)
        if denied:
            return denied
        if key_slot:
            recs = list_records_by_key(key_slot)
        else:
            recs = list_records()
        return JSONResponse({"records": recs})

    @app.get("/api/export/obs/ls")
    def export_obs_ls(request: Request, path: str = ""):
        denied = _require_key_api(request)
        if denied:
            return denied
        if not path or not path.startswith("obs://"):
            return JSONResponse({"detail": "Invalid OBS path"}, status_code=400)
        if not path.endswith("/"):
            path += "/"
        items = obsutil_ls(path)
        return JSONResponse({"path": path, "items": items})
