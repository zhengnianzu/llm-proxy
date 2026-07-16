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
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates

from utils.export_store import (
    append_log,
    create_record,
    get_record,
    list_records,
    list_records_by_key,
    update_status,
)
from utils.export_sync import export_session_index, sync_session_index, _load_session_index
from utils.key_config import load_key_state
from utils.key_store import list_keys, mask_key
from utils.log_paths import get_service_log_dir
from utils.obs_utils import load_obs_base, load_sync_config, obsutil_ls
from utils.stats_index import (
    refresh_index, build_stats_from_index, get_date_to_mtime_map,
    get_current_key_cache, update_key_meta, update_key_records,
)

logger = logging.getLogger(__name__)


def _load_sync_config() -> dict:
    return load_sync_config()


def _require_key_api(request: Request):
    if request.headers.get("x-requested-with") != "XMLHttpRequest":
        return JSONResponse({"detail": "Not found"}, status_code=404)
    return None


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
        return templates.TemplateResponse(request, "export.html", context={"active_page": "export", "user_role": request.session.get("monitor_role", "user"), "user_name": request.session.get("monitor_user", ""), "user_permissions": [p.strip() for p in (request.session.get("monitor_permissions") or "").split(",") if p.strip()]})

    @app.get("/keys/export/report/{record_id}")
    def export_report_page(request: Request, record_id: int):
        return templates.TemplateResponse(request, "export_report.html", context={"active_page": "export", "user_role": request.session.get("monitor_role", "user"), "user_name": request.session.get("monitor_user", ""), "user_permissions": [p.strip() for p in (request.session.get("monitor_permissions") or "").split(",") if p.strip()]})

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

        index = refresh_index(env_dir, threshold)
        stats = build_stats_from_index(index, threshold, env_dir=env_dir)
        cache = get_current_key_cache()

        db_keys_list = list_keys()
        if cache:
            update_key_meta(cache, db_keys_list)
            update_key_records(cache, env_dir)

        keys_result = []
        for row in stats.get("rows", []):
            api_key = row["api_key"]
            mtime_dist = row.get("mtime_cells", {})

            if cache and cache.get("key_meta"):
                meta = cache["key_meta"].get("mapping", {}).get(api_key, {})
                matched_name = meta.get("key_name", "")
                slot = meta.get("key_slot", _key_slot(api_key))
                created_at = meta.get("created_at", "")
            else:
                matched_name = ""
                created_at = ""
                slot = _key_slot(api_key if api_key != "(empty)" else "")
                for k in db_keys_list:
                    masked = k.get("key", "")
                    if api_key != "(empty)" and masked.endswith(api_key[-4:]):
                        matched_name = k.get("name", "")
                        created_at = k.get("created_at", "")
                        break

            if cache and cache.get("key_records"):
                records = cache["key_records"].get("records", {}).get(slot, [])
            else:
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

    # -----------------------------------------------------------------
    # 统一任务执行（导出 / 质检）
    # -----------------------------------------------------------------

    def _run_task(record_id, _env_dir, _env_key_name, obs_prefix, now_tag, mode, force=False):
        from utils.eval.reformat import reformat_and_analyze
        from utils.eval.eval import evaluate_sessions
        from utils.obs_sync import _run_upload_cmd

        _log = lambda msg: append_log(record_id, msg)
        try:
            _run_task_inner(record_id, _env_dir, _env_key_name, obs_prefix, now_tag, mode, force,
                            reformat_and_analyze, evaluate_sessions, _run_upload_cmd, _log)
        except Exception as exc:
            logger.exception("_run_task crashed (record=%s)", record_id)
            try:
                _log(f"任务异常终止: {exc}")
            except Exception:
                logger.warning("append_log also failed for record=%s", record_id)
            update_status(record_id, "failed", error_message=str(exc))

    def _run_task_inner(record_id, _env_dir, _env_key_name, obs_prefix, now_tag, mode, force,
                        reformat_and_analyze, evaluate_sessions, _run_upload_cmd, _log):
        rec = get_record(record_id)
        update_status(record_id, "running")

        sync_cfg = _load_sync_config()
        workers = sync_cfg.get("workers", 4)
        upload_script = sync_cfg.get("upload_script") or None

        mtime_dirs = json.loads(rec.get("mtime_dirs", "[]"))
        api_key = rec.get("api_key", "")
        slot = rec.get("key_slot", "all")

        if mode == "export":
            local_base = _env_dir.parent.parent / "logs_session" / _env_key_name / slot / f"ex-{now_tag}"
            obs_sub = "session"
        else:
            local_base = (_env_dir.parent.parent / "logs_session_analysis" / _env_key_name / slot / f"ex-{now_tag}").resolve()
            obs_sub = "session_analysis"

        obs_dst = f"{obs_prefix}/{obs_sub}/{_env_key_name}/{slot}/ex-{now_tag}/" if obs_prefix else ""
        local_base.mkdir(parents=True, exist_ok=True)

        update_status(record_id, "running", obs_dst=obs_dst, local_copy_dir=str(local_base))

        _log(f"开始{'质检' if mode == 'eval' else '导出'}: key_slot={slot}, mtime_dirs={mtime_dirs}")
        _log(f"本地目录: {local_base}")
        if obs_dst:
            _log(f"OBS 目标: {obs_dst}")

        errors = []
        total_sessions = 0
        total_uploaded = 0
        total_skipped = 0
        all_results = []
        all_entries = []

        for mt in mtime_dirs:
            mt_src = str(_env_dir / mt)
            try:
                _log(f"[{mt}] 生成 session_index...")
                exp_result = export_session_index(mt_src, force=force)
                _log(f"[{mt}] session_index: {exp_result.get('total_sessions', 0)} sessions"
                     + (" (已是最新，跳过)" if exp_result.get("skipped") else ""))

                if mode == "export":
                    _log(f"[{mt}] 开始同步文件" + (f" (按 key 过滤: ...{api_key[-8:]})" if api_key else " (全量)"))
                    sync_result = sync_session_index(
                        mt_src, obs_dst=obs_dst, key=api_key or None,
                        local_copy_dir=str(local_base), force=force,
                    )
                    matched = sync_result.get("matched_sessions", 0)
                    new_sessions = sync_result.get("new_sessions", 0)
                    uploaded = sync_result.get("uploaded", 0)
                    skipped = sync_result.get("skipped", 0)
                    total_sessions += new_sessions
                    total_uploaded += uploaded
                    total_skipped += skipped
                    _log(f"[{mt}] 匹配 {matched} sessions, 新导出 {new_sessions}, 跳过 {skipped}, 文件数 {uploaded}")
                    if sync_result.get("failed", 0) > 0:
                        _log(f"[{mt}] 上传失败!")
                        errors.append(f"{mt}: upload failed")
                else:
                    session_entries = _load_session_index(mt_src)
                    total_before = len(session_entries)
                    if api_key:
                        session_entries = [s for s in session_entries if s.get("api_key") == api_key]
                    if not session_entries:
                        _log(f"[{mt}] 共 {total_before} sessions, 按 key 过滤后 0, 跳过")
                        continue
                    if api_key:
                        _log(f"[{mt}] 共 {total_before} sessions, 按 key 过滤后 {len(session_entries)}")
                    _log(f"[{mt}] reformat+analyze: {len(session_entries)} sessions...")
                    ra_result = reformat_and_analyze(
                        src_dir=mt_src, out_dir=str(local_base),
                        session_entries=session_entries, api_key=api_key, workers=workers,
                        progress_cb=lambda msg, _mt=mt: _log(f"[{_mt}] {msg}"),
                    )
                    all_results.extend(ra_result["results"])
                    all_entries.extend(session_entries)
                    total_sessions += len(ra_result["results"])
                    _log(f"[{mt}] 完成: {ra_result['total_files']} sessions")
            except Exception as e:
                _log(f"[{mt}] 错误: {e}")
                errors.append(f"{mt}: {e}")
                logger.exception("task failed for %s (mode=%s)", mt, mode)

        eval_report_path = ""
        analysis_json_stored = ""
        if mode == "eval":
            if all_results:
                idx_path = local_base / "session_index.jsonl"
                with open(idx_path, "w", encoding="utf-8") as f:
                    for entry in all_entries:
                        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                eval_result = evaluate_sessions(
                    sessions=all_results, report_dir=str(local_base), progress_cb=_log,
                    key_name=f"{_env_key_name}/{slot}",
                    obs_path=obs_dst,
                )
                eval_report_path = eval_result.get("report_path", "")
                analysis_json_path = eval_result.get("analysis_json_path", "")
                if analysis_json_path and Path(analysis_json_path).is_file():
                    # 读取内容并写入外部文件
                    from utils.export_store import _externalize_field
                    analysis_json_content = Path(analysis_json_path).read_text(encoding="utf-8")
                    analysis_json_stored = _externalize_field(record_id, "analysis_json", analysis_json_content)

                from utils.eval.reformat import write_eval_to_cache
                for mt in mtime_dirs:
                    try:
                        write_eval_to_cache(str(_env_dir / mt), all_results)
                    except Exception:
                        logger.debug("write_eval_to_cache failed for %s", mt, exc_info=True)
            else:
                _log("无 session 数据")

            if obs_dst:
                _log(f"同步到 OBS: {obs_dst}")
                obs_parent = obs_dst.rstrip("/").rsplit("/", 1)[0] + "/"
                ok, msg = _run_upload_cmd(str(local_base), obs_parent, upload_script)
                if ok:
                    _log("上传成功")
                else:
                    _log(f"上传失败: {msg}")
                    errors.append(f"OBS upload: {msg}")

        if errors or (mode == "eval" and not all_results):
            _log(f"{'质检' if mode == 'eval' else '导出'}失败: {'; '.join(errors) if errors else '无数据'}")
            update_status(record_id, "failed",
                          error_message="; ".join(errors) if errors else "无 session 数据",
                          total_sessions=total_sessions,
                          files_uploaded=total_uploaded,
                          files_skipped=total_skipped,
                          eval_report_path=eval_report_path,
                          analysis_json=analysis_json_stored)
        else:
            _log(f"{'质检' if mode == 'eval' else '导出'}完成: {total_sessions} sessions")
            update_status(record_id, "success",
                          total_sessions=total_sessions,
                          files_uploaded=total_uploaded,
                          files_skipped=total_skipped,
                          eval_report_path=eval_report_path,
                          analysis_json=analysis_json_stored)

    # -----------------------------------------------------------------
    # API 端点
    # -----------------------------------------------------------------

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
        mode = "eval" if body.get("auto_eval", False) else "export"

        if not mtime_dirs:
            return JSONResponse({"detail": "mtime_dirs is required"}, status_code=400)
        for mt in mtime_dirs:
            if not (env_dir / mt).is_dir():
                return JSONResponse({"detail": f"目录不存在: {mt}"}, status_code=400)

        slot = _key_slot(api_key)
        now_tag = datetime.now().strftime("%y%m%d%H%M%S")

        record_id = create_record(
            api_key=api_key, key_slot=slot,
            mtime_dirs=json.dumps(mtime_dirs),
            obs_dst="", local_copy_dir="",
            mode=mode,
        )

        t = threading.Thread(
            target=_run_task,
            args=(record_id, env_dir, env_key_name, obs_prefix, now_tag, mode),
            kwargs={"force": force},
            daemon=True,
        )
        t.start()

        return JSONResponse({"record_id": record_id, "status": "running", "mode": mode})

    @app.post("/api/export/eval")
    async def export_eval(request: Request):
        denied = _require_key_api(request)
        if denied:
            return denied
        body = await request.json()
        src_record_id = body.get("record_id")
        if not src_record_id:
            return JSONResponse({"detail": "record_id is required"}, status_code=400)

        rec = get_record(src_record_id)
        if not rec:
            return JSONResponse({"detail": "Record not found"}, status_code=404)
        if rec["status"] != "success":
            return JSONResponse({"detail": "导出未完成，无法质检"}, status_code=400)

        obs_prefix = rec.get("obs_dst", "").split("/session/")[0] if rec.get("obs_dst") else ""
        now_tag = datetime.now().strftime("%y%m%d%H%M%S")

        new_record_id = create_record(
            api_key=rec["api_key"], key_slot=rec["key_slot"],
            mtime_dirs=rec["mtime_dirs"],
            mode="eval",
            source_export_id=src_record_id,
        )

        t = threading.Thread(
            target=_run_task,
            args=(new_record_id, env_dir, env_key_name, obs_prefix, now_tag, "eval"),
            daemon=True,
        )
        t.start()

        return JSONResponse({"record_id": new_record_id, "status": "running", "mode": "eval"})

    @app.get("/api/export/status/{record_id}")
    def export_status(request: Request, record_id: int):
        denied = _require_key_api(request)
        if denied:
            return denied
        from utils.export_store import get_record_resolved
        rec = get_record_resolved(record_id)  # 自动解析外部文件
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

    @app.get("/api/export/eval/report/{record_id}")
    def export_eval_report(request: Request, record_id: int):
        from utils.export_store import get_record_resolved
        rec = get_record_resolved(record_id)  # 自动解析外部文件
        if not rec:
            return JSONResponse({"detail": "Not found"}, status_code=404)

        report_path = rec.get("eval_report_path", "")
        html_path = ""
        if report_path:
            html_path = str(Path(report_path).parent / "session_report.html")

        if html_path and Path(html_path).is_file():
            content = Path(html_path).read_text(encoding="utf-8")
            return JSONResponse({"report_html": content, "record_id": record_id})

        analysis_json = rec.get("analysis_json", "")
        if analysis_json:
            try:
                from utils.eval.eval import load_analysis_json, compute_stats, render_html_report_string
                sessions = load_analysis_json(analysis_json)
                stats = compute_stats(sessions)
                key_slot = rec.get("key_slot", "all")
                content = render_html_report_string(
                    sessions, stats,
                    key_name=f"{env_key_name}/{key_slot}",
                    obs_path=rec.get("obs_dst", ""),
                )
                return JSONResponse({"report_html": content, "record_id": record_id})
            except Exception as e:
                logger.exception("Failed to rebuild report from analysis JSON")
                return JSONResponse({"detail": f"报告重建失败: {e}"}, status_code=500)

        if report_path and Path(report_path).is_file():
            content = Path(report_path).read_text(encoding="utf-8")
            return JSONResponse({"report_md": content, "record_id": record_id})

        return JSONResponse({"detail": "报告未生成"}, status_code=404)

    # --- shared 公开导出路由（key+code 验证） ---

    def _check_shared_export(key: str, code: str):
        import hmac as _hmac
        from utils.key_store import find_key
        expected = os.getenv("SHARED_CODE", "shared")
        if not _hmac.compare_digest(code, expected):
            return JSONResponse({"detail": "Invalid code"}, status_code=403)
        if not key or not find_key(key):
            return JSONResponse({"detail": "Key not found"}, status_code=404)
        return None

    @app.post("/api/shared/export")
    async def shared_export(request: Request):
        body = await request.json()
        key_value = body.get("key", "")
        code = body.get("code", "")

        err = _check_shared_export(key_value, code)
        if err:
            return err

        index = refresh_index(env_dir)
        stats = build_stats_from_index(index, env_dir=env_dir)
        mtime_dirs = []
        for row in stats.get("rows", []):
            if row["api_key"] == key_value:
                mtime_dirs = sorted(row.get("mtime_cells", {}).keys(), reverse=True)
                break
        if not mtime_dirs:
            return JSONResponse({"detail": "No session data found for this key"}, status_code=404)

        api_key = key_value
        slot = _key_slot(api_key)
        obs_prefix = body.get("obs_prefix", "").strip().rstrip("/") or load_obs_base()
        mode = "eval"
        now_tag = datetime.now().strftime("%y%m%d%H%M%S")

        record_id = create_record(
            api_key=api_key, key_slot=slot,
            mtime_dirs=json.dumps(mtime_dirs),
            mode=mode,
        )

        obs_dst = f"{obs_prefix}/session_analysis/{env_key_name}/{slot}/ex-{now_tag}/" if obs_prefix else ""

        t = threading.Thread(
            target=_run_task,
            args=(record_id, env_dir, env_key_name, obs_prefix, now_tag, mode),
            daemon=True,
        )
        t.start()

        return JSONResponse({
            "record_id": record_id,
            "session_path": obs_dst,
            "status": "running",
        })

    @app.get("/api/shared/export/status/{record_id}")
    def shared_export_status(record_id: int, key: str = "", code: str = ""):
        err = _check_shared_export(key, code)
        if err:
            return err

        from utils.export_store import get_record_resolved
        rec = get_record_resolved(record_id)  # 自动解析外部文件
        if not rec:
            return JSONResponse({"detail": "Not found"}, status_code=404)

        if rec.get("api_key") and rec["api_key"] != key:
            return JSONResponse({"detail": "Access denied"}, status_code=403)

        return JSONResponse({
            "record_id": rec["id"],
            "status": rec["status"],
            "session_path": rec.get("obs_dst", ""),
            "total_sessions": rec.get("total_sessions", 0),
            "error_message": rec.get("error_message", ""),
        })
