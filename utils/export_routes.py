"""
utils/export_routes.py — Session 导出 Web 路由

与 keys 管理界面共享认证（key_state.yaml 密码），
通过 /keys/export 进入导出页面。
"""

import json
import logging
import os
import queue
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
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
    refresh_index, build_stats_from_index, build_stats_multi, get_date_to_mtime_map,
    get_current_key_cache, update_key_meta, update_key_records,
    get_last_refresh_ts, start_stats_warmer,
)
from utils.logs_config import get_stats_roots

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 全局任务队列（串行执行，支持取消排队中任务）
# ---------------------------------------------------------------------------
# 队列项: (record_id, fn, args)
# fn 是无参可调用，内部捕获所有上下文
#
# 多 worker 说明：任务 fn 是内存闭包，只存在于「接收入队请求」的那个 worker 的
# 队列里，无法跨进程。这里保证两条不变量：
#   1) 取消跨 worker 生效：cancel 请求（可能落到别的 worker）把 DB 状态置为
#      cancelled；执行 worker 出队时以 DB 状态为准重新判定，取消则跳过。
#   2) 全局串行：用文件锁（flock 阻塞）保证任意时刻只有一个 worker 在真正执行
#      导出任务，避免多 worker 各自的 runner 并发跑导出撞 OBS/本地拷贝。
_task_queue: queue.Queue = queue.Queue()
_queue_lock = threading.Lock()

# record_id -> True 表示该任务已被取消，调度线程出队后直接跳过（本进程内快速路径）
_cancelled_ids: set = set()

_EXPORT_LOCK_PATH = os.path.join(get_service_log_dir(), "export_queue.lock")


def _is_cancelled(record_id: int) -> bool:
    """出队时判定任务是否已取消：先查本进程集合，再以 DB 状态为准（跨 worker）。"""
    with _queue_lock:
        if record_id in _cancelled_ids:
            _cancelled_ids.discard(record_id)
            return True
    try:
        rec = get_record(record_id)
        if rec and rec.get("status") == "cancelled":
            return True
    except Exception:
        pass
    return False


def _queue_runner():
    """全局单一调度线程，逐个执行队列中的任务。跨 worker 用文件锁保证全局串行。"""
    import fcntl
    while True:
        record_id, fn = _task_queue.get()
        if _is_cancelled(record_id):
            _task_queue.task_done()
            continue
        lock_fp = None
        try:
            # 阻塞式全局锁：等到其它 worker 的导出任务结束再执行本任务
            os.makedirs(os.path.dirname(_EXPORT_LOCK_PATH), exist_ok=True)
            lock_fp = open(_EXPORT_LOCK_PATH, "w")
            fcntl.flock(lock_fp, fcntl.LOCK_EX)
            # 拿到锁后再确认一次是否在等待期间被取消
            if _is_cancelled(record_id):
                continue
            fn()
        except Exception:
            logger.exception("queue_runner: task %s raised", record_id)
        finally:
            if lock_fp is not None:
                try:
                    fcntl.flock(lock_fp, fcntl.LOCK_UN)
                    lock_fp.close()
                except OSError:
                    pass
            _task_queue.task_done()


def _enqueue_task(record_id: int, fn) -> None:
    """将任务加入队列，并把记录状态设为 queued（如果还不是 running）。"""
    from utils.export_store import update_status
    update_status(record_id, "queued")
    _task_queue.put((record_id, fn))


# 启动调度线程（只启动一次）
_runner_thread = threading.Thread(target=_queue_runner, daemon=True, name="export-queue-runner")
_runner_thread.start()


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

    def _all_roots() -> list:
        """活跃 env_dir + 配置的历史路径（去重）。"""
        return get_stats_roots(str(env_dir))

    def _existing_roots() -> list:
        """只保留实际存在的 root。与 build_stats_multi 收到的列表一致，
        决定 mtime key 是否带 <root_basename>/ 前缀（多 root 才带）。"""
        return [r for r in _all_roots() if Path(r).is_dir()]

    def _list_all_mtimes() -> list:
        """列出所有 root 下含 index.jsonl 的叶子目录，多 root 时带 <root_basename>/ 前缀。

        与 build_stats_multi 的 mtime_cells key 格式一致，供前端 mtime 选择。
        """
        from utils.log_scan import iter_index_dirs, dir_key_for
        roots = _existing_roots()
        multi = len(roots) > 1
        out = []
        for root in roots:
            rp = Path(root)
            base = os.path.basename(os.path.normpath(root))
            for leaf in iter_index_dirs(rp):
                rel = dir_key_for(rp, leaf)
                out.append(f"{base}/{rel}" if multi else rel)
        return sorted(set(out), reverse=True)

    def _resolve_mt(mt: str) -> Optional[str]:
        """把 mtime key（<root_basename>/<rel> 或裸 <rel>）解析为叶子目录绝对路径。"""
        roots = _existing_roots()
        multi = len(roots) > 1
        for root in roots:
            rp = Path(root)
            base = os.path.basename(os.path.normpath(root))
            if multi and mt.startswith(base + "/"):
                cand = rp / mt[len(base) + 1:]
                if cand.is_dir():
                    return str(cand)
            cand2 = rp / mt
            if cand2.is_dir():
                return str(cand2)
        # 兜底：活跃 env_dir 下直接拼
        cand = env_dir / mt
        return str(cand) if cand.is_dir() else None

    def _log_dir_key(mt_src: str) -> str:
        """把叶子目录绝对路径解析为相对 root 的 dir_key（如 "260728/26072813"）。

        与 stats_index / log_scan 的 dir_key 一致，供报告生成 /history 链接。
        找不到归属 root 时退回叶子目录名（单级 native 布局下即正确）。"""
        from utils.log_scan import dir_key_for
        p = Path(mt_src)
        for root in _existing_roots():
            rp = Path(root)
            try:
                if p == rp or rp in p.parents:
                    return dir_key_for(rp, p)
            except (OSError, ValueError):
                continue
        return p.name

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
        mtimes = _list_all_mtimes()
        sync_cfg = _load_sync_config()
        return JSONResponse({
            "port": port,
            "env_key": env_key_name,
            "env_dir": str(env_dir),
            "current_logs_dir": logs_dir,
            "roots": _all_roots(),
            "obs_base": load_obs_base(),
            "workers": sync_cfg.get("workers", 4),
            "interval": sync_cfg.get("interval", 600),
            "upload_script": sync_cfg.get("upload_script", ""),
            "mtimes": mtimes,
        })

    @app.get("/api/export/backfill-status")
    def export_backfill_status(request: Request):
        """new-api 会话回填的后台队列状态：正在跑哪个 root、排队哪些、各自进度。

        供导出页透明展示——回填串行执行，一个 root 跑完再跑下一个。
        """
        denied = _require_key_api(request)
        if denied:
            return denied
        try:
            from utils.newapi_backfill import get_queue_snapshot, get_backfill_status
        except Exception:
            return JSONResponse({"enabled": False, "current": None, "queued": [], "roots": []})

        snap = get_queue_snapshot()
        # 汇总每个 new-api root 的进度（total/done 叶子）
        roots_prog = []
        for r in _existing_roots():
            try:
                from utils.log_scan import detect_format as _df
                if _df(r) != "newapi":
                    continue
            except Exception:
                continue
            st = get_backfill_status(r)
            roots_prog.append({
                "root": r,
                "name": Path(r).name,
                "status": st.get("status", "idle"),
                "total_leaves": st.get("total_leaves", 0),
                "done_leaves": st.get("done_leaves", 0),
                "last_error": st.get("last_error", ""),
            })
        return JSONResponse({
            "enabled": True,
            "current": snap.get("current"),
            "current_name": Path(snap["current"]).name if snap.get("current") else None,
            "current_status": snap.get("current_status"),
            "queued": [Path(q).name for q in snap.get("queued", [])],
            "queue_len": snap.get("queue_len", 0),
            "running": snap.get("running", False),
            "roots": roots_prog,
        })

    @app.get("/api/export/keys")
    def export_keys(request: Request, threshold: int = 5):
        denied = _require_key_api(request)
        if denied:
            return denied
        roots = [r for r in _all_roots() if Path(r).is_dir()]
        if not roots:
            return JSONResponse({"keys": [], "mtimes": []})

        # 跨 root 合并 session 统计（rows 的 mtime_cells key 已带 <root>/ 前缀）
        stats = build_stats_multi(roots, threshold)

        # 每个 key 的模型分布（{api_key: {model: session_count}}）。DB 已按 service 绑定，
        # 一次查询覆盖全部叶子，避免逐叶重扫 index.jsonl（那是「导出一直加载中」的根因）。
        try:
            import utils.session_store as _ss
            model_stats = _ss.get_model_stats_by_key()
        except Exception:
            model_stats = {}
        all_models = sorted({m for dist in model_stats.values() for m in dist})

        db_keys_list = list_keys()

        # 名称/slot/创建时间：直接用 DB key 列表按后 4 位匹配（多 root 不复用单 env 的 key_meta 缓存）
        def _match_meta(api_key: str):
            slot = _key_slot(api_key if api_key != "(empty)" else "")
            matched_name = ""
            created_at = ""
            for k in db_keys_list:
                masked = k.get("key", "")
                if api_key != "(empty)" and masked.endswith(api_key[-4:]):
                    matched_name = k.get("name", "")
                    created_at = k.get("created_at", "")
                    break
            return matched_name, slot, created_at

        keys_result = []
        for row in stats.get("rows", []):
            api_key = row["api_key"]
            matched_name, slot, created_at = _match_meta(api_key)
            records = list_records_by_key(slot, limit=10)

            keys_result.append({
                "api_key": api_key,
                "key_name": matched_name,
                "key_slot": slot,
                "created_at": created_at,
                "total_sessions": row["row_total"],
                "qualified_sessions": row["row_qualified"],
                "mtime_distribution": row.get("mtime_cells", {}),
                "model_distribution": model_stats.get(api_key, {}),
                "records": records,
            })

        keys_result.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        return JSONResponse({"keys": keys_result, "mtimes": _list_all_mtimes(),
                             "models": all_models, "last_refresh_ts": get_last_refresh_ts()})

    # -----------------------------------------------------------------
    # 统一任务执行（导出 / 质检）
    # -----------------------------------------------------------------

    def _run_upload_only(record_id, local_copy_dir, obs_dst):
        """仅执行上传步骤，用于对已完成导出/质检但上传失败的记录进行重试。"""
        from utils.obs_sync import _run_upload_cmd
        _log = lambda msg: append_log(record_id, msg)
        try:
            sync_cfg = _load_sync_config()
            upload_script = sync_cfg.get("upload_script") or None
            obs_parent = obs_dst.rstrip("/").rsplit("/", 1)[0] + "/"
            _log(f"重试上传: {local_copy_dir} -> {obs_parent}")
            ok, msg = _run_upload_cmd(local_copy_dir, obs_parent, upload_script, log_cb=_log)
            if ok:
                _log("上传成功")
                update_status(record_id, "success")
            else:
                _log(f"上传失败: {msg}")
                update_status(record_id, "failed", error_message=f"OBS upload: {msg}")
        except Exception as e:
            logger.exception("_run_upload_only crashed (record=%s)", record_id)
            update_status(record_id, "failed", error_message=str(e))

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
            mt_src = _resolve_mt(mt) or str(_env_dir / mt)
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
                    _log(f"[{mt}] 进入 reformat+analyze: {len(session_entries)} sessions, workers={workers}...")
                    ra_result = reformat_and_analyze(
                        src_dir=mt_src, out_dir=str(local_base),
                        session_entries=session_entries, api_key=api_key, workers=workers,
                        progress_cb=lambda msg, _mt=mt: _log(f"[{_mt}] {msg}"),
                        log_dir=_log_dir_key(mt_src),
                    )
                    all_results.extend(ra_result["results"])
                    all_entries.extend(session_entries)
                    total_sessions += len(ra_result["results"])
                    _log(f"[{mt}] reformat+analyze 返回: results={ra_result['total_files']}, errors={len(ra_result.get('errors', []))}")
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
                _log(f"开始 evaluate_sessions: {len(all_results)} sessions...")
                eval_result = evaluate_sessions(
                    sessions=all_results, report_dir=str(local_base), progress_cb=_log,
                    key_name=f"{_env_key_name}/{slot}",
                    obs_path=obs_dst,
                )
                _log(f"evaluate_sessions 完成: total_sessions={eval_result.get('total_sessions', 0)}")
                eval_report_path = eval_result.get("report_path", "")
                analysis_json_path = eval_result.get("analysis_json_path", "")
                # 只有文件 ≤100MB 才存入 DB（超大文件直接从 local_copy_dir 读取即可）
                _MAX_ANALYSIS_JSON = 100 * 1024 * 1024
                if analysis_json_path and Path(analysis_json_path).is_file():
                    if Path(analysis_json_path).stat().st_size <= _MAX_ANALYSIS_JSON:
                        from utils.export_store import _externalize_field
                        analysis_json_content = Path(analysis_json_path).read_text(encoding="utf-8")
                        analysis_json_stored = _externalize_field(record_id, "analysis_json", analysis_json_content)
                    else:
                        _log(f"analysis_json 过大（{Path(analysis_json_path).stat().st_size // 1024 // 1024}MB），跳过存储，报告直接使用 session_report.html")

                from utils.eval.reformat import write_eval_to_cache
                for mt in mtime_dirs:
                    try:
                        write_eval_to_cache(_resolve_mt(mt) or str(_env_dir / mt), all_results)
                    except Exception:
                        logger.debug("write_eval_to_cache failed for %s", mt, exc_info=True)
            else:
                _log("无 session 数据")

            if obs_dst:
                _log(f"同步到 OBS: {obs_dst}")
                obs_parent = obs_dst.rstrip("/").rsplit("/", 1)[0] + "/"
                ok, msg = _run_upload_cmd(str(local_base), obs_parent, upload_script, log_cb=_log)
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
            if _resolve_mt(mt) is None:
                return JSONResponse({"detail": f"目录不存在: {mt}"}, status_code=400)

        slot = _key_slot(api_key)
        now_tag = datetime.now().strftime("%y%m%d%H%M%S")

        record_id = create_record(
            api_key=api_key, key_slot=slot,
            mtime_dirs=json.dumps(mtime_dirs),
            obs_dst="", local_copy_dir="",
            mode=mode,
        )

        _enqueue_task(record_id, lambda rid=record_id, ed=env_dir, ek=env_key_name, op=obs_prefix, nt=now_tag, m=mode, f=force: _run_task(rid, ed, ek, op, nt, m, force=f))

        return JSONResponse({"record_id": record_id, "status": "queued", "mode": mode})

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

        _enqueue_task(new_record_id, lambda rid=new_record_id, ed=env_dir, ek=env_key_name, op=obs_prefix, nt=now_tag: _run_task(rid, ed, ek, op, nt, "eval"))

        return JSONResponse({"record_id": new_record_id, "status": "queued", "mode": "eval"})

    @app.post("/api/export/upload_retry")
    async def export_upload_retry(request: Request):
        denied = _require_key_api(request)
        if denied:
            return denied
        body = await request.json()
        record_id = body.get("record_id")
        if not record_id:
            return JSONResponse({"detail": "record_id is required"}, status_code=400)
        rec = get_record(record_id)
        if not rec:
            return JSONResponse({"detail": "Record not found"}, status_code=404)
        if rec["status"] != "failed":
            return JSONResponse({"detail": "只能对失败记录重试上传"}, status_code=400)
        local_copy_dir = rec.get("local_copy_dir", "")
        if not local_copy_dir or not Path(local_copy_dir).is_dir():
            return JSONResponse({"detail": "本地文件不存在，无法重试（可能已被清理）"}, status_code=400)
        obs_dst = rec.get("obs_dst", "")
        if not obs_dst:
            return JSONResponse({"detail": "无 OBS 目标路径，无法上传"}, status_code=400)

        def _upload_retry_task(rid=record_id, lcd=local_copy_dir, od=obs_dst):
            update_status(rid, "running", error_message="")
            _run_upload_only(rid, lcd, od)

        _enqueue_task(record_id, _upload_retry_task)
        return JSONResponse({"record_id": record_id, "status": "queued"})

    @app.post("/api/export/cancel")
    async def export_cancel(request: Request):
        denied = _require_key_api(request)
        if denied:
            return denied
        body = await request.json()
        record_id = body.get("record_id")
        if not record_id:
            return JSONResponse({"detail": "record_id is required"}, status_code=400)
        rec = get_record(record_id)
        if not rec:
            return JSONResponse({"detail": "Record not found"}, status_code=404)
        if rec["status"] != "queued":
            return JSONResponse({"detail": "只能取消排队中的任务"}, status_code=400)
        with _queue_lock:
            _cancelled_ids.add(record_id)
        update_status(record_id, "cancelled", error_message="用户取消")
        return JSONResponse({"record_id": record_id, "status": "cancelled"})

    @app.get("/api/export/status/{record_id}")
    def export_status(request: Request, record_id: int):
        denied = _require_key_api(request)
        if denied:
            return denied
        from utils.export_store import get_record_resolved
        rec = get_record_resolved(record_id)
        if not rec:
            return JSONResponse({"detail": "Not found"}, status_code=404)
        rec.pop("analysis_json", None)  # 可达 2GB+，不返回给前端
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
        rec = get_record(record_id)
        if not rec:
            return JSONResponse({"detail": "Not found"}, status_code=404)

        # 优先直接 serve 本地 HTML 文件（FileResponse 流式传输，不占内存）
        report_path = rec.get("eval_report_path", "")
        html_path = Path(report_path).parent / "session_report.html" if report_path else None
        if html_path and html_path.is_file():
            return FileResponse(str(html_path), media_type="text/html")

        # 降级：analysis_json 重建报告（限制文件大小 ≤ 200MB，防止 OOM）
        analysis_json_ref = rec.get("analysis_json", "")
        if analysis_json_ref:
            from utils.export_store import _read_field_content
            # 先检查外部文件大小
            if analysis_json_ref.startswith("file://"):
                fpath = Path(analysis_json_ref[7:])
                if fpath.is_file() and fpath.stat().st_size > 200 * 1024 * 1024:
                    return JSONResponse(
                        {"detail": f"analysis_json 过大（{fpath.stat().st_size // 1024 // 1024}MB），无法在线渲染，请直接访问 OBS"},
                        status_code=413,
                    )
            try:
                from utils.eval.eval import load_analysis_json, compute_stats, render_html_report_string
                analysis_json = _read_field_content(analysis_json_ref)
                sessions = load_analysis_json(analysis_json)
                stats = compute_stats(sessions)
                key_slot = rec.get("key_slot", "all")
                content = render_html_report_string(
                    sessions, stats,
                    key_name=f"{env_key_name}/{key_slot}",
                    obs_path=rec.get("obs_dst", ""),
                )
                return HTMLResponse(content=content)
            except Exception as e:
                logger.exception("Failed to rebuild report from analysis JSON")
                return JSONResponse({"detail": f"报告重建失败: {e}"}, status_code=500)

        if report_path and Path(report_path).is_file():
            return FileResponse(str(report_path), media_type="text/plain")

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

        roots = [r for r in _all_roots() if Path(r).is_dir()]
        stats = build_stats_multi(roots) if roots else {"rows": []}
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

        _enqueue_task(record_id, lambda rid=record_id, ed=env_dir, ek=env_key_name, op=obs_prefix, nt=now_tag, m=mode: _run_task(rid, ed, ek, op, nt, m))

        return JSONResponse({
            "record_id": record_id,
            "session_path": obs_dst,
            "status": "queued",
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

    # start_stats_warmer(str(env_dir))
