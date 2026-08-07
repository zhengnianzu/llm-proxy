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
    cancel_interrupted,
    create_record,
    delete_record,
    restore_record,
    get_record,
    get_leaves_cache,
    list_records,
    list_records_all_slim,
    list_records_by_key,
    save_leaves_cache,
    update_status,
)
from utils.export_jobs import persist_params, persist_upload_retry
from utils.export_sync import export_session_index, sync_session_index, _load_session_index
from utils.key_config import load_key_state
from utils.key_store import list_keys, mask_key
from utils.log_paths import get_service_log_dir, get_log_dir
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

# 全局并发上限：同一时刻最多几个导出任务在真正执行（跨 worker）。
# 用 EXPORT_CONCURRENCY 配置，默认 1（同一时刻只跑一个导出/质检，其余自动排队）。
_EXPORT_CONCURRENCY = max(1, int(os.getenv("EXPORT_CONCURRENCY", "1")))

# N 个槽位锁文件，任务执行前非阻塞抢占任一空槽，抢到才执行 → 全局并发严格 ≤ N。
#
# 关键：锁目录必须是**绝对路径**，且在运行时（而非模块 import 时）解析。
# app 进程与独立 export_worker 进程可能有不同 cwd，若在 import 时用相对路径求值，
# 两者会解析到不同目录 → 跨进程文件锁失效、导出并发失控。因此这里留空，由
# init_export_lock_dir() 在启动时（app 的 register_export_routes / worker 的 _init_env）
# 显式初始化为同一绝对路径。
_EXPORT_LOCK_DIR = ""
_EXPORT_LOCK_PATHS: list = []


def init_export_lock_dir(d: "str | None" = None) -> None:
    """初始化导出槽位锁目录为绝对路径，并生成 N 个槽位锁文件路径。

    app 与 worker 双端在启动时各调一次；只要 get_service_log_dir() 在相同环境变量
    下解析一致（PROXY_PORT/LOG_TASK_TAG/UPSTREAM_API_KEY 相同），两端就落到同一
    绝对目录，跨进程互斥成立。
    """
    global _EXPORT_LOCK_DIR
    _EXPORT_LOCK_DIR = os.path.abspath(d or get_service_log_dir())
    _EXPORT_LOCK_PATHS[:] = [
        os.path.join(_EXPORT_LOCK_DIR, f"export_queue.slot{i}.lock")
        for i in range(_EXPORT_CONCURRENCY)
    ]


def _acquire_slot():
    """非阻塞轮询抢占任一空槽，返回持锁的文件对象；抢不到返回 None。"""
    import fcntl
    if not _EXPORT_LOCK_DIR:
        # 兜底：调用方未显式 init（不应发生），按当前环境解析一次。
        init_export_lock_dir()
    os.makedirs(_EXPORT_LOCK_DIR, exist_ok=True)
    for path in _EXPORT_LOCK_PATHS:
        try:
            fp = open(path, "w")
            fcntl.flock(fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fp
        except OSError:
            try:
                fp.close()
            except OSError:
                pass
            continue
    return None


def _release_slot(fp) -> None:
    import fcntl
    if fp is None:
        return
    try:
        fcntl.flock(fp, fcntl.LOCK_UN)
        fp.close()
    except OSError:
        pass


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
    """调度线程，逐个执行队列中的任务。跨 worker 用 N 槽位文件锁保证全局并发 ≤ N。"""
    import time
    while True:
        record_id, fn = _task_queue.get()
        if _is_cancelled(record_id):
            _task_queue.task_done()
            continue
        lock_fp = None
        try:
            # 抢占任一空槽；所有槽被占（已达全局并发上限）时轮询等待
            while True:
                if _is_cancelled(record_id):
                    break
                lock_fp = _acquire_slot()
                if lock_fp is not None:
                    break
                time.sleep(1)
            if lock_fp is None:  # 等待期间被取消
                continue
            # 拿到槽后再确认一次是否在等待期间被取消
            if _is_cancelled(record_id):
                continue
            fn()
        except Exception:
            logger.exception("queue_runner: task %s raised", record_id)
        finally:
            _release_slot(lock_fp)
            _task_queue.task_done()


# 是否在 app 进程内跑导出队列（旧行为）。默认关闭：导出执行已剥离到独立的
# export_worker 进程。仅在临时排障 / 回退时置 EXPORT_QUEUE_IN_PROCESS=1 恢复旧路径。
_EXPORT_QUEUE_IN_PROCESS = os.getenv("EXPORT_QUEUE_IN_PROCESS", "") in ("1", "true", "True")


def _enqueue_task(record_id: int, fn=None) -> None:
    """入队 = 把记录状态置 queued，交由独立 export_worker 进程从 DB 领取执行。

    任务参数在入队前已由调用方 persist_params() 写进 export_records.task_json，
    worker 靠 record_id + task_json 重建执行，不依赖内存闭包。

    fn 参数仅在 EXPORT_QUEUE_IN_PROCESS=1 回退路径下使用（塞进本进程内存队列）。
    """
    from utils.export_store import update_status
    update_status(record_id, "queued")
    if _EXPORT_QUEUE_IN_PROCESS and fn is not None:
        _task_queue.put((record_id, fn))


# 回退路径：仅当 EXPORT_QUEUE_IN_PROCESS=1 时，才在 app 进程内起 N 个 runner。
# 默认（剥离模式）不启动任何线程，导出执行完全交给 export_worker 进程。
if _EXPORT_QUEUE_IN_PROCESS:
    for _i in range(_EXPORT_CONCURRENCY):
        _t = threading.Thread(target=_queue_runner, daemon=True, name=f"export-queue-runner-{_i}")
        _t.start()


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


def _key_name_snapshot(api_key: str) -> str:
    """建任务时快照 key 名称（按后 4 位匹配 DB keys 列表），供历史记录展示。

    key 后续改名不会影响已建记录；无匹配时返回 ''，前端回退显示 api_key 尾部。
    """
    if not api_key or api_key == "(empty)":
        return ""
    try:
        from utils.key_store import list_keys
        for k in list_keys():
            if k.get("key", "").endswith(api_key[-4:]):
                return k.get("name", "") or ""
    except Exception:
        return ""
    return ""


# ---------------------------------------------------------------------------
# 目录解析 & 任务执行（模块级）
#
# 这些函数原为 register_export_routes 内的闭包，现提到模块级并显式接收 env_dir，
# 使独立的 export_worker 进程无需构造 FastAPI app 即可复用同一套执行逻辑。
# app 内的 register_export_routes 保留同名薄封装委托到这里，端点调用点不变。
# ---------------------------------------------------------------------------

def _all_roots_for(env_dir) -> list:
    """活跃 env_dir + 配置的历史路径（去重）。"""
    return get_stats_roots(str(env_dir))


def _existing_roots_for(env_dir) -> list:
    """只保留实际存在的 root。决定 mtime key 是否带 <root_basename>/ 前缀。"""
    return [r for r in _all_roots_for(env_dir) if Path(r).is_dir()]


def _resolve_mt_for(env_dir, mt: str) -> Optional[str]:
    """把 mtime key（<root_id>/<rel> 或裸 <rel>）解析为叶子目录绝对路径。"""
    from utils.logs_config import get_root_id
    roots = _existing_roots_for(env_dir)
    multi = len(roots) > 1
    for root in roots:
        rp = Path(root)
        rid = get_root_id(root, str(env_dir))
        base = os.path.basename(os.path.normpath(root))
        for pfx in (rid, base):  # 新前缀优先，旧 basename 回退
            if multi and mt.startswith(pfx + "/"):
                cand = rp / mt[len(pfx) + 1:]
                if cand.is_dir():
                    return str(cand)
        cand2 = rp / mt
        if cand2.is_dir():
            return str(cand2)
    cand = Path(env_dir) / mt
    return str(cand) if cand.is_dir() else None


def _log_dir_key_for(env_dir, mt_src: str) -> str:
    """把叶子目录绝对路径解析为相对 root 的 dir_key（如 "260728/26072813"）。"""
    from utils.log_scan import dir_key_for
    p = Path(mt_src)
    for root in _existing_roots_for(env_dir):
        rp = Path(root)
        try:
            if p == rp or rp in p.parents:
                return dir_key_for(rp, p)
        except (OSError, ValueError):
            continue
    return p.name


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
            rec = get_record(record_id) or {}
            update_status(record_id, "success",
                          files_uploaded=rec.get("total_sessions", 0),
                          error_message="")
        else:
            _log(f"上传失败: {msg}")
            update_status(record_id, "failed", error_message=f"OBS upload: {msg}")
    except Exception as e:
        logger.exception("_run_upload_only crashed (record=%s)", record_id)
        update_status(record_id, "failed", error_message=str(e))


def _run_task(record_id, _env_dir, _env_key_name, obs_prefix, now_tag, mode, force=False):
    from utils.eval.reformat import reformat_and_analyze
    from utils.eval.reconstruct import reconstruct_and_export
    from utils.eval.eval import evaluate_sessions
    from utils.obs_sync import _run_upload_cmd

    _log = lambda msg: append_log(record_id, msg)
    try:
        # reconstruct 与 reformat/eval 同框架但 processor 不同：前者逐 session 聚合多个
        # trace 文件（去重 + 保留分支 + 回填 reasoning），后者只处理 latest_file。
        if mode == "reconstruct":
            _run_task_inner(record_id, _env_dir, _env_key_name, obs_prefix, now_tag, mode, force,
                            reconstruct_and_export, evaluate_sessions, _run_upload_cmd, _log)
        else:
            _run_task_inner(record_id, _env_dir, _env_key_name, obs_prefix, now_tag, mode, force,
                            reformat_and_analyze, evaluate_sessions, _run_upload_cmd, _log)
    except Exception as exc:
        logger.exception("_run_task crashed (record=%s)", record_id)
        try:
            _log(f"任务异常终止: {exc}")
        except Exception:
            logger.warning("append_log also failed for record=%s", record_id)
        update_status(record_id, "failed", error_message=str(exc))


def _run_task_from_record(record_id: int, params: dict) -> None:
    """worker 侧入口：从 DB 记录 + task_json 参数重建执行上下文并跑任务。

    env_dir / env_key_name 取自 persist_params 落库的 task_json（绝对路径）。
    这两者不能由 SERVICE_LOG_DIR（get_service_log_dir(): logs/port<P>/<seg>）反推——
    app 侧 register_export_routes 收到的是 get_log_dir("logs_all")（logs_all/<env>），
    base 与日期段都不同，反推会解析到错误目录（历史 bug）。故必须随任务持久化。

    旧记录（升级前入队）task_json 缺 env_dir：兜底用 get_log_dir("logs_all") 的
    父目录，与 app 注册路由时同源，避免旧任务直接失败。
    """
    env_dir_str = params.get("env_dir", "") or ""
    env_key_name = params.get("env_key_name", "") or ""
    if env_dir_str:
        env_dir = Path(env_dir_str)
    else:
        # 兜底：与 app register_export_routes(app, get_log_dir("logs_all")) 同源
        env_dir = Path(get_log_dir("logs_all")).parent
        logger.warning("record=%s task_json 缺 env_dir，兜底用 %s", record_id, env_dir)
    if not env_key_name:
        env_key_name = env_dir.name
    _run_task(record_id, env_dir, env_key_name,
              params.get("obs_prefix", ""), params.get("now_tag", ""),
              params.get("mode", "export"), force=bool(params.get("force", False)))


def _run_task_inner(record_id, _env_dir, _env_key_name, obs_prefix, now_tag, mode, force,
                    processor, evaluate_sessions, _run_upload_cmd, _log):
    """逐 mtime 目录并行导出。processor 按 mode 注入：
    reformat/eval → reformat_and_analyze（analyze 按 mode 开/关）；
    reconstruct → reconstruct_and_export（无 analyze 参数，逐 session 聚合多个 trace）。"""
    rec = get_record(record_id)
    update_status(record_id, "running")

    sync_cfg = _load_sync_config()
    # workers：优先用任务记录里显式配置的并发数（新建任务时可选）；
    # 记录未设（0/缺失，含旧任务）时回退全局 sync 配置默认。
    _rec_workers = rec.get("workers") or 0
    workers = _rec_workers if _rec_workers > 0 else sync_cfg.get("workers", 8)
    # dir_workers：多个 mtime 目录并行处理的并发数（每目录内部仍各用 workers 线程）。
    # 0/缺失（含旧任务）= 默认 8；总线程 ≈ dir_workers × workers。
    dir_workers = int(rec.get("dir_workers") or 0) or 8
    upload_script = sync_cfg.get("upload_script") or None

    mtime_dirs = json.loads(rec.get("mtime_dirs", "[]"))
    api_key = rec.get("api_key", "")
    slot = rec.get("key_slot", "all")

    if mode == "export":
        local_base = _env_dir.parent.parent / "logs_session" / _env_key_name / slot / f"ex-{now_tag}"
        obs_sub = "session"
    elif mode == "reconstruct":
        # hermes 重构聚合：每 session 落多个聚合文件 + _manifest.jsonl，
        # 走 session_reconstruct/（平行于 session_analysis/，互不混放）
        local_base = (_env_dir.parent.parent / "logs_session_reconstruct" / _env_key_name / slot / f"ex-{now_tag}").resolve()
        obs_sub = "session_reconstruct"
    else:
        # eval 与 reformat 都产出合并后的 session JSON，落 session_analysis 目录
        local_base = (_env_dir.parent.parent / "logs_session_analysis" / _env_key_name / slot / f"ex-{now_tag}").resolve()
        obs_sub = "session_analysis"

    obs_dst = f"{obs_prefix}/{obs_sub}/{_env_key_name}/{slot}/ex-{now_tag}/" if obs_prefix else ""
    local_base.mkdir(parents=True, exist_ok=True)

    update_status(record_id, "running", obs_dst=obs_dst, local_copy_dir=str(local_base))

    _mode_label = {"eval": "质检", "reformat": "合并导出", "reconstruct": "重构导出"}.get(mode, "导出")
    _log(f"开始{_mode_label}: key_slot={slot}, mtime_dirs={mtime_dirs}")
    _log(f"本地目录: {local_base}")
    if obs_dst:
        _log(f"OBS 目标: {obs_dst}")

    errors = []
    warnings = []  # 非致命跳过（如 new-api 索引未构建）：记为提示，不判任务失败
    total_sessions = 0
    total_uploaded = 0
    total_skipped = 0
    all_results = []
    all_entries = []

    def _process_mtime(mt):
        """处理单个 mtime 目录，返回一份独立结果 dict（供外层串行合并，避免并发写共享量）。
        每目录各自读/写自己的 session_index.jsonl 与 .sync_state，互不干扰。"""
        res = {"mt": mt, "errors": [], "warnings": [], "skip": False,
               "new_sessions": 0, "uploaded": 0, "skipped": 0,
               "results": [], "entries": []}
        # 协作式取消：用户「终止」后，尚未开始的目录直接跳过（并行/串行同理）。
        if _is_cancelled(record_id):
            res["skip"] = True
            return res
        mt_src = _resolve_mt_for(_env_dir, mt) or str(_env_dir / mt)
        try:
            # 格式检测：new-api 导出不触发回填；reconstruct 仅支持 new-api 合并文件，
            # native 三元组叶子没有合并文件实物（hermes 聚合器读合并文件本身，见
            # doc/REAMDE_traj.md §3.2），在生成 session_index 前就跳过，省一次构建。
            try:
                from utils.log_scan import detect_format as _df
                _fmt = _df(mt_src)
                if mode == "reconstruct" and _fmt != "newapi":
                    _log(f"[{mt}] 目录格式 {_fmt}，重构导出仅支持 new-api 合并文件，跳过")
                    res["warnings"].append(f"{mt}: 非 new-api 格式（{_fmt}），已跳过")
                    res["skip"] = True
                    return res
                if _fmt == "newapi":
                    import utils.newapi_index_db as _nidb
                    if _nidb.needs_build(mt_src):
                        _log(f"[{mt}] 索引未构建/不完整，跳过；请先在数据管理界面手动构建索引后再导出")
                        res["warnings"].append(f"{mt}: 索引未构建/不完整，已跳过（请先在数据管理界面手动构建索引）")
                        res["skip"] = True
                        return res
            except Exception as _pe:
                # 索引状态检查本身异常（如 index.db 存在但 meta 表未建好、
                # 半成品库 no such table: meta 等）与 needs_build=True 是同一语义：
                # 叶子索引尚未就绪。跳过并提示，不判整个任务失败（warning 非致命）。
                _log(f"[{mt}] 索引未就绪（检查异常: {_pe}），跳过；请先在数据管理界面手动构建索引")
                res["warnings"].append(f"{mt}: 索引未构建/不完整，已跳过（检查异常: {_pe}）")
                res["skip"] = True
                return res
            _log(f"[{mt}] 生成 session_index...")
            exp_result = export_session_index(mt_src, force=force)
            _log(f"[{mt}] session_index: {exp_result.get('total_sessions', 0)} sessions"
                 + (" (已是最新，跳过)" if exp_result.get("skipped") else ""))

            if mode == "export":
                _log(f"[{mt}] 开始复制文件" + (f" (按 key 过滤: ...{api_key[-8:]})" if api_key else " (全量)"))
                # 只复制到共享 local_base，不在此上传；收尾统一上传一次（见下方 finalize）。
                # 避免多目录并行时每目录都整目录上传 → 重复上传 + 写入中上传的竞态。
                sync_result = sync_session_index(
                    mt_src, obs_dst=obs_dst, key=api_key or None,
                    local_copy_dir=str(local_base), force=force, workers=workers,
                    upload=False,
                )
                matched = sync_result.get("matched_sessions", 0)
                res["new_sessions"] = sync_result.get("new_sessions", 0)
                res["uploaded"] = sync_result.get("uploaded", 0)
                res["skipped"] = sync_result.get("skipped", 0)
                _log(f"[{mt}] 匹配 {matched} sessions, 新导出 {res['new_sessions']}, 跳过 {res['skipped']}, 复制文件 {res['uploaded']}")
            else:
                session_entries = _load_session_index(mt_src)
                total_before = len(session_entries)
                if api_key:
                    session_entries = [s for s in session_entries if s.get("api_key") == api_key]
                if not session_entries:
                    _log(f"[{mt}] 共 {total_before} sessions, 按 key 过滤后 0, 跳过")
                    res["skip"] = True
                    return res
                if api_key:
                    _log(f"[{mt}] 共 {total_before} sessions, 按 key 过滤后 {len(session_entries)}")
                if mode == "reconstruct":
                    # 格式守卫已在上方 detect_format 处提前拦截（非 new-api 已跳过）。
                    _log(f"[{mt}] 进入 hermes 重构: {len(session_entries)} sessions, workers={workers}...")
                    ra_result = processor(
                        src_dir=mt_src, out_dir=str(local_base),
                        session_entries=session_entries, api_key=api_key, workers=workers,
                        progress_cb=lambda msg, _mt=mt: _log(f"[{_mt}] {msg}"),
                        log_dir=_log_dir_key_for(_env_dir, mt_src),
                    )
                else:
                    _do_analyze = (mode == "eval")
                    _step = "reformat+analyze" if _do_analyze else "reformat"
                    _log(f"[{mt}] 进入 {_step}: {len(session_entries)} sessions, workers={workers}...")
                    ra_result = processor(
                        src_dir=mt_src, out_dir=str(local_base),
                        session_entries=session_entries, api_key=api_key, workers=workers,
                        progress_cb=lambda msg, _mt=mt: _log(f"[{_mt}] {msg}"),
                        log_dir=_log_dir_key_for(_env_dir, mt_src),
                        analyze=_do_analyze,
                    )
                res["results"] = ra_result["results"]
                res["entries"] = session_entries
                _n_err = len(ra_result.get('errors', []))
                _done_word = {"eval": "质检", "reconstruct": "重构"}.get(mode, "导出")
                _log(f"[{mt}] {_done_word}完成：成功 {ra_result['total_files']} 个 session"
                     + (f"，失败 {_n_err}" if _n_err else ""))
        except Exception as e:
            _log(f"[{mt}] 错误: {e}")
            res["errors"].append(f"{mt}: {e}")
            logger.exception("task failed for %s (mode=%s)", mt, mode)
        return res

    # 多目录调度：dir_workers=1 顺序执行；>1 用线程池并行（每目录内部仍各用 workers 线程）。
    # 结果按提交顺序收集后【串行合并】到共享累加量，避免并发写竞态。
    if _is_cancelled(record_id):
        _log("任务已终止（用户取消）")
        update_status(record_id, "cancelled", error_message="用户终止")
        return

    if dir_workers <= 1 or len(mtime_dirs) <= 1:
        mt_results = [_process_mtime(mt) for mt in mtime_dirs]
    else:
        n_par = min(dir_workers, len(mtime_dirs))
        _log(f"多目录并行处理：{len(mtime_dirs)} 个目录，目录并发 {n_par}，每目录 workers={workers}")
        from concurrent.futures import ThreadPoolExecutor as _TPE
        mt_results = [None] * len(mtime_dirs)
        with _TPE(max_workers=n_par) as _ex:
            _fut = {_ex.submit(_process_mtime, mt): i for i, mt in enumerate(mtime_dirs)}
            for f in _fut:
                idx = _fut[f]
                try:
                    mt_results[idx] = f.result()
                except Exception as e:  # noqa: BLE001
                    logger.exception("mtime worker crashed: %s", mtime_dirs[idx])
                    mt_results[idx] = {"mt": mtime_dirs[idx], "errors": [f"{mtime_dirs[idx]}: {e}"],
                                       "warnings": [], "skip": False, "new_sessions": 0,
                                       "uploaded": 0, "skipped": 0, "results": [], "entries": []}

    # 串行合并各目录结果（保持 mtime_dirs 原顺序，结果稳定）
    for res in mt_results:
        if not res:
            continue
        errors.extend(res["errors"])
        warnings.extend(res["warnings"])
        total_uploaded += res["uploaded"]
        total_skipped += res["skipped"]
        if mode == "export":
            total_sessions += res["new_sessions"]
        else:
            all_results.extend(res["results"])
            all_entries.extend(res["entries"])
            total_sessions += len(res["results"])

    # 目录循环结束后、进入收尾（evaluate/上传/判定）前再查一次取消：
    # 若用户在处理最后一个目录期间点了「终止」，这里退出，避免跑完整个 evaluate
    # 与 OBS 上传。
    if _is_cancelled(record_id):
        _log("任务已终止（用户取消）")
        update_status(record_id, "cancelled", error_message="用户终止")
        return

    eval_report_path = ""
    analysis_json_stored = ""
    if mode == "export":
        # 收尾统一上传：所有 mtime 目录已把三元组文件复制到共享 local_base，
        # 这里对整个 local_base 上传一次。obsutil 用 -u 增量，已存在的对象自动跳过，
        # 所以「重试上传」再跑一遍时只补上次失败的文件。
        if total_sessions > 0:
            if obs_dst:
                _log(f"复制完成，开始统一上传到 OBS: {obs_dst}")
                obs_parent = obs_dst.rstrip("/").rsplit("/", 1)[0] + "/"
                ok, msg = _run_upload_cmd(str(local_base), obs_parent, upload_script, log_cb=_log)
                if ok:
                    _log("上传成功")
                else:
                    _log(f"上传失败: {msg}")
                    errors.append(f"OBS upload: {msg}")
            else:
                _log("未配置 OBS 目标，仅本地复制完成")
        else:
            _log("无新 session 需要导出")
    elif mode in ("reformat", "reconstruct"):
        # reformat-only / reconstruct：合并/聚合后的 session JSON 已由 worker 落在
        # local_base/<session>/，这里补一份 session_index.jsonl 清单后整目录上传，
        # 不跑 evaluate/质检。
        if all_results:
            idx_path = local_base / "session_index.jsonl"
            with open(idx_path, "w", encoding="utf-8") as f:
                for entry in all_entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            if obs_dst:
                _log(f"同步到 OBS: {obs_dst}")
                obs_parent = obs_dst.rstrip("/").rsplit("/", 1)[0] + "/"
                ok, msg = _run_upload_cmd(str(local_base), obs_parent, upload_script, log_cb=_log)
                if ok:
                    _log("上传成功")
                    # 整目录上传成功：记入本次合并的 session 数（否则 files_uploaded 恒为 0）
                    total_uploaded = len(all_results)
                else:
                    _log(f"上传失败: {msg}")
                    errors.append(f"OBS upload: {msg}")
        else:
            _log("无 session 数据")
    elif mode == "eval":
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
                    write_eval_to_cache(_resolve_mt_for(_env_dir, mt) or str(_env_dir / mt), all_results)
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
                # 整目录上传成功：记入本次质检的 session 数（否则 files_uploaded 恒为 0）
                total_uploaded = len(all_results)
            else:
                _log(f"上传失败: {msg}")
                errors.append(f"OBS upload: {msg}")

    _done_label = {"eval": "质检", "reformat": "合并导出", "reconstruct": "重构导出"}.get(mode, "导出")

    # 判定失败只看「真错误」(errors)：上传失败、异常等。
    # warnings（如 new-api 索引未构建被跳过）不影响已成功部分的产出，不判失败。
    no_output = (mode in ("eval", "reformat", "reconstruct") and not all_results)
    warn_suffix = f"（{len(warnings)} 个目录跳过：{'; '.join(warnings)}）" if warnings else ""

    if errors or no_output:
        # no_output 且无真错误：说明目录全被跳过（多为索引未构建），如实说明而非笼统“失败”
        if no_output and not errors:
            reason = f"无 session 数据{warn_suffix}" if warnings else "无 session 数据"
        else:
            reason = "; ".join(errors) + warn_suffix
        _log(f"{_done_label}失败: {reason}")
        update_status(record_id, "failed",
                      error_message=reason,
                      total_sessions=total_sessions,
                      files_uploaded=total_uploaded,
                      files_skipped=total_skipped,
                      eval_report_path=eval_report_path,
                      analysis_json=analysis_json_stored)
    else:
        # 成功；若有跳过目录，把提示写进 error_message 供前端展示，但状态是 success
        msg = f"{_done_label}完成: {total_sessions} sessions"
        if warnings:
            msg += f"，{len(warnings)} 个目录跳过（索引未构建）"
        _log(msg)
        update_status(record_id, "success",
                      error_message=(f"部分目录跳过：{'; '.join(warnings)}" if warnings else ""),
                      total_sessions=total_sessions,
                      files_uploaded=total_uploaded,
                      files_skipped=total_skipped,
                      eval_report_path=eval_report_path,
                      analysis_json=analysis_json_stored)



def register_export_routes(app: FastAPI, logs_dir: str) -> None:
    env_dir = Path(logs_dir).parent
    env_key_name = env_dir.name
    templates = Jinja2Templates(directory="templates")

    # 初始化导出槽位锁目录为绝对路径（与独立 export_worker 进程对齐，见 init_export_lock_dir）。
    init_export_lock_dir()

    # keys 统计（跨 root 扫盘）的内存缓存，10 分钟 TTL。前端「刷新统计」传 force=1 强刷。
    _keys_cache = {"ts": 0.0, "data": None}
    _KEYS_TTL = 600

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
        from utils.logs_config import get_root_id
        for root in roots:
            rp = Path(root)
            base = get_root_id(root, str(env_dir))
            for leaf in iter_index_dirs(rp):
                rel = dir_key_for(rp, leaf)
                out.append(f"{base}/{rel}" if multi else rel)
        return sorted(set(out), reverse=True)

    def _resolve_mt(mt: str) -> Optional[str]:
        """把 mtime key（<root_id>/<rel> 或裸 <rel>）解析为叶子目录绝对路径。

        优先按 root_id 前缀匹配；对 export_records 里遗留的旧 <basename>/ 前缀
        保留回退匹配，保证历史记录仍可解析。
        """
        from utils.logs_config import get_root_id
        roots = _existing_roots()
        multi = len(roots) > 1
        for root in roots:
            rp = Path(root)
            rid = get_root_id(root, str(env_dir))
            base = os.path.basename(os.path.normpath(root))
            for pfx in (rid, base):  # 新前缀优先，旧 basename 回退
                if multi and mt.startswith(pfx + "/"):
                    cand = rp / mt[len(pfx) + 1:]
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
        # no-store：导出页模板已重构为任务列表，避免浏览器用旧的缓存页面
        # （页面路由默认无 Cache-Control，浏览器会启发式缓存 HTML）。
        return templates.TemplateResponse(
            request, "export.html",
            context={"active_page": "export", "user_role": request.session.get("monitor_role", "user"), "user_name": request.session.get("monitor_user", ""), "user_permissions": [p.strip() for p in (request.session.get("monitor_permissions") or "").split(",") if p.strip()]},
            headers={"Cache-Control": "no-store"},
        )

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
            "workers": sync_cfg.get("workers", 8),
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
        # 回填在独立 worker 进程，app 内存无运行态；运行/排队真相取自 backfill_requests
        # 表（get_queue_snapshot 已据此拼 current/queued/running）。这里额外用它标注每个
        # root 的 running/queued，避免逐叶进度虽由磁盘探测得出、状态却恒显 idle。
        active_by_root = {}
        for _rr in snap.get("running_roots", []):
            active_by_root[_rr] = "running"
        for _qr in snap.get("queued", []):
            active_by_root.setdefault(_qr, "queued")
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
            active = active_by_root.get(os.path.normpath(r))
            roots_prog.append({
                "root": r,
                "name": Path(r).name,
                "status": active or st.get("status", "idle"),
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
    def export_keys(request: Request, threshold: int = 5, force: int = 0):
        denied = _require_key_api(request)
        if denied:
            return denied
        import time as _time
        now = _time.time()
        # 10 分钟内存缓存：keys 统计是跨 root 扫盘的重接口，避免每次进页面/开新建任务都重扫。
        # force=1（点「刷新统计」）时强制重算并刷新缓存。
        if not force and _keys_cache["data"] is not None and (now - _keys_cache["ts"] < _KEYS_TTL):
            payload = dict(_keys_cache["data"])
            payload["cached"] = True
            payload["cache_age"] = int(now - _keys_cache["ts"])
            return JSONResponse(payload)

        payload = _compute_keys_payload(threshold)
        _keys_cache["ts"] = now
        _keys_cache["data"] = payload
        out = dict(payload)
        out["cached"] = False
        out["cache_age"] = 0
        return JSONResponse(out)

    def _compute_keys_payload(threshold: int = 5) -> dict:
        """组装 /api/export/keys 的完整响应体（跨 root 扫盘统计 + 模型分布 + 每 key 记录）。"""
        roots = [r for r in _all_roots() if Path(r).is_dir()]
        if not roots:
            return {"keys": [], "mtimes": [], "models": [], "last_refresh_ts": get_last_refresh_ts()}

        # 跨 root 合并 session 统计（rows 的 mtime_cells key 已带 <root_id>/ 前缀）
        stats = build_stats_multi(roots, threshold, active_env_dir=str(env_dir))

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

        return {"keys": keys_result, "mtimes": _list_all_mtimes(),
                "models": all_models, "last_refresh_ts": get_last_refresh_ts()}

    # -----------------------------------------------------------------
    # 统一任务执行（导出 / 质检）
    # -----------------------------------------------------------------


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
        # 并发 worker 数：>0 时按任务记录执行；0/缺失回退全局配置默认。
        try:
            workers = int(body.get("workers", 0) or 0)
        except (TypeError, ValueError):
            workers = 0
        if workers < 0:
            workers = 0
        # 目录并发数：多个 mtime 目录并行处理的数量。0/缺失=串行；总线程≈dir_workers×workers。
        try:
            dir_workers = int(body.get("dir_workers", 0) or 0)
        except (TypeError, ValueError):
            dir_workers = 0
        if dir_workers < 0:
            dir_workers = 0
        # 默认立即入队执行；start=False 时只建草稿（draft）记录，由前端手动「启动」。
        start = body.get("start", True)
        # mode 优先显式取值（export / eval / reformat / reconstruct）；兼容旧的 auto_eval 布尔
        mode = body.get("mode")
        if mode not in ("export", "eval", "reformat", "reconstruct"):
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
            # 草稿阶段 obs_dst 存裸前缀，供「启动」时复原 obs_prefix；
            # 真正执行时 _run_task_inner 会用 now_tag 重拼完整 obs_dst 覆盖。
            obs_dst=obs_prefix if not start else "",
            local_copy_dir="",
            mode=mode,
            key_name=_key_name_snapshot(api_key),
            workers=workers,
            dir_workers=dir_workers,
        )

        if not start:
            # 草稿：只登记不入队，等待用户手动「启动」。
            update_status(record_id, "draft")
            return JSONResponse({"record_id": record_id, "status": "draft", "mode": mode})

        persist_params(record_id, mode=mode, obs_prefix=obs_prefix, force=force,
                       now_tag=now_tag, env_dir=str(env_dir), env_key_name=env_key_name,
                       workers=workers, dir_workers=dir_workers)
        _enqueue_task(record_id, lambda rid=record_id, ed=env_dir, ek=env_key_name, op=obs_prefix, nt=now_tag, m=mode, f=force: _run_task(rid, ed, ek, op, nt, m, force=f))

        return JSONResponse({"record_id": record_id, "status": "queued", "mode": mode})

    @app.post("/api/export/start")
    async def export_start(request: Request):
        """启动一条草稿（draft）记录：复原参数后入队执行。

        草稿在创建时不执行，obs_dst 存的是裸 obs_prefix（见 export_run）；
        这里从记录复原 prefix / mode / api_key / mtime_dirs 后入队，_run_task 内部
        会用新 now_tag 重拼完整 obs_dst。
        """
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
        if rec["status"] != "draft":
            return JSONResponse({"detail": "只能启动草稿任务"}, status_code=400)

        mode = rec.get("mode") or "export"
        obs_prefix = (rec.get("obs_dst", "") or "").strip().rstrip("/")
        force = False
        now_tag = datetime.now().strftime("%y%m%d%H%M%S")

        persist_params(record_id, mode=mode, obs_prefix=obs_prefix, force=force,
                       now_tag=now_tag, env_dir=str(env_dir), env_key_name=env_key_name,
                       workers=rec.get("workers") or 0,
                       dir_workers=rec.get("dir_workers") or 0)
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
            key_name=rec.get("key_name", ""),
        )

        persist_params(new_record_id, mode="eval", obs_prefix=obs_prefix, force=False,
                       now_tag=now_tag, env_dir=str(env_dir), env_key_name=env_key_name)
        _enqueue_task(new_record_id, lambda rid=new_record_id, ed=env_dir, ek=env_key_name, op=obs_prefix, nt=now_tag: _run_task(rid, ed, ek, op, nt, "eval"))

        return JSONResponse({"record_id": new_record_id, "status": "queued", "mode": "eval"})

    @app.post("/api/export/retry")
    async def export_retry(request: Request):
        """重试失败/已取消（含用户终止）的任务：用原记录参数新建一条记录重跑。

        原失败记录保留作历史（同 export_eval 的"从原记录重建"模式）。沿用原
        mode（export/reformat/eval）、api_key、mtime_dirs；obs_prefix 从原
        obs_dst 复原，保证与原任务一致。
        """
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
        if rec["status"] not in ("failed", "cancelled"):
            return JSONResponse({"detail": "只能重试失败/已取消的任务"}, status_code=400)

        mode = rec.get("mode") or "export"
        # obs_prefix 复原：obs_dst 形如 <prefix>/session/... 或 <prefix>/session_analysis/...
        obs_dst = rec.get("obs_dst", "") or ""
        obs_prefix = ""
        if obs_dst:
            for _seg in ("/session_analysis/", "/session/"):
                if _seg in obs_dst:
                    obs_prefix = obs_dst.split(_seg)[0]
                    break
            else:
                obs_prefix = obs_dst.rstrip("/").rsplit("/", 1)[0]
        now_tag = datetime.now().strftime("%y%m%d%H%M%S")

        new_record_id = create_record(
            api_key=rec["api_key"], key_slot=rec["key_slot"],
            mtime_dirs=rec["mtime_dirs"],
            obs_dst="", local_copy_dir="",
            mode=mode,
            key_name=rec.get("key_name", ""),
        )

        persist_params(new_record_id, mode=mode, obs_prefix=obs_prefix, force=False,
                       now_tag=now_tag, env_dir=str(env_dir), env_key_name=env_key_name,
                       workers=rec.get("workers") or 0,
                       dir_workers=rec.get("dir_workers") or 0)
        _enqueue_task(new_record_id, lambda rid=new_record_id, ed=env_dir, ek=env_key_name, op=obs_prefix, nt=now_tag, m=mode: _run_task(rid, ed, ek, op, nt, m))

        return JSONResponse({"record_id": new_record_id, "status": "queued", "mode": mode})

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

        persist_upload_retry(record_id)
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
        if rec["status"] not in ("queued", "pending", "running"):
            return JSONResponse({"detail": "只能终止排队中或运行中的任务"}, status_code=400)
        # 标记取消：加入本进程集合（供出队/抢槽/目录循环检查点识别），并把 DB 置
        # cancelled（跨 worker 真相源）。
        #   - queued/pending：出队时被 _is_cancelled 跳过。
        #   - running：执行线程在目录循环顶部的检查点自行退出；即使线程已死
        #     （如被 sqlite 撞崩的僵尸），DB 置 cancelled 后前端"运行中"立即消失。
        with _queue_lock:
            _cancelled_ids.add(record_id)
        update_status(record_id, "cancelled", error_message="用户终止")
        return JSONResponse({"record_id": record_id, "status": "cancelled"})

    @app.post("/api/export/delete")
    async def export_delete(request: Request):
        """删除一条导出记录。

        默认软删除：置 is_delete=1，从列表移除，行与产物元数据全部保留，可 restore 恢复。
        只允许删终态/草稿（draft/success/failed/cancelled）；running/queued/pending 需先取消——
        避免删掉 worker 正在写、或即将出队执行的行。

        purge=true 为永久删除（硬删 DB 行 + 清外置日志）；再叠加 purge_local=true 连本地产物
        目录一并删（不可逆，OBS 云端不动）。
        """
        denied = _require_key_api(request)
        if denied:
            return denied
        body = await request.json()
        record_id = body.get("record_id")
        if not record_id:
            return JSONResponse({"detail": "record_id is required"}, status_code=400)
        purge = bool(body.get("purge", False))
        purge_local = bool(body.get("purge_local", False))
        rec = get_record(record_id)
        if not rec:
            return JSONResponse({"detail": "Record not found"}, status_code=404)
        if rec["status"] not in ("draft", "success", "failed", "cancelled"):
            return JSONResponse(
                {"detail": "只能删除草稿/已完成/失败/已取消的任务，运行中或排队中请先取消"},
                status_code=400)
        delete_record(record_id, purge=purge, purge_local=purge_local)
        return JSONResponse({
            "record_id": record_id, "deleted": True,
            "soft": not purge, "purge": purge, "purge_local": purge_local,
        })

    @app.post("/api/export/restore")
    async def export_restore(request: Request):
        """撤销软删除：is_delete 置 0，记录重新出现在列表。"""
        denied = _require_key_api(request)
        if denied:
            return denied
        body = await request.json()
        record_id = body.get("record_id")
        if not record_id:
            return JSONResponse({"detail": "record_id is required"}, status_code=400)
        rec = restore_record(record_id)
        if not rec:
            return JSONResponse({"detail": "Record not found"}, status_code=404)
        return JSONResponse({"record_id": record_id, "restored": True})

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

    @app.get("/api/export/records-slim")
    def export_records_slim(request: Request, limit_per_key: int = 10):
        """轻量记录列表：只查 export_records（不重扫统计/不读大字段），按 key_slot 分组。

        供前端高频轮询（几秒一次）只更新每张卡片的记录状态/进度，重的
        /api/export/keys（含跨 root 统计扫盘）保持手动或低频刷新。
        """
        denied = _require_key_api(request)
        if denied:
            return denied
        grouped = list_records_all_slim(limit_per_key=limit_per_key)
        # 任务化后记录就是页面主列表，不再按 slot 截断（limit_per_key 仅作兜底）。
        # 记录按 key_slot 分组返回，前端把各组拍平即为完整任务表。
        return JSONResponse({"records_by_slot": grouped})

    def _compute_leaves(rec: dict) -> tuple:
        """把记录的 mtime_dirs 逐叶解析为 [{dir_key, sessions, state, error}], warnings。
        session 数读各叶 session_index.jsonl 的 meta；构建状态来自 log_dir.db。"""
        from utils.export_sync import _read_session_index_meta
        from utils.log_scan import dir_key_for

        mtime_dirs = json.loads(rec.get("mtime_dirs", "[]") or "[]")
        leaves = []
        warnings = []
        for mt in mtime_dirs:
            mt_src = _resolve_mt(mt) or (str(env_dir / mt) if Path(env_dir / mt).is_dir() else "")
            if not mt_src:
                warnings.append(f"{mt}: 目录不存在")
                leaves.append({"dir_key": mt, "sessions": 0, "state": "missing",
                               "built": False, "error": "目录不存在"})
                continue
            leaf = {"dir_key": mt, "sessions": 0, "state": "unknown",
                    "built": False, "error": ""}
            try:
                meta = _read_session_index_meta(mt_src)
                if meta:
                    leaf["sessions"] = int(meta.get("total_sessions", 0) or 0)
                try:
                    import utils.logdir_store as lds
                    from utils.logs_config import get_root_id
                    # dir_key 是相对叶子所属 root 的标识（与 log_dir.db / 叶子详情同口径）。
                    # 遍历已配置 roots，找到包含该叶子的 root 再算 dir_key。
                    dk = ""
                    rid = ""
                    for _r in _existing_roots():
                        _rp = Path(_r)
                        try:
                            if Path(mt_src) == _rp or _rp in Path(mt_src).parents:
                                rid = get_root_id(_r, str(env_dir))
                                dk = dir_key_for(_rp, Path(mt_src))
                                break
                        except (OSError, ValueError):
                            continue
                    info = None
                    try:
                        # bulk_get 返回 list，转成 dir_key -> row 的查找表
                        _by_key = {r.get("dir_key", ""): r for r in lds.bulk_get(rid)} if (rid and lds.has_any(rid)) else {}
                        info = _by_key.get(dk) if dk else None
                    except Exception:
                        info = None
                    if info:
                        leaf["state"] = info.get("state", "unknown")
                        leaf["built"] = bool(info.get("built"))
                        leaf["error"] = info.get("last_error", "") or ""
                except Exception:
                    pass  # log_dir.db 未同步/缺失：状态留 unknown
            except Exception as e:  # noqa: BLE001
                leaf["state"] = "error"
                leaf["error"] = str(e)
            leaves.append(leaf)
        return leaves, warnings

    @app.get("/api/export/leaves")
    def export_leaves(request: Request, record_id: int = 0, refresh: int = 0):
        """某导出记录的节点分布：把 mtime_dirs 逐叶解析为 {dir_key, sessions, state, error}。

        供任务行内「节点分布」展开（叶子详情样式）。终态任务（success/failed/cancelled）
        首次计算后把结果缓存进 DB（leaves_cache 列），之后展开直接读缓存、不再扫盘；
        非终态（running 等）或 refresh=1 时实时重算。
        """
        denied = _require_key_api(request)
        if denied:
            return denied
        if not record_id:
            return JSONResponse({"detail": "record_id is required"}, status_code=400)
        rec = get_record(record_id)
        if not rec:
            return JSONResponse({"detail": "Record not found"}, status_code=404)

        status = rec.get("status", "")
        is_terminal = status in ("success", "failed", "cancelled")

        # 终态任务优先读 DB 缓存（除非显式 refresh）：节点分布对已结束的任务是固定的。
        if is_terminal and not refresh:
            cached = get_leaves_cache(record_id)
            if cached is not None:
                out = dict(cached)
                out["record_id"] = record_id
                out["cached"] = True
                return JSONResponse(out)

        leaves, warnings = _compute_leaves(rec)

        # 终态任务算完落库，供后续直接读；非终态不缓存（数据还会变）。
        if is_terminal:
            try:
                save_leaves_cache(record_id, leaves, warnings)
            except Exception:
                logger.debug("save_leaves_cache failed for record=%s", record_id, exc_info=True)

        return JSONResponse({"record_id": record_id, "leaves": leaves,
                             "warnings": warnings, "cached": False})

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
        stats = build_stats_multi(roots, active_env_dir=str(env_dir)) if roots else {"rows": []}
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
            key_name=_key_name_snapshot(api_key),
        )

        obs_dst = f"{obs_prefix}/session_analysis/{env_key_name}/{slot}/ex-{now_tag}/" if obs_prefix else ""

        persist_params(record_id, mode=mode, obs_prefix=obs_prefix, force=False, now_tag=now_tag,
                       env_dir=str(env_dir), env_key_name=env_key_name)
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

    # 注意：导出执行已剥离到独立 export_worker 进程，app 启动**不再**清理
    # running/queued/pending —— 否则 app 重启会误杀 worker 正在跑/待跑的任务。
    # 队列的「重启即清空」语义改由 export_worker 在它自己的进程启动时执行
    # （见 utils/export_worker.py 的 cancel_interrupted 调用）。

    # start_stats_warmer(str(env_dir))
