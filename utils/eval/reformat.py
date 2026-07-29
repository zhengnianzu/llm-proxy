"""
utils/eval/reformat.py — 格式重整 + 质检一体化

将三元组文件（req/headers/res）合并为单个 JSON，只处理 latest_file。
合并后立即跑 analyze_best_data，返回分析结果，省掉二次读取。
使用多进程并行处理。
"""

import json
import logging
import os
import signal
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from utils.eval.eval import analyze_best_data, fmt_quality
from utils.message_common import parse_response

logger = logging.getLogger(__name__)

# 无进展超时：距上一个 future 完成超过该秒数没有任何新进展，判定卡死
_NO_PROGRESS_TIMEOUT = int(os.getenv("EVAL_NO_PROGRESS_TIMEOUT", "600"))
# 进程池关闭宽限：所有 future 已完成后，等待 executor 正常收尾的最长秒数
_SHUTDOWN_GRACE = int(os.getenv("EVAL_SHUTDOWN_GRACE", "60"))


def _terminate_pool_workers(executor: ProcessPoolExecutor, log: Callable[[str], None]) -> None:
    """强制回收进程池内残留的 worker 进程（参考 backup_routes 的 SIGTERM->SIGKILL 模式）。

    executor.shutdown(wait=False, cancel_futures=True) 之后，仍可能有 worker
    卡在收尾阶段不退出，这里直接对存活进程发信号回收，避免僵尸进程堆积。
    """
    procs = list(getattr(executor, "_processes", {}).values())
    if not procs:
        return
    log(f"强制回收 {len(procs)} 个 worker 进程")
    for p in procs:
        pid = getattr(p, "pid", None)
        if not pid:
            continue
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    deadline = time.monotonic() + 10
    for p in procs:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            p.join(timeout=remaining)
        except Exception:
            pass
    for p in procs:
        if p.is_alive():
            pid = getattr(p, "pid", None)
            if pid:
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_triplet(src_dir: Path, stem: str) -> Dict[str, Path]:
    tri: Dict[str, Path] = {}
    for suffix, key in [("-req.json", "req"), ("-headers.json", "headers"), ("-res.json", "res")]:
        p = src_dir / f"{stem}{suffix}"
        if p.is_file():
            tri[key] = p
    return tri


def _load_merged(src_dir: Path, stem: str, latest_file: str,
                 tri: Dict[str, Path]) -> Optional[dict]:
    """加载并组装 merged（messages + response + header）。

    优先三元组（本项目格式）；找不到 req 时回退 new-api 合并单文件
    （latest_file 是 .json 但非 -req.json），用 newapi_format 拆解。
    返回 None 表示无法加载。
    """
    if "req" in tri:
        try:
            merged = _load_json(tri["req"])
        except Exception:
            return None
        if "headers" in tri:
            try:
                merged["header"] = _load_json(tri["headers"])
            except Exception:
                merged["header"] = {}
        else:
            merged["header"] = {}
        if "res" in tri:
            try:
                merged["response"] = parse_response(_load_json(tri["res"]))
            except Exception:
                merged["response"] = {}
        else:
            merged["response"] = {}
        return merged

    # new-api：合并单文件
    if latest_file.endswith(".json") and not latest_file.endswith("-req.json"):
        combined = src_dir / latest_file
        if combined.is_file():
            try:
                from utils.newapi_format import build_merged_for_eval
                return build_merged_for_eval(combined)
            except Exception:
                return None
    return None


def _process_one(args: tuple) -> Optional[dict]:
    """单个 session 的 reformat + analyze，多进程可调用。

    返回分析结果 dict（含 session/start_time 等字段），失败返回 None。
    """
    src_dir_s, out_dir_s, first_ts, latest_file, trace_list, api_key, models = args

    src_dir = Path(src_dir_s)
    out_dir = Path(out_dir_s)

    stem = latest_file
    if stem.endswith("-req.json"):
        stem = stem[:-len("-req.json")]
    elif stem.endswith(".json"):
        stem = stem[:-len(".json")]

    tri = _resolve_triplet(src_dir, stem)
    merged = _load_merged(src_dir, stem, latest_file, tri)
    if merged is None:
        return None

    session_dir = out_dir / first_ts
    session_dir.mkdir(parents=True, exist_ok=True)
    out_file = session_dir / f"{stem}.json"
    with open(out_file, "w", encoding="utf-8") as fh:
        json.dump(merged, fh, ensure_ascii=False, separators=(",", ":"))

    # 直接分析，不再二次读取
    analyzed = analyze_best_data(merged)

    resp = merged.get("response") or {}
    api_errors = 0
    if isinstance(resp, dict) and isinstance(resp.get("status_code"), int):
        if resp["status_code"] >= 400:
            api_errors = 1

    api_call_count = len(trace_list) if trace_list else 1

    from utils.eval.eval import _parse_folder_ts
    start_ts = _parse_folder_ts(first_ts)
    end_ts = _parse_folder_ts(stem)
    duration_s = None
    if start_ts and end_ts and end_ts >= start_ts:
        duration_s = (end_ts - start_ts).total_seconds()

    # models 列表优先取传入值，退回 merged 的单值
    models_list = list(models) if models else ([merged.get("model")] if merged.get("model") else [])

    return {
        "session": first_ts,
        "start_time": start_ts.strftime("%Y-%m-%d %H:%M:%S") if start_ts else None,
        "end_time": end_ts.strftime("%Y-%m-%d %H:%M:%S") if end_ts else None,
        "duration_s": duration_s,
        "api_call_count": api_call_count,
        "api_errors": api_errors,
        "user_turns": analyzed["user_turns"],
        "total_messages": analyzed["total_messages"],
        "tool_use_count": analyzed["tool_use_count"],
        "tool_result_count": analyzed["tool_result_count"],
        "tool_success": analyzed["tool_success"],
        "tool_fail_flag": analyzed["tool_fail_flag"],
        "tool_fail_keyword": analyzed["tool_fail_keyword"],
        "tool_fail_total": analyzed["tool_fail_total"],
        "tool_success_rate": analyzed["tool_success_rate"],
        "model": merged.get("model", ""),
        "q1": analyzed["q1"],
        "latest_file": latest_file,
        "log_dir": Path(src_dir_s).name,
        "tool_use_detail": analyzed["tool_use_detail"],
        "tool_success_detail": analyzed["tool_success_detail"],
        "tool_fail_detail": analyzed["tool_fail_detail"],
        "skills_used": analyzed["skills_used"],
        # 统一 schema 导出字段 + 兼容别名
        "api_key": api_key,
        "models": models_list,
        "trace_list": trace_list or [],
        "_key": first_ts,
        "first_ts": first_ts,
        "last_ts": stem,
        "msg_count": analyzed["total_messages"],
        **dict(zip(("completed", "completed_note"), fmt_quality(analyzed["quality_errors"]))),
    }


def _copy_session_to_outdir(args: tuple) -> None:
    """缓存命中时，仍需将合并后的 session JSON 写到 out_dir。"""
    src_dir_s, out_dir_s, first_ts, latest_file, trace_list, api_key, models = args
    src_dir = Path(src_dir_s)
    out_dir = Path(out_dir_s)

    stem = latest_file
    if stem.endswith("-req.json"):
        stem = stem[:-len("-req.json")]
    elif stem.endswith(".json"):
        stem = stem[:-len(".json")]

    tri = _resolve_triplet(src_dir, stem)
    merged = _load_merged(src_dir, stem, latest_file, tri)
    if merged is None:
        return

    session_dir = out_dir / first_ts
    session_dir.mkdir(parents=True, exist_ok=True)
    out_file = session_dir / f"{stem}.json"
    with open(out_file, "w", encoding="utf-8") as fh:
        json.dump(merged, fh, ensure_ascii=False, separators=(",", ":"))


def _rebuild_from_cache(entry: dict) -> Optional[dict]:
    ev = entry.get("_eval")
    if not ev or not isinstance(ev, dict):
        return None
    first_ts = entry.get("_key") or entry.get("first_ts", "")
    if not first_ts:
        return None
    from utils.eval.eval import _parse_folder_ts
    start_ts = _parse_folder_ts(first_ts)
    models = entry.get("models", []) or []
    total_messages = ev.get("total_messages", entry.get("msg_count", 0))
    return {
        "session": first_ts,
        "latest_file": entry.get("latest_file", ""),
        "start_time": start_ts.strftime("%Y-%m-%d %H:%M:%S") if start_ts else None,
        "end_time": ev.get("end_time"),
        "duration_s": ev.get("duration_s"),
        "api_call_count": ev.get("api_call_count", len(entry.get("trace_list", []) or []) or 1),
        "api_errors": ev.get("api_errors", 0),
        "user_turns": ev.get("user_turns", 0),
        "total_messages": total_messages,
        "tool_use_count": ev.get("tool_use_count", 0),
        "tool_result_count": ev.get("tool_result_count", 0),
        "tool_success": ev.get("tool_success", 0),
        "tool_fail_flag": ev.get("tool_fail_flag", 0),
        "tool_fail_keyword": ev.get("tool_fail_keyword", 0),
        "tool_fail_total": ev.get("tool_fail_total", 0),
        "tool_success_rate": ev.get("tool_success_rate"),
        "model": ev.get("model", models[0] if models else ""),
        "q1": ev.get("q1", entry.get("q1", "")),
        "tool_use_detail": ev.get("tool_use_detail", {}),
        "tool_success_detail": ev.get("tool_success_detail", {}),
        "tool_fail_detail": ev.get("tool_fail_detail", {}),
        "skills_used": ev.get("skills_used", {}),
        "completed": ev.get("completed", 0),
        "completed_note": ev.get("completed_note", ""),
        # 统一 schema 导出字段 + 兼容别名
        "api_key": entry.get("api_key", ""),
        "models": models,
        "trace_list": entry.get("trace_list", []) or [],
        "_key": first_ts,
        "first_ts": first_ts,
        "last_ts": entry.get("last_ts", ""),
        "msg_count": total_messages,
    }


def write_eval_to_cache(logs_dir: str, results: List[dict]) -> None:
    cache_path = Path(logs_dir) / ".session_cache.jsonl"
    if not cache_path.is_file() or not results:
        return
    result_map = {}
    for r in results:
        sid = r.get("session", "")
        if sid:
            result_map[sid] = r
    if not result_map:
        return

    lines = []
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return

    from datetime import datetime as _dt
    now_str = _dt.now().strftime("%Y-%m-%dT%H:%M:%S")
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            new_lines.append(line)
            continue
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            new_lines.append(line)
            continue
        key = obj.get("_key", "")
        if key and key in result_map:
            r = result_map[key]
            obj["_eval"] = {
                "ts": now_str,
                "completed": r.get("completed", 0),
                "completed_note": r.get("completed_note", ""),
                "tool_use_count": r.get("tool_use_count", 0),
                "tool_result_count": r.get("tool_result_count", 0),
                "tool_success": r.get("tool_success", 0),
                "tool_fail_flag": r.get("tool_fail_flag", 0),
                "tool_fail_keyword": r.get("tool_fail_keyword", 0),
                "tool_fail_total": r.get("tool_fail_total", 0),
                "tool_success_rate": r.get("tool_success_rate"),
                "user_turns": r.get("user_turns", 0),
                "total_messages": r.get("total_messages", 0),
                "duration_s": r.get("duration_s"),
                "end_time": r.get("end_time"),
                "api_call_count": r.get("api_call_count", 0),
                "api_errors": r.get("api_errors", 0),
                "model": r.get("model", ""),
                "q1": r.get("q1", ""),
                "tool_use_detail": r.get("tool_use_detail", {}),
                "tool_success_detail": r.get("tool_success_detail", {}),
                "tool_fail_detail": r.get("tool_fail_detail", {}),
                "skills_used": r.get("skills_used", {}),
            }
            new_lines.append(json.dumps(obj, ensure_ascii=False) + "\n")
        else:
            new_lines.append(line)

    import os
    tmp = str(cache_path) + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
        os.replace(tmp, str(cache_path))
    except OSError:
        pass


def reformat_and_analyze(
    src_dir: str,
    out_dir: str,
    session_entries: List[dict],
    api_key: Optional[str] = None,
    workers: int = 4,
    progress_cb: Optional[Callable[[str], None]] = None,
    log_dir: Optional[str] = None,
) -> dict:
    """
    多进程并行: 对每个 session 的 latest_file 做 reformat + analyze_best_data。

    log_dir: 写入每个 result 的历史目录标识（相对 root 的 posix 路径，
        如 "260728/26072813"）；供报告生成 /history 链接用。缺省时退回
        Path(src_dir).name（仅在单级 native 布局下正确）。

    Returns:
        {"total_sessions": N, "total_files": M, "errors": [...], "results": [analyzed_dict, ...]}
    """
    _log = progress_cb or (lambda msg: None)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 统一的历史目录标识：new-api 布局为 "<day>/<hour>"，native 为 "<hour>"。
    # worker 内只能拿到 Path(src_dir).name（丢日期层），故在此统一覆盖。
    resolved_log_dir = log_dir if log_dir is not None else Path(src_dir).name

    entries = session_entries
    if api_key:
        entries = [s for s in entries if s.get("api_key") == api_key]

    total = len(entries)
    if total == 0:
        return {"total_sessions": 0, "total_files": 0, "errors": [], "results": []}

    tasks = []
    cached_results = []
    copy_tasks = []
    for sess in entries:
        first_ts = sess.get("first_ts") or sess.get("_key", "")
        latest_file = sess.get("latest_file", "")
        trace_list = sess.get("trace_list", [])
        s_api_key = sess.get("api_key", "")
        s_models = sess.get("models", []) or []
        if not first_ts or not latest_file:
            continue
        cached = _rebuild_from_cache(sess)
        if cached:
            cached["log_dir"] = resolved_log_dir
            cached_results.append(cached)
            copy_tasks.append((src_dir, out_dir, first_ts, latest_file, trace_list, s_api_key, s_models))
        else:
            tasks.append((src_dir, out_dir, first_ts, latest_file, trace_list, s_api_key, s_models))

    for ct in copy_tasks:
        _copy_session_to_outdir(ct)

    results = list(cached_results)
    if cached_results:
        _log(f"缓存命中 {len(cached_results)} sessions, 需处理 {len(tasks)}")
    errors = []
    done = 0
    n_tasks = len(tasks)

    if n_tasks == 0:
        return {"total_sessions": total, "total_files": len(results), "errors": errors, "results": results}

    executor = ProcessPoolExecutor(max_workers=workers)
    try:
        futures = {executor.submit(_process_one, t): t for t in tasks}
        pending = set(futures.keys())
        last_progress = time.monotonic()

        # 逐批等待完成；用 wait(timeout) 做无进展看门狗，避免某个 future
        # 永不返回时整个循环无限阻塞（历史卡死正是停在这里之后）。
        from concurrent.futures import wait as _fwait, FIRST_COMPLETED
        while pending:
            completed, pending = _fwait(pending, timeout=30, return_when=FIRST_COMPLETED)
            if not completed:
                # 本轮无任何 future 完成，检查是否超过无进展阈值
                if time.monotonic() - last_progress > _NO_PROGRESS_TIMEOUT:
                    _log(f"reformat+analyze 无进展超时（{_NO_PROGRESS_TIMEOUT}s）: "
                         f"已完成 {done}/{n_tasks}, 待处理 {len(pending)}，强制终止")
                    executor.shutdown(wait=False, cancel_futures=True)
                    _terminate_pool_workers(executor, _log)
                    raise TimeoutError(
                        f"reformat_and_analyze no progress for {_NO_PROGRESS_TIMEOUT}s "
                        f"(done={done}/{n_tasks}, pending={len(pending)})"
                    )
                continue

            last_progress = time.monotonic()
            for future in completed:
                done += 1
                task = futures[future]
                first_ts = task[2]
                try:
                    result = future.result()
                    if result:
                        # worker 内 log_dir=Path(src_dir).name 会丢失日期层，统一覆盖
                        result["log_dir"] = resolved_log_dir
                        results.append(result)
                    else:
                        errors.append(f"处理失败: {first_ts}")
                except Exception as e:
                    errors.append(f"{first_ts}: {e}")

                if done % 10 == 0 or done == n_tasks:
                    _log(f"reformat+analyze: {done}/{n_tasks}, 成功 {len(results)}")
    finally:
        # 显式关闭进程池，并对收尾挂起做宽限超时兜底。
        # 历史卡死表现为：所有 future 已完成，但 with 块隐式 shutdown 永久阻塞。
        _log("reformat+analyze 完成，开始关闭进程池")
        shutdown_done = {"ok": False}

        import threading as _threading

        def _do_shutdown():
            try:
                executor.shutdown(wait=True)
                shutdown_done["ok"] = True
            except Exception:
                pass

        t = _threading.Thread(target=_do_shutdown, daemon=True)
        t.start()
        t.join(timeout=_SHUTDOWN_GRACE)
        if not shutdown_done["ok"]:
            _log(f"进程池关闭超时（{_SHUTDOWN_GRACE}s），强制回收 worker")
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            _terminate_pool_workers(executor, _log)
        else:
            _log("进程池已关闭")

    return {"total_sessions": total, "total_files": len(results), "errors": errors, "results": results}
