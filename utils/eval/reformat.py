"""
utils/eval/reformat.py — 格式重整 + 质检一体化

将三元组文件（req/headers/res）合并为单个 JSON，只处理 latest_file。
合并后立即跑 analyze_best_data，返回分析结果，省掉二次读取。
使用多进程并行处理。
"""

import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from utils.eval.eval import analyze_best_data, fmt_quality
from utils.message_common import parse_response

logger = logging.getLogger(__name__)


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


def _process_one(args: tuple) -> Optional[dict]:
    """单个 session 的 reformat + analyze，多进程可调用。

    返回分析结果 dict（含 session/start_time 等字段），失败返回 None。
    """
    src_dir_s, out_dir_s, first_ts, latest_file, trace_list = args

    src_dir = Path(src_dir_s)
    out_dir = Path(out_dir_s)

    stem = latest_file
    if stem.endswith("-req.json"):
        stem = stem[:-len("-req.json")]
    elif stem.endswith(".json"):
        stem = stem[:-len(".json")]

    tri = _resolve_triplet(src_dir, stem)
    if "req" not in tri:
        return None

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
        **dict(zip(("completed", "completed_note"), fmt_quality(analyzed["quality_errors"]))),
    }


def _copy_session_to_outdir(args: tuple) -> None:
    """缓存命中时，仍需将合并后的 session JSON 写到 out_dir。"""
    src_dir_s, out_dir_s, first_ts, latest_file, trace_list = args
    src_dir = Path(src_dir_s)
    out_dir = Path(out_dir_s)

    stem = latest_file
    if stem.endswith("-req.json"):
        stem = stem[:-len("-req.json")]
    elif stem.endswith(".json"):
        stem = stem[:-len(".json")]

    tri = _resolve_triplet(src_dir, stem)
    if "req" not in tri:
        return

    try:
        merged = _load_json(tri["req"])
    except Exception:
        return

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
    models = entry.get("models", [])
    return {
        "session": first_ts,
        "latest_file": entry.get("latest_file", ""),
        "start_time": start_ts.strftime("%Y-%m-%d %H:%M:%S") if start_ts else None,
        "end_time": ev.get("end_time"),
        "duration_s": ev.get("duration_s"),
        "api_call_count": ev.get("api_call_count", len(entry.get("trace_list", []) or []) or 1),
        "api_errors": ev.get("api_errors", 0),
        "user_turns": ev.get("user_turns", 0),
        "total_messages": ev.get("total_messages", entry.get("msg_count", 0)),
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
) -> dict:
    """
    多进程并行: 对每个 session 的 latest_file 做 reformat + analyze_best_data。

    Returns:
        {"total_sessions": N, "total_files": M, "errors": [...], "results": [analyzed_dict, ...]}
    """
    _log = progress_cb or (lambda msg: None)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

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
        if not first_ts or not latest_file:
            continue
        cached = _rebuild_from_cache(sess)
        if cached:
            cached["log_dir"] = Path(src_dir).name
            cached_results.append(cached)
            copy_tasks.append((src_dir, out_dir, first_ts, latest_file, trace_list))
        else:
            tasks.append((src_dir, out_dir, first_ts, latest_file, trace_list))

    for ct in copy_tasks:
        _copy_session_to_outdir(ct)

    results = list(cached_results)
    if cached_results:
        _log(f"缓存命中 {len(cached_results)} sessions, 需处理 {len(tasks)}")
    errors = []
    done = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_process_one, t): t for t in tasks}
        for future in as_completed(futures):
            done += 1
            task = futures[future]
            first_ts = task[2]
            try:
                result = future.result()
                if result:
                    results.append(result)
                else:
                    errors.append(f"处理失败: {first_ts}")
            except Exception as e:
                errors.append(f"{first_ts}: {e}")

            if done % 10 == 0 or done == len(tasks):
                _log(f"reformat+analyze: {done}/{len(tasks)}, 成功 {len(results)}")

    return {"total_sessions": total, "total_files": len(results), "errors": errors, "results": results}
