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
        "tool_use_detail": analyzed["tool_use_detail"],
        "tool_success_detail": analyzed["tool_success_detail"],
        "tool_fail_detail": analyzed["tool_fail_detail"],
        "skills_used": analyzed["skills_used"],
        **dict(zip(("completed", "completed_note"), fmt_quality(analyzed["quality_errors"]))),
    }


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
    for sess in entries:
        first_ts = sess.get("first_ts") or sess.get("_key", "")
        latest_file = sess.get("latest_file", "")
        trace_list = sess.get("trace_list", [])
        if not first_ts or not latest_file:
            continue
        tasks.append((src_dir, out_dir, first_ts, latest_file, trace_list))

    results = []
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
