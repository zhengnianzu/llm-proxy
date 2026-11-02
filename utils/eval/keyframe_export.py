#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""关键帧导出：按上下文连续性把会话切成若干段，每段落盘代表文件原文。

产物形态：**原始 merged JSON 透传**，不做格式改写。每段一个文件：

    <out>/<first_ts>/<stem>.json         段代表文件的原文（与源文件逐字节一致）
    <out>/<first_ts>/_keyframes.jsonl    每帧的边界与证据，供审计/下游关联

为什么不做格式改写：下游训练方按自己的 harness 约定消费原始请求；把 messages
规范化成统一形态反而会丢失 harness 特有的字段（system block、tools、
context_management 等）。关键帧只负责「挑哪几条」，不负责「改成什么样」。

切分逻辑在 ``utils/eval/keyframes``，harness 差异在 ``utils/eval/harness``。
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from utils.eval.harness import HarnessAdapter, detect_adapter, get_adapter
from utils.eval.keyframes import KeyFrame, TraceNode, build_node, split_keyframes
from utils.eval.runner import run_tasks
from utils.session_store import _fmt_ts

logger = logging.getLogger(__name__)

DEFAULT_HARNESS = "auto"


def _stem_of(filename: str) -> str:
    if filename.endswith("-req.json"):
        return filename[:-len("-req.json")]
    if filename.endswith(".json"):
        return filename[:-len(".json")]
    return filename


def _trace_files(session: dict) -> List[str]:
    """trace_list 里的文件名（去重保序）。空则回退 latest_file。"""
    out: List[str] = []
    seen = set()
    for trace in session.get("trace_list") or []:
        if not isinstance(trace, dict):
            continue
        name = trace.get("filename")
        if isinstance(name, str) and name and name not in seen:
            seen.add(name)
            out.append(name)
    if not out and session.get("latest_file"):
        out = [session["latest_file"]]
    return out


def _resolve_path(src: Path, filename: str, file_dirs: Optional[Dict[str, Path]]) -> Optional[Path]:
    """定位一个 trace 文件。跨小时导出时文件可能不在 src 下。"""
    if file_dirs:
        base = file_dirs.get(filename)
        if base is not None:
            candidate = base / filename
            if candidate.is_file():
                return candidate
    candidate = src / filename
    return candidate if candidate.is_file() else None


def _load_raw(path: Path) -> Optional[dict]:
    """原样读入，不做任何结构改写。"""
    try:
        with open(path, "r", encoding="utf-8-sig") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def analyze_traces(
    src: Path,
    filenames: List[str],
    file_dirs: Optional[Dict[str, Path]] = None,
    harness: str = DEFAULT_HARNESS,
) -> tuple[List[TraceNode], Dict[str, dict], List[str]]:
    """加载 trace 文件并切成关键帧所需的节点视图。**纯计算，不落盘。**

    返回 (nodes, raws, errors)：
      nodes  按 trace 顺序的判定视图
      raws   {filename: 原始 obj}，供调用方决定要不要落盘
      errors 逐文件的解析错误（不中断整体）

    抽出来是为了让「对话浏览器的按需分析」复用同一套加载逻辑——
    那边只要结果，不写文件。
    """
    fixed_adapter: Optional[HarnessAdapter] = get_adapter(harness) if harness != "auto" else None
    nodes: List[TraceNode] = []
    raws: Dict[str, dict] = {}
    errors: List[str] = []

    for index, filename in enumerate(filenames):
        path = _resolve_path(src, filename, file_dirs)
        if path is None:
            errors.append(f"{filename}: missing or damaged file")
            continue
        obj = _load_raw(path)
        if obj is None:
            errors.append(f"{filename}: missing or damaged file")
            continue

        adapter = fixed_adapter or detect_adapter(obj)
        node = build_node(obj, index, filename, adapter)
        if node is None:
            errors.append(f"{filename}: request messages missing")
            continue
        nodes.append(node)
        raws[filename] = obj

    return nodes, raws, errors


def _one_session(task: tuple) -> tuple[Optional[dict], List[str]]:
    """单个会话：加载 → 切关键帧 → 落盘代表文件。"""
    src_s, out_s, sess, log_dir, file_dirs_s, harness = task
    src, out = Path(src_s), Path(out_s)
    first = str(sess.get("first_ts") or sess.get("_key") or "session")

    file_dirs = {k: Path(v) for k, v in file_dirs_s.items() if v} if file_dirs_s else None
    filenames = _trace_files(sess)
    if not filenames:
        return None, [f"{first}: no trace files"]

    nodes, raws, errors = analyze_traces(src, filenames, file_dirs, harness)
    if not nodes:
        return None, errors or [f"{first}: no usable trace"]

    frames = split_keyframes(nodes)

    session_dir = out / _safe_name(first)
    session_dir.mkdir(parents=True, exist_ok=True)

    written: List[str] = []
    for frame in frames:
        obj = raws.get(frame.filename)
        if obj is None:
            continue
        target = session_dir / (_stem_of(frame.filename) + ".json")
        with open(target, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, ensure_ascii=False, separators=(",", ":"))
        written.append(target.name)

    manifest = session_dir / "_keyframes.jsonl"
    with open(manifest, "w", encoding="utf-8") as fh:
        for frame in frames:
            fh.write(json.dumps(frame.as_dict(), ensure_ascii=False) + "\n")

    record = _result_record(sess, src, log_dir, first)
    record["request_files"] = [f"{_safe_name(first)}/{n}" for n in written]
    record["keyframe_count"] = len(frames)
    record["boundary_counts"] = _count_boundaries(frames)
    record["msg_count"] = sum(f.messages for f in frames)
    record["total_messages"] = record["msg_count"]
    record["api_errors"] = len(errors)
    record["completed"] = 1 if not errors else 0
    record["completed_note"] = "; ".join(errors[:3])
    return record, errors


def _count_boundaries(frames: List[KeyFrame]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for frame in frames:
        counts[frame.boundary] = counts.get(frame.boundary, 0) + 1
    return counts


def _safe_name(raw: str) -> str:
    import re
    cleaned = re.sub(r"[^\w.-]+", "_", str(raw)).strip("._")[:96]
    return cleaned or "session"


def _result_record(sess: dict, src: Path, log_dir: str, first: str) -> dict:
    """与其它处理器对齐的键集合，下游无需判 key 存在性。"""
    models = sess.get("models") or []
    traces = sess.get("trace_list") or []
    return {
        "session": first,
        "start_time": _fmt_ts(sess.get("first_ts") or sess.get("_key", "")),
        "end_time": _fmt_ts(sess.get("last_ts", "")),
        "duration_s": None,
        "api_call_count": len(traces) if traces else 1,
        "api_errors": 0,
        "user_turns": 0,
        "total_messages": 0,
        "tool_use_count": 0,
        "tool_result_count": 0,
        "tool_success": 0,
        "tool_fail_flag": 0,
        "tool_fail_keyword": 0,
        "tool_fail_total": 0,
        "tool_success_rate": None,
        "model": models[0] if models else "",
        "q1": "",
        "latest_file": sess.get("latest_file", ""),
        "log_dir": log_dir or src.name,
        "tool_use_detail": {},
        "tool_success_detail": {},
        "tool_fail_detail": {},
        "skills_used": {},
        "api_key": sess.get("api_key", ""),
        "models": models,
        "trace_list": traces,
        "_key": sess.get("first_ts") or sess.get("_key", ""),
        "first_ts": sess.get("first_ts") or sess.get("_key", ""),
        "last_ts": sess.get("last_ts", ""),
        "msg_count": 0,
        "completed": 0,
        "completed_note": "",
        "request_files": [],
        "keyframe_count": 0,
        "boundary_counts": {},
    }


def keyframe_export(
    src_dir: str,
    out_dir: str,
    session_entries: List[dict],
    api_key: Optional[str] = None,
    workers: int = 4,
    progress_cb: Optional[Callable[[str], None]] = None,
    log_dir: Optional[str] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
    on_session_done: Optional[Callable[[], None]] = None,
    file_dirs: Optional[Dict[str, str]] = None,
    profile: Optional[str] = None,
) -> dict:
    """按上下文连续性切关键帧，每帧落盘代表文件的原始 JSON。

    profile: harness 名（auto / claude_code / openclaw / generic）。由
        settings/export_modes.yaml 的 profile 字段传入；缺省 auto 表示逐请求识别。
        非法值直接抛 ValueError —— 配置写错时宁可失败，也不要静默用默认值
        产出一份语义不对的数据。
    """
    harness = (profile or DEFAULT_HARNESS).strip() or DEFAULT_HARNESS
    if harness != "auto":
        get_adapter(harness)          # 校验，非法值抛 ValueError

    _log = progress_cb or (lambda message: None)
    _should_cancel = should_cancel or (lambda: False)

    workers = max(1, min(int(workers), 32))
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    entries = session_entries if not api_key else [
        s for s in session_entries if s.get("api_key") == api_key
    ]

    result = {
        "total_sessions": len(entries),
        "total_files": 0,
        "errors": [],
        "results": [],
        "cancelled": False,
    }
    if not entries:
        return result
    if _should_cancel():
        result["cancelled"] = True
        return result

    tasks = [
        (src_dir, out_dir, sess, log_dir if log_dir is not None else Path(src_dir).name,
         file_dirs, harness)
        for sess in entries
        if sess.get("first_ts") or sess.get("_key")
    ]
    if not tasks:
        return result

    _log(f"进入关键帧导出: {len(tasks)} sessions, harness={harness}, workers={workers}...")

    run = run_tasks(
        tasks,
        _one_session,
        workers=workers,
        label="keyframes",
        progress_cb=_log,
        should_cancel=_should_cancel,
        on_task_done=on_session_done,
    )

    for record, errors in run["payload"]:
        result["errors"].extend(errors)
        if record is not None:
            result["results"].append(record)
            result["total_files"] += len(record["request_files"])

    result["errors"].extend(run["errors"])
    if run["cancelled"]:
        result["cancelled"] = True
    return result
