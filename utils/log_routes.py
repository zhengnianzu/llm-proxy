"""
/logs/* 路由：列表、聚合、单文件读取（Anthropic + OpenAI）
优先使用 index.jsonl 的最近窗口；缺失时降级为目录扫描。
"""

import json
import logging
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse

from utils.log_paths import build_index_path, get_log_dir
from utils.message_common import (
    build_chain_key,
    compute_q1_hash,
    count_real_user_turns,
    extract_res_usage,
    get_first_user_text,
    get_text_from_content,
    load_json_safe,
    parse_openai_streaming_response,
    parse_streaming_response_content,
)
from utils.q1_index import get_effective_q1, should_update_q1, update_q1
import utils.session_store as _ss

_CACHE_LOCK = threading.Lock()
_LOG_CACHE: Dict[Tuple[str, str], Dict[str, Any]] = {}


def _resolve_req_path(root: Path, req_file: str) -> Optional[Path]:
    raw = (req_file or "").strip()
    if not raw:
        return None

    rf = Path(raw)
    candidates = []
    if rf.is_absolute():
        candidates.append(rf.resolve())
    else:
        candidates.append((root.parent / rf).resolve())
        candidates.append((root / rf.name).resolve())
        candidates.append(rf.resolve())

    for path in candidates:
        if path.is_file():
            return path
    return None


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    return load_json_safe(path)


def _format_time(ts: str) -> str:
    return ts.replace("_", " ", 1).replace("_", ".").replace("-", ":") if ts else ""


def _read_new_index_entries(
    index_path: Path, root: Path, byte_offset: int
) -> Tuple[List[Dict[str, Any]], int]:
    """从 byte_offset 处增量读取 index.jsonl 新增行。
    返回 (rows, new_byte_offset)。
    文件被截断时返回 ([], 0) 表示需要全量重建。
    """
    try:
        file_size = index_path.stat().st_size
    except OSError:
        return [], 0

    if byte_offset > file_size:
        return [], 0

    if byte_offset == file_size:
        return [], byte_offset

    rows: List[Dict[str, Any]] = []
    try:
        with index_path.open("rb") as f:
            f.seek(byte_offset)
            raw_data = f.read()
            new_offset = f.tell()
    except OSError:
        return [], byte_offset

    for raw_line in raw_data.split(b"\n"):
        line = raw_line.decode("utf-8", errors="ignore").strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        req_path = _resolve_req_path(root, str(entry.get("req_file", "")))
        if req_path is None:
            continue
        rows.append({"entry": entry, "req_path": req_path})

    return rows, new_offset


def _collect_req_files(root: Path) -> List[Path]:
    return sorted(root.glob("*-req.json"))


def _extract_anthropic_res_content(res_path: Path):
    data = _load_json(res_path)
    if not data:
        return None

    rtype = data.get("type")
    if rtype == "anthropic_passthrough_sse_capture":
        chunks = data.get("chunks", [])
        return parse_streaming_response_content(
            [c for c in chunks if c.get("type") != "anthropic_passthrough_sse_meta"]
        )

    if isinstance(data.get("json"), dict):
        msg = data["json"]
        if msg.get("content") is not None:
            return msg["content"]
    return None


def _extract_openai_res_content(res_path: Path):
    data = _load_json(res_path)
    if not data:
        return None

    if data.get("type") == "openai_passthrough_sse_capture":
        chunks = data.get("chunks", [])
        if not chunks:
            return None
        msg = parse_openai_streaming_response(chunks)
        if msg.get("role") == "assistant":
            return msg
        return None

    msg = data.get("json", {}).get("choices", [{}])[0].get("message")
    if isinstance(msg, dict) and msg.get("role") == "assistant":
        return msg
    return None


def _get_text_from_content(content) -> str:
    return get_text_from_content(content)


def _build_state(root_dir: str) -> Dict[str, Any]:
    return {
        "root_dir": root_dir,
        "index_path": build_index_path(root_dir),
        "initialized": False,
        "line_count": 0,
        "byte_offset": 0,
        "known_keys": set(),
        "_chain_map": {},  # lookup_key -> session_key，启动时从 DB 重建
        "_last_refresh_ts": 0.0,
    }


def _load_state_from_db(state: Dict[str, Any]) -> bool:
    """从 DB 恢复内存 chain_map 和 index 进度。"""
    root_dir = state["root_dir"]
    try:
        byte_offset, line_count = _ss.get_progress(root_dir)
        state["byte_offset"] = byte_offset
        state["line_count"] = line_count

        chain_map = _ss.get_all_chain_index(root_dir)
        state["_chain_map"] = chain_map

        # known_keys 从 sessions 表重建
        sessions = _ss.list_sessions(root_dir)
        state["known_keys"] = {s.get("api_key", "") for s in sessions if s.get("api_key")}

        return bool(byte_offset > 0 or chain_map)
    except Exception:
        return False


def _state_key(kind: str, root_dir: str) -> Tuple[str, str]:
    return kind, os.path.normpath(root_dir)


def _state(kind: str, root_dir: str) -> Dict[str, Any]:
    key = _state_key(kind, root_dir)
    state = _LOG_CACHE.get(key)
    if state is None:
        state = _build_state(root_dir)
        _LOG_CACHE[key] = state
    return state


def _process_req_row(kind: str, state: Dict[str, Any], req_path: Path, index_entry: Optional[Dict[str, Any]] = None) -> bool:
    _noise_prefixes = ("(session bootstrap)",)
    ie = index_entry or {}

    # 快速路径：index_entry 已含 msg_count/user_turns/chain_key，跳过读 req.json
    _idx_msg_count = ie.get("msg_count", 0)
    _idx_user_turns = ie.get("user_turns")
    _idx_chain_key = ie.get("chain_key", "")
    _idx_q1_hash = ie.get("q1_hash", "")
    _needs_req = (
        not _idx_msg_count  # 旧格式 index 没有 msg_count
        or not _idx_chain_key
        or not _idx_q1_hash  # 旧格式 index 没有 q1_hash，需读 req.json 现算
        or (_idx_chain_key and any(_idx_chain_key.startswith(p) for p in _noise_prefixes))
        or (_idx_user_turns is None)
    )

    messages = None
    data = None
    if _needs_req:
        data = _load_json(req_path)
        if not data:
            return False
        messages = data.get("messages")
        if not isinstance(messages, list):
            return False

    root_dir = state["root_dir"]
    filename = req_path.name
    ts = str(ie.get("ts") or filename.replace("-req.json", ""))
    model = str((data.get("model", "") if data else "") or ie.get("model", "") or "")
    api_key = str(ie.get("api_key", "") or "")

    if _idx_chain_key and not any(_idx_chain_key.startswith(p) for p in _noise_prefixes):
        chain_key = _idx_chain_key
    else:
        chain_key = build_chain_key(messages)

    # 聚合键：优先用 index 里的 q1_hash（完整 Q1 的 md5）；旧 index 无此字段时
    # 从 messages 现算，保证不误聚合。messages 在 _needs_req 分支已加载。
    if _idx_q1_hash:
        q1_hash = _idx_q1_hash
    else:
        q1_hash = compute_q1_hash(messages) if messages is not None else ""

    lookup_key = f"{api_key}||{q1_hash}"
    q1_preview = ie.get("q1_preview", "")
    if q1_preview and any(q1_preview.startswith(p) for p in _noise_prefixes):
        q1_preview = get_first_user_text(messages)[:200] if messages is not None else ""

    if _idx_msg_count and not _needs_req:
        message_count = _idx_msg_count
        res_path = req_path.with_name(filename.replace("-req.json", "-res.json"))
        has_res = res_path.is_file()
        full_message_count = message_count + (1 if has_res else 0)
    else:
        message_count = len(messages)
        res_path = req_path.with_name(filename.replace("-req.json", "-res.json"))
        has_res = res_path.is_file()
        full_message_count = message_count + (1 if has_res else 0)

    trace_entry: Dict[str, Any] = {
        "filename": filename,
        "model": model,
        "msg_count": full_message_count,
        "ts": ts,
    }
    if index_entry:
        if not ie.get("success", True):
            trace_entry["success"] = False
            trace_entry["total_attempts"] = ie.get("total_attempts", 1)
        if ie.get("debug_file"):
            trace_entry["debug_file"] = ie["debug_file"]

    if _idx_user_turns is not None and not _needs_req:
        real_user_turns = _idx_user_turns
    else:
        real_user_turns = count_real_user_turns(messages)

    # 用 _CACHE_LOCK 短暂读取内存 chain_map（微秒级）
    with _CACHE_LOCK:
        state["known_keys"].add(api_key)
        session_key = state["_chain_map"].get(lookup_key)

    # DB 读在锁外（不阻塞其他请求）
    session: Optional[Dict[str, Any]] = None
    if session_key:
        session = _ss.get_session(session_key)

    if session is not None and real_user_turns <= 1 and \
       real_user_turns < session.get("_max_real_turns", 1):
        # 新会话检测：需要找一个未被占用的 suffix key
        with _CACHE_LOCK:
            suffix = 1
            new_lookup = f"{lookup_key}##session_{suffix}"
            while new_lookup in state["_chain_map"]:
                suffix += 1
                new_lookup = f"{lookup_key}##session_{suffix}"
        lookup_key = new_lookup
        session = None
        session_key = None

    if session is None:
        session_key = ts
        new_session = {
            "q1": q1_preview or chain_key[:200],
            "models": [model] if model else [],
            "latest_file": filename,
            "msg_count": full_message_count,
            "api_key": api_key,
            "first_ts": ts,
            "last_ts": ts,
            "_best_req_count": message_count,
            "_max_real_turns": real_user_turns,
        }
        # DB 写在锁外
        _ss.create_session(root_dir, session_key, new_session)
        _ss.append_trace(root_dir, session_key, trace_entry)
        _ss.set_chain_index(root_dir, lookup_key, session_key)
        # 内存 chain_map 更新需要锁
        with _CACHE_LOCK:
            state["_chain_map"][lookup_key] = session_key
    else:
        models = session.get("models", [])
        if model and model not in models:
            models = models + [model]

        max_real_turns = max(real_user_turns, session.get("_max_real_turns", 0))
        best_req_count = session.get("_best_req_count", 0)
        latest_file = session.get("latest_file", "")
        msg_count = session.get("msg_count", 0)

        if message_count > best_req_count or \
           (message_count == best_req_count and has_res):
            latest_file = filename
            msg_count = full_message_count
            best_req_count = message_count

        updated = {
            "last_ts": ts,
            "models": models,
            "latest_file": latest_file,
            "msg_count": msg_count,
            "_max_real_turns": max_real_turns,
            "_best_req_count": best_req_count,
        }
        # DB 写在锁外
        _ss.update_session(session_key, updated)
        _ss.append_trace(root_dir, session_key, trace_entry)

    return True

    return True


_REFRESH_TTL = 10  # 秒：已初始化的目录在此时间内跳过重复刷新
_INIT_LOCK: Dict[str, threading.Lock] = {}  # root_dir -> 首次初始化锁
_INIT_LOCK_GUARD = threading.Lock()


def _get_init_lock(root_dir: str) -> threading.Lock:
    with _INIT_LOCK_GUARD:
        if root_dir not in _INIT_LOCK:
            _INIT_LOCK[root_dir] = threading.Lock()
        return _INIT_LOCK[root_dir]


def _refresh_state(kind: str, root_dir: str, force: bool = False) -> None:
    # 快速路径：TTL 内直接返回，只读内存变量，不需要全局锁
    with _CACHE_LOCK:
        state = _state(kind, root_dir)
        if not force and state["initialized"] and time.time() - state["_last_refresh_ts"] < _REFRESH_TTL:
            return
        already_initialized = state["initialized"]

    root = Path(root_dir)
    index_path = Path(state["index_path"])

    # Phase 1：首次初始化 —— 用 per-root 锁防止并发重复初始化，主锁不参与
    if not already_initialized:
        init_lock = _get_init_lock(root_dir)
        with init_lock:
            # 二次检查，防止排队的第二个请求重复初始化
            with _CACHE_LOCK:
                if state["initialized"]:
                    return
            if not _load_state_from_db(state):
                state["_chain_map"].clear()
                state["known_keys"].clear()
                state["byte_offset"] = 0
                state["line_count"] = 0

    # Phase 2：锁外读 index.jsonl 新增内容（纯文件 I/O，不竞争）
    with _CACHE_LOCK:
        current_offset = state["byte_offset"]

    if index_path.is_file():
        rows, new_offset = _read_new_index_entries(index_path, root, current_offset)

        if new_offset == 0 and current_offset > 0:
            # 文件被截断/轮转 — 全量重建，需要锁保护内存清理
            with _CACHE_LOCK:
                state["_chain_map"].clear()
                state["known_keys"].clear()
                state["byte_offset"] = 0
                state["line_count"] = 0
            rows, new_offset = _read_new_index_entries(index_path, root, 0)

        if rows:
            # 锁外处理每行：读 req.json（文件 I/O）+ 写 DB
            # chain_map 读写在 _process_req_row 内部用 _CACHE_LOCK 短暂保护
            for row in rows:
                _process_req_row(kind, state, row["req_path"], row["entry"])
            with _CACHE_LOCK:
                state["line_count"] += len(rows)
                state["byte_offset"] = new_offset
            _ss.set_progress(root_dir, new_offset, state["line_count"])
        elif new_offset != current_offset:
            with _CACHE_LOCK:
                state["byte_offset"] = new_offset
            _ss.set_progress(root_dir, new_offset, state["line_count"])
    else:
        if not already_initialized:
            for req_path in _collect_req_files(root):
                _process_req_row(kind, state, req_path)

    with _CACHE_LOCK:
        state["initialized"] = True
        state["_last_refresh_ts"] = time.time()


def _list_payload(kind: str, root_dir: str, min_messages: int, offset: int = 0, limit: int = 50, api_key: str = "", refresh: bool = False, model: str = "") -> Dict[str, Any]:
    _refresh_state(kind, root_dir, force=refresh)

    sessions = _ss.list_sessions(root_dir, api_key=api_key, model=model, min_msg_count=min_messages)
    session_keys = [s["session_key"] for s in sessions]
    traces_by_key = _ss.get_traces_batch(session_keys)

    known_models = _ss.get_known_models(root_dir)
    items = []
    for session in sessions:
        sk = session["session_key"]
        for trace in traces_by_key.get(sk, []):
            if trace.get("msg_count", 0) >= min_messages:
                if model and trace.get("model", "") != model:
                    continue
                item: Dict[str, Any] = {
                    "filename": trace["filename"],
                    "message_count": trace["msg_count"],
                    "model": trace.get("model", ""),
                    "api_key": session.get("api_key", ""),
                }
                if not trace.get("success", True):
                    item["success"] = False
                    item["total_attempts"] = trace.get("total_attempts", 1)
                if trace.get("debug_file"):
                    item["debug_file"] = trace["debug_file"]
                items.append(item)

    items.sort(key=lambda x: x["filename"], reverse=True)
    total = len(items)
    paged = items[offset:offset + limit] if limit > 0 else items[offset:]

    with _CACHE_LOCK:
        current_state = _state(kind, root_dir)
        known_keys = sorted(current_state["known_keys"])
        last_ts = current_state.get("_last_refresh_ts", 0)

    return {"items": paged, "total": total, "known_keys": known_keys, "known_models": known_models, "last_refresh_ts": last_ts}


def _content_contains_keyword(content, kw: str) -> bool:
    if isinstance(content, str):
        return kw in content.lower()
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                text = block.get("text", "")
                if isinstance(text, str) and kw in text.lower():
                    return True
    return False


def _match_messages_content(root_dir: str, filename: str, keyword: str) -> bool:
    req_path = Path(root_dir) / filename
    if not req_path.is_file():
        return False
    try:
        kw = keyword.lower()
        with open(req_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        messages = data.get("messages")
        if isinstance(messages, list):
            for msg in messages:
                if _content_contains_keyword(msg.get("content", ""), kw):
                    return True
        res_path = req_path.with_name(filename.replace("-req.json", "-res.json"))
        res_content = _extract_anthropic_res_content(res_path)
        if res_content is not None and _content_contains_keyword(res_content, kw):
            return True
        if res_content is None:
            openai_content = _extract_openai_res_content(res_path)
            if isinstance(openai_content, dict):
                if _content_contains_keyword(openai_content.get("content", ""), kw):
                    return True
        return False
    except Exception:
        return False


def _aggregate_payload(kind: str, root_dir: str, min_messages: int, offset: int = 0, limit: int = 50, api_key: str = "", refresh: bool = False, model: str = "", search: str = "") -> Dict[str, Any]:
    _refresh_state(kind, root_dir, force=refresh)

    search = (search or "").strip()

    if search:
        # search 需要全量加载再过滤（无法下推 SQL），分页在过滤后处理
        sessions = _ss.list_sessions(root_dir, api_key=api_key, model=model, min_msg_count=min_messages)
        sessions = [s for s in sessions if _match_messages_content(root_dir, s.get("latest_file", ""), search)]
        total = len(sessions)
        paged = sessions[offset:offset + limit] if limit > 0 else sessions[offset:]
    else:
        # 无 search：COUNT + 分页完全下推 SQL
        total = _ss.count_sessions(root_dir, api_key=api_key, model=model, min_msg_count=min_messages)
        paged = _ss.list_sessions(root_dir, api_key=api_key, model=model, min_msg_count=min_messages, offset=offset, limit=limit if limit > 0 else 0)

    known_models = _ss.get_known_models(root_dir)
    paged_keys = [s["session_key"] for s in paged]
    traces_by_key = _ss.get_traces_batch(paged_keys)

    items = []
    for session in paged:
        sk = session["session_key"]
        trace_list = traces_by_key.get(sk, [])
        payload: Dict[str, Any] = {
            "first_time": _format_time(session["first_ts"]),
            "last_time": _format_time(session["last_ts"]),
            "file_count": len(trace_list),
            "message_count": session.get("msg_count", 0),
            "models": session.get("models", []),
            "latest_file": session.get("latest_file", ""),
            "api_key": session.get("api_key", ""),
            "q1_preview": session.get("q1", ""),
            "trace_list": trace_list,
        }
        if any(not t.get("success", True) for t in trace_list):
            payload["has_failure"] = True
        items.append(payload)

    with _CACHE_LOCK:
        current_state = _state(kind, root_dir)
        known_keys = sorted(current_state["known_keys"])
        last_refresh_ts = current_state.get("_last_refresh_ts", 0)

    return {"items": items, "total": total, "known_keys": known_keys, "known_models": known_models, "last_refresh_ts": last_refresh_ts}


def _aggregate_all_payload(kind: str, env_dir: str, min_messages: int, offset: int = 0, limit: int = 50, api_key: str = "", refresh: bool = False, model: str = "", search: str = "") -> Dict[str, Any]:
    env_path = Path(env_dir)
    if not env_path.is_dir():
        return {"items": [], "total": 0, "known_keys": [], "known_models": []}

    sub_dirs = sorted(
        [d for d in env_path.iterdir() if d.is_dir()],
        key=lambda d: d.name, reverse=True,
    )

    all_sessions = []
    all_known_keys: set = set()
    all_known_models: set = set()

    for sub in sub_dirs:
        root_dir = str(sub)
        _refresh_state(kind, root_dir, force=refresh)

        sessions = _ss.list_sessions(root_dir, api_key=api_key, model=model, min_msg_count=min_messages)
        for session in sessions:
            for m in session.get("models", []):
                if m:
                    all_known_models.add(m)
            all_known_keys.add(session.get("api_key", ""))
            all_sessions.append((session, root_dir))

    if search and search.strip():
        search = search.strip()
        all_sessions = [(s, rd) for s, rd in all_sessions if _match_messages_content(rd, s.get("latest_file", ""), search)]

    all_sessions.sort(key=lambda t: t[0].get("last_ts", ""), reverse=True)
    total = len(all_sessions)
    paged = all_sessions[offset:offset + limit] if limit > 0 else all_sessions[offset:]

    # 按 root_dir 分组批量获取 traces
    from collections import defaultdict
    by_root: Dict[str, List[str]] = defaultdict(list)
    for session, rd in paged:
        by_root[rd].append(session["session_key"])

    traces_map: Dict[str, Dict[str, List]] = {}
    for rd, keys in by_root.items():
        traces_map[rd] = _ss.get_traces_batch(keys)

    items = []
    for session, rd in paged:
        sk = session["session_key"]
        trace_list = traces_map.get(rd, {}).get(sk, [])
        models = session.get("models", [])
        payload: Dict[str, Any] = {
            "first_time": _format_time(session["first_ts"]),
            "last_time": _format_time(session["last_ts"]),
            "file_count": len(trace_list),
            "message_count": session.get("msg_count", 0),
            "models": models,
            "latest_file": session.get("latest_file", ""),
            "api_key": session.get("api_key", ""),
            "q1_preview": session.get("q1", ""),
            "trace_list": trace_list,
        }
        if any(not t.get("success", True) for t in trace_list):
            payload["has_failure"] = True
        items.append(payload)

    return {"items": items, "total": total, "known_keys": sorted(all_known_keys), "known_models": sorted(all_known_models)}


def _find_file_in_all_dirs(env_dir: str, filename: str) -> Optional[str]:
    env_path = Path(env_dir)
    if not env_path.is_dir():
        return None
    for sub in sorted(env_path.iterdir(), key=lambda d: d.name, reverse=True):
        if sub.is_dir():
            candidate = sub / filename
            if candidate.is_file():
                return str(candidate)
    return None


# ---------------------------------------------------------------------------
# 后台预热线程 — 进程启动后自动预热所有 mtime 目录的 session 缓存
# ---------------------------------------------------------------------------

_warmer_thread = None
_warmer_stop = threading.Event()

_logger = logging.getLogger("session-warmer")


def _warmer_loop(env_dir: str, current_log_dir: str, interval: float = 30.0) -> None:
    env_path = Path(env_dir)
    if env_path.is_dir():
        subs = sorted(env_path.iterdir())
        _logger.info("session-warmer: 开始预热 %d 个目录", len(subs))
        for sub in subs:
            if _warmer_stop.is_set():
                return
            if not sub.is_dir():
                continue
            with _CACHE_LOCK:
                _refresh_state("anthropic", str(sub), force=True)
        _logger.info("session-warmer: 预热完成")

    while not _warmer_stop.is_set():
        _warmer_stop.wait(interval)
        if _warmer_stop.is_set():
            break
        with _CACHE_LOCK:
            _refresh_state("anthropic", current_log_dir, force=True)


def start_session_cache_warmer(env_dir: str, current_log_dir: str) -> None:
    global _warmer_thread
    if _warmer_thread is not None and _warmer_thread.is_alive():
        return
    _warmer_stop.clear()
    _warmer_thread = threading.Thread(
        target=_warmer_loop,
        args=(env_dir, current_log_dir),
        daemon=True,
        name="session-cache-warmer",
    )
    _warmer_thread.start()


def register_log_routes(app: FastAPI) -> None:
    _current_log_dir = get_log_dir("logs_all")
    _env_dir = str(Path(_current_log_dir).parent)

    def unified_log_dir() -> str:
        return _current_log_dir

    def resolve_log_dir(log_dir: str = "") -> str:
        if not log_dir:
            return _current_log_dir
        # 只允许选择同 env-key 下的时间戳子目录
        candidate = os.path.join(_env_dir, log_dir)
        if os.path.isdir(candidate):
            return candidate
        return _current_log_dir

    def anthropic_log_dir() -> str:
        return get_log_dir("logs_anthropic")

    def openai_log_dir() -> str:
        return get_log_dir("logs_openai")

    @app.get("/logs/dirs")
    def logs_dirs():
        """列出 env-key 目录下所有时间戳子目录，当前目录排在第一个。"""
        from utils.stats_index import refresh_index, get_dir_counts
        from utils.token_index import refresh_token_index
        current_tag = Path(_current_log_dir).name
        dirs = []
        total_count = 0
        env_path = Path(_env_dir)
        if env_path.is_dir():
            index = refresh_index(env_path)
            tok_index = refresh_token_index(str(env_path))
            counts = get_dir_counts(index, tok_index)
            for sub in sorted(env_path.iterdir(), reverse=True):
                if sub.is_dir():
                    count = counts.get(sub.name, 0)
                    total_count += count
                    dirs.append({
                        "name": sub.name,
                        "current": sub.name == current_tag,
                        "count": count,
                    })
        # __ALL__ 已移除：文件数过多时全目录聚合会导致超时
        return JSONResponse({"dirs": dirs, "current": current_tag})

    # --- 统一路由（新） ---

    @app.get("/logs/list")
    def logs_list(min_messages: int = 10, offset: int = 0, limit: int = 50, api_key: str = "", refresh: bool = False, model: str = "", log_dir: str = ""):
        return JSONResponse(_list_payload("anthropic", resolve_log_dir(log_dir), min_messages, offset, limit, api_key, refresh, model))

    @app.get("/logs/file")
    def logs_file(filename: str, log_dir: str = ""):
        if not filename.endswith("-req.json") or "/" in filename or "\\" in filename or ".." in filename:
            return JSONResponse({"error": "invalid filename"}, status_code=400)
        if log_dir == "__ALL__":
            found = _find_file_in_all_dirs(_env_dir, filename)
            if not found:
                return JSONResponse({"error": "file not found"}, status_code=404)
            path = found
        else:
            target_dir = resolve_log_dir(log_dir)
            path = os.path.join(target_dir, filename)
            if not os.path.isfile(path):
                # fallback: 在当前目录和 legacy 目录中搜索
                for search_dir in [unified_log_dir(), anthropic_log_dir(), openai_log_dir()]:
                    alt = os.path.join(search_dir, filename)
                    if os.path.isfile(alt):
                        path = alt
                        break
                else:
                    return JSONResponse({"error": "file not found"}, status_code=404)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        res_path = Path(path).with_name(filename.replace("-req.json", "-res.json"))
        res_content = _extract_anthropic_res_content(res_path)
        if res_content is not None and isinstance(data.get("messages"), list):
            data["messages"].append({"role": "assistant", "content": res_content, "_from_res": True})
        elif isinstance(data.get("messages"), list):
            openai_content = _extract_openai_res_content(res_path)
            if openai_content is not None:
                data["messages"].append({**openai_content, "_from_res": True})
        res_data = _load_json(res_path)
        usage = extract_res_usage(res_data) if res_data else None
        if usage:
            data["_usage"] = usage
        return JSONResponse(data)

    @app.get("/logs/file/download")
    def logs_file_download(filename: str, log_dir: str = ""):
        if not filename.endswith("-req.json") or "/" in filename or "\\" in filename or ".." in filename:
            return JSONResponse({"error": "invalid filename"}, status_code=400)
        if log_dir == "__ALL__":
            found = _find_file_in_all_dirs(_env_dir, filename)
            if not found:
                return JSONResponse({"error": "file not found"}, status_code=404)
            path = found
        else:
            target_dir = resolve_log_dir(log_dir)
            path = os.path.join(target_dir, filename)
            if not os.path.isfile(path):
                for search_dir in [unified_log_dir(), anthropic_log_dir(), openai_log_dir()]:
                    alt = os.path.join(search_dir, filename)
                    if os.path.isfile(alt):
                        path = alt
                        break
                else:
                    return JSONResponse({"error": "file not found"}, status_code=404)
        return FileResponse(path, filename=filename, media_type="application/json")

    _RAW_SUFFIXES = ("-req.json", "-headers.json", "-res.json")

    @app.get("/logs/file/raw")
    def logs_file_raw(filename: str, log_dir: str = ""):
        """返回原始 JSON 文件内容（支持 req / headers / res 三种后缀）。"""
        if not any(filename.endswith(s) for s in _RAW_SUFFIXES):
            return JSONResponse({"error": "invalid filename suffix"}, status_code=400)
        if "/" in filename or "\\" in filename or ".." in filename:
            return JSONResponse({"error": "invalid filename"}, status_code=400)
        if log_dir == "__ALL__":
            req_name = filename
            for s in _RAW_SUFFIXES:
                if filename.endswith(s):
                    req_name = filename[:-len(s)] + "-req.json"
                    break
            found = _find_file_in_all_dirs(_env_dir, req_name)
            if found:
                path = os.path.join(os.path.dirname(found), filename)
            else:
                return JSONResponse({"error": "file not found"}, status_code=404)
        else:
            target_dir = resolve_log_dir(log_dir)
            path = os.path.join(target_dir, filename)
            if not os.path.isfile(path):
                for search_dir in [unified_log_dir(), anthropic_log_dir(), openai_log_dir()]:
                    alt = os.path.join(search_dir, filename)
                    if os.path.isfile(alt):
                        path = alt
                        break
        if not os.path.isfile(path):
            return JSONResponse({"error": "file not found"}, status_code=404)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return JSONResponse(data)

    @app.get("/logs/aggregate")
    def logs_aggregate(min_messages: int = 1, offset: int = 0, limit: int = 50, api_key: str = "", refresh: bool = False, model: str = "", log_dir: str = "", search: str = ""):
        return JSONResponse(_aggregate_payload("anthropic", resolve_log_dir(log_dir), min_messages, offset, limit, api_key, refresh, model, search))

    # --- 旧路由（别名，向后兼容） ---

    @app.get("/logs/anthropic/list")
    def logs_anthropic_list(min_messages: int = 10, offset: int = 0, limit: int = 50, api_key: str = "", refresh: bool = False, model: str = ""):
        return JSONResponse(_list_payload("anthropic", unified_log_dir(), min_messages, offset, limit, api_key, refresh, model))

    @app.get("/logs/anthropic/file")
    def logs_anthropic_file(filename: str):
        return logs_file(filename)

    @app.get("/logs/anthropic/aggregate")
    def logs_anthropic_aggregate(min_messages: int = 1, offset: int = 0, limit: int = 50, api_key: str = "", refresh: bool = False, model: str = "", search: str = ""):
        return JSONResponse(_aggregate_payload("anthropic", unified_log_dir(), min_messages, offset, limit, api_key, refresh, model, search))

    @app.get("/logs/openai/list")
    def logs_openai_list(min_messages: int = 10, offset: int = 0, limit: int = 50, api_key: str = "", refresh: bool = False, model: str = ""):
        return JSONResponse(_list_payload("anthropic", unified_log_dir(), min_messages, offset, limit, api_key, refresh, model))

    @app.get("/logs/openai/file")
    def logs_openai_file(filename: str):
        return logs_file(filename)

    @app.get("/logs/openai/aggregate")
    def logs_openai_aggregate(min_messages: int = 1, offset: int = 0, limit: int = 50, api_key: str = "", refresh: bool = False, model: str = "", search: str = ""):
        return JSONResponse(_aggregate_payload("anthropic", unified_log_dir(), min_messages, offset, limit, api_key, refresh, model, search))

    # --- shared 公开路由（key+code 验证） ---

    def _check_shared(key: str, code: str) -> Optional[JSONResponse]:
        import hmac as _hmac
        from utils.key_store import find_key
        expected = os.getenv("SHARED_CODE", "shared")
        if not _hmac.compare_digest(code, expected):
            return JSONResponse({"detail": "Invalid code"}, status_code=403)
        if not key or not find_key(key):
            return JSONResponse({"detail": "Key not found"}, status_code=404)
        return None

    @app.get("/api/shared/logs/dirs")
    def shared_logs_dirs(key: str = "", code: str = ""):
        err = _check_shared(key, code)
        if err:
            return err
        return logs_dirs()

    @app.get("/api/shared/logs/aggregate")
    def shared_logs_aggregate(key: str = "", code: str = "", min_messages: int = 1, offset: int = 0, limit: int = 50, refresh: bool = False, model: str = "", log_dir: str = "", search: str = ""):
        err = _check_shared(key, code)
        if err:
            return err
        target_dir = resolve_log_dir(log_dir) if log_dir and log_dir != "__ALL__" else resolve_log_dir("")
        return JSONResponse(_aggregate_payload("anthropic", target_dir, min_messages, offset, limit, key, refresh, model, search))

    @app.get("/api/shared/logs/file")
    def shared_logs_file(key: str = "", code: str = "", filename: str = "", log_dir: str = ""):
        err = _check_shared(key, code)
        if err:
            return err
        return logs_file(filename, log_dir=log_dir or "__ALL__")

    @app.get("/api/shared/logs/file/download")
    def shared_logs_file_download(key: str = "", code: str = "", filename: str = "", log_dir: str = ""):
        err = _check_shared(key, code)
        if err:
            return err
        return logs_file_download(filename, log_dir=log_dir or "__ALL__")

    @app.get("/api/shared/logs/file/raw")
    def shared_logs_file_raw(key: str = "", code: str = "", filename: str = "", log_dir: str = ""):
        err = _check_shared(key, code)
        if err:
            return err
        return logs_file_raw(filename, log_dir=log_dir or "__ALL__")

    # start_session_cache_warmer(_env_dir, _current_log_dir)
