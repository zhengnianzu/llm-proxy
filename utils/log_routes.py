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
from concurrent.futures import ThreadPoolExecutor
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
import utils.newapi_index_db as nidb

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

_FMT_CACHE: Dict[str, str] = {}  # root_dir -> 'native'/'newapi'（缓存 detect_format）


def _root_format(root_dir: str) -> str:
    key = os.path.normpath(root_dir)
    fmt = _FMT_CACHE.get(key)
    if fmt is None:
        try:
            from utils.log_scan import detect_format
            fmt = detect_format(root_dir)
        except Exception:
            fmt = "native"
        _FMT_CACHE[key] = fmt
    return fmt


def _get_init_lock(root_dir: str) -> threading.Lock:
    with _INIT_LOCK_GUARD:
        if root_dir not in _INIT_LOCK:
            _INIT_LOCK[root_dir] = threading.Lock()
        return _INIT_LOCK[root_dir]


def _refresh_state(kind: str, root_dir: str, force: bool = False) -> None:
    # new-api 叶子：视图只读 index.db。构建/补 meta 一律经「日志管理」批量路径
    # （dispatcher + leader_lock 单写者）；此处不写库、不解析合并文件，保证视图请求恒亚秒。
    # 待构建/待补 meta 的信息由 _attach_backfill_status 以徽标形式带给前端。
    if _root_format(root_dir) == "newapi":
        return

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


def _attach_backfill_status(payload: Dict[str, Any], root_dir: str) -> None:
    """new-api 叶子给 payload 带上 index.db 构建状态，供前端区分「未构建 / 待补 meta /
    有新增待重建 / 就绪」并给出「去日志管理批量构建」的提示。native 目录不附带此字段。"""
    if _root_format(root_dir) != "newapi":
        return
    try:
        st = nidb.status(root_dir)
        # index.jsonl 是否有超出已摄取偏移的新增行（stale：已构建但落后于最新）
        stale = st.get("built", False) and nidb.needs_build(root_dir)
        payload["backfill"] = {
            "format": "newapi",
            "built": st.get("built", False),
            "ingested": st.get("ingested", 0),
            "pending": st.get("pending", 0),
            "sessions": st.get("sessions", 0),
            "stale": bool(stale),
        }
    except Exception:
        pass


class _NidbBackend:
    """把 newapi 叶子 index.db 适配成 session_store 同名读 API，供 payload 逻辑无差别复用。

    session_store 是「单库按 session_key 全局寻址」，故其 get_traces_batch 只收 session_keys；
    index.db 是「每叶子一库」，故这里把 root_dir 记在实例上，get_traces_batch 内部带上叶子路径。
    """

    def __init__(self, root_dir: str):
        self._root = root_dir

    def list_sessions(self, root_dir, api_key="", model="", min_msg_count=0, offset=0, limit=0):
        return nidb.list_sessions(root_dir, api_key=api_key, model=model,
                                  min_msg_count=min_msg_count, offset=offset, limit=limit)

    def count_sessions(self, root_dir, api_key="", model="", min_msg_count=0):
        return nidb.count_sessions(root_dir, api_key=api_key, model=model, min_msg_count=min_msg_count)

    def get_traces_batch(self, session_keys):
        return nidb.get_traces_batch(self._root, session_keys)

    def get_known_models(self, root_dir):
        return nidb.get_known_models(root_dir)


def _read_backend(root_dir: str):
    """按格式选读后端：newapi → 叶子 index.db；native → 全局 session_cache.db（_ss）。"""
    if _root_format(root_dir) == "newapi":
        return _NidbBackend(root_dir)
    return _ss


def _payload_known_keys(kind: str, root_dir: str) -> Tuple[List[str], float]:
    """返回 (known_keys, last_refresh_ts)。newapi 从 index.db 取，native 从内存 state 取。"""
    if _root_format(root_dir) == "newapi":
        try:
            keys = nidb.get_known_keys(root_dir)
            ts = float(nidb.status(root_dir).get("updated_at", 0) or 0)
        except Exception:
            keys, ts = [], 0.0
        return keys, ts
    with _CACHE_LOCK:
        current_state = _state(kind, root_dir)
        return sorted(current_state["known_keys"]), current_state.get("_last_refresh_ts", 0)


def _list_payload(kind: str, root_dir: str, min_messages: int, offset: int = 0, limit: int = 50, api_key: str = "", refresh: bool = False, model: str = "") -> Dict[str, Any]:
    _refresh_state(kind, root_dir, force=refresh)
    backend = _read_backend(root_dir)

    sessions = backend.list_sessions(root_dir, api_key=api_key, model=model, min_msg_count=min_messages)
    session_keys = [s["session_key"] for s in sessions]
    traces_by_key = backend.get_traces_batch(session_keys)

    known_models = backend.get_known_models(root_dir)
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

    known_keys, last_ts = _payload_known_keys(kind, root_dir)

    payload = {"items": paged, "total": total, "known_keys": known_keys, "known_models": known_models, "last_refresh_ts": last_ts}
    _attach_backfill_status(payload, root_dir)
    return payload


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


def _match_messages_content(root_dir: str, filename: str, keyword: str, newapi: bool = False) -> bool:
    req_path = Path(root_dir) / filename
    if not req_path.is_file():
        return False
    kw = keyword.lower()
    # new-api：合并单文件（键 req/resp/up_req/up_resp，无顶层 messages、无 -res.json 兄弟文件）。
    # 直接对整份文件字节做子串扫描——覆盖请求+响应全部内容，免 json 解析与 utf-8 解码（快得多）。
    # 关键词含 ASCII 字母时才做大小写折叠（bytes.lower 只影响 ASCII 字节，多字节 UTF-8 不受影响，
    # 故中文关键词走零拷贝的直接字节查找）。
    if newapi:
        try:
            kwb = keyword.encode("utf-8", "ignore")
            fold = any(("a" <= ch.lower() <= "z") for ch in keyword if ch.isascii())
            with open(req_path, "rb") as f:
                data = f.read()
            return (kwb.lower() in data.lower()) if fold else (kwb in data)
        except Exception:
            return False
    try:
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


def _filter_sessions_by_content(root_dir: str, sessions: List[dict], keyword: str, newapi: bool) -> List[dict]:
    """并行对每个 session 的 latest_file 做内容匹配，返回命中的 session。

    单线程逐个 open+json.load 数千个合并文件会长时间卡住（前端一直转圈）；这里用线程池并发，
    I/O 等待可重叠，配合 newapi 的免解析子串扫描，把「深度搜索」从数分钟降到秒级。
    """
    files = [(i, s.get("latest_file", "")) for i, s in enumerate(sessions)]
    workers = min(32, (os.cpu_count() or 4) * 4)

    def _hit(item):
        i, fn = item
        return i if (fn and _match_messages_content(root_dir, fn, keyword, newapi)) else -1

    matched_idx = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(_hit, files):
            if r >= 0:
                matched_idx.append(r)
    matched_idx.sort()
    return [sessions[i] for i in matched_idx]


def _aggregate_payload(kind: str, root_dir: str, min_messages: int, offset: int = 0, limit: int = 50, api_key: str = "", refresh: bool = False, model: str = "", search: str = "", q1search: str = "") -> Dict[str, Any]:
    _refresh_state(kind, root_dir, force=refresh)
    backend = _read_backend(root_dir)

    search = (search or "").strip()
    q1search = (q1search or "").strip()

    if search:
        # search 需要全量加载再过滤（无法下推 SQL），分页在过滤后处理。
        # 内容扫描并行化 + newapi 免解析，避免前端长时间转圈。
        sessions = backend.list_sessions(root_dir, api_key=api_key, model=model, min_msg_count=min_messages)
        is_newapi = _root_format(root_dir) == "newapi"
        sessions = _filter_sessions_by_content(root_dir, sessions, search, is_newapi)
        total = len(sessions)
        paged = sessions[offset:offset + limit] if limit > 0 else sessions[offset:]
    elif q1search:
        # 首句过滤：只匹配 session 的 q1（内存过滤，不读合并文件），覆盖全部会话后再分页。
        needle = q1search.lower()
        sessions = backend.list_sessions(root_dir, api_key=api_key, model=model, min_msg_count=min_messages)
        sessions = [s for s in sessions if needle in (s.get("q1", "") or "").lower()]
        total = len(sessions)
        paged = sessions[offset:offset + limit] if limit > 0 else sessions[offset:]
    else:
        # 无 search：COUNT + 分页完全下推 SQL
        total = backend.count_sessions(root_dir, api_key=api_key, model=model, min_msg_count=min_messages)
        paged = backend.list_sessions(root_dir, api_key=api_key, model=model, min_msg_count=min_messages, offset=offset, limit=limit if limit > 0 else 0)

    known_models = backend.get_known_models(root_dir)
    paged_keys = [s["session_key"] for s in paged]
    traces_by_key = backend.get_traces_batch(paged_keys)

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

    known_keys, last_refresh_ts = _payload_known_keys(kind, root_dir)

    agg_payload = {"items": items, "total": total, "known_keys": known_keys, "known_models": known_models, "last_refresh_ts": last_refresh_ts}
    _attach_backfill_status(agg_payload, root_dir)
    return agg_payload


def _aggregate_all_payload(kind: str, env_dir: str, min_messages: int, offset: int = 0, limit: int = 50, api_key: str = "", refresh: bool = False, model: str = "", search: str = "", q1search: str = "") -> Dict[str, Any]:
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

    q1search = (q1search or "").strip()
    if q1search:
        needle = q1search.lower()
        all_sessions = [p for p in all_sessions if needle in (p[0].get("q1", "") or "").lower()]

    if search and search.strip():
        search = search.strip()

        def _hit(pair):
            s, rd = pair
            fn = s.get("latest_file", "")
            ok = bool(fn) and _match_messages_content(rd, fn, search, _root_format(rd) == "newapi")
            return pair if ok else None

        workers = min(32, (os.cpu_count() or 4) * 4)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            all_sessions = [p for p in ex.map(_hit, all_sessions) if p is not None]

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


def _export_view_leaves(roots, env_dir: str) -> List[str]:
    """枚举导出浏览要覆盖的叶子目录（跨全部来源，去掉 mtime 层级语义）。

    只从 leaf_status（同步产物，含绝对 leaf_path）读，不再扫盘兜底：每个登记
    根在 sources 表里都有行，leaf_path 与 leaf_status 同源，保证「列表里有」≡
    「DB 有 leaf_path」。root_id 用 sources 里按 root_path 归一化匹配出的
    **实际存储** root_id（而非 get_root_id(root, env_dir)——那会把活跃 env 目录
    强制折叠成 'default'，偏偏本系统里活跃目录的叶子挂在 md5 哈希 root_id 下，
    bulk_get('default') 恒为空，旧实现只得退回 iter_index_dirs 全盘扫）。
    返回去重后的叶子绝对路径 list。
    """
    import utils.logdir_store as _lds
    from utils.logs_config import get_root_id
    leaves: List[str] = []
    seen = set()
    for root in roots or []:
        if not root:
            continue
        try:
            rp = Path(root)
        except (TypeError, ValueError):
            continue
        if not rp.is_dir():
            continue
        rid = None
        try:
            # 按 root_path 归一化匹配 sources 行，取其真实 root_id（不折叠活跃目录）
            norm = os.path.normpath(str(root))
            src = _lds.get_source_by_path(norm)
            if src:
                rid = src.get("root_id")
            if not rid:
                rid = get_root_id(str(root), env_dir)
        except Exception:  # noqa: BLE001
            rid = None
        if not rid:
            continue
        try:
            for row in _lds.bulk_get(rid):
                lp = row.get("leaf_path") or ""
                if not lp:
                    continue
                full = lp if os.path.isabs(lp) else os.path.join(str(root), lp)
                if Path(full).is_dir():
                    key = os.path.normpath(full)
                    if key not in seen:
                        seen.add(key)
                        leaves.append(full)
        except Exception:  # noqa: BLE001
            continue
    return leaves


def _export_key_leaves(roots, env_dir: str, api_key: str) -> List[str]:
    """导出浏览按单个 key 取叶子：直接用 build_stats_multi **已算好的** key→mtime 分布。

    不再对全部叶子逐叶 list_sessions（导出浏览覆盖到 /mnt 上 1900+ 叶子，逐叶连库是
    分钟级）。build_stats_multi 的 rows[].mtime_cells 正是「该 key 落在哪些 mtime 目录」，
    与 public_export_submit 取 mtime_dirs 同一来源；用 _resolve_mt_for 把 mtime key
    （<root_id>/<rel> 或裸 <rel>）解析回叶子绝对路径。api_key 为空 / 分布未命中 / 解析
    失败时退回全量 _export_view_leaves（保证正确性优先于速度）。
    """
    if not api_key:
        return _export_view_leaves(roots, env_dir)
    existing = [r for r in roots if Path(r).is_dir()]
    if not existing:
        return []
    try:
        from utils.stats_index import build_stats_multi
        stats = build_stats_multi(existing, active_env_dir=env_dir)
    except Exception:  # noqa: BLE001
        return _export_view_leaves(roots, env_dir)
    mts = []
    for row in stats.get("rows", []):
        if row.get("api_key") == api_key:
            mts = sorted(row.get("mtime_cells", {}).keys(), reverse=True)
            break
    if not mts:
        return []
    try:
        from utils.export_routes import _resolve_mt_for
    except Exception:  # noqa: BLE001
        return _export_view_leaves(roots, env_dir)
    leaves: List[str] = []
    seen = set()
    for mt in mts:
        try:
            lp = _resolve_mt_for(env_dir, mt)
        except Exception:  # noqa: BLE001
            lp = None
        if lp:
            key = os.path.normpath(lp)
            if key not in seen:
                seen.add(key)
                leaves.append(lp)
    return leaves or _export_view_leaves(roots, env_dir)


def export_view_aggregate_payload(roots, env_dir: str, min_messages: int = 1, offset: int = 0, limit: int = 50,
                                  api_key: str = "", model: str = "", search: str = "", q1search: str = "",
                                  refresh: bool = False) -> Dict[str, Any]:
    """导出浏览聚合：按 key 汇总会话，分页返回。

    与 _aggregate_all_payload 的区别：_read_backend 按叶子选后端（newapi→index.db，
    native→session_cache.db），api_key 下推过滤。叶子集用 _export_key_leaves ——
    直接取 build_stats_multi **已算好的** key→mtime 分布（rows[].mtime_cells），
    只加载该 key 落到的叶子，而非跨全部登记根逐叶扫（那要连 /mnt 上 1900+ 个叶子，
    分钟级）。首句(q1search)/内容(search)过滤在合并后做。分页前先按 last_ts 全局
    倒序，traces 仅对分页切片按叶子批量取。
    """
    leaves = _export_key_leaves(roots, env_dir, api_key)
    all_sessions: List[Tuple[dict, str]] = []
    known_keys: set = set()
    known_models: set = set()

    for leaf in leaves:
        try:
            _refresh_state("anthropic", leaf, force=refresh)
        except Exception:  # noqa: BLE001
            pass
        try:
            backend = _read_backend(leaf)
            sels = backend.list_sessions(leaf, api_key=api_key, model=model, min_msg_count=min_messages)
            for s in sels:
                all_sessions.append((s, leaf))
                if s.get("api_key"):
                    known_keys.add(s["api_key"])
                for m in s.get("models", []):
                    if m:
                        known_models.add(m)
            try:
                for m in backend.get_known_models(leaf):
                    if m:
                        known_models.add(m)
            except Exception:  # noqa: BLE001
                pass
        except Exception:  # noqa: BLE001
            continue

    q1search = (q1search or "").strip()
    if q1search:
        needle = q1search.lower()
        all_sessions = [p for p in all_sessions if needle in (p[0].get("q1", "") or "").lower()]

    search = (search or "").strip()
    if search:
        def _hit(pair):
            s, ld = pair
            fn = s.get("latest_file", "")
            try:
                ok = bool(fn) and _match_messages_content(ld, fn, search, _root_format(ld) == "newapi")
            except Exception:  # noqa: BLE001
                ok = False
            return pair if ok else None
        workers = min(32, (os.cpu_count() or 4) * 4)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            all_sessions = [p for p in ex.map(_hit, all_sessions) if p is not None]

    all_sessions.sort(key=lambda t: t[0].get("last_ts", "") or t[0].get("first_ts", "") or "", reverse=True)
    total = len(all_sessions)
    paged = all_sessions[offset:offset + limit] if limit > 0 else all_sessions[offset:]

    from collections import defaultdict
    by_root: Dict[str, List[str]] = defaultdict(list)
    for session, ld in paged:
        by_root[ld].append(session["session_key"])
    traces_map: Dict[str, Dict[str, List]] = {}
    for ld, keys in by_root.items():
        try:
            traces_map[ld] = _read_backend(ld).get_traces_batch(keys)
        except Exception:  # noqa: BLE001
            traces_map[ld] = {}

    items: List[Dict[str, Any]] = []
    for session, ld in paged:
        sk = session["session_key"]
        trace_list = traces_map.get(ld, {}).get(sk, [])
        payload: Dict[str, Any] = {
            "first_time": _format_time(session.get("first_ts", "")),
            "last_time": _format_time(session.get("last_ts", "")),
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

    return {"items": items, "total": total, "known_keys": sorted(known_keys), "known_models": sorted(known_models)}


def export_view_list_payload(roots, env_dir: str, min_messages: int = 1, offset: int = 0, limit: int = 50,
                             api_key: str = "", model: str = "", search: str = "", q1search: str = "",
                             refresh: bool = False) -> Dict[str, Any]:
    """导出浏览「单文件列表」：把每 session 的 trace 展开为文件级条目，分页返回。

    叶子集来自 _export_key_leaves（build_stats_multi 的 key→mtime 分布），只加载
    该 key 落到的叶子。先聚合会话、对 session 做 q1search/search 过滤（与
    export_view_aggregate_payload 同口径），再把命中的 session 展开成文件行；
    文件行保留所属 session_key 供 trace 归属。
    """
    leaves = _export_key_leaves(roots, env_dir, api_key)
    sessions_by_leaf: List[Tuple[dict, str]] = []
    known_keys: set = set()
    known_models: set = set()

    for leaf in leaves:
        try:
            _refresh_state("anthropic", leaf, force=refresh)
        except Exception:  # noqa: BLE001
            pass
        try:
            backend = _read_backend(leaf)
            sels = backend.list_sessions(leaf, api_key=api_key, model=model, min_msg_count=min_messages)
        except Exception:  # noqa: BLE001
            continue
        for s in sels:
            sessions_by_leaf.append((s, leaf))
            if s.get("api_key"):
                known_keys.add(s["api_key"])
            for m in s.get("models", []):
                if m:
                    known_models.add(m)

    q1search = (q1search or "").strip()
    if q1search:
        needle = q1search.lower()
        sessions_by_leaf = [p for p in sessions_by_leaf if needle in (p[0].get("q1", "") or "").lower()]

    search = (search or "").strip()
    if search:
        def _hit(pair):
            s, ld = pair
            fn = s.get("latest_file", "")
            ok = bool(fn) and _match_messages_content(ld, fn, search, _root_format(ld) == "newapi")
            return pair if ok else None
        workers = min(32, (os.cpu_count() or 4) * 4)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            sessions_by_leaf = [p for p in ex.map(_hit, sessions_by_leaf) if p is not None]

    # 按叶子批量取 trace，展开为文件行
    from collections import defaultdict
    by_leaf: Dict[str, List[str]] = defaultdict(list)
    for s, ld in sessions_by_leaf:
        by_leaf[ld].append(s["session_key"])
    traces_map: Dict[str, Dict[str, List]] = {}
    for ld, keys in by_leaf.items():
        try:
            traces_map[ld] = _read_backend(ld).get_traces_batch(keys)
        except Exception:  # noqa: BLE001
            traces_map[ld] = {}

    raw: List[Dict[str, Any]] = []
    for s, ld in sessions_by_leaf:
        for t in traces_map.get(ld, {}).get(s["session_key"], []):
            fn = t.get("filename", "")
            if not fn:
                continue
            raw.append({
                "filename": fn,
                "message_count": t.get("msg_count", 0),
                "model": t.get("model", ""),
                "api_key": s.get("api_key", ""),
                "_ts": t.get("ts", "") or s.get("last_ts", ""),
                "_session_key": s["session_key"],
                "_leaf": ld,
            })

    raw.sort(key=lambda r: r.get("_ts", ""), reverse=True)
    total = len(raw)
    paged = raw[offset:offset + limit] if limit > 0 else raw[offset:]
    items = [{k: v for k, v in r.items() if not k.startswith("_")} for r in paged]
    return {"items": items, "total": total, "known_keys": sorted(known_keys), "known_models": sorted(known_models)}


def _export_view_find_file(roots, env_dir: str, filename: str, api_key: str = "") -> Optional[str]:
    """在导出浏览覆盖的叶子下查找文件，返回绝对路径或 None。

    传 api_key 时按该 key 的 mtime 分布（_export_key_leaves）限定加载的叶子；
    不传则回退全量 _export_view_leaves。
    """
    leaves = _export_key_leaves(roots, env_dir, api_key) if api_key else _export_view_leaves(roots, env_dir)
    for ld in leaves:
        p = os.path.join(ld, filename)
        if os.path.isfile(p):
            return p
    return None


def _load_conversation_file(path: str) -> Optional[dict]:
    """读取请求-响应合并文件，返回与「对话浏览」一致的 payload（含 _usage）。

    path 为叶子内绝对路径。newapi 单文件用 build_preview_payload 拆解；
    native 三元组则加载 -req.json 并把响应内容作为 assistant 消息追加。
    解析失败返回 None。
    """
    leaf = os.path.dirname(path)
    filename = os.path.basename(path)
    is_newapi = _root_format(leaf) == "newapi"
    if (filename.endswith(".json") and not filename.endswith("-req.json")
            and "/" not in filename and "\\" not in filename and ".." not in filename):
        if is_newapi:
            from utils.newapi_format import build_preview_payload
            return build_preview_payload(Path(path))
        return None
    if not filename.endswith("-req.json") or "/" in filename or "\\" in filename or ".." in filename:
        return None
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
    return data


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

    def _all_roots() -> list:
        """活跃 env_dir + 配置的历史路径。"""
        from utils.logs_config import get_stats_roots
        return get_stats_roots(_env_dir)

    def unified_log_dir() -> str:
        return _current_log_dir

    def resolve_log_dir(log_dir: str = "") -> str:
        if not log_dir:
            return _current_log_dir
        # log_dir 形如 "26072520" 或跨 root 时 "<root_id>::<rel_key>"
        # root_part 现为稳定 root_id；旧链接可能是 basename/normpath，保留回退匹配。
        if "::" in log_dir:
            from utils.logs_config import get_root_id
            from utils.log_scan import resolve_leaf
            root_part, _, rel = log_dir.partition("::")
            for root in _all_roots():
                if (get_root_id(root, _env_dir) == root_part
                        or os.path.basename(os.path.normpath(root)) == root_part
                        or os.path.normpath(root) == root_part):
                    candidate = str(resolve_leaf(root, rel))
                    if os.path.isdir(candidate):
                        return candidate
        # 兼容旧格式：同 env-key 下的时间戳子目录
        candidate = os.path.join(_env_dir, log_dir)
        if os.path.isdir(candidate):
            return candidate
        # 再在所有 root 下按相对路径找
        for root in _all_roots():
            c = os.path.join(root, log_dir)
            if os.path.isdir(c):
                return c
        # 兼容旧报告：new-api 布局 <root>/<day>/<hour>，但旧报告只存了裸 <hour>
        # （历史 bug：log_dir=Path(src_dir).name 丢了日期层）。这里在每个 root
        # 下一层里搜同名子目录补回。
        if os.sep not in log_dir and "/" not in log_dir:
            for root in _all_roots():
                try:
                    for day in os.listdir(root):
                        c = os.path.join(root, day, log_dir)
                        if os.path.isdir(c):
                            return c
                except OSError:
                    continue
        return _current_log_dir

    def anthropic_log_dir() -> str:
        return get_log_dir("logs_anthropic")

    def openai_log_dir() -> str:
        return get_log_dir("logs_openai")

    @app.get("/logs/dirs")
    def logs_dirs():
        """列出所有配置 root 下含 index.jsonl 的叶子目录，当前活跃目录排在第一个。

        单 root（仅活跃目录）时 name 用叶子相对路径（兼容旧行为）；
        多 root 时 name 用 "<root_basename>::<rel_key>" 以区分来源。
        """
        from utils.stats_index import refresh_index, get_dir_counts
        from utils.token_index import refresh_token_index
        from utils.log_scan import dir_key_for
        from utils.logs_config import get_path_name, get_root_id
        current_tag = Path(_current_log_dir).name
        roots = _all_roots()
        multi = len(roots) > 1
        dirs = []
        current_name = current_tag

        for root in roots:
            root_path = Path(root)
            if not root_path.is_dir():
                continue
            # 多 root 时用稳定 root_id 作前缀区分来源（消除同 basename 冲突）
            root_id = get_root_id(root, _env_dir)
            # 活跃 env_dir 名称固定 default；历史根取配置名称
            is_active_root = (os.path.normpath(root) == os.path.normpath(_env_dir))
            root_label = "default" if is_active_root else get_path_name(root)
            index = refresh_index(root_path)
            tok_index = refresh_token_index(root)
            counts = get_dir_counts(index, tok_index)
            for rel_key, count in counts.items():
                is_current = (os.path.normpath(root) == os.path.normpath(_env_dir)
                              and rel_key == current_tag)
                name = f"{root_id}::{rel_key}" if multi else rel_key
                if is_current:
                    current_name = name
                dirs.append({
                    "name": name,
                    "current": is_current,
                    "count": count,
                    "root": root_id,
                    "root_label": root_label,
                })
        # 当前目录优先，其余按名称倒序
        dirs.sort(key=lambda d: (not d["current"], d["name"]), reverse=False)
        dirs[1:] = sorted(dirs[1:], key=lambda d: d["name"], reverse=True)
        return JSONResponse({"dirs": dirs, "current": current_name})

    # --- 统一路由（新） ---

    @app.get("/logs/list")
    def logs_list(min_messages: int = 10, offset: int = 0, limit: int = 50, api_key: str = "", refresh: bool = False, model: str = "", log_dir: str = ""):
        return JSONResponse(_list_payload("anthropic", resolve_log_dir(log_dir), min_messages, offset, limit, api_key, refresh, model))

    @app.get("/logs/file")
    def logs_file(filename: str, log_dir: str = ""):
        # new-api：合并文件（文件名以 .json 结尾但非 -req.json），拆解为 messages + assistant
        if (filename.endswith(".json") and not filename.endswith("-req.json")
                and "/" not in filename and "\\" not in filename and ".." not in filename):
            target_dir = resolve_log_dir(log_dir)
            if _root_format(target_dir) == "newapi":
                from utils.newapi_format import build_preview_payload
                path = os.path.join(target_dir, filename)
                if not os.path.isfile(path):
                    return JSONResponse({"error": "file not found"}, status_code=404)
                payload = build_preview_payload(Path(path))
                if payload is None:
                    return JSONResponse({"error": "parse failed"}, status_code=500)
                return JSONResponse(payload)

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
        """返回原始 JSON 文件内容（支持 req / headers / res 三种后缀，及 new-api 合并文件）。"""
        if "/" in filename or "\\" in filename or ".." in filename:
            return JSONResponse({"error": "invalid filename"}, status_code=400)
        # new-api 合并文件：文件名以 .json 结尾但非三元组后缀，限定 new-api root
        if (not any(filename.endswith(s) for s in _RAW_SUFFIXES)
                and filename.endswith(".json")):
            target_dir = resolve_log_dir(log_dir)
            if _root_format(target_dir) == "newapi":
                path = os.path.join(target_dir, filename)
                if not os.path.isfile(path):
                    return JSONResponse({"error": "file not found"}, status_code=404)
                with open(path, "r", encoding="utf-8") as f:
                    return JSONResponse(json.load(f))
            return JSONResponse({"error": "invalid filename suffix"}, status_code=400)
        if not any(filename.endswith(s) for s in _RAW_SUFFIXES):
            return JSONResponse({"error": "invalid filename suffix"}, status_code=400)
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
    def logs_aggregate(min_messages: int = 1, offset: int = 0, limit: int = 50, api_key: str = "", refresh: bool = False, model: str = "", log_dir: str = "", search: str = "", q1search: str = ""):
        return JSONResponse(_aggregate_payload("anthropic", resolve_log_dir(log_dir), min_messages, offset, limit, api_key, refresh, model, search, q1search))

    # --- 旧路由（别名，向后兼容） ---

    @app.get("/logs/anthropic/list")
    def logs_anthropic_list(min_messages: int = 10, offset: int = 0, limit: int = 50, api_key: str = "", refresh: bool = False, model: str = ""):
        return JSONResponse(_list_payload("anthropic", unified_log_dir(), min_messages, offset, limit, api_key, refresh, model))

    @app.get("/logs/anthropic/file")
    def logs_anthropic_file(filename: str):
        return logs_file(filename)

    @app.get("/logs/anthropic/aggregate")
    def logs_anthropic_aggregate(min_messages: int = 1, offset: int = 0, limit: int = 50, api_key: str = "", refresh: bool = False, model: str = "", search: str = "", q1search: str = ""):
        return JSONResponse(_aggregate_payload("anthropic", unified_log_dir(), min_messages, offset, limit, api_key, refresh, model, search, q1search))

    @app.get("/logs/openai/list")
    def logs_openai_list(min_messages: int = 10, offset: int = 0, limit: int = 50, api_key: str = "", refresh: bool = False, model: str = ""):
        return JSONResponse(_list_payload("anthropic", unified_log_dir(), min_messages, offset, limit, api_key, refresh, model))

    @app.get("/logs/openai/file")
    def logs_openai_file(filename: str):
        return logs_file(filename)

    @app.get("/logs/openai/aggregate")
    def logs_openai_aggregate(min_messages: int = 1, offset: int = 0, limit: int = 50, api_key: str = "", refresh: bool = False, model: str = "", search: str = "", q1search: str = ""):
        return JSONResponse(_aggregate_payload("anthropic", unified_log_dir(), min_messages, offset, limit, api_key, refresh, model, search, q1search))

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
    def shared_logs_aggregate(key: str = "", code: str = "", min_messages: int = 1, offset: int = 0, limit: int = 50, refresh: bool = False, model: str = "", log_dir: str = "", search: str = "", q1search: str = ""):
        err = _check_shared(key, code)
        if err:
            return err
        target_dir = resolve_log_dir(log_dir) if log_dir and log_dir != "__ALL__" else resolve_log_dir("")
        return JSONResponse(_aggregate_payload("anthropic", target_dir, min_messages, offset, limit, key, refresh, model, search, q1search))

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
