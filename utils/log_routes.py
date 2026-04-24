"""
/logs/* 路由：列表、聚合、单文件读取（Anthropic + OpenAI）
优先使用 index.jsonl 的最近窗口；缺失时降级为目录扫描。
"""

import json
import os
import re as _re
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from utils.log_paths import build_index_path, get_log_dir

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
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


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
        # 流式 SSE 响应：从 chunks 重建完整 message
        chunks = data.get("chunks", [])
        message = {}
        blocks = {}
        block_json_buf = {}
        for chunk in chunks:
            t = chunk.get("type")
            if t == "anthropic_passthrough_sse_meta":
                continue
            if t == "message_start":
                message = dict(chunk.get("message", {}))
                message["content"] = []
            elif t == "content_block_start":
                idx = chunk.get("index", 0)
                cb = dict(chunk.get("content_block", {}))
                blocks[idx] = cb
                if cb.get("type") == "tool_use":
                    block_json_buf[idx] = ""
            elif t == "content_block_delta":
                idx = chunk.get("index", 0)
                delta = chunk.get("delta", {})
                dtype = delta.get("type")
                if dtype == "text_delta":
                    blocks.setdefault(idx, {"type": "text", "text": ""})
                    blocks[idx]["text"] = blocks[idx].get("text", "") + delta.get("text", "")
                elif dtype == "input_json_delta":
                    block_json_buf[idx] = block_json_buf.get(idx, "") + delta.get("partial_json", "")
            elif t == "content_block_stop":
                idx = chunk.get("index", 0)
                if idx in block_json_buf:
                    try:
                        blocks[idx]["input"] = json.loads(block_json_buf[idx])
                    except json.JSONDecodeError:
                        blocks[idx]["input"] = block_json_buf[idx]
        if blocks:
            message["content"] = [blocks[i] for i in sorted(blocks)]
        content = message.get("content")
        if content:
            return content
        return None

    # 非流式响应
    if isinstance(data.get("json"), dict):
        msg = data["json"]
        if msg.get("content") is not None:
            return msg["content"]
    return None


def _extract_openai_res_content(res_path: Path):
    data = _load_json(res_path)
    if not data:
        return None
    msg = data.get("json", {}).get("choices", [{}])[0].get("message")
    if isinstance(msg, dict) and msg.get("role") == "assistant":
        return msg
    return None


def _get_text_from_content(content) -> str:
    """从 message content 中提取纯文本"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                return block.get("text", "")
        return str(content[0])[:500] if content else ""
    return str(content)[:500]


def _strip_sender_prefix(text: str) -> str:
    """去掉 OpenClaw 的 Sender (untrusted metadata) 前缀"""
    m = _re.search(r'\[.*?\]\s*', text)
    if m:
        return text[m.end():].strip()
    return text.strip()


def _find_real_user_query_anthropic(messages, return_index=False):
    """找到真正的用户 query 及其在 messages 中的索引。
    return_index=True 时返回 (text, index) 元组。
    """
    if not messages:
        return ("", 0) if return_index else ""
    first_text = _get_text_from_content(messages[0].get("content", ""))
    is_new_session = first_text.startswith("A new session was started via /new") or first_text.startswith("A new session was started via /reset")

    if is_new_session:
        for i, msg in enumerate(messages[1:], 1):
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if isinstance(content, list):
                if not any(b.get("type") == "text" for b in content if isinstance(b, dict)):
                    continue
            real_text = _get_text_from_content(content)
            if real_text.startswith("Sender (untrusted metadata)"):
                real_text = _strip_sender_prefix(real_text)
            return (real_text, i) if return_index else real_text
        return ("", 0) if return_index else ""
    else:
        real_text = first_text
        if real_text.startswith("Sender (untrusted metadata)"):
            real_text = _strip_sender_prefix(real_text)
        return (real_text, 0) if return_index else real_text


def _anthropic_chain_key(messages: List[Dict[str, Any]]) -> str:
    if not messages:
        return ""
    content = messages[0].get("content", "")
    if isinstance(content, list):
        content = "|".join(
            block.get("text") or block.get("id") or str(block)[:200]
            for block in content
            if isinstance(block, dict)
        )
    return str(content)


def _openai_chain_key(messages: List[Dict[str, Any]]) -> str:
    for msg in messages:
        if msg.get("role") == "system":
            match = _re.search(r"# Origin_query\s*\n+(.*?)(\n---|\Z)", str(msg.get("content", "")), _re.DOTALL)
            if match:
                return "oq:" + match.group(1).strip()
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                content = "|".join(block.get("text", "") for block in content if isinstance(block, dict))
            return "u:" + str(content)
    return str(messages[0].get("content", "")) if messages else ""


def _build_state(root_dir: str) -> Dict[str, Any]:
    return {
        "root_dir": root_dir,
        "index_path": build_index_path(root_dir),
        "initialized": False,
        "line_count": 0,
        "byte_offset": 0,
        "known_keys": set(),
        "sessions": OrderedDict(),  # key=first_ts, value={q1, model, ...}
        "_chain_map": {},  # chain_key -> first_ts (内存映射，不持久化)
    }


_CACHE_FILE = ".session_cache.json"


def _cache_file_path(root_dir: str) -> str:
    return os.path.join(root_dir, _CACHE_FILE)


def _save_state_to_disk(state: Dict[str, Any]) -> None:
    """持久化计算状态到磁盘，供服务重启后恢复。"""
    root_dir = state["root_dir"]
    try:
        os.makedirs(root_dir, exist_ok=True)
        # sessions 持久化时去掉 _best_req_count（内部字段）
        sessions_out = {}
        for ts_key, s in state["sessions"].items():
            sessions_out[ts_key] = {k: v for k, v in s.items() if not k.startswith("_")}
        payload = {
            "byte_offset": state["byte_offset"],
            "line_count": state["line_count"],
            "known_keys": list(state["known_keys"]),
            "sessions": sessions_out,
        }
        tmp = _cache_file_path(root_dir) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp, _cache_file_path(root_dir))
    except OSError:
        pass


def _load_state_from_disk(state: Dict[str, Any]) -> bool:
    """从磁盘恢复持久化状态。成功返回 True，失败返回 False（调用方应全量重建）。"""
    root_dir = state["root_dir"]
    cache_path = _cache_file_path(root_dir)

    if not os.path.isfile(cache_path):
        return False

    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            sp = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False

    byte_offset = sp.get("byte_offset", 0)
    if not isinstance(byte_offset, int) or byte_offset < 0:
        return False

    state["byte_offset"] = byte_offset
    state["line_count"] = sp.get("line_count", 0)
    state["known_keys"] = set(sp.get("known_keys", []))

    sessions = OrderedDict()
    chain_map = {}
    raw_sessions = sp.get("sessions", {})
    if isinstance(raw_sessions, dict):
        for ts_key, s in raw_sessions.items():
            sessions[ts_key] = s
            # 重建 _chain_map: api_key||q1 -> ts_key
            ck = f"{s.get('api_key', '')}||{s.get('q1', '')}"
            chain_map[ck] = ts_key
    state["sessions"] = sessions
    state["_chain_map"] = chain_map

    return True


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
    data = _load_json(req_path)
    if not data:
        return False

    messages = data.get("messages")
    if not isinstance(messages, list):
        return False

    filename = req_path.name
    ts = str((index_entry or {}).get("ts") or filename.replace("-req.json", ""))
    model = str(data.get("model", "") or (index_entry or {}).get("model", "") or "")
    message_count = len(messages)
    api_key = str((index_entry or {}).get("api_key", "") or "")
    state["known_keys"].add(api_key)

    if index_entry and index_entry.get("chain_key"):
        chain_key = index_entry["chain_key"]
    elif kind == "anthropic":
        chain_key = _anthropic_chain_key(messages)
    else:
        chain_key = _openai_chain_key(messages)

    lookup_key = f"{api_key}||{chain_key}"
    q1_preview = (index_entry or {}).get("q1_preview", "")

    res_path = req_path.with_name(filename.replace("-req.json", "-res.json"))
    has_res = res_path.is_file()
    full_message_count = message_count + (1 if has_res else 0)

    trace_entry = {"filename": filename, "model": model, "msg_count": full_message_count, "ts": ts}

    chain_map = state["_chain_map"]
    session_key = chain_map.get(lookup_key)
    session = state["sessions"].get(session_key) if session_key else None

    # Detect new session: 真实轮次 = message_count - q1_base
    # 真实轮次为 1（只有 q1 本身）且已有同 q1 的 session，说明是新会话
    if kind == "anthropic":
        st = (index_entry or {}).get("start_turn")
        if st is not None:
            q1_base = int(st)
        else:
            _, q1_base = _find_real_user_query_anthropic(messages, return_index=True)
    else:
        q1_base = 0
    real_turns = message_count - q1_base
    if session is not None and real_turns <= 1:
        suffix = 1
        new_lookup = f"{lookup_key}##session_{suffix}"
        while new_lookup in chain_map:
            suffix += 1
            new_lookup = f"{lookup_key}##session_{suffix}"
        lookup_key = new_lookup
        session = None

    if session is None:
        session_key = ts
        session = {
            "q1": q1_preview or chain_key[:200],
            "model": model,
            "latest_file": filename,
            "msg_count": full_message_count,
            "api_key": api_key,
            "first_ts": ts,
            "last_ts": ts,
            "trace_list": [trace_entry],
            "_best_req_count": message_count,
        }
        state["sessions"][session_key] = session
        chain_map[lookup_key] = session_key
    else:
        session["last_ts"] = ts
        session["trace_list"].append(trace_entry)
        if message_count > session.get("_best_req_count", 0) or \
           (message_count == session.get("_best_req_count", 0) and has_res):
            session["latest_file"] = filename
            session["msg_count"] = full_message_count
            session["_best_req_count"] = message_count
            session["model"] = model or session["model"]

    return True


def _refresh_state(kind: str, root_dir: str) -> None:
    state = _state(kind, root_dir)
    root = Path(root_dir)
    index_path = Path(state["index_path"])

    # Phase 1: 首次调用时尝试从磁盘恢复持久化状态
    if not state["initialized"]:
        if not _load_state_from_disk(state):
            state["sessions"].clear()
            state["_chain_map"].clear()
            state["byte_offset"] = 0
            state["line_count"] = 0

    # Phase 2: 增量读取 index.jsonl
    if index_path.is_file():
        rows, new_offset = _read_new_index_entries(index_path, root, state["byte_offset"])

        if new_offset == 0 and state["byte_offset"] > 0:
            # 文件被截断/轮转 — 全量重建
            state["sessions"].clear()
            state["_chain_map"].clear()
            state["byte_offset"] = 0
            state["line_count"] = 0
            rows, new_offset = _read_new_index_entries(index_path, root, 0)

        if rows:
            for row in rows:
                _process_req_row(kind, state, row["req_path"], row["entry"])
            state["line_count"] += len(rows)
            state["byte_offset"] = new_offset
            _save_state_to_disk(state)
        elif new_offset != state["byte_offset"]:
            state["byte_offset"] = new_offset
    else:
        # 无 index.jsonl — 降级为目录扫描（保持原有行为）
        if not state["initialized"]:
            for req_path in _collect_req_files(root):
                _process_req_row(kind, state, req_path)

    state["initialized"] = True


def _list_payload(kind: str, root_dir: str, min_messages: int, offset: int = 0, limit: int = 50, api_key: str = "", refresh: bool = False) -> Dict[str, Any]:
    with _CACHE_LOCK:
        current_state = _state(kind, root_dir)
        if refresh or not current_state["initialized"]:
            _refresh_state(kind, root_dir)
        # 从所有 session 的 trace_list 展开
        items = []
        for session in current_state["sessions"].values():
            if api_key and (session.get("api_key", "") or "") != api_key:
                continue
            for trace in session.get("trace_list", []):
                if trace.get("msg_count", 0) >= min_messages:
                    items.append({
                        "filename": trace["filename"],
                        "message_count": trace["msg_count"],
                        "model": trace.get("model", ""),
                        "api_key": session.get("api_key", ""),
                    })
        items.sort(key=lambda x: x["filename"], reverse=True)
        total = len(items)
        paged = items[offset:offset + limit] if limit > 0 else items[offset:]
        return {"items": paged, "total": total, "known_keys": sorted(current_state["known_keys"])}


def _aggregate_payload(kind: str, root_dir: str, min_messages: int, offset: int = 0, limit: int = 50, api_key: str = "", refresh: bool = False) -> Dict[str, Any]:
    with _CACHE_LOCK:
        current_state = _state(kind, root_dir)
        if refresh or not current_state["initialized"]:
            _refresh_state(kind, root_dir)
        sessions = []
        for session in current_state["sessions"].values():
            if api_key and (session.get("api_key", "") or "") != api_key:
                continue
            if session.get("msg_count", 0) < min_messages:
                continue
            sessions.append(session)

        sessions.sort(key=lambda s: s.get("last_ts", ""), reverse=True)
        total = len(sessions)
        paged = sessions[offset:offset + limit] if limit > 0 else sessions[offset:]

        items = []
        for session in paged:
            payload = {
                "first_time": _format_time(session["first_ts"]),
                "last_time": _format_time(session["last_ts"]),
                "file_count": len(session.get("trace_list", [])),
                "message_count": session.get("msg_count", 0),
                "model": session["model"],
                "latest_file": session.get("latest_file", ""),
                "api_key": session.get("api_key", ""),
                "q1_preview": session.get("q1", ""),
            }
            items.append(payload)

        return {"items": items, "total": total, "known_keys": sorted(current_state["known_keys"])}

def register_log_routes(app: FastAPI) -> None:
    def anthropic_log_dir() -> str:
        return get_log_dir("logs_anthropic")

    def openai_log_dir() -> str:
        return get_log_dir("logs_openai")

    @app.get("/logs/anthropic/list")
    def logs_anthropic_list(min_messages: int = 10, offset: int = 0, limit: int = 50, api_key: str = "", refresh: bool = False):
        return JSONResponse(_list_payload("anthropic", anthropic_log_dir(), min_messages, offset, limit, api_key, refresh))

    @app.get("/logs/anthropic/file")
    def logs_anthropic_file(filename: str):
        if not filename.endswith("-req.json") or "/" in filename or "\\" in filename or ".." in filename:
            return JSONResponse({"error": "invalid filename"}, status_code=400)
        path = os.path.join(anthropic_log_dir(), filename)
        if not os.path.isfile(path):
            return JSONResponse({"error": "file not found"}, status_code=404)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Append res content as assistant message
        res_path = Path(path).with_name(filename.replace("-req.json", "-res.json"))
        res_content = _extract_anthropic_res_content(res_path)
        if res_content is not None and isinstance(data.get("messages"), list):
            data["messages"].append({"role": "assistant", "content": res_content, "_from_res": True})
        return JSONResponse(data)

    @app.get("/logs/anthropic/aggregate")
    def logs_anthropic_aggregate(min_messages: int = 1, offset: int = 0, limit: int = 50, api_key: str = "", refresh: bool = False):
        return JSONResponse(_aggregate_payload("anthropic", anthropic_log_dir(), min_messages, offset, limit, api_key, refresh))

    @app.get("/logs/openai/list")
    def logs_openai_list(min_messages: int = 10, offset: int = 0, limit: int = 50, api_key: str = "", refresh: bool = False):
        return JSONResponse(_list_payload("openai", openai_log_dir(), min_messages, offset, limit, api_key, refresh))

    @app.get("/logs/openai/file")
    def logs_openai_file(filename: str):
        if not filename.endswith("-req.json") or "/" in filename or "\\" in filename or ".." in filename:
            return JSONResponse({"error": "invalid filename"}, status_code=400)
        path = os.path.join(openai_log_dir(), filename)
        if not os.path.isfile(path):
            return JSONResponse({"error": "file not found"}, status_code=404)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Append res content as assistant message
        res_path = Path(path).with_name(filename.replace("-req.json", "-res.json"))
        res_content = _extract_openai_res_content(res_path)
        if res_content is not None and isinstance(data.get("messages"), list):
            data["messages"].append({**res_content, "_from_res": True})
        return JSONResponse(data)

    @app.get("/logs/openai/aggregate")
    def logs_openai_aggregate(min_messages: int = 1, offset: int = 0, limit: int = 50, api_key: str = "", refresh: bool = False):
        return JSONResponse(_aggregate_payload("openai", openai_log_dir(), min_messages, offset, limit, api_key, refresh))
