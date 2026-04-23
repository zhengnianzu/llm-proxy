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
_RECENT_INDEX_LIMIT_DEFAULT = 50000
_RECENT_INDEX_LIMIT_MAX = 100000


def get_recent_index_limit() -> int:
    raw = os.getenv("LOG_RECENT_INDEX_LIMIT", str(_RECENT_INDEX_LIMIT_DEFAULT)).strip()
    try:
        value = int(raw)
    except ValueError:
        value = _RECENT_INDEX_LIMIT_DEFAULT
    return max(1, min(value, _RECENT_INDEX_LIMIT_MAX))


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


def _tail_index_entries(index_path: Path, root: Path, limit: int) -> List[Dict[str, Any]]:
    if limit <= 0:
        return []

    try:
        with index_path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            pos = f.tell()
            if pos <= 0:
                return []

            buffer = b""
            needed_newlines = limit + 1
            chunk_size = 64 * 1024
            while pos > 0 and buffer.count(b"\n") < needed_newlines:
                read_size = min(chunk_size, pos)
                pos -= read_size
                f.seek(pos)
                buffer = f.read(read_size) + buffer
    except OSError:
        return []

    rows: List[Dict[str, Any]] = []
    for raw in buffer.splitlines()[-limit:]:
        line = raw.decode("utf-8", errors="ignore").strip()
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
    return rows


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
        "scanned": set(),
        "list_items": {},
        "agg_chains": OrderedDict(),
    }


_STATE_FILE = ".session_state.json"
_CHAINS_FILE = ".session_chain_keys.json"


def _state_file_path(root_dir: str) -> str:
    return os.path.join(root_dir, _STATE_FILE)


def _chains_file_path(root_dir: str) -> str:
    return os.path.join(root_dir, _CHAINS_FILE)


def _save_state_to_disk(state: Dict[str, Any]) -> None:
    """持久化计算状态到磁盘，供服务重启后恢复。"""
    root_dir = state["root_dir"]
    try:
        os.makedirs(root_dir, exist_ok=True)

        state_payload = {
            "byte_offset": state["byte_offset"],
            "line_count": state["line_count"],
            "list_items": state["list_items"],
            "scanned_list": list(state["scanned"]),
        }
        tmp = _state_file_path(root_dir) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state_payload, f, ensure_ascii=False)
        os.replace(tmp, _state_file_path(root_dir))

        chains_payload = {
            "byte_offset": state["byte_offset"],
            "chains": [[k, v] for k, v in state["agg_chains"].items()],
        }
        tmp = _chains_file_path(root_dir) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(chains_payload, f, ensure_ascii=False)
        os.replace(tmp, _chains_file_path(root_dir))
    except OSError:
        pass


def _load_state_from_disk(state: Dict[str, Any]) -> bool:
    """从磁盘恢复持久化状态。成功返回 True，失败返回 False（调用方应全量重建）。"""
    root_dir = state["root_dir"]
    state_path = _state_file_path(root_dir)
    chains_path = _chains_file_path(root_dir)

    if not os.path.isfile(state_path) or not os.path.isfile(chains_path):
        return False

    try:
        with open(state_path, "r", encoding="utf-8") as f:
            sp = json.load(f)
        with open(chains_path, "r", encoding="utf-8") as f:
            cp = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False

    if sp.get("byte_offset") != cp.get("byte_offset"):
        return False

    byte_offset = sp.get("byte_offset", 0)
    if not isinstance(byte_offset, int) or byte_offset < 0:
        return False

    state["byte_offset"] = byte_offset
    state["line_count"] = sp.get("line_count", 0)
    state["list_items"] = sp.get("list_items", {})
    state["scanned"] = set(sp.get("scanned_list", []))

    agg = OrderedDict()
    for pair in cp.get("chains", []):
        if isinstance(pair, list) and len(pair) == 2:
            agg[pair[0]] = pair[1]
    state["agg_chains"] = agg

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
    req_key = str(req_path.resolve())
    if req_key in state["scanned"]:
        return False

    data = _load_json(req_path)
    if not data:
        state["scanned"].add(req_key)
        return False

    messages = data.get("messages")
    if not isinstance(messages, list):
        state["scanned"].add(req_key)
        return False

    filename = req_path.name
    ts = str((index_entry or {}).get("ts") or filename.replace("-req.json", ""))
    model = str(data.get("model", "") or (index_entry or {}).get("model", "") or "")
    message_count = len(messages)
    api_key = str((index_entry or {}).get("api_key", "") or "")

    state["list_items"][filename] = {
        "filename": filename,
        "message_count": message_count,
        "model": model,
        "ts": ts,
        "api_key": api_key,
    }

    # Prefer pre-computed chain_key from index_entry; fallback to parsing messages
    if index_entry and index_entry.get("chain_key"):
        chain_key = index_entry["chain_key"]
    elif kind == "anthropic":
        chain_key = _anthropic_chain_key(messages)
    else:
        chain_key = _openai_chain_key(messages)

    # Prepend api_key so different keys aggregate separately
    chain_key = f"{api_key}||{chain_key}"

    q1_preview = (index_entry or {}).get("q1_preview", "")

    if kind == "anthropic":
        res_content = _extract_anthropic_res_content(req_path.with_name(filename.replace("-req.json", "-res.json")))
    else:
        res_content = _extract_openai_res_content(req_path.with_name(filename.replace("-req.json", "-res.json")))

    chain = state["agg_chains"].get(chain_key)
    full_message_count = message_count + (1 if res_content is not None else 0)

    # Detect new session: if current request has fewer messages than the
    # existing chain's best, it's a brand-new conversation starting over
    # with the same q1 — split into a separate chain.
    if chain is not None and message_count < chain["best_req_count"]:
        suffix = 1
        new_key = f"{chain_key}##session_{suffix}"
        while new_key in state["agg_chains"]:
            suffix += 1
            new_key = f"{chain_key}##session_{suffix}"
        chain_key = new_key
        chain = None

    if chain is None:
        chain = {
            "chain_id": len(state["agg_chains"]),
            "chain_key": chain_key if kind == "openai" else None,
            "messages": list(messages),
            "res_content": res_content,
            "file_count": 0,
            "message_count": full_message_count,
            "model": model,
            "first_ts": ts,
            "last_ts": ts,
            "best_req_count": message_count,
            "api_key": api_key,
            "q1_preview": q1_preview,
        }
        state["agg_chains"][chain_key] = chain
    else:
        if ts and (not chain["first_ts"] or ts < chain["first_ts"]):
            chain["first_ts"] = ts
        if ts and (not chain["last_ts"] or ts > chain["last_ts"]):
            chain["last_ts"] = ts
        has_better_payload = (
            message_count > chain["best_req_count"]
            or (message_count == chain["best_req_count"] and res_content is not None and chain["res_content"] is None)
        )
        if has_better_payload:
            chain["messages"] = list(messages)
            chain["res_content"] = res_content
            chain["best_req_count"] = message_count
            chain["message_count"] = full_message_count
            chain["model"] = model or chain["model"]

    chain["file_count"] += 1
    if full_message_count > chain["message_count"]:
        chain["message_count"] = full_message_count
    if model and not chain["model"]:
        chain["model"] = model

    state["scanned"].add(req_key)
    return True


def _refresh_state(kind: str, root_dir: str) -> None:
    state = _state(kind, root_dir)
    root = Path(root_dir)
    index_path = Path(state["index_path"])

    # Phase 1: 首次调用时尝试从磁盘恢复持久化状态
    if not state["initialized"]:
        if not _load_state_from_disk(state):
            state["list_items"].clear()
            state["agg_chains"].clear()
            state["scanned"].clear()
            state["byte_offset"] = 0
            state["line_count"] = 0

    # Phase 2: 增量读取 index.jsonl
    if index_path.is_file():
        rows, new_offset = _read_new_index_entries(index_path, root, state["byte_offset"])

        if new_offset == 0 and state["byte_offset"] > 0:
            # 文件被截断/轮转 — 全量重建
            state["list_items"].clear()
            state["agg_chains"].clear()
            state["scanned"].clear()
            state["byte_offset"] = 0
            state["line_count"] = 0
            rows = _tail_index_entries(index_path, root, get_recent_index_limit())
            try:
                new_offset = index_path.stat().st_size
            except OSError:
                new_offset = 0

        if rows:
            for row in rows:
                _process_req_row(kind, state, row["req_path"], row["entry"])
            state["line_count"] += len(rows)
            state["byte_offset"] = new_offset
            # Phase 3: 窗口裁剪 — 防止无限累积
            recent_limit = get_recent_index_limit()
            if state["line_count"] > int(recent_limit * 1.2):
                state["list_items"].clear()
                state["agg_chains"].clear()
                state["scanned"].clear()
                pruned_rows = _tail_index_entries(index_path, root, recent_limit)
                for row in pruned_rows:
                    _process_req_row(kind, state, row["req_path"], row["entry"])
                state["line_count"] = len(pruned_rows)
                try:
                    state["byte_offset"] = index_path.stat().st_size
                except OSError:
                    pass
            _save_state_to_disk(state)
        elif new_offset != state["byte_offset"]:
            state["byte_offset"] = new_offset
    else:
        # 无 index.jsonl — 降级为目录扫描（保持原有行为）
        if not state["initialized"]:
            for req_path in _collect_req_files(root):
                _process_req_row(kind, state, req_path)

    state["initialized"] = True


def _list_payload(kind: str, root_dir: str, min_messages: int, offset: int = 0, limit: int = 50) -> Dict[str, Any]:
    with _CACHE_LOCK:
        _refresh_state(kind, root_dir)
        current_state = _state(kind, root_dir)
        items = [
            {k: v for k, v in item.items() if k != "ts"}
            for item in current_state["list_items"].values()
            if item.get("message_count", 0) >= min_messages
        ]
        items.sort(
            key=lambda item: current_state["list_items"][item["filename"]].get("ts", item["filename"]),
            reverse=True,
        )
        total = len(items)
        paged = items[offset:offset + limit] if limit > 0 else items[offset:]
        return {"items": paged, "total": total}


def _aggregate_payload(kind: str, root_dir: str, min_messages: int, offset: int = 0, limit: int = 50) -> Dict[str, Any]:
    with _CACHE_LOCK:
        _refresh_state(kind, root_dir)
        current_state = _state(kind, root_dir)
        chains = []
        for chain in current_state["agg_chains"].values():
            full_messages = list(chain["messages"])
            if chain["res_content"] is not None:
                if kind == "anthropic":
                    full_messages.append({
                        "role": "assistant",
                        "content": chain["res_content"],
                        "_from_res": True,
                    })
                else:
                    full_messages.append({**chain["res_content"], "_from_res": True})
            if len(full_messages) < min_messages:
                continue
            payload = {
                "chain_id": chain["chain_id"],
                "first_time": _format_time(chain["first_ts"]),
                "last_time": _format_time(chain["last_ts"]),
                "file_count": chain["file_count"],
                "message_count": len(full_messages),
                "model": chain["model"],
                "messages": full_messages,
                "api_key": chain.get("api_key", ""),
                "q1_preview": chain.get("q1_preview", ""),
            }
            if kind == "openai":
                payload["chain_key"] = chain["chain_key"]
            chains.append(payload)

        chains.sort(key=lambda item: item["last_time"], reverse=True)
        total = len(chains)
        paged = chains[offset:offset + limit] if limit > 0 else chains[offset:]
        return {"items": paged, "total": total}


def register_log_routes(app: FastAPI) -> None:
    def anthropic_log_dir() -> str:
        return get_log_dir("logs_anthropic")

    def openai_log_dir() -> str:
        return get_log_dir("logs_openai")

    @app.get("/logs/anthropic/list")
    def logs_anthropic_list(min_messages: int = 10, offset: int = 0, limit: int = 50):
        return JSONResponse(_list_payload("anthropic", anthropic_log_dir(), min_messages, offset, limit))

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
    def logs_anthropic_aggregate(min_messages: int = 1, offset: int = 0, limit: int = 50):
        return JSONResponse(_aggregate_payload("anthropic", anthropic_log_dir(), min_messages, offset, limit))

    @app.get("/logs/openai/list")
    def logs_openai_list(min_messages: int = 10, offset: int = 0, limit: int = 50):
        return JSONResponse(_list_payload("openai", openai_log_dir(), min_messages, offset, limit))

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
    def logs_openai_aggregate(min_messages: int = 1, offset: int = 0, limit: int = 50):
        return JSONResponse(_aggregate_payload("openai", openai_log_dir(), min_messages, offset, limit))
