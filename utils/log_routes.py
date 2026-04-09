"""
/logs/* 路由：列表、聚合、单文件读取（Anthropic + OpenAI）
优先使用 index.jsonl 做增量刷新；缺失时降级为目录扫描。
"""

import glob
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


def _index_entries(index_path: Path, root: Path, start_line: int = 0) -> Tuple[List[Dict[str, Any]], int]:
    rows: List[Dict[str, Any]] = []
    total = 0
    with index_path.open("r", encoding="utf-8") as f:
        for i, raw in enumerate(f):
            total += 1
            if i < start_line:
                continue
            line = raw.strip()
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
    return rows, total


def _collect_req_files(root: Path) -> List[Path]:
    return sorted(root.glob("*-req.json"))


def _extract_anthropic_res_content(res_path: Path):
    data = _load_json(res_path)
    if not data:
        return None
    if isinstance(data.get("json"), dict):
        msg = data["json"]
        if msg.get("role") == "assistant":
            return msg.get("content")
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
        "scanned": set(),
        "list_items": {},
        "agg_chains": OrderedDict(),
    }


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

    state["list_items"][filename] = {
        "filename": filename,
        "message_count": message_count,
        "model": model,
        "ts": ts,
    }

    if kind == "anthropic":
        chain_key = _anthropic_chain_key(messages)
        res_content = _extract_anthropic_res_content(req_path.with_name(filename.replace("-req.json", "-res.json")))
    else:
        chain_key = _openai_chain_key(messages)
        res_content = _extract_openai_res_content(req_path.with_name(filename.replace("-req.json", "-res.json")))

    chain = state["agg_chains"].get(chain_key)
    full_message_count = message_count + (1 if res_content is not None else 0)
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

    if not state["initialized"]:
        state["list_items"].clear()
        state["agg_chains"].clear()
        state["scanned"].clear()
        state["line_count"] = 0

    if index_path.is_file():
        start_line = state["line_count"] if state["initialized"] else 0
        rows, total_lines = _index_entries(index_path, root, start_line=start_line)
        state["line_count"] = total_lines
        for row in rows:
            _process_req_row(kind, state, row["req_path"], row["entry"])
    else:
        for req_path in _collect_req_files(root):
            _process_req_row(kind, state, req_path)

    state["initialized"] = True


def _list_payload(kind: str, root_dir: str, min_messages: int) -> List[Dict[str, Any]]:
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
        return items


def _aggregate_payload(kind: str, root_dir: str, min_messages: int) -> List[Dict[str, Any]]:
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
            }
            if kind == "openai":
                payload["chain_key"] = chain["chain_key"]
            chains.append(payload)

        chains.sort(key=lambda item: item["last_time"], reverse=True)
        return chains


def register_log_routes(app: FastAPI) -> None:
    def anthropic_log_dir() -> str:
        return get_log_dir("logs_anthropic")

    def openai_log_dir() -> str:
        return get_log_dir("logs_openai")

    @app.get("/logs/anthropic/list")
    def logs_anthropic_list(min_messages: int = 10):
        return JSONResponse(_list_payload("anthropic", anthropic_log_dir(), min_messages))

    @app.get("/logs/anthropic/file")
    def logs_anthropic_file(filename: str):
        if not filename.endswith("-req.json") or "/" in filename or "\\" in filename or ".." in filename:
            return JSONResponse({"error": "invalid filename"}, status_code=400)
        path = os.path.join(anthropic_log_dir(), filename)
        if not os.path.isfile(path):
            return JSONResponse({"error": "file not found"}, status_code=404)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return JSONResponse(data)

    @app.get("/logs/anthropic/aggregate")
    def logs_anthropic_aggregate(min_messages: int = 1):
        return JSONResponse(_aggregate_payload("anthropic", anthropic_log_dir(), min_messages))

    @app.get("/logs/openai/list")
    def logs_openai_list(min_messages: int = 10):
        return JSONResponse(_list_payload("openai", openai_log_dir(), min_messages))

    @app.get("/logs/openai/file")
    def logs_openai_file(filename: str):
        if not filename.endswith("-req.json") or "/" in filename or "\\" in filename or ".." in filename:
            return JSONResponse({"error": "invalid filename"}, status_code=400)
        path = os.path.join(openai_log_dir(), filename)
        if not os.path.isfile(path):
            return JSONResponse({"error": "file not found"}, status_code=404)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return JSONResponse(data)

    @app.get("/logs/openai/aggregate")
    def logs_openai_aggregate(min_messages: int = 1):
        return JSONResponse(_aggregate_payload("openai", openai_log_dir(), min_messages))
