"""
Chat Log Viewer — standalone backend
Usage:
    python server.py --dir /path/to/logs [--port 8080] [--host 0.0.0.0]
    python server.py --session-dir /path/to/logs_session_anthropic [--port 8080]
"""

import argparse
import json
import logging
import os
import re
import sys
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger("chat-log-viewer")

# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Chat Log Viewer server")
parser.add_argument("--dir", "-d", default=None, help="Root directory to scan for JSON trace files")
parser.add_argument("--session-dir", "-s", default=None, help="Pre-exported session directory with index.json")
parser.add_argument("--port", "-p", type=int, default=8080)
parser.add_argument("--host", default="0.0.0.0")
args = parser.parse_args()

if not args.dir and not args.session_dir:
    sys.exit("[error] Must specify --dir or --session-dir")

ROOT_DIR: Optional[Path] = Path(args.dir).resolve() if args.dir else None
if ROOT_DIR and not ROOT_DIR.is_dir():
    sys.exit(f"[error] Directory not found: {ROOT_DIR}")

SESSION_DIR: Optional[Path] = Path(args.session_dir).resolve() if args.session_dir else None
if SESSION_DIR and not SESSION_DIR.is_dir():
    sys.exit(f"[error] Session directory not found: {SESSION_DIR}")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Chat Log Viewer")
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

_cache: List[Dict] = []
_best: OrderedDict = OrderedDict()
_scanned: Set[str] = set()
_index_line_count: int = 0  # 已处理的 index.jsonl 行数（用于增量刷新）

_SKIP_SUFFIXES = ("-res.json", "-headers.json")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_messages(data: Any) -> Optional[List[dict]]:
    if isinstance(data, dict) and isinstance(data.get("messages"), list):
        return data["messages"]
    if isinstance(data, list) and data and isinstance(data[0], dict) and "role" in data[0]:
        return data
    return None


def get_first_user_text(messages: List[dict]) -> str:
    for msg in messages:
        if msg.get("role") != "user":
            continue
        c = msg.get("content", "")
        if isinstance(c, str):
            text = c
        elif isinstance(c, list):
            parts = [b.get("text") or b.get("id") or "" for b in c if isinstance(b, dict)]
            text = "\n".join(p for p in parts if p)
        else:
            text = str(c)

        while True:
            prev = text
            text = re.sub(r"^\s*\[[^\]]*\]\s*", "", text)
            text = re.sub(r"^Sender\s*(?:\([^)]*\))?:\s*```json\s*\{[\s\S]*?\}\s*```\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"^Sender\s*(?:\([^)]*\))?:[^\n]*\n?", "", text, flags=re.IGNORECASE)
            if text == prev:
                break
        return text.strip()
    return ""


def _extract_res_content(res_path: Path) -> Optional[List[dict]]:
    """从同级 -res.json 中提取 assistant content 列表。"""
    try:
        with open(res_path, "r", encoding="utf-8") as f:
            res = json.load(f)
    except Exception:
        return None

    if res.get("type") == "anthropic_passthrough_sse_capture":
        blocks: dict = {}
        json_buf: dict = {}
        for chunk in res.get("chunks", []):
            t = chunk.get("type")
            if t == "content_block_start":
                idx = chunk.get("index", 0)
                cb = dict(chunk.get("content_block", {}))
                blocks[idx] = cb
                if cb.get("type") == "tool_use":
                    json_buf[idx] = ""
            elif t == "content_block_delta":
                idx = chunk.get("index", 0)
                delta = chunk.get("delta", {})
                if delta.get("type") == "text_delta":
                    blocks.setdefault(idx, {"type": "text", "text": ""})
                    blocks[idx]["text"] = blocks[idx].get("text", "") + delta.get("text", "")
                elif delta.get("type") == "input_json_delta":
                    json_buf[idx] = json_buf.get(idx, "") + delta.get("partial_json", "")
            elif t == "content_block_stop":
                idx = chunk.get("index", 0)
                if idx in json_buf:
                    try:
                        blocks[idx]["input"] = json.loads(json_buf[idx])
                    except json.JSONDecodeError:
                        blocks[idx]["input"] = json_buf[idx]
        return [blocks[i] for i in sorted(blocks)] if blocks else None
    else:
        content = (res.get("json") or {}).get("content")
        return content if isinstance(content, list) else None


def _parse_file(abs_path: Path) -> Optional[Dict]:
    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    messages = extract_messages(data)
    if not messages:
        return None
    label = get_first_user_text(messages)
    extra = 0
    if isinstance(data.get("response"), dict) and data["response"].get("content"):
        extra = 1
    elif abs_path.name.endswith("-req.json"):
        res_path = abs_path.with_name(abs_path.name[: -len("-req.json")] + "-res.json")
        if res_path.exists():
            extra = 1
    return {
        "label": label,
        "label_short": label[:120] if label else "(empty)",
        "msg_count": len(messages) + extra,
        "model": data.get("model") or data.get("request", {}).get("model") or "",
        "rel_path": str(abs_path.relative_to(ROOT_DIR)),
    }


def _collect_json_paths(root: Path) -> List[Path]:
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.endswith(".json") and not name.endswith(_SKIP_SUFFIXES):
                paths.append(Path(dirpath) / name)
    return paths


def _get_req_paths_from_index(root: Path, start_line: int = 0):
    """从 index.jsonl 读取 req 文件路径，从 start_line 行开始。
    返回 (new_paths: List[Path], total_lines: int)。"""
    index_path = root / "index.jsonl"
    paths: List[Path] = []
    total = 0
    with open(index_path, "r", encoding="utf-8") as f:
        for i, raw in enumerate(f):
            total += 1
            if i < start_line:
                continue
            line = raw.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                req_file = entry.get("req_file", "")
                if not req_file:
                    continue
                # req_file 是相对于项目根目录的路径（root 的上级目录）
                abs_path = (root.parent / req_file).resolve()
                if abs_path.is_file():
                    paths.append(abs_path)
            except json.JSONDecodeError:
                pass
    return paths, total


# ---------------------------------------------------------------------------
# Scan modes
# ---------------------------------------------------------------------------

def scan_directory(incremental: bool = False) -> List[Dict]:
    global _best, _scanned, _index_line_count

    index_path = ROOT_DIR / "index.jsonl"
    if index_path.exists():
        # 快速路径：从 index.jsonl 获取 req 文件列表，无需遍历目录
        start = _index_line_count if incremental else 0
        if not incremental:
            _best.clear()
            _scanned.clear()
        new_paths, total = _get_req_paths_from_index(ROOT_DIR, start)
        _index_line_count = total
        logger.info(f"[index] {'Incremental' if incremental else 'Full'} load: {len(new_paths)} new entries (total lines: {total})")
    else:
        # 降级路径：遍历目录
        all_paths = _collect_json_paths(ROOT_DIR)
        new_paths = [p for p in all_paths if str(p) not in _scanned] if incremental else all_paths

    for p in new_paths:
        _scanned.add(str(p))

    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(_parse_file, p): p for p in new_paths}
        for future in as_completed(futures):
            entry = future.result()
            if entry is None:
                continue
            key = entry["label"]
            if key not in _best or entry["msg_count"] > _best[key]["msg_count"]:
                _best[key] = entry

    result = sorted(_best.values(), key=lambda x: x["rel_path"], reverse=True)
    for i, item in enumerate(result):
        item["id"] = i
    return result


def scan_session_dir() -> List[Dict]:
    index_path = SESSION_DIR / "index.json"
    if not index_path.exists():
        logger.warning(f"[session] index.json not found: {index_path}")
        return []
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            entries = json.load(f)
    except Exception as e:
        logger.error(f"[session] Failed to load index.json: {e}")
        return []

    result = []
    for idx, entry in enumerate(entries):
        folder = entry.get("folder", "")
        latest = entry.get("latest_file", "")
        if not folder or not latest:
            continue
        result.append({
            "label": entry.get("q1", ""),
            "label_short": (entry.get("q1") or "")[:120],
            "msg_count": entry.get("msg_count", 0),
            "model": entry.get("model", ""),
            "rel_path": f"{folder}/{latest}",
            "id": idx,
        })
    logger.info(f"[session] Loaded {len(result)} sessions from index")
    return result


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/list")
def api_list():
    return JSONResponse(_cache)


@app.get("/api/refresh")
def api_refresh():
    global _cache
    _cache = scan_session_dir() if SESSION_DIR else scan_directory(incremental=True)
    logger.info(f"Refreshed: {len(_cache)} conversations")
    return JSONResponse({"count": len(_cache)})


@app.get("/api/file")
def api_file(rel_path: str = Query(...)):
    base_dir = SESSION_DIR if SESSION_DIR else ROOT_DIR
    try:
        abs_path = (base_dir / rel_path).resolve()
        abs_path.relative_to(base_dir)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid path")

    if not abs_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    messages = data.get("messages")
    if isinstance(messages, list):
        assistant_content = None
        response = data.get("response")
        if isinstance(response, dict) and response.get("content"):
            assistant_content = response["content"]
        elif abs_path.name.endswith("-req.json"):
            res_path = abs_path.with_name(abs_path.name[: -len("-req.json")] + "-res.json")
            if res_path.exists():
                assistant_content = _extract_res_content(res_path)
        if assistant_content:
            data = dict(data)
            data["messages"] = messages + [{"role": "assistant", "content": assistant_content}]

    return JSONResponse(data)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if SESSION_DIR:
        print(f"[chat-log-viewer] Mode       : session (--session-dir)")
        print(f"[chat-log-viewer] Session dir: {SESSION_DIR}")
        _cache = scan_session_dir()
    else:
        print(f"[chat-log-viewer] Mode       : scan (--dir)")
        print(f"[chat-log-viewer] Root dir   : {ROOT_DIR}")
        _cache = scan_directory()
    print(f"[chat-log-viewer] Listening on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
