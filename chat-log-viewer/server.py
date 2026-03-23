"""
Chat Log Viewer — standalone backend
Usage:
    python server.py --dir /path/to/logs [--port 8080] [--host 0.0.0.0]

Recursively scans the given directory for JSON files that contain a `messages`
field.  Files are deduplicated by the first user query (keeps the longest
trajectory for each unique first-user-message).
"""

import argparse
import asyncio
import json
import logging
import re
import sys
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger("chat-log-viewer")

# ---------------------------------------------------------------------------
# CLI args (parsed before app creation so we can validate early)
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Chat Log Viewer server")
parser.add_argument(
    "--dir", "-d", required=True,
    help="Root directory to scan for JSON trace files"
)
parser.add_argument("--port", "-p", type=int, default=8080)
parser.add_argument("--host", default="0.0.0.0")
args = parser.parse_args()

ROOT_DIR = Path(args.dir).resolve()
if not ROOT_DIR.is_dir():
    sys.exit(f"[error] Directory not found: {ROOT_DIR}")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Chat Log Viewer")

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# In-memory cache: populated at startup, incrementally updated on /api/refresh
_cache: List[Dict] = []
# dedup map: first-user-query key -> cache entry (for incremental merge)
_best: OrderedDict = OrderedDict()
# set of abs_path strings already scanned
_scanned: Set[str] = set()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_messages(data: Any) -> Optional[List[dict]]:
    """Extract the messages list from a JSON payload (multiple formats)."""
    if isinstance(data, dict) and isinstance(data.get("messages"), list):
        return data["messages"]
    if isinstance(data, list) and data and isinstance(data[0], dict) and "role" in data[0]:
        return data
    return None


def get_first_user_text(messages: List[dict]) -> str:
    """Return the cleaned text of the first user message.

    Strips leading system-injected tag blocks such as:
      [Mon Mar 17 2025 10:30:00 GMT+0800 (China Standard Time)]
      [Subagent Context]
      ... etc.
    Matches the stripping logic used in chat-viewer.html.
    """
    for msg in messages:
        if msg.get("role") != "user":
            continue
        c = msg.get("content", "")
        if isinstance(c, str):
            text = c
        elif isinstance(c, list):
            parts = []
            for b in c:
                if isinstance(b, dict):
                    parts.append(b.get("text") or b.get("id") or "")
            text = "\n".join(p for p in parts if p)
        else:
            text = str(c)

        # Repeatedly strip leading noise until stable:
        #   - [tag] blocks: "[Wed 2026-03-11 15:20 GMT+8]", "[Subagent Context]", etc.
        #   - Sender metadata prefix + JSON block
        while True:
            prev = text
            # strip leading [tag] blocks
            text = re.sub(r"^\s*\[[^\]]*\]\s*", "", text)
            # strip "Sender (...): ```json {...} ```" prefix
            text = re.sub(r"^Sender\s*(?:\([^)]*\))?:\s*```json\s*\{[\s\S]*?\}\s*```\s*", "", text, flags=re.IGNORECASE)
            # strip bare "Sender: ..." line (fallback)
            text = re.sub(r"^Sender\s*(?:\([^)]*\))?:[^\n]*\n?", "", text, flags=re.IGNORECASE)
            if text == prev:
                break

        return text.strip()
    return ""


def _parse_file(abs_path: Path) -> Optional[Dict]:
    """Read and parse a single JSON file. Returns entry dict or None."""
    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    messages = extract_messages(data)
    if not messages:
        return None
    label = get_first_user_text(messages)
    # 若存在 response 字段（导出格式），将其计为额外一条 assistant 消息
    extra = 1 if (isinstance(data.get("response"), dict) and data["response"].get("content")) else 0
    return {
        "label": label,
        "label_short": label[:120] if label else "(empty)",
        "msg_count": len(messages) + extra,
        "model": data.get("model") or data.get("request", {}).get("model") or "",
        "rel_path": str(abs_path.relative_to(ROOT_DIR)),
    }


def scan_directory(incremental: bool = False) -> List[Dict]:
    """
    Scan ROOT_DIR for JSON files with a `messages` field.
    If incremental=True, only process files not yet in _scanned.
    Uses a thread pool for parallel file I/O.
    """
    global _best, _scanned

    # Collect paths to process
    all_paths = sorted(ROOT_DIR.rglob("*.json"))
    if incremental:
        new_paths = [p for p in all_paths if str(p) not in _scanned]
    else:
        new_paths = all_paths

    # Mark as scanned before processing (avoids double-scan on concurrent refresh)
    for p in new_paths:
        _scanned.add(str(p))

    # Parallel parse
    with ThreadPoolExecutor(max_workers=8) as executor:
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


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/list")
def api_list():
    """Return deduplicated file list from cache."""
    return JSONResponse(_cache)


@app.get("/api/refresh")
def api_refresh():
    """Incrementally scan for new files and update cache."""
    global _cache
    _cache = scan_directory(incremental=True)
    logger.info(f"Refreshed: {len(_cache)} conversations")
    return JSONResponse({"count": len(_cache)})


@app.get("/api/scan-stream")
async def api_scan_stream(mode: str = Query("incremental")):
    """
    Stream scan results as Server-Sent Events.

    mode=incremental  先推送已缓存条目，再增量扫描新文件
    mode=full         清空缓存，重新扫描全部文件
    """
    loop = asyncio.get_event_loop()
    q: asyncio.Queue = asyncio.Queue()

    def _worker():
        global _best, _scanned, _cache

        if mode == "full":
            _best = OrderedDict()
            _scanned.clear()
            new_paths = sorted(ROOT_DIR.rglob("*.json"))
        else:
            # incremental: emit cached items first, then scan new files
            for item in list(_cache):
                loop.call_soon_threadsafe(q.put_nowait, {"type": "item", "data": item})
            all_paths = sorted(ROOT_DIR.rglob("*.json"))
            new_paths = [p for p in all_paths if str(p) not in _scanned]

        for p in new_paths:
            _scanned.add(str(p))

        # local snapshot of _best for dedup decisions
        local_best: dict = {k: dict(v) for k, v in _best.items()}

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(_parse_file, p): p for p in new_paths}
            for future in as_completed(futures):
                entry = future.result()
                if entry is None:
                    continue
                key = entry["label"]
                is_replace = key in local_best
                if not is_replace or entry["msg_count"] > local_best[key]["msg_count"]:
                    local_best[key] = entry
                    _best[key] = entry
                    evt = {"type": "replace" if is_replace else "item", "data": entry}
                    loop.call_soon_threadsafe(q.put_nowait, evt)

        # rebuild global cache
        result = sorted(_best.values(), key=lambda x: x["rel_path"], reverse=True)
        for i, item in enumerate(result):
            item["id"] = i
        _cache = result
        loop.call_soon_threadsafe(q.put_nowait, None)  # sentinel

    threading.Thread(target=_worker, daemon=True).start()

    async def generate():
        while True:
            evt = await q.get()
            if evt is None:
                yield f"data: {json.dumps({'type': 'done', 'count': len(_cache)})}\n\n"
                break
            yield f"data: {json.dumps(evt)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/file")
def api_file(rel_path: str = Query(..., description="Relative path from root dir")):
    """Return the parsed JSON of a single file."""
    # security: prevent path traversal
    try:
        abs_path = (ROOT_DIR / rel_path).resolve()
        abs_path.relative_to(ROOT_DIR)  # raises ValueError if outside
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid path")

    if not abs_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # 若存在 response 字段（导出格式），将其追加为最后一条 assistant 消息
    response = data.get("response")
    if isinstance(response, dict) and response.get("content"):
        messages = data.get("messages")
        if isinstance(messages, list):
            data = dict(data)
            data["messages"] = messages + [{
                "role": "assistant",
                "content": response["content"],
            }]

    return JSONResponse(data)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"[chat-log-viewer] Root dir : {ROOT_DIR}")
    print(f"[chat-log-viewer] Listening on http://{args.host}:{args.port}")
    # 后台启动初次扫描，服务器立即可用
    threading.Thread(
        target=lambda: scan_directory(incremental=False),
        daemon=True,
    ).start()
    uvicorn.run(app, host=args.host, port=args.port)
