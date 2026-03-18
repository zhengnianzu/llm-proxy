"""
Chat Log Viewer — standalone backend
Usage:
    python server.py --dir /path/to/logs [--port 8080] [--host 0.0.0.0]

Recursively scans the given directory for JSON files that contain a `messages`
field.  Files are deduplicated by the first user query (keeps the longest
trajectory for each unique first-user-message).
"""

import argparse
import json
import logging
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
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

# In-memory cache: populated at startup and on /api/refresh
_cache: List[Dict] = []


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


def scan_directory() -> List[Dict]:
    """
    Recursively scan ROOT_DIR for JSON files with a `messages` field.
    Deduplicate by first-user-query; keep only the longest trajectory per key.

    Returns a list of dicts sorted by label (first user query).
    """
    # key -> { label, msg_count, model, rel_path, abs_path }
    best: OrderedDict[str, dict] = OrderedDict()

    for abs_path in sorted(ROOT_DIR.rglob("*.json")):
        try:
            with open(abs_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        messages = extract_messages(data)
        if not messages:
            continue

        label = get_first_user_text(messages)
        key = label  # full text as dedup key

        rel_path = str(abs_path.relative_to(ROOT_DIR))
        entry = {
            "label": label,
            "label_short": label[:120] if label else "(empty)",
            "msg_count": len(messages),
            "model": data.get("model") or data.get("request", {}).get("model") or "",
            "rel_path": rel_path,
        }

        if key not in best or len(messages) > best[key]["msg_count"]:
            best[key] = entry

    result = sorted(best.values(), key=lambda x: x["rel_path"], reverse=True)
    # assign stable numeric ids after sort
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
    """Re-scan directory and update cache."""
    global _cache
    _cache = scan_directory()
    logger.info(f"Refreshed: {len(_cache)} conversations")
    return JSONResponse({"count": len(_cache)})


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

    return JSONResponse(data)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"[chat-log-viewer] Scanning: {ROOT_DIR}")
    _cache = scan_directory()
    print(f"[chat-log-viewer] Loaded {len(_cache)} conversations")
    print(f"[chat-log-viewer] Listening on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
