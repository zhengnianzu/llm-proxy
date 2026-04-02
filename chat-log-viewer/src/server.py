"""
Chat Log Viewer — standalone backend
Usage:
    python server.py --dir /path/to/logs /path/to/other_logs [--port 8080] [--host 0.0.0.0]
    python server.py --dirs /path/to/parents [--dirs /path/to/more_parents] [--port 8080]
    python server.py --session-dir /path/to/logs_session_a /path/to/logs_session_b [--port 8080]
    python server.py --session-dirs /path/to/session_parents [--port 8080]
"""

import argparse
import hashlib
import json
import logging
import os
import sys
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.utils.message_utils import (
    extract_messages,
    get_first_user_text,
    load_json,
    parse_response,
)

logger = logging.getLogger("chat-log-viewer")

# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Chat Log Viewer server")
parser.add_argument("--dir", "-d", action="append", nargs="+", default=None,
                    help="Root directory to scan for JSON trace files; can be specified multiple times")
parser.add_argument("--dirs", action="append", nargs="+", default=None,
                    help="Parent directory whose immediate subdirectories are treated as scan roots; can be specified multiple times")
parser.add_argument("--session-dir", "-s", action="append", nargs="+", default=None,
                    help="Pre-exported session directory with index.json; can be specified multiple times")
parser.add_argument("--session-dirs", action="append", nargs="+", default=None,
                    help="Parent directory whose immediate subdirectories are treated as session roots; can be specified multiple times")
parser.add_argument("--port", "-p", type=int, default=8080)
parser.add_argument("--host", default="0.0.0.0")
args = parser.parse_args()

has_scan_args = bool(args.dir or args.dirs)
has_session_args = bool(args.session_dir or args.session_dirs)
if not has_scan_args and not has_session_args:
    sys.exit("[error] Must specify --dir, --dirs, --session-dir or --session-dirs")
if has_scan_args and has_session_args:
    sys.exit("[error] Scan mode and session mode cannot be used together")

ROOT_DIRS: List[Path] = [Path(p).resolve() for group in (args.dir or []) for p in group]
for root_dir in ROOT_DIRS:
    if not root_dir.is_dir():
        sys.exit(f"[error] Directory not found: {root_dir}")
ROOT_PARENT_DIRS: List[Path] = [Path(p).resolve() for group in (args.dirs or []) for p in group]
for parent_dir in ROOT_PARENT_DIRS:
    if not parent_dir.is_dir():
        sys.exit(f"[error] Parent directory not found: {parent_dir}")

SESSION_DIRS: List[Path] = [Path(p).resolve() for group in (args.session_dir or []) for p in group]
for session_dir in SESSION_DIRS:
    if not session_dir.is_dir():
        sys.exit(f"[error] Session directory not found: {session_dir}")
SESSION_PARENT_DIRS: List[Path] = [Path(p).resolve() for group in (args.session_dirs or []) for p in group]
for parent_dir in SESSION_PARENT_DIRS:
    if not parent_dir.is_dir():
        sys.exit(f"[error] Session parent directory not found: {parent_dir}")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Chat Log Viewer")
STATIC_DIR = Path(__file__).parent.parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

_cache: Dict[str, List[Dict]] = {}
_best: Dict[str, OrderedDict] = {}
_scanned: Dict[str, Set[str]] = {}
_index_line_count: Dict[str, int] = {}  # dir_key -> 已处理的 index.jsonl 行数（用于增量刷新）

_SKIP_SUFFIXES = ("-res.json", "-headers.json")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_res_content(res_path: Path) -> Optional[List[dict]]:
    """从同级 -res.json 中提取 assistant content 列表。"""
    try:
        res = load_json(res_path)
    except Exception:
        return None
    content = parse_response(res).get("content")
    return content if isinstance(content, list) else None


def _dir_key(root: Path) -> str:
    digest = hashlib.md5(str(root).encode("utf-8")).hexdigest()[:12]
    return f"dir-{digest}"


def _dir_label(root: Path) -> str:
    return f"{root.name or root} - {root}"


def _session_key(root: Path) -> str:
    digest = hashlib.md5(str(root).encode("utf-8")).hexdigest()[:12]
    return f"session-{digest}"


def _session_label(root: Path) -> str:
    return f"{root.name or root} - {root}"


def _discover_root_dirs() -> List[Path]:
    roots: List[Path] = []
    seen: Set[Path] = set()

    for root in ROOT_DIRS:
        if root not in seen and root.is_dir():
            roots.append(root)
            seen.add(root)

    for parent in ROOT_PARENT_DIRS:
        try:
            children = sorted(p.resolve() for p in parent.iterdir() if p.is_dir())
        except Exception:
            continue
        for child in children:
            if child not in seen:
                roots.append(child)
                seen.add(child)

    return roots


def _sync_root_dirs() -> List[Path]:
    roots = _discover_root_dirs()
    valid_keys = {_dir_key(root) for root in roots}

    for stale_key in list(_cache.keys()):
        if stale_key not in valid_keys:
            _cache.pop(stale_key, None)
            _best.pop(stale_key, None)
            _scanned.pop(stale_key, None)
            _index_line_count.pop(stale_key, None)

    for root in roots:
        key = _dir_key(root)
        if key not in _cache:
            logger.info(f"[dirs] New root detected: {root}")
            _cache[key] = scan_directory(root)

    return roots


def _discover_session_dirs() -> List[Path]:
    roots: List[Path] = []
    seen: Set[Path] = set()

    for root in SESSION_DIRS:
        if root not in seen and root.is_dir():
            roots.append(root)
            seen.add(root)

    for parent in SESSION_PARENT_DIRS:
        try:
            children = sorted(p.resolve() for p in parent.iterdir() if p.is_dir())
        except Exception:
            continue
        for child in children:
            if child not in seen:
                roots.append(child)
                seen.add(child)

    return roots


def _sync_session_dirs() -> List[Path]:
    roots = _discover_session_dirs()
    valid_keys = {_session_key(root) for root in roots}

    for stale_key in list(_cache.keys()):
        if stale_key.startswith("session-") and stale_key not in valid_keys:
            _cache.pop(stale_key, None)

    for root in roots:
        key = _session_key(root)
        if key not in _cache:
            logger.info(f"[session-dirs] New root detected: {root}")
            _cache[key] = scan_session_dir(root)

    return roots


def _get_root_by_key(dir_key: Optional[str]) -> Path:
    if has_session_args:
        raise HTTPException(status_code=400, detail="Directory selection is not available in session mode")
    roots = _sync_root_dirs()
    if not roots:
        raise HTTPException(status_code=500, detail="No scan directories configured")
    if dir_key is None:
        if len(roots) == 1:
            return roots[0]
        raise HTTPException(status_code=400, detail="Missing dir")
    for root in roots:
        if _dir_key(root) == dir_key:
            return root
    raise HTTPException(status_code=404, detail="Directory not found")


def _get_session_root_by_key(dir_key: Optional[str]) -> Path:
    if not has_session_args:
        raise HTTPException(status_code=400, detail="Session directory selection is not available in scan mode")
    roots = _sync_session_dirs()
    if not roots:
        raise HTTPException(status_code=500, detail="No session directories configured")
    if dir_key is None:
        if len(roots) == 1:
            return roots[0]
        raise HTTPException(status_code=400, detail="Missing dir")
    for root in roots:
        if _session_key(root) == dir_key:
            return root
    raise HTTPException(status_code=404, detail="Directory not found")


def _parse_file(abs_path: Path, root_dir: Path, dir_key: str, dir_label: str) -> Optional[Dict]:
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
    try:
        rel_path = str(abs_path.relative_to(root_dir))
    except ValueError:
        rel_path = str(abs_path)
    return {
        "label": label,
        "label_short": label[:120] if label else "(empty)",
        "msg_count": len(messages) + extra,
        "model": data.get("model") or data.get("request", {}).get("model") or "",
        "rel_path": rel_path,
        "dir_key": dir_key,
        "dir_label": dir_label,
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
                # req_file 可能是绝对路径，也可能是相对于 root.parent 的相对路径
                rf = Path(req_file)
                abs_path = rf.resolve() if rf.is_absolute() else (root.parent / req_file).resolve()
                if abs_path.is_file():
                    paths.append(abs_path)
            except json.JSONDecodeError:
                pass
    return paths, total


# ---------------------------------------------------------------------------
# Scan modes
# ---------------------------------------------------------------------------

def scan_directory(root_dir: Path, incremental: bool = False) -> List[Dict]:
    dir_key = _dir_key(root_dir)
    dir_label = _dir_label(root_dir)
    best = _best.setdefault(dir_key, OrderedDict())
    scanned = _scanned.setdefault(dir_key, set())

    index_path = root_dir / "index.jsonl"
    if index_path.exists():
        # 快速路径：从 index.jsonl 获取 req 文件列表，无需遍历目录
        start = _index_line_count.get(dir_key, 0) if incremental else 0
        if not incremental:
            best.clear()
            scanned.clear()
        new_paths, total = _get_req_paths_from_index(root_dir, start)
        _index_line_count[dir_key] = total
        logger.info(f"[index] {dir_key}: {'Incremental' if incremental else 'Full'} load: {len(new_paths)} new entries (total lines: {total})")
    else:
        # 降级路径：遍历目录
        all_paths = _collect_json_paths(root_dir)
        new_paths = [p for p in all_paths if str(p) not in scanned] if incremental else all_paths

    for p in new_paths:
        scanned.add(str(p))

    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {
            executor.submit(_parse_file, p, root_dir, dir_key, dir_label): p
            for p in new_paths
        }
        for future in as_completed(futures):
            entry = future.result()
            if entry is None:
                continue
            key = entry["label"]
            if key not in best or entry["msg_count"] > best[key]["msg_count"]:
                best[key] = entry

    result = sorted(best.values(), key=lambda x: x["rel_path"], reverse=True)
    for i, item in enumerate(result):
        item["id"] = i
    return result


def scan_session_dir(session_dir: Path) -> List[Dict]:
    index_path = session_dir / "index.json"
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
            "dir_key": _session_key(session_dir),
            "dir_label": _session_label(session_dir),
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
def api_list(dir: Optional[str] = Query(default=None)):
    if has_session_args:
        session_dir = _get_session_root_by_key(dir)
        return JSONResponse(_cache.get(_session_key(session_dir), []))
    root_dir = _get_root_by_key(dir)
    return JSONResponse(_cache.get(_dir_key(root_dir), []))


@app.get("/api/refresh")
def api_refresh(dir: Optional[str] = Query(default=None)):
    global _cache
    if has_session_args:
        _sync_session_dirs()
        session_dir = _get_session_root_by_key(dir)
        session_key = _session_key(session_dir)
        _cache[session_key] = scan_session_dir(session_dir)
        logger.info(f"Refreshed {session_key}: {len(_cache[session_key])} conversations")
        return JSONResponse({"count": len(_cache[session_key]), "dir": session_key})

    root_dir = _get_root_by_key(dir)
    dir_key = _dir_key(root_dir)
    _cache[dir_key] = scan_directory(root_dir, incremental=True)
    logger.info(f"Refreshed {dir_key}: {len(_cache[dir_key])} conversations")
    return JSONResponse({"count": len(_cache[dir_key]), "dir": dir_key})


@app.get("/api/file")
def api_file(rel_path: str = Query(...), dir: Optional[str] = Query(default=None)):
    base_dir = _get_session_root_by_key(dir) if has_session_args else _get_root_by_key(dir)
    try:
        abs_path = (base_dir / rel_path).resolve()
        # 兼容 index.jsonl 中 req_file 指向其他已注册根目录的情况
        all_roots = ROOT_DIRS + [d for p in ROOT_PARENT_DIRS for d in p.iterdir() if d.is_dir()]
        if not any(abs_path == r or r in abs_path.parents for r in [base_dir] + all_roots):
            raise ValueError("path not under any root")
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


@app.get("/api/config")
def api_config():
    if has_session_args:
        roots = _sync_session_dirs()
        return JSONResponse({
            "mode": "session",
            "dirs": [
                {
                    "key": _session_key(root),
                    "label": _session_label(root),
                    "path": str(root),
                }
                for root in roots
            ],
            "active_dir": _session_key(roots[0]) if len(roots) == 1 else None,
        })
    roots = _sync_root_dirs()
    dirs = [
        {
            "key": _dir_key(root),
            "label": _dir_label(root),
            "path": str(root),
        }
        for root in roots
    ]
    return JSONResponse({
        "mode": "scan",
        "dirs": dirs,
        "active_dir": dirs[0]["key"] if len(dirs) == 1 else None,
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if has_session_args:
        mode_desc = []
        if SESSION_DIRS:
            mode_desc.append("--session-dir")
        if SESSION_PARENT_DIRS:
            mode_desc.append("--session-dirs")
        print(f"[chat-log-viewer] Mode       : session ({', '.join(mode_desc)})")
        for parent_dir in SESSION_PARENT_DIRS:
            print(f"[chat-log-viewer] Session parent: {parent_dir}")
        for session_dir in _sync_session_dirs():
            print(f"[chat-log-viewer] Session dir: {_session_key(session_dir)} -> {session_dir}")
    else:
        mode_desc = []
        if ROOT_DIRS:
            mode_desc.append("--dir")
        if ROOT_PARENT_DIRS:
            mode_desc.append("--dirs")
        print(f"[chat-log-viewer] Mode       : scan ({', '.join(mode_desc)})")
        for parent_dir in ROOT_PARENT_DIRS:
            print(f"[chat-log-viewer] Parent dir : {parent_dir}")
        for root_dir in _sync_root_dirs():
            print(f"[chat-log-viewer] Root dir   : {_dir_key(root_dir)} -> {root_dir}")
    print(f"[chat-log-viewer] Listening on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
