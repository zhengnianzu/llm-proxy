"""
Chat Log Viewer — standalone backend
Usage:
    python server.py --dir /path/to/logs /path/to/other_logs [--port 8080] [--host 0.0.0.0]
    python server.py --dirs /path/to/parents [--dirs /path/to/more_parents] [--port 8080]
    python server.py --session-dir /path/to/logs_session_a /path/to/logs_session_b [--port 8080]
    python server.py --session-dir /path/to/logs_session_a --report-dir /path/to/report_a [--port 8080]
    python server.py --session-dirs /path/to/session_parents [--port 8080]
"""

import argparse
import dataclasses
import hashlib
import json
import logging
import os
import sys
from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from src.utils.message_utils import (
    extract_messages,
    get_first_user_text,
    load_json,
    parse_response,
)
from src.utils.trajectory_utils import build_session_trajectory, build_trajectory_diff

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
parser.add_argument("--report-dir", action="append", nargs="+", default=None,
                    help="Analysis/report directory paired with --session-dir; defaults to <session_dir>/stat")
parser.add_argument("--label", action="append", nargs="+", default=None,
                    help="Display label paired with --session-dir; defaults to directory name")
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
REPORT_DIRS: List[Path] = [Path(p).resolve() for group in (args.report_dir or []) for p in group]
LABELS: List[str] = [label for group in (args.label or []) for label in group]
if REPORT_DIRS and not SESSION_DIRS:
    sys.exit("[error] --report-dir requires --session-dir")
if len(REPORT_DIRS) > len(SESSION_DIRS):
    sys.exit("[error] --report-dir count cannot exceed --session-dir count")
if len(LABELS) > len(SESSION_DIRS):
    sys.exit("[error] --label count cannot exceed --session-dir count")

# ---------------------------------------------------------------------------
# Source registry: each entry is a distinct source (even if session_dir is the same)
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class SourceEntry:
    key: str
    label: str
    session_dir: Path
    report_dir: Path

SOURCE_ENTRIES: List[SourceEntry] = []
_SOURCE_BY_KEY: Dict[str, SourceEntry] = {}

def _build_source_entries():
    """Build source entries from CLI args. Called once at startup."""
    SOURCE_ENTRIES.clear()
    _SOURCE_BY_KEY.clear()
    for idx, session_dir in enumerate(SESSION_DIRS):
        report_dir = REPORT_DIRS[idx] if idx < len(REPORT_DIRS) else (session_dir / "stat").resolve()
        label = LABELS[idx] if idx < len(LABELS) else f"{session_dir.name or session_dir}"
        # Use index in key to allow duplicate paths with different labels
        digest = hashlib.md5(f"{idx}:{session_dir}".encode("utf-8")).hexdigest()[:12]
        key = f"session-{digest}"
        entry = SourceEntry(key=key, label=label, session_dir=session_dir, report_dir=report_dir)
        SOURCE_ENTRIES.append(entry)
        _SOURCE_BY_KEY[key] = entry

if has_session_args:
    _build_source_entries()

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
_report_meta_cache: Dict[str, Dict[str, Any]] = {}
_analysis_summary_cache: Dict[str, Dict[str, Any]] = {}

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
    """Legacy fallback — prefer using SourceEntry.key directly."""
    digest = hashlib.md5(str(root).encode("utf-8")).hexdigest()[:12]
    return f"session-{digest}"


def _session_label(root: Path) -> str:
    return f"{root.name or root} - {root}"


def _source_report_dir(source: SourceEntry) -> Path:
    return source.report_dir


def _source_report_meta(source: SourceEntry) -> Dict[str, Any]:
    report_dir = _source_report_dir(source)
    meta = {
        "report_dir": str(report_dir),
        "has_analysis": (report_dir / "session_analysis.json").is_file(),
        "has_report_html": (report_dir / "session_report.html").is_file(),
        "has_report_md": (report_dir / "session_report.md").is_file(),
        "has_report_xlsx": (report_dir / "session_report.xlsx").is_file(),
    }
    _report_meta_cache[source.key] = meta
    return meta


def _get_report_file(source: SourceEntry, name: str) -> Path:
    report_dir = _source_report_dir(source)
    path = (report_dir / name).resolve()
    if path.parent != report_dir.resolve():
        raise HTTPException(status_code=400, detail="Invalid report path")
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Report file not found")
    return path


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


def _discover_session_sources() -> List[SourceEntry]:
    """Return all configured source entries, including dynamic ones from SESSION_PARENT_DIRS."""
    sources: List[SourceEntry] = list(SOURCE_ENTRIES)
    seen_keys: Set[str] = {s.key for s in sources}

    for parent in SESSION_PARENT_DIRS:
        try:
            children = sorted(p.resolve() for p in parent.iterdir() if p.is_dir())
        except Exception:
            continue
        for child in children:
            # Dynamic sources from parent dirs use path-based key (legacy behavior)
            digest = hashlib.md5(str(child).encode("utf-8")).hexdigest()[:12]
            key = f"session-{digest}"
            if key not in seen_keys:
                sources.append(SourceEntry(
                    key=key,
                    label=f"{child.name or child}",
                    session_dir=child,
                    report_dir=(child / "stat").resolve(),
                ))
                seen_keys.add(key)

    return sources


def _sync_session_dirs() -> List[SourceEntry]:
    sources = _discover_session_sources()
    valid_keys = {s.key for s in sources}

    for stale_key in list(_cache.keys()):
        if stale_key.startswith("session-") and stale_key not in valid_keys:
            _cache.pop(stale_key, None)
            _report_meta_cache.pop(stale_key, None)
            _analysis_summary_cache.pop(stale_key, None)

    for source in sources:
        if source.key not in _cache:
            logger.info(f"[session-dirs] New source detected: {source.label} -> {source.session_dir}")
            _cache[source.key] = scan_session_dir(source)
            _analysis_summary_cache[source.key] = _build_analysis_summary(_cache[source.key])

    return sources


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


def _get_source_by_key(dir_key: Optional[str]) -> SourceEntry:
    if not has_session_args:
        raise HTTPException(status_code=400, detail="Session directory selection is not available in scan mode")
    sources = _sync_session_dirs()
    if not sources:
        raise HTTPException(status_code=500, detail="No session directories configured")
    if dir_key is None:
        if len(sources) == 1:
            return sources[0]
        raise HTTPException(status_code=400, detail="Missing dir")
    for source in sources:
        if source.key == dir_key:
            return source
    raise HTTPException(status_code=404, detail="Directory not found")


def _load_session_analysis(report_dir: Path) -> Dict[str, Dict[str, Any]]:
    analysis_path = report_dir / "session_analysis.json"
    if not analysis_path.exists():
        return {}
    try:
        payload = json.loads(analysis_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"[analysis] Failed to load {analysis_path}: {e}")
        return {}

    sessions = payload.get("sessions") if isinstance(payload, dict) else payload
    if not isinstance(sessions, list):
        logger.warning(f"[analysis] Invalid payload format: {analysis_path}")
        return {}

    result: Dict[str, Dict[str, Any]] = {}
    for item in sessions:
        if not isinstance(item, dict):
            continue
        session = item.get("session")
        if not session:
            continue
        result[str(session)] = item
    return result


def _pct(values: List[float], p: int) -> float:
    if not values:
        return 0.0
    sv = sorted(values)
    idx = min(int(len(sv) * p / 100), len(sv) - 1)
    return float(sv[idx])


def _bucket_counts(values: List[float], buckets: List[tuple]) -> List[Dict[str, Any]]:
    total = len(values)
    rows: List[Dict[str, Any]] = []
    for label, lo, hi in buckets:
        count = sum(1 for v in values if v is not None and lo <= v < hi)
        rows.append({
            "label": label,
            "count": count,
            "pct": round(count / total * 100, 1) if total else 0.0,
        })
    return rows


def _build_analysis_summary(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    analyzed = [item for item in items if item.get("analysis_available")]
    total_sessions = len(items)
    analysis_sessions = len(analyzed)
    turns_vals = [int(item["user_turns"]) for item in analyzed if item.get("user_turns") is not None]
    tool_use_vals = [int(item["tool_use_count"]) for item in analyzed if item.get("tool_use_count") is not None]
    rate_vals = [float(item["tool_success_rate"]) for item in analyzed if item.get("tool_success_rate") is not None]
    duration_vals = [float(item["duration_s"]) for item in analyzed if item.get("duration_s") is not None]
    model_dist = Counter((item.get("model") or "(未知)") for item in analyzed)
    completed_dist = Counter((item.get("completed") or "(未知)") for item in analyzed)
    fail_tool_counter: Counter = Counter()
    skills_counter: Counter = Counter()
    for item in analyzed:
        fail_tool_counter.update(item.get("tool_fail_detail") or {})
        skills_counter.update(item.get("skills_used") or {})

    ok_count = sum(1 for item in analyzed if item.get("completed") == 0)

    return {
        "total_sessions": total_sessions,
        "analysis_sessions": analysis_sessions,
        "kpis": {
            "quality_passed": ok_count,
            "quality_passed_rate": round(ok_count / analysis_sessions * 100, 1) if analysis_sessions else None,
            "avg_tool_success_rate": round(sum(rate_vals) / len(rate_vals), 1) if rate_vals else None,
            "p90_duration_s": round(_pct(duration_vals, 90), 1) if duration_vals else None,
            "total_skills": len(skills_counter),
        },
        "charts": {
            "user_turns_hist": _bucket_counts(turns_vals, [
                ("1", 1, 2), ("2-3", 2, 4), ("4-7", 4, 8), ("8-15", 8, 16), (">15", 16, 10 ** 9),
            ]),
            "tool_use_hist": _bucket_counts(tool_use_vals, [
                ("1-5", 1, 6), ("6-15", 6, 16), ("16-30", 16, 31), ("31-50", 31, 51), (">50", 51, 10 ** 9),
            ]),
            "tool_success_rate_hist": _bucket_counts(rate_vals, [
                ("0-50%", 0, 50), ("50-80%", 50, 80), ("80-95%", 80, 95), ("95-99%", 95, 100), ("100%", 100, 101),
            ]),
            "duration_hist": _bucket_counts(duration_vals, [
                ("<1min", 0, 60), ("1-5min", 60, 300), ("5-15min", 300, 900), ("15-30min", 900, 1800), (">30min", 1800, 10 ** 9),
            ]),
            "model_dist": [
                {"label": label, "count": count, "pct": round(count / analysis_sessions * 100, 1) if analysis_sessions else 0.0}
                for label, count in model_dist.most_common(8)
            ],
            "completed_dist": [
                {"label": str(label), "count": count, "pct": round(count / analysis_sessions * 100, 1) if analysis_sessions else 0.0}
                for label, count in completed_dist.most_common(8)
            ],
            "tool_fail_top": [
                {"label": label, "count": count}
                for label, count in fail_tool_counter.most_common(10)
            ],
            "skills_dist": [
                {"label": label, "count": count}
                for label, count in skills_counter.most_common(15)
            ],
        },
    }


def _build_source_summary(source: SourceEntry, items: List[Dict[str, Any]]) -> Dict[str, Any]:
    analyzed = [item for item in items if item.get("analysis_available")]
    total_sessions = len(items)
    analysis_sessions = len(analyzed)
    ok_count = sum(1 for item in analyzed if item.get("completed") == 0)
    success_rate = round(ok_count / analysis_sessions * 100, 1) if analysis_sessions else None
    rate_vals = [float(item["tool_success_rate"]) for item in analyzed if item.get("tool_success_rate") is not None]
    duration_vals = [float(item["duration_s"]) for item in analyzed if item.get("duration_s") is not None]
    skills_set: set = set()
    for item in analyzed:
        skills_set.update((item.get("skills_used") or {}).keys())
    return {
        "key": source.key,
        "label": source.label,
        "session_dir": str(source.session_dir),
        "report_dir": str(source.report_dir),
        "total_sessions": total_sessions,
        "analysis_sessions": analysis_sessions,
        "success_rate": success_rate,
        "avg_tool_success_rate": round(sum(rate_vals) / len(rate_vals), 1) if rate_vals else None,
        "quality_passed": ok_count,
        "total_skills": len(skills_set),
        "p90_duration_s": round(_pct(duration_vals, 90), 1) if duration_vals else None,
        "has_analysis": any(item.get("analysis_available") for item in items),
        "has_report_html": (source.report_dir / "session_report.html").is_file(),
    }


def _merge_session_analysis(base_item: Dict[str, Any], analysis_item: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = dict(base_item)
    merged["session"] = str(base_item.get("rel_path", "")).split("/", 1)[0] if base_item.get("rel_path") else ""
    merged["q1"] = base_item.get("label", "")
    merged["analysis_available"] = bool(analysis_item)
    if not analysis_item:
        return merged

    for field in (
        "start_time",
        "end_time",
        "duration_s",
        "api_call_count",
        "api_errors",
        "user_turns",
        "total_messages",
        "tool_use_count",
        "tool_result_count",
        "tool_success",
        "tool_fail_flag",
        "tool_fail_keyword",
        "tool_fail_total",
        "tool_success_rate",
        "completed",
        "completed_note",
        "tool_use_detail",
        "tool_success_detail",
        "tool_fail_detail",
        "skills_used",
    ):
        if field in analysis_item:
            merged[field] = analysis_item[field]
    if analysis_item.get("q1") and not merged.get("label"):
        merged["label"] = analysis_item["q1"]
        merged["label_short"] = str(analysis_item["q1"])[:120]
        merged["q1"] = analysis_item["q1"]
    return merged


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


def scan_session_dir(source: SourceEntry) -> List[Dict]:
    session_dir = source.session_dir
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

    analysis_by_session = _load_session_analysis(_source_report_dir(source))
    result = []
    for idx, entry in enumerate(entries):
        folder = entry.get("folder", "")
        latest = entry.get("latest_file", "")
        if not folder or not latest:
            continue
        item = {
            "label": entry.get("q1", ""),
            "label_short": (entry.get("q1") or "")[:120],
            "msg_count": entry.get("msg_count", 0),
            "model": entry.get("model", ""),
            "rel_path": f"{folder}/{latest}",
            "dir_key": source.key,
            "dir_label": source.label,
            "id": idx,
        }
        result.append(_merge_session_analysis(item, analysis_by_session.get(folder)))
    logger.info(
        f"[session] Loaded {len(result)} sessions from index"
        f" (analysis matched: {sum(1 for item in result if item.get('analysis_available'))})"
    )
    return result


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/list")
def api_list(
    dir: Optional[str] = Query(default=None),
    q: Optional[str] = Query(default=None),
    filters: Optional[str] = Query(default=None),
    sort_field: Optional[str] = Query(default=None),
    sort_order: Optional[str] = Query(default="desc"),
    limit: Optional[int] = Query(default=None),
    offset: int = Query(default=0),
):
    if has_session_args:
        source = _get_source_by_key(dir)
        items = _cache.get(source.key, [])
    else:
        root_dir = _get_root_by_key(dir)
        items = _cache.get(_dir_key(root_dir), [])

    # --- search ---
    if q:
        ql = q.lower()
        items = [it for it in items if
                 ql in str(it.get("label_short") or "").lower() or
                 ql in str(it.get("rel_path") or "").lower()]

    # --- filters ---
    if filters:
        try:
            filter_list = json.loads(filters)
        except json.JSONDecodeError:
            filter_list = []
        for f in filter_list:
            items = [it for it in items if _match_filter(it, f)]

    # --- sort ---
    if sort_field:
        reverse = sort_order != "asc"
        items = sorted(items, key=lambda it: _sort_key(it, sort_field), reverse=reverse)

    # --- pagination ---
    total = len(items)
    has_analysis = any(it.get("analysis_available") for it in items)
    if limit is not None:
        items = items[offset:offset + limit]

    return JSONResponse({"items": items, "total": total, "has_analysis": has_analysis})


def _match_filter(item: dict, f: dict) -> bool:
    """Server-side replica of the front-end _matchFilter logic."""
    field = f.get("field", "")
    op = f.get("op", "")
    value = f.get("value", "")
    raw = item.get(field)

    if op == "range":
        try:
            left = float(raw) if raw is not None else None
        except (TypeError, ValueError):
            return False
        if left is None:
            return False
        lo = float(f.get("min", "-inf")) if f.get("min") is not None else float("-inf")
        hi = float(f.get("max", "inf")) if f.get("max") is not None else float("inf")
        return lo <= left < hi

    if op == "map_has_key":
        key = f.get("key", "")
        return isinstance(raw, dict) and key in raw

    if op == "contains":
        return value.lower() in str(raw or "").lower()

    if op == "=":
        return str(raw or "") == value

    # numeric comparisons
    try:
        left = float(raw) if raw is not None else None
        right = float(value)
    except (TypeError, ValueError):
        return False
    if left is None:
        return False
    if op == ">=":
        return left >= right
    if op == "<=":
        return left <= right
    return False


def _sort_key(item: dict, field: str):
    """Return a sort key; numeric fields sort numerically, else string."""
    v = item.get(field)
    if v is None:
        return (1, 0)  # nulls last
    try:
        return (0, float(v))
    except (TypeError, ValueError):
        return (0, str(v))


@app.get("/api/refresh")
def api_refresh(dir: Optional[str] = Query(default=None)):
    global _cache
    if has_session_args:
        _sync_session_dirs()
        source = _get_source_by_key(dir)
        _cache[source.key] = scan_session_dir(source)
        _analysis_summary_cache[source.key] = _build_analysis_summary(_cache[source.key])
        logger.info(f"Refreshed {source.key}: {len(_cache[source.key])} conversations")
        return JSONResponse({"count": len(_cache[source.key]), "dir": source.key})

    root_dir = _get_root_by_key(dir)
    dir_key = _dir_key(root_dir)
    _cache[dir_key] = scan_directory(root_dir, incremental=True)
    logger.info(f"Refreshed {dir_key}: {len(_cache[dir_key])} conversations")
    return JSONResponse({"count": len(_cache[dir_key]), "dir": dir_key})


@app.get("/api/file")
def api_file(rel_path: str = Query(...), dir: Optional[str] = Query(default=None)):
    if has_session_args:
        source = _get_source_by_key(dir)
        base_dir = source.session_dir
    else:
        base_dir = _get_root_by_key(dir)
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


@app.get("/api/session/trajectory")
def api_session_trajectory(rel_path: str = Query(...), dir: Optional[str] = Query(default=None)):
    if not has_session_args:
        raise HTTPException(status_code=400, detail="Trajectory is only available in session mode")

    source = _get_source_by_key(dir)
    session_name = str(rel_path).split("/", 1)[0]
    if not session_name:
        raise HTTPException(status_code=400, detail="Invalid session path")

    session_dir = (source.session_dir / session_name).resolve()
    try:
        if source.session_dir not in session_dir.parents and session_dir != source.session_dir:
            raise ValueError("path not under session dir")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid session path")

    if not session_dir.is_dir():
        raise HTTPException(status_code=404, detail="Session directory not found")

    try:
        return JSONResponse(build_session_trajectory(session_dir))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/trajectory/diff")
def api_trajectory_diff(session: str = Query(...), dir: Optional[str] = Query(default=None)):
    if not has_session_args:
        raise HTTPException(status_code=400, detail="Trajectory is only available in session mode")

    source = _get_source_by_key(dir)
    session_dir = (source.session_dir / session).resolve()
    try:
        if source.session_dir not in session_dir.parents and session_dir != source.session_dir:
            raise ValueError("path not under session dir")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid session path")

    if not session_dir.is_dir():
        raise HTTPException(status_code=404, detail="Session directory not found")

    try:
        return JSONResponse(build_trajectory_diff(session_dir))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/report/meta")
def api_report_meta(dir: Optional[str] = Query(default=None)):
    if not has_session_args:
        raise HTTPException(status_code=400, detail="Reports are only available in session mode")
    source = _get_source_by_key(dir)
    meta = _source_report_meta(source)
    return JSONResponse({
        "dir": source.key,
        "report_dir": meta["report_dir"],
        "html": {
            "exists": meta["has_report_html"],
            "url": f"/report/view?dir={source.key}" if meta["has_report_html"] else None,
        },
        "md": {
            "exists": meta["has_report_md"],
            "url": f"/report/raw/md?dir={source.key}" if meta["has_report_md"] else None,
        },
        "xlsx": {
            "exists": meta["has_report_xlsx"],
            "url": f"/report/raw/xlsx?dir={source.key}" if meta["has_report_xlsx"] else None,
        },
    })


@app.get("/report/view", response_class=HTMLResponse)
def report_view(dir: Optional[str] = Query(default=None)):
    if not has_session_args:
        raise HTTPException(status_code=400, detail="Reports are only available in session mode")
    source = _get_source_by_key(dir)
    html_path = _get_report_file(source, "session_report.html")
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/report/raw/md", response_class=PlainTextResponse)
def report_raw_md(dir: Optional[str] = Query(default=None)):
    if not has_session_args:
        raise HTTPException(status_code=400, detail="Reports are only available in session mode")
    source = _get_source_by_key(dir)
    md_path = _get_report_file(source, "session_report.md")
    return PlainTextResponse(md_path.read_text(encoding="utf-8"))


@app.get("/report/raw/xlsx")
def report_raw_xlsx(dir: Optional[str] = Query(default=None)):
    if not has_session_args:
        raise HTTPException(status_code=400, detail="Reports are only available in session mode")
    source = _get_source_by_key(dir)
    xlsx_path = _get_report_file(source, "session_report.xlsx")
    return FileResponse(
        str(xlsx_path),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=xlsx_path.name,
    )


@app.get("/api/analysis/summary")
def api_analysis_summary(dir: Optional[str] = Query(default=None)):
    if not has_session_args:
        raise HTTPException(status_code=400, detail="Analysis is only available in session mode")
    source = _get_source_by_key(dir)
    if source.key not in _analysis_summary_cache:
        _cache[source.key] = scan_session_dir(source)
        _analysis_summary_cache[source.key] = _build_analysis_summary(_cache[source.key])
    return JSONResponse(_analysis_summary_cache[source.key])


@app.get("/api/summary")
def api_summary():
    if not has_session_args:
        raise HTTPException(status_code=400, detail="Summary is only available in session mode")
    sources = _sync_session_dirs()
    result = []
    for source in sources:
        if source.key not in _cache:
            _cache[source.key] = scan_session_dir(source)
        if source.key not in _analysis_summary_cache:
            _analysis_summary_cache[source.key] = _build_analysis_summary(_cache[source.key])
        result.append(_build_source_summary(source, _cache[source.key]))
    return JSONResponse({
        "sources": result,
        "total_sources": len(result),
    })


@app.get("/api/config")
def api_config():
    if has_session_args:
        sources = _sync_session_dirs()
        return JSONResponse({
            "mode": "session",
            "dirs": [
                dict({
                    "key": source.key,
                    "label": source.label,
                    "path": str(source.session_dir),
                }, **_source_report_meta(source))
                for source in sources
            ],
            "active_dir": sources[0].key if len(sources) == 1 else None,
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
# Favorites
# ---------------------------------------------------------------------------
_FAVORITES_FILE = Path(__file__).resolve().parent.parent / "favorites.json"
_favorites: List[Dict] = []


def _load_favorites():
    global _favorites
    if _FAVORITES_FILE.is_file():
        try:
            with open(_FAVORITES_FILE, "r", encoding="utf-8") as f:
                _favorites = json.load(f)
        except Exception:
            _favorites = []
    else:
        _favorites = []


def _save_favorites():
    with open(_FAVORITES_FILE, "w", encoding="utf-8") as f:
        json.dump(_favorites, f, ensure_ascii=False, indent=2)


_load_favorites()


@app.get("/api/favorites")
def api_favorites():
    # Enrich dir_label from source registry
    enriched = []
    for fav in _favorites:
        entry = _SOURCE_BY_KEY.get(fav.get("dir_key"))
        item = {**fav, "dir_label": entry.label if entry else fav.get("dir_label", "")}
        enriched.append(item)
    return JSONResponse(enriched)


@app.post("/api/favorites/add")
async def api_favorites_add(request: Request):
    body = await request.json()
    dir_key = body.get("dir_key", "")
    rel_path = body.get("rel_path", "")
    if not dir_key or not rel_path:
        raise HTTPException(status_code=400, detail="dir_key and rel_path required")
    # Dedup
    for fav in _favorites:
        if fav.get("dir_key") == dir_key and fav.get("rel_path") == rel_path:
            return JSONResponse({"status": "already_exists"})
    _favorites.append({
        "dir_key": dir_key,
        "rel_path": rel_path,
        "label_short": body.get("label_short", ""),
        "model": body.get("model", ""),
        "msg_count": body.get("msg_count", 0),
        "session": body.get("session", ""),
        "dir_label": body.get("dir_label", ""),
        "added_at": datetime.now().isoformat(),
    })
    _save_favorites()
    return JSONResponse({"status": "ok"})


@app.post("/api/favorites/remove")
async def api_favorites_remove(request: Request):
    body = await request.json()
    dir_key = body.get("dir_key", "")
    rel_path = body.get("rel_path", "")
    _favorites[:] = [f for f in _favorites if not (f.get("dir_key") == dir_key and f.get("rel_path") == rel_path)]
    _save_favorites()
    return JSONResponse({"status": "ok"})


@app.get("/api/favorites/export")
def api_favorites_export(fmt: str = Query(default="csv")):
    """Export favorites as CSV."""
    import csv
    import io

    fields = ["dir_key", "dir_label", "session", "label_short", "model", "msg_count", "rel_path", "added_at"]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for fav in _favorites:
        entry = _SOURCE_BY_KEY.get(fav.get("dir_key"))
        row = {**fav, "dir_label": entry.label if entry else fav.get("dir_label", "")}
        writer.writerow(row)
    content = "\ufeff" + buf.getvalue()  # UTF-8 BOM for Excel
    return PlainTextResponse(
        content,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=bookmark_favorites.csv"},
    )


@app.get("/api/trajectory")
def api_trajectory(dir: Optional[str] = Query(default=None), session: Optional[str] = Query(default=None)):
    """Return all round files for a session folder."""
    if not has_session_args:
        raise HTTPException(status_code=400, detail="Trajectory is only available in session mode")
    if not session:
        raise HTTPException(status_code=400, detail="Missing session parameter")

    source = _get_source_by_key(dir)
    session_folder = source.session_dir / session
    if not session_folder.is_dir():
        raise HTTPException(status_code=404, detail="Session folder not found")

    # Collect all JSON files in the session folder
    json_files = sorted([f for f in session_folder.iterdir() if f.suffix == ".json"])
    rounds = []
    for idx, file_path in enumerate(json_files):
        try:
            data = load_json(file_path)
            msg_count = len(data.get("messages", []))
            rounds.append({
                "round": idx + 1,
                "file": file_path.name,
                "rel_path": f"{session}/{file_path.name}",
                "msg_count": msg_count,
            })
        except Exception:
            continue

    return JSONResponse({"rounds": rounds})


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
        if REPORT_DIRS:
            mode_desc.append("--report-dir")
        print(f"[chat-log-viewer] Mode       : session ({', '.join(mode_desc)})")
        for parent_dir in SESSION_PARENT_DIRS:
            print(f"[chat-log-viewer] Session parent: {parent_dir}")
        for source in _sync_session_dirs():
            print(f"[chat-log-viewer] Source     : {source.key} [{source.label}] -> {source.session_dir}")
            print(f"[chat-log-viewer] Report dir : {source.report_dir}")
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
