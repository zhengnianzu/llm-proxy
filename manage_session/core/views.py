"""
core/views.py — 状态视图生成

生成两类视图：
1. views/batch_status/<task_id>.json   — 以批次为主的状态
2. views/global_summary.json           — 全局汇总
"""

import json
from pathlib import Path
from datetime import datetime

from core.config import views_dir, pair_cache_dir


def _views_dir() -> Path:
    return views_dir()


def _batch_status_dir() -> Path:
    d = _views_dir() / "batch_status"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _w(path: Path, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _now():
    return datetime.now().isoformat(timespec="seconds")


# ── batch status ──────────────────────────────────────────────────────────────

def update_batch_status(task_id: str):
    """
    汇总该 task 对应的所有 pair_cache，生成批次状态视图。
    """
    from core import cache as cache_mod, manifest as manifest_mod

    task_meta = manifest_mod.get_task(task_id)
    if not task_meta:
        raise KeyError(f"Task not registered: {task_id}")

    total_tasks = task_meta["count"]
    task_hash   = task_meta["hash"]

    matched_count     = 0
    matched_indexes   = {}  # index_id -> count
    all_unmatched     = []  # 仅记录前 1000 条，避免文件过大
    matched_topics: dict[str, int] = {}
    unmatched_topics: dict[str, int] = {}

    for cache_file in sorted(pair_cache_dir().glob("*.json")):
        with open(cache_file, encoding="utf-8") as f:
            data = json.load(f)

        if data.get("task_id") != task_id:
            continue

        index_id = data.get("index_id", "unknown")
        mc       = data.get("matched_count", 0)
        matched_count += mc
        if mc:
            matched_indexes[index_id] = mc

        for m in data.get("matches", []):
            t = m.get("task_row_data", {}).get("topic") or ""
            matched_topics[t] = matched_topics.get(t, 0) + 1

        for ut in data.get("unmatched_tasks", []):
            t = ut.get("topic") or ""
            unmatched_topics[t] = unmatched_topics.get(t, 0) + 1
            if len(all_unmatched) < 1000:
                all_unmatched.append({
                    "query": ut.get("query", "")[:120],
                    "topic": ut.get("topic"),
                    "env_name": ut.get("env_name"),
                })

    view = {
        "task_id":          task_id,
        "task_file":        task_meta["path"],
        "total_tasks":      total_tasks,
        "matched_count":    matched_count,
        "unmatched_count":  total_tasks - matched_count,
        "completion_rate":  round(matched_count / total_tasks, 4) if total_tasks else 0,
        "matched_indexes":  matched_indexes,
        "topics":           task_meta.get("topics", {}),
        "matched_topics":   dict(sorted(matched_topics.items(), key=lambda x: -x[1])),
        "unmatched_topics": dict(sorted(unmatched_topics.items(), key=lambda x: -x[1])),
        "unmatched_sample": all_unmatched[:100],
        "updated_at":       _now(),
    }

    safe_id = task_id.replace("/", "__")
    _w(_batch_status_dir() / f"{safe_id}.json", view)
    return view


# ── global summary ────────────────────────────────────────────────────────────

def update_global_summary():
    """
    汇总所有批次状态，生成全局视图。
    """
    from core import manifest as manifest_mod

    all_tasks   = manifest_mod.get_all_tasks()
    all_indexes = manifest_mod.get_all_indexes()

    total_index_count = sum(v["count"] for v in all_indexes.values())

    batches = []
    total_matched = 0

    for task_id, task_meta in all_tasks.items():
        batch_view_path = _batch_status_dir() / f"{task_id.replace('/', '__')}.json"
        if batch_view_path.exists():
            with open(batch_view_path, encoding="utf-8") as f:
                bv = json.load(f)
        else:
            bv = {
                "matched_count":   0,
                "unmatched_count": task_meta["count"],
                "completion_rate": 0,
                "matched_indexes": {},
            }

        matched = bv.get("matched_count", 0)
        total_matched += matched
        batches.append({
            "task_id":        task_id,
            "task_file":      task_meta["path"],
            "total_tasks":    task_meta["count"],
            "matched_count":  matched,
            "unmatched_count": bv.get("unmatched_count", task_meta["count"] - matched),
            "completion_rate": bv.get("completion_rate", 0),
            "topics":         task_meta.get("topics", {}),
            "added_at":       task_meta.get("added_at"),
        })

    indexes_list = []
    for idx_id, idx_meta in all_indexes.items():
        html_path = None
        if idx_meta.get("format") == "xlsx":
            candidate = Path(idx_meta["abs_path"]).parent / "session_report.html"
            if candidate.is_file():
                html_path = str(candidate)
        indexes_list.append({
            "index_id":        idx_id,
            "index_file":      idx_meta["path"],
            "format":          idx_meta.get("format", "json"),
            "count":           idx_meta["count"],
            "added_at":        idx_meta.get("added_at"),
            "report_html_path": html_path,
        })

    summary = {
        "total_task_batches":      len(all_tasks),
        "total_tasks":             sum(v["count"] for v in all_tasks.values()),
        "total_index_files":       len(all_indexes),
        "total_trajectories":      total_index_count,
        "total_matched":           total_matched,
        "total_unmatched_tasks":   sum(v["count"] for v in all_tasks.values()) - total_matched,
        "total_unmatched_indexes": total_index_count - total_matched,
        "batches":                 batches,
        "indexes":                 indexes_list,
        "updated_at":              _now(),
    }

    _w(_views_dir() / "global_summary.json", summary)
    return summary
