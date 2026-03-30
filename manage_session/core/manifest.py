"""
core/manifest.py — 清单管理

维护 tasks 和 indexes 的元数据，记录路径、hash、记录数等。
文件变化时自动更新 hash，供 cache 层判断是否需要重新匹配。
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path

from core.config import manifests_dir, tasks_dir, raw_index_dir


def _manifests_dir() -> Path:
    return manifests_dir()


def _tasks_manifest() -> Path:
    return _manifests_dir() / "tasks.json"


def _indexes_manifest() -> Path:
    return _manifests_dir() / "indexes.json"


# ── helpers ──────────────────────────────────────────────────────────────────

def _file_hash(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_manifest(path: Path) -> dict:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_manifest(path: Path, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ── tasks ─────────────────────────────────────────────────────────────────────

def add_task(src_path: str | Path, copy_to_tasks: bool = True) -> dict:
    """
    注册一个任务文件到 tasks/ 并更新 manifests/tasks.json。
    src_path: 可以是绝对路径。
    返回该任务的清单条目。
    """
    src = Path(src_path).resolve()
    if not src.exists():
        raise FileNotFoundError(f"Task file not found: {src}")

    task_id = src.name  # e.g. "260325-7840-blue-linux.json"

    if copy_to_tasks:
        dest = tasks_dir() / task_id
        if not dest.exists() or dest.resolve() != src:
            import shutil
            shutil.copy2(src, dest)
        working_path = dest
    else:
        working_path = src

    records = _load_task_records(working_path)
    file_hash = _file_hash(working_path)

    manifest = _load_manifest(_tasks_manifest())
    entry = {
        "task_id": task_id,
        "path": str(working_path),
        "abs_path": str(working_path),
        "hash": file_hash,
        "count": len(records),
        "queries_sample": [r.get("query", "")[:80] for r in records[:5]],
        "topics": _count_field(records, "topic"),
        "added_at": manifest.get(task_id, {}).get("added_at") or _now(),
        "updated_at": _now(),
    }
    manifest[task_id] = entry
    _save_manifest(_tasks_manifest(), manifest)
    return entry


def _scan_and_register_tasks() -> dict:
    """自动扫描 tasks/ 目录，注册新文件或更新变化文件，返回完整清单。"""
    manifest = _load_manifest(_tasks_manifest())
    changed = False

    for src in sorted(tasks_dir().glob("*.json")):
        task_id = src.name
        file_hash = _file_hash(src)
        existing = manifest.get(task_id)
        if existing and existing.get("hash") == file_hash:
            continue  # 未变化，跳过
        records = _load_task_records(src)
        entry = {
            "task_id":        task_id,
            "path":           str(src),
            "abs_path":       str(src),
            "hash":           file_hash,
            "count":          len(records),
            "queries_sample": [r.get("query", "")[:80] for r in records[:5]],
            "topics":         _count_field(records, "topic"),
            "added_at":       (existing or {}).get("added_at") or _now(),
            "updated_at":     _now(),
        }
        manifest[task_id] = entry
        changed = True

    if changed:
        _save_manifest(_tasks_manifest(), manifest)
    return manifest


def get_all_tasks() -> dict:
    return _scan_and_register_tasks()


def get_task(task_id: str) -> dict | None:
    return get_all_tasks().get(task_id)


def _load_task_records(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    r = json.loads(line)
                    if "query" in r:
                        records.append(r)
                except json.JSONDecodeError:
                    pass
    return records


def load_task_queries(task_id: str) -> dict[str, dict]:
    """Return {query_text: record} for a task."""
    m = get_task(task_id)
    if not m:
        raise KeyError(f"Task not registered: {task_id}")
    records = _load_task_records(Path(m["abs_path"]))
    return {r["query"].strip(): r for r in records if r.get("query")}


# ── indexes ───────────────────────────────────────────────────────────────────

def add_index(src_path: str | Path, index_id: str | None = None) -> dict:
    """
    注册一个 index 文件并更新 manifests/indexes.json。
    支持 .json（直接复制到 raw_index/）和 .xlsx（原路径引用，不复制）。
    index_id: 可选标识符；默认为 "父目录名/文件名"，如 "test_session2/index.json"。
    返回该 index 的清单条目。
    """
    src = Path(src_path).resolve()
    if not src.exists():
        raise FileNotFoundError(f"Index file not found: {src}")

    if index_id is None:
        index_id = f"{src.parent.name}/{src.name}"

    is_xlsx = src.suffix.lower() == ".xlsx"

    if is_xlsx:
        # xlsx 直接引用原始路径，不复制
        working_path = src
    else:
        dest_dir = raw_index_dir() / Path(index_id).parent
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / src.name
        if not dest.exists() or dest.resolve() != src:
            import shutil
            shutil.copy2(src, dest)
        working_path = dest

    entries = _load_index_entries(working_path)
    file_hash = _file_hash(working_path)

    manifest = _load_manifest(_indexes_manifest())
    entry = {
        "index_id": index_id,
        "path": str(working_path),
        "abs_path": str(working_path),
        "format": "xlsx" if is_xlsx else "json",
        "hash": file_hash,
        "count": len(entries),
        "q1s_sample": [e.get("q1", "")[:80] for e in entries[:5]],
        "added_at": manifest.get(index_id, {}).get("added_at") or _now(),
        "updated_at": _now(),
    }
    manifest[index_id] = entry
    _save_manifest(_indexes_manifest(), manifest)
    return entry


def _scan_and_register_indexes() -> dict:
    """
    自动扫描 raw_index/ 目录下的 **/*.json 和 **/*.xlsx 文件，
    注册新文件或更新变化文件，返回完整清单。
    """
    manifest = _load_manifest(_indexes_manifest())
    changed = False

    for src in sorted(raw_index_dir().rglob("*")):
        if src.suffix.lower() not in (".json", ".xlsx"):
            continue
        try:
            index_id = f"{src.parent.name}/{src.name}"
        except Exception:
            index_id = src.name
        file_hash = _file_hash(src)
        existing = manifest.get(index_id)
        if existing and existing.get("hash") == file_hash:
            continue
        is_xlsx = src.suffix.lower() == ".xlsx"
        entries = _load_index_entries(src)
        entry = {
            "index_id":    index_id,
            "path":        str(src),
            "abs_path":    str(src),
            "format":      "xlsx" if is_xlsx else "json",
            "hash":        file_hash,
            "count":       len(entries),
            "q1s_sample":  [e.get("q1", "")[:80] for e in entries[:5]],
            "added_at":    (existing or {}).get("added_at") or _now(),
            "updated_at":  _now(),
        }
        manifest[index_id] = entry
        changed = True

    if changed:
        _save_manifest(_indexes_manifest(), manifest)
    return manifest


def get_all_indexes() -> dict:
    return _scan_and_register_indexes()


def get_index(index_id: str) -> dict | None:
    return get_all_indexes().get(index_id)


def _load_index_entries(path: Path) -> list[dict]:
    if path.suffix.lower() == ".xlsx":
        return _load_index_from_xlsx(path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def _load_index_from_xlsx(path: Path) -> list[dict]:
    """从 xlsx 读取 Q1首问 列，返回 [{q1: ...}, ...] 格式。"""
    try:
        import openpyxl
    except ImportError:
        raise ImportError("openpyxl required for xlsx support: pip install openpyxl")

    wb = openpyxl.load_workbook(path, data_only=True)
    sheet = wb.active

    # 找 Q1首问 列
    q1_col = None
    for col_idx, cell in enumerate(sheet[1], 1):
        if cell.value == "Q1首问":
            q1_col = col_idx
            break

    if q1_col is None:
        return []

    entries = []
    for row_idx, row in enumerate(sheet.iter_rows(min_row=2, values_only=False), 2):
        q1_cell = row[q1_col - 1]
        q1_val = q1_cell.value
        if q1_val:
            entries.append({"q1": str(q1_val).strip()})

    return entries


def load_index_q1s(index_id: str) -> dict[str, dict]:
    """Return {q1_text: entry} for an index."""
    m = get_index(index_id)
    if not m:
        raise KeyError(f"Index not registered: {index_id}")
    entries = _load_index_entries(Path(m["abs_path"]))
    return {e["q1"].strip(): e for e in entries if e.get("q1")}


# ── utils ─────────────────────────────────────────────────────────────────────

def is_changed(file_path: str | Path, old_hash: str) -> bool:
    return _file_hash(Path(file_path)) != old_hash


def _count_field(records: list[dict], field: str) -> dict:
    from collections import Counter
    return dict(Counter(r.get(field, "") for r in records).most_common(10))


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")
