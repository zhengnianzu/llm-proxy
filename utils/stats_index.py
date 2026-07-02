"""
utils/stats_index.py — 增量汇总索引

持久化文件: {env_dir}/.stats_index.json
进程级内存缓存: _MEM_TTL 秒内直接返回内存数据，跳过磁盘读取和 stat。

结构:
{
  "_version": 3,
  "dirs": {
    "26070100": {
      "cache_mtime": 1719820000.0,
      "cache_size": 123456,
      "req_count": 4200,
      "req_count_mtime": 1719820000.0,
      "frozen": true,
      "sessions": { "sk-abcd|2026-06-28": {"total": 5, "qualified": 3} }
    }
  },
  "updated_at": 1719820100.0
}
"""

import json
import os
import time
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_INDEX_FILE = ".stats_index.json"
_VERSION = 3
_lock = threading.Lock()

QUALIFIED_THRESHOLD_DEFAULT = 5

# ---------------------------------------------------------------------------
# 进程级内存缓存
# ---------------------------------------------------------------------------
_MEM_TTL = 10  # 秒
_mem_index: Optional[dict] = None
_mem_index_ts: float = 0
_mem_env_dir: Optional[str] = None  # 绑定的 env_dir，切换时失效


def _index_path(env_dir: Path) -> str:
    return str(env_dir / _INDEX_FILE)


def _load_index(env_dir: Path) -> dict:
    path = _index_path(env_dir)
    if not os.path.isfile(path):
        return {"_version": _VERSION, "dirs": {}, "updated_at": 0}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("_version", 0) < _VERSION:
            old_dirs = data.get("dirs", {})
            for d in old_dirs.values():
                d.setdefault("frozen", False)
            return {"_version": _VERSION, "dirs": old_dirs, "updated_at": 0}
        return data
    except (OSError, json.JSONDecodeError):
        return {"_version": _VERSION, "dirs": {}, "updated_at": 0}


def _save_index(env_dir: Path, data: dict) -> None:
    path = _index_path(env_dir)
    try:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp, path)
    except OSError:
        pass


def _scan_session_cache(cache_file: Path, threshold: int) -> dict:
    """读一个 .session_cache.jsonl，返回 {(api_key|date): {total, qualified}}。"""
    buckets: Dict[str, Dict[str, int]] = {}
    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("_meta"):
                    continue
                api_key = obj.get("api_key", "") or "(empty)"
                ts = obj.get("first_ts", "")
                date_str = ts.split("_")[0] if ts else "unknown"
                bucket_key = f"{api_key}|{date_str}"
                if bucket_key not in buckets:
                    buckets[bucket_key] = {"total": 0, "qualified": 0}
                buckets[bucket_key]["total"] += 1
                if len(obj.get("trace_list", [])) >= threshold:
                    buckets[bucket_key]["qualified"] += 1
    except OSError:
        pass
    return buckets


def _count_req_files(dir_path: Path) -> int:
    try:
        return sum(1 for f in dir_path.iterdir() if f.name.endswith("-req.json"))
    except OSError:
        return 0


def _get_active_tag() -> str:
    """获取当前启动目录标签（如 '26070115'），用于判断活跃目录。"""
    try:
        from utils.log_paths import STARTUP_DATE_TAG
        return STARTUP_DATE_TAG
    except ImportError:
        return ""


def refresh_index(env_dir: Path, threshold: int = QUALIFIED_THRESHOLD_DEFAULT,
                  force: bool = False) -> dict:
    """增量刷新索引，返回最新的 index data。

    - TTL 内直接返回内存缓存（<1ms）
    - 过期后只 stat 活跃目录，frozen 目录跳过
    - force=True 跳过 TTL + frozen，全量检查
    """
    global _mem_index, _mem_index_ts, _mem_env_dir

    env_dir_str = str(env_dir)
    now = time.time()

    # 快速路径：内存缓存命中
    if (not force
            and _mem_index is not None
            and _mem_env_dir == env_dir_str
            and (now - _mem_index_ts) < _MEM_TTL):
        return _mem_index

    with _lock:
        # double-check：另一个线程可能刚刷新过
        if (not force
                and _mem_index is not None
                and _mem_env_dir == env_dir_str
                and (time.time() - _mem_index_ts) < _MEM_TTL):
            return _mem_index

        index = _mem_index if (_mem_env_dir == env_dir_str and _mem_index) else _load_index(env_dir)
        dirs_cache = index.get("dirs", {})
        changed = False
        active_tag = _get_active_tag()

        current_dirs = set()
        if env_dir.is_dir():
            for sub in env_dir.iterdir():
                if not sub.is_dir():
                    continue
                dir_name = sub.name
                current_dirs.add(dir_name)

                prev = dirs_cache.get(dir_name)

                # frozen 目录：已扫描过的历史目录，跳过 stat
                if (not force and prev
                        and prev.get("frozen")
                        and dir_name != active_tag):
                    continue

                cache_file = sub / ".session_cache.jsonl"
                if not cache_file.is_file():
                    # 没有 session cache 的目录
                    if prev and not force and prev.get("req_count", 0) > 0 and dir_name != active_tag:
                        # 历史目录已有 req_count，标记 frozen 并跳过
                        if not prev.get("frozen"):
                            prev["frozen"] = True
                            changed = True
                        continue
                    try:
                        dir_mt = os.path.getmtime(str(sub))
                    except OSError:
                        dir_mt = 0
                    if prev and not force and prev.get("req_count_mtime") == dir_mt:
                        continue
                    req_count = _count_req_files(sub)
                    dirs_cache[dir_name] = {
                        "cache_mtime": 0,
                        "cache_size": 0,
                        "req_count": req_count,
                        "req_count_mtime": dir_mt,
                        "frozen": dir_name != active_tag and req_count > 0,
                        "sessions": {},
                    }
                    changed = True
                    continue

                try:
                    st = cache_file.stat()
                    c_mtime = st.st_mtime
                    c_size = st.st_size
                except OSError:
                    continue

                if (not force and prev
                        and prev.get("cache_mtime") == c_mtime
                        and prev.get("cache_size") == c_size):
                    # 无变化，如果是历史目录，标记 frozen
                    if dir_name != active_tag and not prev.get("frozen"):
                        prev["frozen"] = True
                        changed = True
                    continue

                sessions = _scan_session_cache(cache_file, threshold)

                try:
                    dir_mt = os.path.getmtime(str(sub))
                except OSError:
                    dir_mt = 0
                if prev and prev.get("req_count_mtime") == dir_mt and prev.get("req_count", 0) > 0:
                    req_count = prev["req_count"]
                    req_mt = prev["req_count_mtime"]
                else:
                    req_count = _count_req_files(sub)
                    req_mt = dir_mt

                dirs_cache[dir_name] = {
                    "cache_mtime": c_mtime,
                    "cache_size": c_size,
                    "req_count": req_count,
                    "req_count_mtime": req_mt,
                    "frozen": dir_name != active_tag,
                    "sessions": sessions,
                }
                changed = True

        # 清理已删除的目录
        removed = set(dirs_cache.keys()) - current_dirs
        if removed:
            for r in removed:
                del dirs_cache[r]
            changed = True

        if changed:
            index["dirs"] = dirs_cache
            index["updated_at"] = time.time()
            _save_index(env_dir, index)

        # 写入内存缓存
        _mem_index = index
        _mem_index_ts = time.time()
        _mem_env_dir = env_dir_str

        return index


def build_stats_from_index(index: dict, threshold: int = QUALIFIED_THRESHOLD_DEFAULT) -> dict:
    """从索引数据构建与原 _build_stats_json 兼容的统计结果。"""
    table: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"total": 0, "qualified": 0})
    )
    mtime_table: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"total": 0, "qualified": 0})
    )
    all_dates: set = set()
    all_mtimes: set = set()

    total = 0
    qualified = 0

    for dir_name, dir_info in index.get("dirs", {}).items():
        sessions = dir_info.get("sessions", {})
        if not sessions:
            continue
        all_mtimes.add(dir_name)
        for bucket_key, counts in sessions.items():
            parts = bucket_key.split("|", 1)
            if len(parts) != 2:
                continue
            api_key, date_str = parts
            t = counts["total"]
            q = counts["qualified"]
            total += t
            qualified += q
            all_dates.add(date_str)
            table[api_key][date_str]["total"] += t
            table[api_key][date_str]["qualified"] += q
            mtime_table[api_key][dir_name]["total"] += t
            mtime_table[api_key][dir_name]["qualified"] += q

    dates = sorted(all_dates)
    keys = sorted(table.keys())

    rows = []
    for key in keys:
        row_total = 0
        row_qualified = 0
        cells = {}
        for d in dates:
            cell = table[key][d]
            cells[d] = {"total": cell["total"], "qualified": cell["qualified"]}
            row_total += cell["total"]
            row_qualified += cell["qualified"]
        mtime_cells = {}
        for mt in sorted(all_mtimes):
            mc = mtime_table[key][mt]
            if mc["total"] > 0:
                mtime_cells[mt] = {"total": mc["total"], "qualified": mc["qualified"]}
        rows.append({
            "api_key": key,
            "cells": cells,
            "mtime_cells": mtime_cells,
            "row_total": row_total,
            "row_qualified": row_qualified,
        })

    return {
        "total_sessions": total,
        "qualified_sessions": qualified,
        "threshold": threshold,
        "dates": dates,
        "rows": rows,
    }


def get_dir_counts(index: dict) -> Dict[str, int]:
    """从索引中提取每个 mtime_dir 的 req_count，供 /logs/dirs 使用。"""
    result = {}
    for dir_name, dir_info in index.get("dirs", {}).items():
        result[dir_name] = dir_info.get("req_count", 0)
    return result


def get_date_to_mtime_map(index: dict) -> Dict[str, List[str]]:
    """从索引中构建 date -> [mtime_dir, ...] 映射，替代 export_routes._date_to_mtime_map。"""
    mapping: Dict[str, List[str]] = {}
    for dir_name, dir_info in index.get("dirs", {}).items():
        dates_in_dir: set = set()
        for bucket_key in dir_info.get("sessions", {}):
            parts = bucket_key.split("|", 1)
            if len(parts) == 2:
                dates_in_dir.add(parts[1])
        for d in dates_in_dir:
            if d not in mapping:
                mapping[d] = []
            if dir_name not in mapping[d]:
                mapping[d].append(dir_name)
    return mapping
