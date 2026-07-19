"""
utils/stats_index.py — 增量汇总索引

持久化文件:
  {env_dir}/.stats_index.json       — 目录级扫描索引
  {env_dir}/.session_key_cache.json  — key×date / key×mtime 预聚合缓存

进程级内存缓存: _MEM_TTL 秒内直接返回内存数据，跳过磁盘读取和 stat。

增量机制:
  refresh_index 扫描时对比 old/new sessions 生成 _changed_buckets；
  build_stats_from_index 根据 _changed_buckets 做 bucket 级定向更新，
  结果持久化到 .session_key_cache.json，进程重启后直接加载。
"""

import hashlib
import json
import logging
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
# 进程级内存缓存 — 目录扫描索引
# ---------------------------------------------------------------------------
_MEM_TTL = 10  # 秒
_mem_index: Optional[dict] = None
_mem_index_ts: float = 0
_mem_env_dir: Optional[str] = None  # 绑定的 env_dir，切换时失效

# ---------------------------------------------------------------------------
# 进程级内存缓存 — key 聚合表
# ---------------------------------------------------------------------------
_KEY_CACHE_FILE = ".session_key_cache.json"
_KEY_CACHE_VERSION = 2
_mem_key_cache: Optional[dict] = None
_mem_key_cache_env: Optional[str] = None


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
    """读 session 统计数据，优先从 DB 查询，降级到读 .session_cache.jsonl 文件。"""
    root_dir = str(cache_file.parent)
    try:
        import utils.session_store as _ss
        count = _ss.get_session_count_by_root(root_dir)
        if count > 0:
            return _ss.get_session_stats(root_dir, threshold)
    except Exception:
        pass

    # 降级：直接读文件（DB 未初始化或该目录无数据时）
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
    try:
        from utils.log_paths import STARTUP_DATE_TAG
        return STARTUP_DATE_TAG
    except ImportError:
        return ""


def _diff_sessions(old_sessions: dict, new_sessions: dict, dir_name: str) -> list:
    """对比两个 sessions dict，返回 [(dir_name, bucket_key, old_counts, new_counts), ...]"""
    diffs = []
    all_keys = set(old_sessions) | set(new_sessions)
    for bk in all_keys:
        old_c = old_sessions.get(bk)
        new_c = new_sessions.get(bk)
        if old_c != new_c:
            diffs.append((dir_name, bk, old_c, new_c))
    return diffs


def refresh_index(env_dir: Path, threshold: int = QUALIFIED_THRESHOLD_DEFAULT,
                  force: bool = False) -> dict:
    """增量刷新索引，返回最新的 index data。

    - TTL 内直接返回内存缓存（<1ms）
    - 过期后只 stat 活跃目录，frozen 目录跳过
    - force=True 跳过 TTL + frozen，全量检查

    index["_changed_buckets"] 记录本次刷新中变化的 bucket 列表，
    供 build_stats_from_index 做定向增量更新。
    """
    global _mem_index, _mem_index_ts, _mem_env_dir

    env_dir_str = str(env_dir)
    now = time.time()

    if (not force
            and _mem_index is not None
            and _mem_env_dir == env_dir_str
            and (now - _mem_index_ts) < _MEM_TTL):
        _mem_index["_changed_buckets"] = []
        return _mem_index

    with _lock:
        if (not force
                and _mem_index is not None
                and _mem_env_dir == env_dir_str
                and (time.time() - _mem_index_ts) < _MEM_TTL):
            _mem_index["_changed_buckets"] = []
            return _mem_index

        index = _mem_index if (_mem_env_dir == env_dir_str and _mem_index) else _load_index(env_dir)
        dirs_cache = index.get("dirs", {})
        changed = False
        active_tag = _get_active_tag()
        changed_buckets: list = []

        current_dirs = set()
        if env_dir.is_dir():
            for sub in env_dir.iterdir():
                if not sub.is_dir():
                    continue
                dir_name = sub.name
                current_dirs.add(dir_name)

                prev = dirs_cache.get(dir_name)

                if (not force and prev
                        and prev.get("frozen")
                        and dir_name != active_tag):
                    continue

                cache_file = sub / ".session_cache.jsonl"
                if not cache_file.is_file():
                    if prev and not force and dir_name != active_tag:
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
                    old_sessions = prev.get("sessions", {}) if prev else {}
                    new_sessions: dict = {}
                    changed_buckets.extend(_diff_sessions(old_sessions, new_sessions, dir_name))
                    req_count = _count_req_files(sub)
                    dirs_cache[dir_name] = {
                        "cache_mtime": 0,
                        "cache_size": 0,
                        "req_count": req_count,
                        "req_count_mtime": dir_mt,
                        "frozen": dir_name != active_tag,
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
                    if dir_name != active_tag and not prev.get("frozen"):
                        prev["frozen"] = True
                        changed = True
                    continue

                sessions = _scan_session_cache(cache_file, threshold)

                old_sessions = prev.get("sessions", {}) if prev else {}
                changed_buckets.extend(_diff_sessions(old_sessions, sessions, dir_name))

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

        removed = set(dirs_cache.keys()) - current_dirs
        if removed:
            for r in removed:
                old_sessions = dirs_cache[r].get("sessions", {})
                for bk, counts in old_sessions.items():
                    changed_buckets.append((r, bk, counts, None))
                del dirs_cache[r]
            changed = True

        if changed:
            index["dirs"] = dirs_cache
            index["updated_at"] = time.time()
            _save_index(env_dir, index)

        index["_changed_buckets"] = changed_buckets

        _mem_index = index
        _mem_index_ts = time.time()
        _mem_env_dir = env_dir_str

        return index


# ---------------------------------------------------------------------------
# key 聚合缓存 — 持久化 + 增量更新
# ---------------------------------------------------------------------------

def _key_cache_path(env_dir: Path) -> str:
    return str(env_dir / _KEY_CACHE_FILE)


def _load_key_cache(env_dir: Path) -> Optional[dict]:
    path = _key_cache_path(env_dir)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        v = data.get("_version", 0)
        if v < 1:
            return None
        if v < _KEY_CACHE_VERSION:
            data["_version"] = _KEY_CACHE_VERSION
            data.setdefault("key_meta", {})
            data.setdefault("key_records", {})
        return data
    except (OSError, json.JSONDecodeError):
        return None


def _save_key_cache(env_dir: Path, cache: dict) -> None:
    path = _key_cache_path(env_dir)
    try:
        os.makedirs(str(env_dir), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)
        os.replace(tmp, path)
    except OSError:
        pass


def _ensure_cell(table: dict, api_key: str, sub_key: str) -> dict:
    if api_key not in table:
        table[api_key] = {}
    if sub_key not in table[api_key]:
        table[api_key][sub_key] = {"total": 0, "qualified": 0}
    return table[api_key][sub_key]


def _apply_delta(cache: dict, dir_name: str, bucket_key: str,
                 old_c: Optional[dict], new_c: Optional[dict]) -> None:
    """对 cache 的 table/mtime_table/totals 做一次 bucket 级增量。"""
    parts = bucket_key.split("|", 1)
    if len(parts) != 2:
        return
    api_key, date_str = parts

    table = cache["table"]
    mtime_table = cache["mtime_table"]
    totals = cache["totals"]

    if old_c:
        cell = _ensure_cell(table, api_key, date_str)
        cell["total"] -= old_c.get("total", 0)
        cell["qualified"] -= old_c.get("qualified", 0)

        mt_cell = _ensure_cell(mtime_table, api_key, dir_name)
        mt_cell["total"] -= old_c.get("total", 0)
        mt_cell["qualified"] -= old_c.get("qualified", 0)

        totals["total"] -= old_c.get("total", 0)
        totals["qualified"] -= old_c.get("qualified", 0)

    if new_c:
        cell = _ensure_cell(table, api_key, date_str)
        cell["total"] += new_c.get("total", 0)
        cell["qualified"] += new_c.get("qualified", 0)

        mt_cell = _ensure_cell(mtime_table, api_key, dir_name)
        mt_cell["total"] += new_c.get("total", 0)
        mt_cell["qualified"] += new_c.get("qualified", 0)

        totals["total"] += new_c.get("total", 0)
        totals["qualified"] += new_c.get("qualified", 0)


def _cleanup_zeros(cache: dict) -> None:
    """清理 total=0 的空条目。"""
    for tbl in (cache["table"], cache["mtime_table"]):
        empty_keys = []
        for api_key, sub in tbl.items():
            empty_subs = [k for k, v in sub.items() if v.get("total", 0) <= 0]
            for es in empty_subs:
                del sub[es]
            if not sub:
                empty_keys.append(api_key)
        for ek in empty_keys:
            del tbl[ek]


def _full_build_key_cache(index: dict) -> dict:
    """从 index 全量构建 key cache。"""
    table: Dict[str, Dict[str, Dict[str, int]]] = {}
    mtime_table: Dict[str, Dict[str, Dict[str, int]]] = {}
    total = 0
    qualified = 0

    for dir_name, dir_info in index.get("dirs", {}).items():
        sessions = dir_info.get("sessions", {})
        if not sessions:
            continue
        for bucket_key, counts in sessions.items():
            parts = bucket_key.split("|", 1)
            if len(parts) != 2:
                continue
            api_key, date_str = parts
            t = counts["total"]
            q = counts["qualified"]
            total += t
            qualified += q

            cell = _ensure_cell(table, api_key, date_str)
            cell["total"] += t
            cell["qualified"] += q

            mt_cell = _ensure_cell(mtime_table, api_key, dir_name)
            mt_cell["total"] += t
            mt_cell["qualified"] += q

    return {
        "_version": _KEY_CACHE_VERSION,
        "updated_at": time.time(),
        "table": table,
        "mtime_table": mtime_table,
        "totals": {"total": total, "qualified": qualified},
        "key_meta": {},
        "key_records": {},
    }


def _build_rows(cache: dict, threshold: int) -> dict:
    """从 key cache 构建 build_stats_from_index 的输出格式。"""
    table = cache.get("table", {})
    mtime_table = cache.get("mtime_table", {})
    totals = cache.get("totals", {})

    all_dates: set = set()
    for sub in table.values():
        all_dates.update(sub.keys())
    all_mtimes: set = set()
    for sub in mtime_table.values():
        all_mtimes.update(sub.keys())

    dates = sorted(all_dates)
    keys = sorted(table.keys())

    rows = []
    for key in keys:
        row_total = 0
        row_qualified = 0
        cells = {}
        for d in dates:
            c = table[key].get(d, {"total": 0, "qualified": 0})
            if c["total"] > 0:
                cells[d] = {"total": c["total"], "qualified": c["qualified"]}
                row_total += c["total"]
                row_qualified += c["qualified"]
        mtime_cells = {}
        for mt in sorted(all_mtimes):
            mc = mtime_table.get(key, {}).get(mt, {"total": 0, "qualified": 0})
            if mc["total"] > 0:
                mtime_cells[mt] = {"total": mc["total"], "qualified": mc["qualified"]}
        if row_total > 0:
            rows.append({
                "api_key": key,
                "cells": cells,
                "mtime_cells": mtime_cells,
                "row_total": row_total,
                "row_qualified": row_qualified,
            })

    return {
        "total_sessions": totals.get("total", 0),
        "qualified_sessions": totals.get("qualified", 0),
        "threshold": threshold,
        "dates": dates,
        "rows": rows,
    }


def build_stats_from_index(index: dict, threshold: int = QUALIFIED_THRESHOLD_DEFAULT,
                           env_dir: Optional[Path] = None) -> dict:
    """从索引数据构建统计结果，支持 bucket 级增量更新 + 磁盘持久化。

    三条路径:
      A) _changed_buckets 为空 → 直接从内存 key cache 构建 rows
      B) _changed_buckets 非空 → 定向增量更新 key cache，持久化后构建 rows
      C) 首次 / env 切换 → 全量构建，持久化后构建 rows
    """
    global _mem_key_cache, _mem_key_cache_env

    changed_buckets = index.get("_changed_buckets")
    _env_dir = env_dir
    if _env_dir is None:
        _env_dir = Path(_mem_env_dir) if _mem_env_dir else None
    env_dir_str = str(_env_dir) if _env_dir else ""

    is_same_env = (_mem_key_cache is not None and _mem_key_cache_env == env_dir_str)

    if is_same_env and changed_buckets is not None and len(changed_buckets) == 0:
        return _build_rows(_mem_key_cache, threshold)

    if is_same_env and changed_buckets:
        for dir_name, bk, old_c, new_c in changed_buckets:
            _apply_delta(_mem_key_cache, dir_name, bk, old_c, new_c)
        _cleanup_zeros(_mem_key_cache)
        _mem_key_cache["updated_at"] = time.time()
        if _env_dir:
            _save_key_cache(_env_dir, _mem_key_cache)
        return _build_rows(_mem_key_cache, threshold)

    if _env_dir and not is_same_env:
        disk_cache = _load_key_cache(_env_dir)
        if disk_cache:
            idx_dirs = {d for d, info in index.get("dirs", {}).items() if info.get("sessions")}
            cache_dirs = set()
            for sub in disk_cache.get("mtime_table", {}).values():
                cache_dirs.update(sub.keys())
            if idx_dirs - cache_dirs:
                disk_cache = None

        if disk_cache:
            _mem_key_cache = disk_cache
            _mem_key_cache_env = env_dir_str
            if changed_buckets:
                for dir_name, bk, old_c, new_c in changed_buckets:
                    _apply_delta(_mem_key_cache, dir_name, bk, old_c, new_c)
                _cleanup_zeros(_mem_key_cache)
                _mem_key_cache["updated_at"] = time.time()
                _save_key_cache(_env_dir, _mem_key_cache)
            return _build_rows(_mem_key_cache, threshold)

    cache = _full_build_key_cache(index)
    _mem_key_cache = cache
    _mem_key_cache_env = env_dir_str
    if _env_dir:
        _save_key_cache(_env_dir, cache)
    return _build_rows(cache, threshold)


# ---------------------------------------------------------------------------
# key 元数据 + records 缓存（供 /api/export/keys 使用）
# ---------------------------------------------------------------------------

def get_current_key_cache() -> Optional[dict]:
    """返回当前内存中的 key cache（引用，非拷贝），供外部读取 key_meta / key_records。"""
    return _mem_key_cache


def _key_slot(api_key: str) -> str:
    if not api_key or api_key == "(empty)":
        return "all"
    return "key-" + api_key[-4:]


def _compute_keys_hash(db_keys_list: list) -> str:
    raw = "|".join(
        f"{k.get('key', '')}:{k.get('name', '')}:{k.get('created_at', '')}"
        for k in db_keys_list
    )
    return hashlib.md5(raw.encode()).hexdigest()


def update_key_meta(cache: dict, db_keys_list: list) -> bool:
    """用 list_keys() 的结果更新 cache 中的 key_meta。返回是否有变更。

    db_keys_list: list_keys() 返回的 [{key, name, created_at, ...}, ...]
    """
    if cache is None:
        return False

    meta = cache.get("key_meta") or {}
    new_hash = _compute_keys_hash(db_keys_list)

    if meta.get("db_keys_hash") == new_hash and meta.get("mapping"):
        return False

    mapping: dict = {}
    for row_key in cache.get("table", {}):
        matched_name = ""
        created_at = ""
        for k in db_keys_list:
            masked = k.get("key", "")
            if row_key != "(empty)" and len(row_key) >= 4 and masked.endswith(row_key[-4:]):
                matched_name = k.get("name", "")
                created_at = k.get("created_at", "")
                break
        mapping[row_key] = {
            "key_name": matched_name,
            "key_slot": _key_slot(row_key),
            "created_at": created_at,
        }

    cache["key_meta"] = {"db_keys_hash": new_hash, "mapping": mapping}
    return True


def update_key_records(cache: dict, env_dir: Optional[Path] = None) -> bool:
    """检查 export_records 是否有变化，有变化则一次性刷新 records 缓存。返回是否有变更。"""
    if cache is None:
        return False

    try:
        from utils.export_store import get_records_summary, list_records_all_slim
    except ImportError:
        return False

    max_id, count, running = get_records_summary()
    rec_cache = cache.get("key_records") or {}

    if (rec_cache.get("db_max_id") == max_id
            and rec_cache.get("db_count") == count
            and rec_cache.get("db_running") == running
            and rec_cache.get("records")):
        return False

    grouped = list_records_all_slim(limit_per_key=10)
    cache["key_records"] = {
        "db_max_id": max_id,
        "db_count": count,
        "db_running": running,
        "records": grouped,
    }

    if env_dir:
        _save_key_cache(env_dir, cache)

    return True


# ---------------------------------------------------------------------------
# 查询接口
# ---------------------------------------------------------------------------

def get_dir_counts(index: dict, token_index: Optional[dict] = None) -> Dict[str, int]:
    """从索引中提取每个 mtime_dir 的请求数，供 /logs/dirs 使用。

    优先用 token_index 的 entry_count（已增量维护，无需 iterdir），
    token_index 中没有该目录时回退到 stats_index 的 req_count。
    """
    tok_dirs = token_index.get("dirs", {}) if token_index else {}
    result = {}
    for dir_name, dir_info in index.get("dirs", {}).items():
        tok_info = tok_dirs.get(dir_name)
        if tok_info is not None and tok_info.get("entry_count", 0) > 0:
            result[dir_name] = tok_info["entry_count"]
        else:
            result[dir_name] = dir_info.get("req_count", 0)
    return result


def get_date_to_mtime_map(index: dict) -> Dict[str, List[str]]:
    """从索引中构建 date -> [mtime_dir, ...] 映射。"""
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


# ---------------------------------------------------------------------------
# 后台预热线程 — stats_index
# ---------------------------------------------------------------------------

_stats_warmer_thread = None
_stats_warmer_stop = threading.Event()
_stats_logger = logging.getLogger("stats-warmer")

_last_refresh_ts: float = 0.0


def get_last_refresh_ts() -> float:
    return _last_refresh_ts


def _stats_warmer_loop(env_dir: str, threshold: int, interval: float = 30.0) -> None:
    global _last_refresh_ts
    env_path = Path(env_dir)
    _stats_logger.info("stats-warmer: 开始预热 env_dir=%s", env_dir)

    index = refresh_index(env_path, threshold, force=True)
    build_stats_from_index(index, threshold, env_dir=env_path)
    _last_refresh_ts = time.time()
    _stats_logger.info("stats-warmer: 首次预热完成")

    while not _stats_warmer_stop.is_set():
        _stats_warmer_stop.wait(interval)
        if _stats_warmer_stop.is_set():
            break
        index = refresh_index(env_path, threshold, force=True)
        build_stats_from_index(index, threshold, env_dir=env_path)
        _last_refresh_ts = time.time()


def start_stats_warmer(env_dir: str, threshold: int = QUALIFIED_THRESHOLD_DEFAULT) -> None:
    global _stats_warmer_thread
    if _stats_warmer_thread is not None and _stats_warmer_thread.is_alive():
        return
    _stats_warmer_stop.clear()
    _stats_warmer_thread = threading.Thread(
        target=_stats_warmer_loop,
        args=(env_dir, threshold),
        daemon=True,
        name="stats-cache-warmer",
    )
    _stats_warmer_thread.start()
