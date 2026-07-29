"""
utils/token_index.py — Token 用量增量索引

持久化文件: {env_dir}/.token_index.jsonl (JSONL, 第一行 meta)
进程级内存缓存: 10s TTL
frozen 机制: index.jsonl 文件 mtime/size 不变时标记 frozen，下次跳过

扫描 {env_dir}/{mtime}/index.jsonl，按 (model|date, status) / (api_key|date) /
(channel_key|date) 三个维度预聚合。查询时从内存中按日期范围过滤汇总。
"""

import hashlib
import json
import os
import time
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.log_scan import iter_index_dirs, dir_key_for, normalize_entry

_INDEX_FILE = ".token_index.jsonl"
_VERSION = 4  # v4: 多格式归一化（new-api usage.token_in / 无 success 字段）
_lock = threading.Lock()

_MEM_TTL = 10
# 按 root 缓存，避免多目录轮询时单槽缓存互相驱逐
_mem_cache: Dict[str, dict] = {}      # root_str -> index dict
_mem_cache_ts: Dict[str, float] = {}  # root_str -> 最后刷新时间

# 项目内缓存目录（当 root 不可写，如只读挂载的 new-api 目录时使用）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_FALLBACK_CACHE_DIR = _PROJECT_ROOT / "logs" / ".token_index_cache"


def _resolve_cache_path(env_dir: Path) -> str:
    """索引缓存文件路径。root 可写则存 root 内（兼容 logs_all 旧行为），
    否则退回项目内 logs/.token_index_cache/{hash}.jsonl。"""
    in_root = env_dir / _INDEX_FILE
    try:
        if env_dir.is_dir() and os.access(str(env_dir), os.W_OK):
            return str(in_root)
    except OSError:
        pass
    h = hashlib.md5(os.path.normpath(str(env_dir)).encode()).hexdigest()[:16]
    try:
        os.makedirs(str(_FALLBACK_CACHE_DIR), exist_ok=True)
    except OSError:
        pass
    return str(_FALLBACK_CACHE_DIR / f"{h}.jsonl")


def _index_path(env_dir: Path) -> str:
    return _resolve_cache_path(env_dir)


def _load_index(env_dir: Path) -> dict:
    path = _index_path(env_dir)
    if not os.path.isfile(path):
        return {"version": _VERSION, "dirs": {}, "updated_at": 0}
    try:
        dirs = {}
        meta = None
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("_meta"):
                    meta = obj
                else:
                    dir_key = obj.get("dir", "")
                    if dir_key:
                        dirs[dir_key] = obj
        version = meta.get("version", 0) if meta else 0
        if version != _VERSION:
            return {"version": _VERSION, "dirs": {}, "updated_at": 0}
        return {
            "version": _VERSION,
            "dirs": dirs,
            "updated_at": meta.get("updated_at", 0) if meta else 0,
        }
    except OSError:
        return {"version": _VERSION, "dirs": {}, "updated_at": 0}


def _save_index(env_dir: Path, data: dict) -> None:
    path = _index_path(env_dir)
    try:
        os.makedirs(str(env_dir), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            meta = {"_meta": True, "version": _VERSION, "updated_at": data.get("updated_at", 0)}
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
            for dir_key, dir_info in sorted(data.get("dirs", {}).items()):
                row = dict(dir_info)
                row["dir"] = dir_key
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
    except OSError:
        pass


def _scan_index_file(index_file: Path, offset: int = 0,
                     prev: Optional[dict] = None) -> dict:
    """读一个 index.jsonl，预聚合为 models/keys/channels 三个维度（按日期分桶）。"""
    if prev and offset > 0:
        models = {k: dict(v) for k, v in prev.get("models", {}).items()}
        keys = {k: dict(v) for k, v in prev.get("keys", {}).items()}
        channels = {k: dict(v) for k, v in prev.get("channels", {}).items()}
        channel_keys_set = set(prev.get("channel_keys_set", []))
        api_keys_set = set(prev.get("api_keys_set", []))
        dates_set = set(prev.get("dates", []))
        entry_count = prev.get("entry_count", 0)
    else:
        models = {}
        keys = {}
        channels = {}
        channel_keys_set = set()
        api_keys_set = set()
        dates_set = set()
        entry_count = 0

    new_entries = 0
    end_offset = offset

    try:
        with open(index_file, "r", encoding="utf-8") as f:
            if offset > 0:
                f.seek(offset)
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    entry = json.loads(stripped)
                except json.JSONDecodeError:
                    continue

                entry = normalize_entry(entry)

                new_entries += 1
                entry_count += 1
                date_str = entry["date"]
                dates_set.add(date_str)

                model = entry["model"]
                tok_in = entry["tok_in"]
                tok_out = entry["tok_out"]
                is_success = entry["success"]

                # models dimension
                if model:
                    mk = f"{model}|{date_str}"
                    if mk not in models:
                        models[mk] = {
                            "s_count": 0, "s_tok_in": 0, "s_tok_out": 0,
                            "e_count": 0, "e_tok_in": 0,
                        }
                    if is_success:
                        models[mk]["s_count"] += 1
                        models[mk]["s_tok_in"] += tok_in
                        models[mk]["s_tok_out"] += tok_out
                    else:
                        models[mk]["e_count"] += 1
                        models[mk]["e_tok_in"] += tok_in

                # keys dimension
                raw_key = entry["api_key"]
                if raw_key:
                    api_keys_set.add(raw_key)
                kk = f"{raw_key}|{date_str}"
                if kk not in keys:
                    keys[kk] = {"count": 0, "tok_in": 0, "tok_out": 0, "sessions": 0}
                keys[kk]["count"] += 1
                keys[kk]["tok_in"] += tok_in
                keys[kk]["tok_out"] += tok_out
                sess_id = entry["q1_hash"] or entry["chain_key"]
                if sess_id:
                    keys[kk]["sessions"] += 1

                # channels dimension
                ch_key = entry["channel_key"]
                if ch_key:
                    channel_keys_set.add(ch_key)
                ck = f"{ch_key or '(default)'}|{date_str}"
                if ck not in channels:
                    channels[ck] = {"count": 0, "tok_in": 0, "tok_out": 0, "sessions": 0}
                channels[ck]["count"] += 1
                channels[ck]["tok_in"] += tok_in
                channels[ck]["tok_out"] += tok_out
                if sess_id:
                    channels[ck]["sessions"] += 1

            end_offset = f.tell()

    except OSError:
        pass

    return {
        "entry_count": entry_count,
        "new_entries": new_entries,
        "scan_offset": end_offset,
        "models": models,
        "keys": keys,
        "channels": channels,
        "channel_keys_set": sorted(channel_keys_set),
        "api_keys_set": sorted(api_keys_set),
        "dates": sorted(dates_set),
    }


def refresh_token_index(env_dir: str, force: bool = False) -> dict:
    """增量刷新单个 root 的 token 索引。

    root 下用可变深度扫描找到所有含 index.jsonl 的叶子目录，
    dir_key 为叶子相对 root 的路径（如 '26072520' 或 '260727/26072717'）。
    每个 root 独立缓存，避免多目录轮询时互相驱逐。
    """
    env_path = Path(env_dir)
    env_dir_str = os.path.normpath(str(env_path))
    now = time.time()

    cached = _mem_cache.get(env_dir_str)
    if (not force and cached is not None
            and (now - _mem_cache_ts.get(env_dir_str, 0)) < _MEM_TTL):
        return cached

    with _lock:
        cached = _mem_cache.get(env_dir_str)
        if (not force and cached is not None
                and (time.time() - _mem_cache_ts.get(env_dir_str, 0)) < _MEM_TTL):
            return cached

        index = cached if cached else _load_index(env_path)
        dirs_cache = index.get("dirs", {})
        changed = False

        current_dirs = set()
        for leaf in iter_index_dirs(env_path):
            dir_key = dir_key_for(env_path, leaf)
            current_dirs.add(dir_key)

            prev = dirs_cache.get(dir_key)

            if not force and prev and prev.get("frozen"):
                continue

            index_file = leaf / "index.jsonl"
            try:
                st = index_file.stat()
                f_mtime = st.st_mtime
                f_size = st.st_size
            except OSError:
                continue

            if (not force and prev
                    and prev.get("index_mtime") == f_mtime
                    and prev.get("index_size") == f_size):
                if not prev.get("frozen"):
                    prev["frozen"] = True
                    changed = True
                continue

            prev_offset = prev.get("scan_offset", 0) if prev else 0
            if (not force and prev
                    and prev_offset > 0
                    and f_size > prev.get("index_size", 0)):
                scan_result = _scan_index_file(index_file, offset=prev_offset, prev=prev)
            else:
                scan_result = _scan_index_file(index_file)
            dirs_cache[dir_key] = {
                "index_mtime": f_mtime,
                "index_size": f_size,
                "scan_offset": scan_result.get("scan_offset", f_size),
                "frozen": False,
                **{k: v for k, v in scan_result.items() if k not in ("new_entries", "scan_offset")},
            }
            changed = True

        # 只删除非 frozen 的消失目录；frozen 代表数据已稳定，本地删除后仍保留统计
        removed = set(dirs_cache.keys()) - current_dirs
        for r in removed:
            if not dirs_cache[r].get("frozen"):
                del dirs_cache[r]
                changed = True

        if changed:
            index["dirs"] = dirs_cache
            index["updated_at"] = time.time()
            _save_index(env_path, index)

        _mem_cache[env_dir_str] = index
        _mem_cache_ts[env_dir_str] = time.time()

        return index


# ---------------------------------------------------------------------------
# 查询接口（从索引中按参数过滤聚合）
# ---------------------------------------------------------------------------

def _mask_api_key(key: str) -> str:
    if not key or len(key) <= 8:
        return key or "(empty)"
    return f"{key[:4]}...{key[-4:]}"


def _as_roots(env_dir) -> List[str]:
    """把 env_dir 归一化为 root 列表（接受 str 或 list）。"""
    if env_dir is None:
        return []
    if isinstance(env_dir, (str, os.PathLike)):
        return [str(env_dir)]
    return [str(x) for x in env_dir if x]


def _combined_index(env_dir, force: bool = False) -> dict:
    """刷新一个或多个 root，合并为一个 index（dir_key 加 root 前缀防冲突）。"""
    roots = _as_roots(env_dir)
    merged: Dict[str, Any] = {}
    for root in roots:
        idx = refresh_token_index(root, force=force)
        prefix = os.path.normpath(root)
        for dir_key, dir_info in idx.get("dirs", {}).items():
            merged[f"{prefix}::{dir_key}"] = dir_info
    return {"dirs": merged}


def query_token_stats(env_dir, model: str = '', date_start: str = '2000-01-01',
                      date_end: str = '9999-12-31', status: str = '',
                      channel_key: str = '', api_key: str = '',
                      force: bool = False) -> dict:
    """从索引中聚合 token 统计。env_dir 可为单个 root 或 root 列表。"""
    index = _combined_index(env_dir, force=force)

    model_filter = model.lower() if model else ""
    model_agg: Dict[str, Dict[str, Dict[str, int]]] = {}

    for dir_info in index.get("dirs", {}).values():
        dir_dates = dir_info.get("dates", [])
        if dir_dates and dir_dates[-1] < date_start:
            continue
        if dir_dates and dir_dates[0] > date_end:
            continue

        if channel_key and channel_key not in (dir_info.get("channel_keys_set") or []):
            continue

        if api_key and api_key not in (dir_info.get("api_keys_set") or []):
            continue

        for mk, counts in dir_info.get("models", {}).items():
            parts = mk.rsplit("|", 1)
            if len(parts) != 2:
                continue
            m_name, m_date = parts

            if not (date_start <= m_date <= date_end):
                continue
            if model_filter and model_filter not in m_name.lower():
                continue

            if m_name not in model_agg:
                model_agg[m_name] = {
                    "success": {"count": 0, "tok_in": 0, "tok_out": 0},
                    "error": {"count": 0, "tok_in": 0},
                }
            model_agg[m_name]["success"]["count"] += counts.get("s_count", 0)
            model_agg[m_name]["success"]["tok_in"] += counts.get("s_tok_in", 0)
            model_agg[m_name]["success"]["tok_out"] += counts.get("s_tok_out", 0)
            model_agg[m_name]["error"]["count"] += counts.get("e_count", 0)
            model_agg[m_name]["error"]["tok_in"] += counts.get("e_tok_in", 0)

    res_data = []
    for m_name, agg in model_agg.items():
        if status in ("", "全部", "success", "成功"):
            if agg["success"]["count"] > 0:
                res_data.append({
                    "model": m_name, "date_start": date_start, "date_end": date_end,
                    "status": "success", "count": agg["success"]["count"],
                    "input_token_num": agg["success"]["tok_in"],
                    "output_token_num": agg["success"]["tok_out"],
                })
        if status in ("", "全部", "error", "失败"):
            if agg["error"]["count"] > 0:
                res_data.append({
                    "model": m_name, "date_start": date_start, "date_end": date_end,
                    "status": "error", "count": agg["error"]["count"],
                    "input_token_num": agg["error"]["tok_in"],
                    "output_token_num": 0,
                })

    s_count = sum(d["count"] for d in res_data if d["status"] == "success")
    s_tok_in = sum(d["input_token_num"] for d in res_data if d["status"] == "success")
    s_tok_out = sum(d["output_token_num"] for d in res_data if d["status"] == "success")
    e_count = sum(d["count"] for d in res_data if d["status"] == "error")
    e_tok_in = sum(d["input_token_num"] for d in res_data if d["status"] == "error")

    summary = [
        {"status": "success", "count": s_count, "total_input": s_tok_in, "total_output": s_tok_out},
        {"status": "error", "count": e_count, "total_input": e_tok_in, "total_output": 0},
    ]

    return {"data": res_data, "summary": summary}


def query_key_stats(env_dir, date_start: str = '2000-01-01', date_end: str = '9999-12-31',
                    force: bool = False) -> dict:
    """从索引中聚合 key 统计。env_dir 可为单个 root 或 root 列表。"""
    index = _combined_index(env_dir, force=force)

    key_agg: Dict[str, Dict[str, Any]] = {}

    for dir_info in index.get("dirs", {}).values():
        dir_dates = dir_info.get("dates", [])
        if dir_dates and dir_dates[-1] < date_start:
            continue
        if dir_dates and dir_dates[0] > date_end:
            continue

        for kk, counts in dir_info.get("keys", {}).items():
            parts = kk.rsplit("|", 1)
            if len(parts) != 2:
                continue
            raw_key, k_date = parts

            if not (date_start <= k_date <= date_end):
                continue

            if not raw_key:
                raw_key = "(unknown)"
            if raw_key not in key_agg:
                key_agg[raw_key] = {"count": 0, "tok_in": 0, "tok_out": 0, "sessions": 0}
            key_agg[raw_key]["count"] += counts.get("count", 0)
            key_agg[raw_key]["tok_in"] += counts.get("tok_in", 0)
            key_agg[raw_key]["tok_out"] += counts.get("tok_out", 0)
            key_agg[raw_key]["sessions"] += counts.get("sessions", 0)

    keys_list = [
        {
            "key": _mask_api_key(k),
            "count": v["count"], "tok_in": v["tok_in"],
            "tok_out": v["tok_out"],
            "sessions": v["sessions"] or v["count"],
        }
        for k, v in key_agg.items()
    ]
    keys_list.sort(key=lambda x: x["count"], reverse=True)
    return {"keys": keys_list}


def query_channel_stats(env_dir, date_start: str = '2000-01-01', date_end: str = '9999-12-31',
                        force: bool = False) -> dict:
    """从索引中聚合 channel 统计。env_dir 可为单个 root 或 root 列表。"""
    index = _combined_index(env_dir, force=force)

    ch_agg: Dict[str, Dict[str, Any]] = {}

    for dir_info in index.get("dirs", {}).values():
        dir_dates = dir_info.get("dates", [])
        if dir_dates and dir_dates[-1] < date_start:
            continue
        if dir_dates and dir_dates[0] > date_end:
            continue

        for ck, counts in dir_info.get("channels", {}).items():
            parts = ck.rsplit("|", 1)
            if len(parts) != 2:
                continue
            ch_key, c_date = parts

            if not (date_start <= c_date <= date_end):
                continue

            if ch_key not in ch_agg:
                ch_agg[ch_key] = {"count": 0, "tok_in": 0, "tok_out": 0, "sessions": 0}
            ch_agg[ch_key]["count"] += counts.get("count", 0)
            ch_agg[ch_key]["tok_in"] += counts.get("tok_in", 0)
            ch_agg[ch_key]["tok_out"] += counts.get("tok_out", 0)
            ch_agg[ch_key]["sessions"] += counts.get("sessions", 0)

    ch_list = [
        {
            "key": _mask_api_key(k),
            "count": v["count"], "tok_in": v["tok_in"],
            "tok_out": v["tok_out"],
            "sessions": v["sessions"] or v["count"],
        }
        for k, v in ch_agg.items()
    ]
    ch_list.sort(key=lambda x: x["count"], reverse=True)
    return {"channels": ch_list}


def query_channel_keys(env_dir, force: bool = False) -> List[str]:
    """从索引中提取所有 channel key。env_dir 可为单个 root 或 root 列表。"""
    index = _combined_index(env_dir, force=force)
    keys: set = set()
    for dir_info in index.get("dirs", {}).values():
        for ck in dir_info.get("channel_keys_set", []):
            keys.add(ck)
    return sorted(keys)


def query_api_keys(env_dir, force: bool = False) -> List[str]:
    """从索引中提取所有 api key。env_dir 可为单个 root 或 root 列表。"""
    index = _combined_index(env_dir, force=force)
    keys: set = set()
    for dir_info in index.get("dirs", {}).values():
        for ak in dir_info.get("api_keys_set", []):
            keys.add(ak)
    return sorted(keys)
