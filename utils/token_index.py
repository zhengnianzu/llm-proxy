"""
utils/token_index.py — Token 用量增量索引

持久化文件: logs_all/.token_index.jsonl (JSONL, 第一行 meta)
进程级内存缓存: 10s TTL
frozen 机制: 历史目录一旦扫描过就不再 stat

扫描 logs_all/{env}/{mtime}/index.jsonl，按 (model|date, status) / (api_key|date) /
(channel_key|date) 三个维度预聚合。查询时从内存中按日期范围过滤汇总。
"""

import json
import os
import time
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

_INDEX_FILE = ".token_index.jsonl"
_VERSION = 1
_lock = threading.Lock()

# 进程级内存缓存
_MEM_TTL = 10
_mem_index: Optional[dict] = None
_mem_index_ts: float = 0

LOGS_ALL = Path("logs_all")


def _get_active_tag() -> str:
    try:
        from utils.log_paths import STARTUP_DATE_TAG
        return STARTUP_DATE_TAG
    except ImportError:
        return ""


def _index_path() -> str:
    return str(LOGS_ALL / _INDEX_FILE)


def _load_index() -> dict:
    path = _index_path()
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


def _save_index(data: dict) -> None:
    path = _index_path()
    try:
        os.makedirs(str(LOGS_ALL), exist_ok=True)
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
    """读一个 index.jsonl，预聚合为 models/keys/channels 三个维度（按日期分桶）。

    支持增量读取：传入 offset（上次读到的字节位置）和 prev（上次的聚合结果），
    只读新追加的行并合并到已有数据中。
    """
    if prev and offset > 0:
        models = {k: dict(v) for k, v in prev.get("models", {}).items()}
        keys = {k: dict(v) for k, v in prev.get("keys", {}).items()}
        channels = {k: dict(v) for k, v in prev.get("channels", {}).items()}
        channel_keys_set = set(prev.get("channel_keys_set", []))
        dates_set = set(prev.get("dates", []))
        entry_count = prev.get("entry_count", 0)
        # sessions 字段：持久化时存的是 count(int)，需要保留为 int 继续累加
    else:
        models = {}
        keys = {}
        channels = {}
        channel_keys_set = set()
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

                new_entries += 1
                entry_count += 1
                ts = entry.get("ts", "")
                date_str = ts[:10] if len(ts) >= 10 else "unknown"
                dates_set.add(date_str)

                model = entry.get("model", "") or ""
                tok_in = entry.get("tok_in", 0) or 0
                tok_out = entry.get("tok_out", 0) or 0

                if "valid" in entry:
                    is_success = bool(entry["valid"]) and tok_out > 0
                else:
                    is_success = bool(entry.get("success", False)) and tok_out > 0

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
                raw_key = entry.get("api_key", "") or ""
                kk = f"{raw_key}|{date_str}"
                if kk not in keys:
                    keys[kk] = {"count": 0, "tok_in": 0, "tok_out": 0, "sessions": 0}
                keys[kk]["count"] += 1
                keys[kk]["tok_in"] += tok_in
                keys[kk]["tok_out"] += tok_out
                chain_key = entry.get("chain_key", "")
                if chain_key:
                    keys[kk]["sessions"] += 1

                # channels dimension
                ch_key = entry.get("channel_key", "") or ""
                if ch_key:
                    channel_keys_set.add(ch_key)
                ck = f"{ch_key or '(default)'}|{date_str}"
                if ck not in channels:
                    channels[ck] = {"count": 0, "tok_in": 0, "tok_out": 0, "sessions": 0}
                channels[ck]["count"] += 1
                channels[ck]["tok_in"] += tok_in
                channels[ck]["tok_out"] += tok_out
                if chain_key:
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
        "dates": sorted(dates_set),
    }


def refresh_token_index(force: bool = False) -> dict:
    """增量刷新 token 索引。"""
    global _mem_index, _mem_index_ts

    now = time.time()
    if not force and _mem_index is not None and (now - _mem_index_ts) < _MEM_TTL:
        return _mem_index

    with _lock:
        if not force and _mem_index is not None and (time.time() - _mem_index_ts) < _MEM_TTL:
            return _mem_index

        index = _mem_index if _mem_index is not None else _load_index()
        dirs_cache = index.get("dirs", {})
        changed = False
        active_tag = _get_active_tag()

        current_dirs = set()
        if LOGS_ALL.is_dir():
            for env_dir in LOGS_ALL.iterdir():
                if not env_dir.is_dir() or env_dir.name.startswith("logs_") or env_dir.name.startswith("."):
                    continue
                for mtime_dir in env_dir.iterdir():
                    if not mtime_dir.is_dir():
                        continue

                    dir_key = f"{env_dir.name}/{mtime_dir.name}"
                    current_dirs.add(dir_key)

                    prev = dirs_cache.get(dir_key)

                    # frozen: 已扫描过的历史目录跳过
                    if (not force and prev
                            and prev.get("frozen")
                            and mtime_dir.name != active_tag):
                        continue

                    index_file = mtime_dir / "index.jsonl"
                    if not index_file.is_file():
                        if not prev:
                            dirs_cache[dir_key] = {
                                "index_mtime": 0, "index_size": 0,
                                "frozen": mtime_dir.name != active_tag,
                                "entry_count": 0,
                                "models": {}, "keys": {}, "channels": {},
                                "channel_keys_set": [], "dates": [],
                            }
                            changed = True
                        continue

                    try:
                        st = index_file.stat()
                        f_mtime = st.st_mtime
                        f_size = st.st_size
                    except OSError:
                        continue

                    if (not force and prev
                            and prev.get("index_mtime") == f_mtime
                            and prev.get("index_size") == f_size):
                        if mtime_dir.name != active_tag and not prev.get("frozen"):
                            prev["frozen"] = True
                            changed = True
                        continue

                    # 增量读取：文件变大但未被截断，只读新追加的部分
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
                        "frozen": mtime_dir.name != active_tag,
                        **{k: v for k, v in scan_result.items() if k not in ("new_entries", "scan_offset")},
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
            _save_index(index)

        _mem_index = index
        _mem_index_ts = time.time()

        return index


# ---------------------------------------------------------------------------
# 查询接口（从索引中按参数过滤聚合）
# ---------------------------------------------------------------------------

def _mask_api_key(key: str) -> str:
    if not key or len(key) <= 8:
        return key or "(empty)"
    return f"{key[:4]}...{key[-4:]}"


def query_token_stats(model: str = '', date_start: str = '2000-01-01',
                      date_end: str = '9999-12-31', status: str = '',
                      channel_key: str = '', force: bool = False) -> dict:
    """替代 statistic_tokens()，从索引中聚合。"""
    index = refresh_token_index(force=force)

    model_filter = model.lower() if model else ""
    model_agg: Dict[str, Dict[str, Dict[str, int]]] = {}

    for dir_info in index.get("dirs", {}).values():
        dir_dates = dir_info.get("dates", [])
        if dir_dates and dir_dates[-1] < date_start:
            continue
        if dir_dates and dir_dates[0] > date_end:
            continue

        # channel_key 过滤: 如果指定了 channel_key 但该目录没有这个 key，可以跳过
        if channel_key and channel_key not in (dir_info.get("channel_keys_set") or []):
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

    # 构建兼容 statistic_tokens 的返回格式
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


def query_key_stats(date_start: str = '2000-01-01', date_end: str = '9999-12-31',
                    force: bool = False) -> dict:
    """替代 statistic_keys()，从索引中聚合。"""
    index = refresh_token_index(force=force)

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


def query_channel_stats(date_start: str = '2000-01-01', date_end: str = '9999-12-31',
                        force: bool = False) -> dict:
    """替代 statistic_channels()，从索引中聚合。"""
    index = refresh_token_index(force=force)

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


def query_channel_keys(force: bool = False) -> List[str]:
    """替代 list_known_channel_keys()，从索引中提取。"""
    index = refresh_token_index(force=force)
    keys: set = set()
    for dir_info in index.get("dirs", {}).values():
        for ck in dir_info.get("channel_keys_set", []):
            keys.add(ck)
    return sorted(keys)
