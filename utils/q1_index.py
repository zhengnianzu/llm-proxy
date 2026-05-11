"""
utils/q1_index.py — Q1 修正机制

当多轮请求到达时，如果 session 已存在且当前 q1 为空或明显不准确，
自动用更准确的 q1 覆盖。修正存储在 q1_overrides.json 侧文件中，
不改写 index.jsonl 本身。
"""

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional

_lock = threading.Lock()
_overrides_cache: Dict[str, Dict[str, Any]] = {}
_overrides_mtime: Dict[str, float] = {}


def _overrides_path(log_dir: str) -> Path:
    return Path(log_dir) / "q1_overrides.json"


def _chain_key_hash(chain_key: str) -> str:
    return hashlib.md5(chain_key.encode("utf-8")).hexdigest()


def _load_overrides(log_dir: str) -> Dict[str, Any]:
    """加载 overrides，带 mtime 缓存避免频繁读盘。"""
    path = _overrides_path(log_dir)
    path_str = str(path)

    try:
        mtime = path.stat().st_mtime
    except OSError:
        return {}

    with _lock:
        if path_str in _overrides_cache and _overrides_mtime.get(path_str) == mtime:
            return _overrides_cache[path_str]

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        data = {}

    with _lock:
        _overrides_cache[path_str] = data
        _overrides_mtime[path_str] = mtime

    return data


def _save_overrides(log_dir: str, data: Dict[str, Any]) -> None:
    path = _overrides_path(log_dir)
    tmp = str(path) + ".tmp"
    try:
        os.makedirs(path.parent, exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=None)
        os.replace(tmp, str(path))
    except OSError:
        return

    with _lock:
        _overrides_cache[str(path)] = data
        try:
            _overrides_mtime[str(path)] = path.stat().st_mtime
        except OSError:
            pass


def should_update_q1(old_q1: str, new_q1: str) -> bool:
    """判断是否应该用 new_q1 替换 old_q1。"""
    if not new_q1:
        return False
    if not old_q1:
        return True
    if len(new_q1) > len(old_q1) * 1.5 and len(old_q1) < 20:
        return True
    if old_q1 == new_q1:
        return False
    if old_q1.startswith("A new session was started"):
        return True
    if old_q1.startswith("[") or old_q1.startswith("Sender"):
        return True
    return False


def update_q1(log_dir: str, chain_key: str, new_q1: str) -> None:
    """写入 q1 修正。"""
    key = _chain_key_hash(chain_key)
    overrides = _load_overrides(log_dir)
    overrides[key] = {"q1": new_q1[:200]}
    _save_overrides(log_dir, overrides)


def get_effective_q1(log_dir: str, chain_key: str, original_q1: str) -> str:
    """获取生效的 q1：优先 override，否则返回原始值。"""
    overrides = _load_overrides(log_dir)
    key = _chain_key_hash(chain_key)
    entry = overrides.get(key)
    if entry and entry.get("q1"):
        return entry["q1"]
    return original_q1


def get_all_overrides(log_dir: str) -> Dict[str, str]:
    """返回所有 override 的 {hash: q1} 映射，供批量合并用。"""
    overrides = _load_overrides(log_dir)
    return {k: v.get("q1", "") for k, v in overrides.items() if v.get("q1")}
