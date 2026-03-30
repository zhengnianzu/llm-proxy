"""
core/cache.py — 缓存管理

记录 task × index 的匹配结果，避免重复计算。
文件 hash 变化时自动失效相关缓存。
"""

import json
from pathlib import Path
from datetime import datetime

from core.config import pair_cache_dir


def _cache_dir():
    return pair_cache_dir()


def _pair_key(task_hash, index_hash):
    return f"{task_hash[:16]}__{index_hash[:16]}"


def _cache_path(pair_key):
    return _cache_dir() / f"{pair_key}.json"


def get_cache(task_hash, index_hash):
    """获取缓存，如果不存在返回 None。"""
    path = _cache_path(_pair_key(task_hash, index_hash))
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def save_cache(result_dict):
    """保存匹配结果到缓存。"""
    pair_key = result_dict.get("pair_key")
    if not pair_key:
        raise ValueError("result_dict must have 'pair_key'")

    path = _cache_path(pair_key)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=2)


def invalidate_by_hash(file_hash):
    """
    文件变化时，失效所有包含该 hash 的缓存。
    """
    for cache_file in _cache_dir().glob("*.json"):
        with open(cache_file, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("task_hash") == file_hash or data.get("index_hash") == file_hash:
            cache_file.unlink()


def list_caches():
    """列出所有缓存文件的元数据。"""
    caches = []
    for cache_file in sorted(_cache_dir().glob("*.json")):
        with open(cache_file, encoding="utf-8") as f:
            data = json.load(f)
        caches.append({
            "pair_key": data.get("pair_key"),
            "task_id": data.get("task_id"),
            "index_id": data.get("index_id"),
            "matched_count": data.get("matched_count", 0),
            "generated_at": data.get("generated_at"),
        })
    return caches


def clear_all():
    """清空所有缓存（谨慎使用）。"""
    for cache_file in _cache_dir().glob("*.json"):
        cache_file.unlink()
