"""
utils/export_sync.py — 导出 session_index.jsonl 并同步到 OBS

export_session_index(logs_dir):
  读取 logs_dir/.session_cache.jsonl → 生成 logs_dir/session_index.jsonl
  输出统计信息（session 总数、平均 msg 轮数）

sync_session_index(logs_dir, obs_dst, ...):
  读取 logs_dir/session_index.jsonl → 上传 latest_file 三元组到 OBS
  目录结构: obs_dst/raw/{文件}, obs_dst/session/{文件}
"""

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from utils.obs_sync import _run_upload_cmd

logger = logging.getLogger(__name__)

SESSION_CACHE_NAME = ".session_cache.jsonl"
SESSION_INDEX_NAME = "session_index.jsonl"
SYNC_STATE_NAME = ".sync_export_state.json"


def _read_session_cache(logs_dir: str) -> List[dict]:
    cache_path = Path(logs_dir) / SESSION_CACHE_NAME
    if not cache_path.is_file():
        return []
    sessions = []
    with open(cache_path, "r", encoding="utf-8") as f:
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
            sessions.append(obj)
    return sessions


def _read_session_index_meta(logs_dir: str) -> Optional[dict]:
    index_path = Path(logs_dir) / SESSION_INDEX_NAME
    if not index_path.is_file():
        return None
    meta = None
    with open(index_path, "r", encoding="utf-8") as f:
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
    return meta


def _refresh_session_cache(logs_dir: str) -> bool:
    """从 index.jsonl 增量刷新 .session_cache.jsonl。
    复用 log_routes 中的 session 聚合逻辑。
    """
    from utils.log_routes import _refresh_state

    index_path = Path(logs_dir) / "index.jsonl"
    if not index_path.is_file():
        return False

    _refresh_state("export", logs_dir)
    return Path(logs_dir, SESSION_CACHE_NAME).is_file()


def export_session_index(logs_dir: str, force: bool = False) -> dict:
    """
    将 logs_dir/.session_cache.jsonl 转为 logs_dir/session_index.jsonl。

    每次先从 index.jsonl 增量刷新 cache，确保不遗漏新数据。
    Returns dict with total_sessions, avg_msg_count, skipped (bool).
    """
    # 先刷新 cache（增量），确保包含 index.jsonl 中的最新数据
    _refresh_session_cache(logs_dir)

    cache_path = Path(logs_dir) / SESSION_CACHE_NAME
    if not cache_path.is_file():
        logger.warning("未找到 %s 且无法从 index.jsonl 生成", cache_path)
        return {"total_sessions": 0, "avg_msg_count": 0, "skipped": True}

    cache_mtime = cache_path.stat().st_mtime
    if not force:
        meta = _read_session_index_meta(logs_dir)
        if meta and meta.get("source_mtime") == cache_mtime:
            logger.info("session_cache 未变更，跳过 export (mtime=%.1f)", cache_mtime)
            return {
                "total_sessions": meta.get("total_sessions", 0),
                "avg_msg_count": meta.get("avg_msg_count", 0),
                "skipped": True,
            }

    sessions = _read_session_cache(logs_dir)
    total = len(sessions)
    if total == 0:
        logger.info("无 session 数据")
        return {"total_sessions": 0, "avg_msg_count": 0, "skipped": False}

    total_msg = sum(s.get("msg_count", 0) for s in sessions)
    avg_msg = round(total_msg / total)

    index_path = Path(logs_dir) / SESSION_INDEX_NAME
    tmp_path = str(index_path) + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for s in sessions:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
        meta_line = {
            "_meta": True,
            "total_sessions": total,
            "avg_msg_count": avg_msg,
            "source_mtime": cache_mtime,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        f.write(json.dumps(meta_line, ensure_ascii=False) + "\n")
    os.replace(tmp_path, str(index_path))

    logger.info("已导出 session_index.jsonl: %d 条 session, 平均 %d 轮 msg", total, avg_msg)
    return {"total_sessions": total, "avg_msg_count": avg_msg, "skipped": False}


def _load_session_index(logs_dir: str) -> List[dict]:
    index_path = Path(logs_dir) / SESSION_INDEX_NAME
    if not index_path.is_file():
        return []
    sessions = []
    with open(index_path, "r", encoding="utf-8") as f:
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
            sessions.append(obj)
    return sessions


def _collect_triplet_files(logs_dir: str, latest_file: str) -> List[Path]:
    base = Path(logs_dir)
    stem = latest_file
    if stem.endswith("-req.json"):
        stem = stem[: -len("-req.json")]
    elif stem.endswith(".json"):
        stem = stem[: -len(".json")]

    files = []
    for suffix in ("-req.json", "-headers.json", "-res.json"):
        p = base / f"{stem}{suffix}"
        if p.is_file():
            files.append(p)
    return files


def _upload_file(
    local_path: Path,
    obs_dst: str,
    upload_script: Optional[str],
) -> Tuple[str, bool, str]:
    dst = obs_dst.rstrip("/") + "/" + local_path.name
    ok, msg = _run_upload_cmd(str(local_path), dst, upload_script)
    return local_path.name, ok, msg


def _load_sync_state(logs_dir: str) -> dict:
    sp = Path(logs_dir) / SYNC_STATE_NAME
    if sp.exists():
        try:
            with open(sp, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"uploaded_keys": []}


def _save_sync_state(logs_dir: str, state: dict):
    sp = Path(logs_dir) / SYNC_STATE_NAME
    state["last_sync_at"] = datetime.now().isoformat(timespec="seconds")
    with open(sp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def sync_session_index(
    logs_dir: str,
    obs_dst: str,
    workers: int = 4,
    upload_script: Optional[str] = None,
) -> dict:
    """
    读取 session_index.jsonl，上传每个 session 的 latest_file 三元组到 OBS。

    上传到 obs_dst 下（调用方负责 raw/session 路径区分）。
    """
    sessions = _load_session_index(logs_dir)
    if not sessions:
        logger.info("无 session 数据，跳过 sync")
        return {"uploaded": 0, "skipped": 0, "failed": 0}

    state = _load_sync_state(logs_dir)
    uploaded_keys = set(state.get("uploaded_keys", []))

    upload_tasks: List[Tuple[Path, str]] = []
    new_keys: List[str] = []

    for s in sessions:
        key = s.get("_key", "")
        if key in uploaded_keys:
            continue
        latest_file = s.get("latest_file", "")
        if not latest_file:
            continue
        new_keys.append(key)
        triplet_files = _collect_triplet_files(logs_dir, latest_file)
        for fp in triplet_files:
            upload_tasks.append((fp, obs_dst))

    if not upload_tasks and not new_keys:
        logger.info("无新 session 需要上传")
        return {"uploaded": 0, "skipped": len(uploaded_keys), "failed": 0}

    index_file = Path(logs_dir) / SESSION_INDEX_NAME
    if index_file.is_file():
        upload_tasks.append((index_file, obs_dst))

    ok_count = 0
    fail_count = 0
    logger.info("准备上传 %d 个文件 (%d 个新 session)", len(upload_tasks), len(new_keys))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_upload_file, fp, dst, upload_script): (fp, dst)
            for fp, dst in upload_tasks
        }
        for future in as_completed(futures):
            name, ok, msg = future.result()
            if ok:
                ok_count += 1
            else:
                fail_count += 1
                logger.error("上传失败 %s: %s", name, msg)

    uploaded_keys.update(new_keys)
    state["uploaded_keys"] = sorted(uploaded_keys)
    _save_sync_state(logs_dir, state)

    logger.info("上传完成: %d 成功, %d 失败", ok_count, fail_count)
    return {"uploaded": ok_count, "skipped": len(uploaded_keys) - len(new_keys), "failed": fail_count}
