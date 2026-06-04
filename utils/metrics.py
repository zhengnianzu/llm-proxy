"""
实时指标：后台线程增量扫描 index.jsonl，按分钟聚合 RPM/TPM/Rate。
内存环形缓冲保留最近 24 小时的每分钟数据，rpm.log/rate.log 作为磁盘缓存。

每个 metrics bucket: {"ts": "2026-03-19T10:05", "rpm": N, "tpm_in": N, "tpm_out": N, "errors": N, "models": {...}}
每个 rate bucket:    {"ts": "...", "first": N, "valid": N, "rate": 0.xx, "models": {...}}
"""

import os
import json
import time
import logging
import threading
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from utils.log_paths import get_service_log_dir, INDEX_FILENAME

_METRICS_LOCK = threading.Lock()
_MAX_UI_HOURS = 24
_DEFAULT_WINDOW = _MAX_UI_HOURS * 60
_METRICS_WINDOW = max(int(os.getenv("METRICS_WINDOW_MINUTES", str(_DEFAULT_WINDOW))), _DEFAULT_WINDOW)

_metrics_buckets: deque = deque(maxlen=_METRICS_WINDOW)

_RATE_WINDOW = max(int(os.getenv("RATE_WINDOW_MINUTES", str(_DEFAULT_WINDOW))), _DEFAULT_WINDOW)
_rate_buckets: deque = deque(maxlen=_RATE_WINDOW)

_SERVICE_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), get_service_log_dir())
_RPM_LOG = os.path.join(_SERVICE_LOG_DIR, "rpm.log")
_RATE_LOG = os.path.join(_SERVICE_LOG_DIR, "rate.log")
_SCANNER_STATE_PATH = os.path.join(_SERVICE_LOG_DIR, "scanner_state.json")


# ---------------------------------------------------------------------------
# 磁盘缓存：rpm.log / rate.log 读写
# ---------------------------------------------------------------------------

def _flush_to_disk(bucket: dict, path: str):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(bucket, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _load_buckets_from_disk(path: str, window: int) -> dict:
    """从磁盘缓存加载 bucket，返回 {ts_key: bucket_dict}。"""
    result: dict = {}
    if not os.path.exists(path):
        return result
    cutoff_key = time.strftime("%Y-%m-%dT%H:%M", time.localtime(time.time() - window * 60))
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    bucket = json.loads(line)
                except Exception:
                    continue
                ts = bucket.get("ts", "")
                if ts >= cutoff_key:
                    result[ts] = bucket
    except Exception:
        pass
    return result


# ---------------------------------------------------------------------------
# Index.jsonl 解析
# ---------------------------------------------------------------------------

def _parse_index_ts(ts_str: str) -> str:
    """'2026-06-01_15-41-46_182' -> '2026-06-01T15:41'"""
    date_part = ts_str[:10]
    hour = ts_str[11:13]
    minute = ts_str[14:16]
    return f"{date_part}T{hour}:{minute}"


def _process_entry(entry: dict, metrics_map: dict, rate_map: dict):
    ts_raw = entry.get("ts", "")
    if len(ts_raw) < 16:
        return
    minute_key = _parse_index_ts(ts_raw)
    model = entry.get("model", "")
    tok_in = entry.get("tok_in", 0) or 0
    tok_out = entry.get("tok_out", 0) or 0
    success = entry.get("success", True)

    # metrics bucket
    bucket = metrics_map.get(minute_key)
    if bucket is None:
        bucket = {"ts": minute_key, "rpm": 0, "tpm_in": 0, "tpm_out": 0, "errors": 0, "models": {}}
        metrics_map[minute_key] = bucket
    bucket["rpm"] += 1
    bucket["tpm_in"] += tok_in
    bucket["tpm_out"] += tok_out
    if not success:
        bucket["errors"] += 1
    if model:
        m = bucket["models"].setdefault(model, {"rpm": 0, "tpm_in": 0, "tpm_out": 0})
        m["rpm"] += 1
        m["tpm_in"] += tok_in
        m["tpm_out"] += tok_out

    # rate bucket (所有 provider)
    rb = rate_map.get(minute_key)
    if rb is None:
        rb = {"ts": minute_key, "first": 0, "valid": 0, "rate": 0.0, "models": {}}
        rate_map[minute_key] = rb
    rb["first"] += 1
    if success:
        rb["valid"] += 1
    rb["rate"] = round(rb["valid"] / rb["first"], 4) if rb["first"] else 0.0
    if model:
        rm = rb["models"].setdefault(model, {"first": 0, "valid": 0})
        rm["first"] += 1
        if success:
            rm["valid"] += 1


# ---------------------------------------------------------------------------
# 零值填充
# ---------------------------------------------------------------------------

def _fill_zero_buckets(metrics_map: dict, rate_map: dict):
    all_keys = sorted(set(metrics_map.keys()) | set(rate_map.keys()))
    if not all_keys:
        return

    now_key = time.strftime("%Y-%m-%dT%H:%M")
    if now_key > all_keys[-1]:
        all_keys.append(now_key)

    if len(all_keys) < 2:
        return

    fmt = "%Y-%m-%dT%H:%M"
    first_dt = datetime.strptime(all_keys[0], fmt)
    last_dt = datetime.strptime(all_keys[-1], fmt)
    current = first_dt + timedelta(minutes=1)
    while current <= last_dt:
        key = current.strftime(fmt)
        if key not in metrics_map:
            metrics_map[key] = {"ts": key, "rpm": 0, "tpm_in": 0, "tpm_out": 0, "errors": 0, "models": {}}
        if key not in rate_map:
            rate_map[key] = {"ts": key, "first": 0, "valid": 0, "rate": 0.0, "models": {}}
        current += timedelta(minutes=1)


# ---------------------------------------------------------------------------
# Deque 重建 + 增量刷盘
# ---------------------------------------------------------------------------

def _rebuild_deques(metrics_map: dict, rate_map: dict,
                    flushed_rpm_keys: set, flushed_rate_keys: set):
    """从 map 重建 deque（加锁），并将新增 bucket 追加到磁盘缓存。"""
    cutoff_metrics = time.strftime(
        "%Y-%m-%dT%H:%M",
        time.localtime(time.time() - _METRICS_WINDOW * 60)
    )
    cutoff_rate = time.strftime(
        "%Y-%m-%dT%H:%M",
        time.localtime(time.time() - _RATE_WINDOW * 60)
    )

    m_keys = sorted(k for k in metrics_map if k >= cutoff_metrics)
    r_keys = sorted(k for k in rate_map if k >= cutoff_rate)

    with _METRICS_LOCK:
        _metrics_buckets.clear()
        for k in m_keys:
            _metrics_buckets.append(metrics_map[k])
        _rate_buckets.clear()
        for k in r_keys:
            _rate_buckets.append(rate_map[k])

    for k in m_keys:
        if k not in flushed_rpm_keys:
            _flush_to_disk(metrics_map[k], _RPM_LOG)
            flushed_rpm_keys.add(k)
    for k in r_keys:
        if k not in flushed_rate_keys:
            _flush_to_disk(rate_map[k], _RATE_LOG)
            flushed_rate_keys.add(k)


def _prune_maps(metrics_map: dict, rate_map: dict):
    cutoff = time.strftime(
        "%Y-%m-%dT%H:%M",
        time.localtime(time.time() - max(_METRICS_WINDOW, _RATE_WINDOW) * 60)
    )
    for m in (metrics_map, rate_map):
        stale = [k for k in m if k < cutoff]
        for k in stale:
            del m[k]


# ---------------------------------------------------------------------------
# Scanner state 持久化
# ---------------------------------------------------------------------------

def _load_scanner_state() -> dict:
    if not os.path.exists(_SCANNER_STATE_PATH):
        return {"files": {}}
    try:
        with open(_SCANNER_STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"files": {}}


def _save_scanner_state(state: dict):
    try:
        os.makedirs(os.path.dirname(_SCANNER_STATE_PATH), exist_ok=True)
        with open(_SCANNER_STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception:
        logging.exception("Failed to save scanner state")


# ---------------------------------------------------------------------------
# 发现 index 文件
# ---------------------------------------------------------------------------

def _discover_index_files(logs_all_base: str) -> list:
    """遍历 logs_all/{env-segment}/{startup_hour}/index.jsonl"""
    result = []
    base = Path(logs_all_base)
    if not base.is_dir():
        return result
    for env_dir in sorted(base.iterdir()):
        if not env_dir.is_dir():
            continue
        for hour_dir in sorted(env_dir.iterdir()):
            if not hour_dir.is_dir():
                continue
            idx = hour_dir / INDEX_FILENAME
            if idx.exists():
                result.append(str(idx))
    return result


# ---------------------------------------------------------------------------
# 增量读取单个 index 文件
# ---------------------------------------------------------------------------

def _incremental_read(index_path: str, offset: int, metrics_map: dict, rate_map: dict) -> int:
    """从 offset 开始增量读取 index_path，返回新的 offset。"""
    try:
        file_size = os.path.getsize(index_path)
    except OSError:
        return offset

    if file_size < offset:
        offset = 0

    if file_size <= offset:
        return offset

    try:
        with open(index_path, "rb") as f:
            f.seek(offset)
            for raw_line in f:
                if not raw_line.endswith(b"\n"):
                    break
                offset += len(raw_line)
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    _process_entry(entry, metrics_map, rate_map)
                except json.JSONDecodeError:
                    continue
    except Exception:
        logging.exception(f"Error reading index file: {index_path}")

    return offset


# ---------------------------------------------------------------------------
# 后台 scanner 线程
# ---------------------------------------------------------------------------

_scanner_stop = threading.Event()
_scanner_thread: Optional[threading.Thread] = None


def _scanner_loop(logs_all_base: str, interval: float):
    metrics_map: dict = {}
    rate_map: dict = {}
    flushed_rpm_keys: set = set()
    flushed_rate_keys: set = set()

    # 1. 从 rpm.log/rate.log 加载缓存到 map
    cached_metrics = _load_buckets_from_disk(_RPM_LOG, _METRICS_WINDOW)
    cached_rate = _load_buckets_from_disk(_RATE_LOG, _RATE_WINDOW)
    metrics_map.update(cached_metrics)
    rate_map.update(cached_rate)
    flushed_rpm_keys.update(cached_metrics.keys())
    flushed_rate_keys.update(cached_rate.keys())

    # 2. 加载 scanner state
    state = _load_scanner_state()
    file_states = state.get("files", {})

    # 首次构建 deque
    _fill_zero_buckets(metrics_map, rate_map)
    _rebuild_deques(metrics_map, rate_map, flushed_rpm_keys, flushed_rate_keys)

    while not _scanner_stop.is_set():
        try:
            changed = False
            index_files = _discover_index_files(logs_all_base)

            for idx_path in index_files:
                fs = file_states.get(idx_path, {"offset": 0, "mtime": 0.0})
                try:
                    current_mtime = os.path.getmtime(idx_path)
                except OSError:
                    continue

                if current_mtime <= fs["mtime"]:
                    continue

                new_offset = _incremental_read(idx_path, fs["offset"], metrics_map, rate_map)
                file_states[idx_path] = {"offset": new_offset, "mtime": current_mtime}
                changed = True

            if changed:
                _fill_zero_buckets(metrics_map, rate_map)
                _rebuild_deques(metrics_map, rate_map, flushed_rpm_keys, flushed_rate_keys)
                _prune_maps(metrics_map, rate_map)
                _save_scanner_state({"files": file_states})

        except Exception:
            logging.exception("metrics scanner error")

        _scanner_stop.wait(interval)


def start_metrics_scanner(logs_all_base: str, interval: float = 10.0):
    global _scanner_thread
    if _scanner_thread is not None and _scanner_thread.is_alive():
        return
    _scanner_stop.clear()
    _scanner_thread = threading.Thread(
        target=_scanner_loop,
        args=(logs_all_base, interval),
        daemon=True,
        name="metrics-scanner",
    )
    _scanner_thread.start()


def stop_metrics_scanner():
    _scanner_stop.set()
    if _scanner_thread is not None:
        _scanner_thread.join(timeout=5)


# ---------------------------------------------------------------------------
# 公共查询 API（与旧接口兼容）
# ---------------------------------------------------------------------------

def get_metrics_snapshot() -> list:
    with _METRICS_LOCK:
        return list(_metrics_buckets)


def get_rate_history() -> list:
    with _METRICS_LOCK:
        return list(_rate_buckets)


def get_metrics_storage_info() -> dict:
    return {
        "rpm_log": _RPM_LOG,
        "rate_log": _RATE_LOG,
        "service_log_dir": _SERVICE_LOG_DIR,
        "metrics_window_minutes": _METRICS_WINDOW,
        "rate_window_minutes": _RATE_WINDOW,
        "scanner_alive": _scanner_thread is not None and _scanner_thread.is_alive(),
    }
