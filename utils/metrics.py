"""
实时指标：内存环形缓冲，默认保留最近 24 小时的每分钟数据
每个 bucket: {"ts": "2026-03-19T10:05", "rpm": N, "tpm_in": N, "tpm_out": N, "errors": N}

有效率时序：独立环形缓冲，默认保留最近 24 小时的每分钟有效率数据
每个 bucket: {"ts": "...", "first": N, "valid": N, "rate": 0.xx}
"""

import os
import json
import time
import threading
from collections import deque

from utils.log_paths import get_log_task_tag, get_upstream_key_prefix

_METRICS_LOCK = threading.Lock()
_MAX_UI_HOURS = 24
_DEFAULT_WINDOW = _MAX_UI_HOURS * 60
_METRICS_WINDOW = max(int(os.getenv("METRICS_WINDOW_MINUTES", str(_DEFAULT_WINDOW))), _DEFAULT_WINDOW)

_metrics_buckets: deque = deque(maxlen=_METRICS_WINDOW)
_current_bucket: dict = {}

_RATE_WINDOW = max(int(os.getenv("RATE_WINDOW_MINUTES", str(_DEFAULT_WINDOW))), _DEFAULT_WINDOW)
_rate_buckets: deque = deque(maxlen=_RATE_WINDOW)
_current_rate_bucket: dict = {}

_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")


def _service_metrics_suffix() -> str:
    parts = []
    task_tag = get_log_task_tag()
    if task_tag:
        parts.append(task_tag)
    port = (os.getenv("PROXY_PORT") or "").strip()
    if port:
        parts.append(f"port{port}")
    upstream = get_upstream_key_prefix()
    if upstream:
        parts.append(upstream)
    return "-".join(parts) if parts else "default"


_SERVICE_SUFFIX = _service_metrics_suffix()
_RPM_LOG = os.path.join(_LOG_DIR, f"rpm-{_SERVICE_SUFFIX}.log")
_RATE_LOG = os.path.join(_LOG_DIR, f"rate-{_SERVICE_SUFFIX}.log")


def _flush_to_disk(bucket: dict, path: str):
    """将一个已完成的 bucket 追加写入日志文件。"""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(bucket, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _load_buckets_from_disk(path: str, window: int, target: deque):
    if not os.path.exists(path):
        return
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
                if bucket.get("ts", "") >= cutoff_key:
                    target.append(bucket)
    except Exception:
        pass


def load_metrics_from_disk():
    """启动时从当前服务自己的 metrics 文件加载历史数据，仅保留窗口内的数据。"""
    _load_buckets_from_disk(_RPM_LOG, _METRICS_WINDOW, _metrics_buckets)
    _load_buckets_from_disk(_RATE_LOG, _RATE_WINDOW, _rate_buckets)


def _bucket_key() -> str:
    return time.strftime("%Y-%m-%dT%H:%M")


def record_request(input_tokens: int = 0, output_tokens: int = 0, success: bool = True, model: str = ""):
    """每次请求完成后调用，记录到当前分钟 bucket"""
    key = _bucket_key()
    with _METRICS_LOCK:
        global _current_bucket
        if _current_bucket.get("ts") != key:
            if _current_bucket:
                completed = dict(_current_bucket)
                _metrics_buckets.append(completed)
                _flush_to_disk(completed, _RPM_LOG)
            _current_bucket = {"ts": key, "rpm": 0, "tpm_in": 0, "tpm_out": 0, "errors": 0, "models": {}}
        _current_bucket["rpm"] += 1
        _current_bucket["tpm_in"] += input_tokens
        _current_bucket["tpm_out"] += output_tokens
        if not success:
            _current_bucket["errors"] += 1
        if model:
            m = _current_bucket["models"].setdefault(model, {"rpm": 0, "tpm_in": 0, "tpm_out": 0})
            m["rpm"] += 1
            m["tpm_in"] += input_tokens
            m["tpm_out"] += output_tokens


def record_validity(valid: bool, model: str = ""):
    """每次 Anthropic 请求完成后调用，记录有效率到当前分钟 bucket"""
    key = _bucket_key()
    with _METRICS_LOCK:
        global _current_rate_bucket
        if _current_rate_bucket.get("ts") != key:
            if _current_rate_bucket:
                completed = dict(_current_rate_bucket)
                _rate_buckets.append(completed)
                _flush_to_disk(completed, _RATE_LOG)
            _current_rate_bucket = {"ts": key, "first": 0, "valid": 0, "rate": 0.0, "models": {}}
        _current_rate_bucket["first"] += 1
        if valid:
            _current_rate_bucket["valid"] += 1
        f = _current_rate_bucket["first"]
        _current_rate_bucket["rate"] = round(_current_rate_bucket["valid"] / f, 4) if f else 0.0
        if model:
            m = _current_rate_bucket["models"].setdefault(model, {"first": 0, "valid": 0})
            m["first"] += 1
            if valid:
                m["valid"] += 1


def get_metrics_snapshot() -> list:
    """返回历史 buckets + 当前未完成 bucket 的快照"""
    with _METRICS_LOCK:
        result = list(_metrics_buckets)
        if _current_bucket:
            result.append(dict(_current_bucket))
    return result


def get_rate_history() -> list:
    """返回有效率时序快照。"""
    with _METRICS_LOCK:
        result = list(_rate_buckets)
        if _current_rate_bucket:
            result.append(dict(_current_rate_bucket))
    return result


def get_metrics_storage_info() -> dict:
    return {
        "rpm_log": _RPM_LOG,
        "rate_log": _RATE_LOG,
        "service_suffix": _SERVICE_SUFFIX,
        "metrics_window_minutes": _METRICS_WINDOW,
        "rate_window_minutes": _RATE_WINDOW,
    }
