"""
实时指标：内存环形缓冲，保留最近 120 分钟的每分钟数据
每个 bucket: {"ts": "2026-03-19T10:05", "rpm": N, "tpm_in": N, "tpm_out": N, "errors": N}
"""

import time
import threading
from collections import deque

_METRICS_LOCK = threading.Lock()
_METRICS_WINDOW = 120  # 保留分钟数

_metrics_buckets: deque = deque(maxlen=_METRICS_WINDOW)
_current_bucket: dict = {}


def _bucket_key() -> str:
    return time.strftime("%Y-%m-%dT%H:%M")


def record_request(input_tokens: int = 0, output_tokens: int = 0, success: bool = True):
    """每次请求完成后调用，记录到当前分钟 bucket"""
    key = _bucket_key()
    with _METRICS_LOCK:
        global _current_bucket
        if _current_bucket.get("ts") != key:
            if _current_bucket:
                _metrics_buckets.append(dict(_current_bucket))
            _current_bucket = {"ts": key, "rpm": 0, "tpm_in": 0, "tpm_out": 0, "errors": 0}
        _current_bucket["rpm"] += 1
        _current_bucket["tpm_in"] += input_tokens
        _current_bucket["tpm_out"] += output_tokens
        if not success:
            _current_bucket["errors"] += 1


def get_metrics_snapshot() -> list:
    """返回历史 buckets + 当前未完成 bucket 的快照"""
    with _METRICS_LOCK:
        result = list(_metrics_buckets)
        if _current_bucket:
            result.append(dict(_current_bucket))
    return result
