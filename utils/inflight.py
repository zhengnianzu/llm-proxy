"""
utils/inflight.py — 进程内实时请求状态跟踪（in-flight gauge）

回答「此刻系统在干什么」：在途请求数、分别卡在哪个阶段、以此定位阻塞点。

阶段划分（对应转发链路 接收→透传→厂商→透传→回复）：
  - connecting:       正在与厂商建连 / 发出请求、等首字节前
  - waiting_upstream: 已连上厂商，等待厂商生成响应（非流式等整体，流式等 message_start 前）
  - streaming:        正在把厂商响应透传回客户端

单进程单事件循环下，普通 int 自增/自减是原子的（GIL 保证），无需锁。
多 worker 部署时每个进程各有一份，聚合需在读取端合并（见 /metrics/live）。

用法（上下文管理器，保证异常/取消也能正确回收计数）：
    async with track("waiting_upstream"):
        ... 等厂商 ...
或手动切换阶段：
    tk = Tracker(); tk.enter("connecting"); ...; tk.switch("streaming"); ...; tk.done()
"""

import atexit
import glob
import json
import os
from utils.atomic_write import safe_replace
import threading
import time
from contextlib import asynccontextmanager
from threading import Lock
from typing import Dict

# 各阶段当前在途数
_stage_counts: Dict[str, int] = {
    "connecting": 0,
    "waiting_upstream": 0,
    "streaming": 0,
}
# 累计计数（用于观察总量，不清零）
_started_total = 0
_finished_total = 0
_process_start = time.time()

# 单进程 GIL 下 int +=1 原子，但涉及多字段一致读取时用锁保护读快照
_lock = Lock()


class Tracker:
    """单个请求的阶段跟踪器。切换阶段时自动增减对应计数。"""

    __slots__ = ("_stage",)

    def __init__(self):
        self._stage = None

    def enter(self, stage: str):
        global _started_total
        with _lock:
            if self._stage is not None:
                _stage_counts[self._stage] = max(0, _stage_counts[self._stage] - 1)
            if stage in _stage_counts:
                _stage_counts[stage] += 1
                self._stage = stage
            else:
                self._stage = None
            _started_total += 1

    def switch(self, stage: str):
        with _lock:
            if self._stage is not None:
                _stage_counts[self._stage] = max(0, _stage_counts[self._stage] - 1)
            if stage in _stage_counts:
                _stage_counts[stage] += 1
                self._stage = stage
            else:
                self._stage = None

    def done(self):
        global _finished_total
        with _lock:
            if self._stage is not None:
                _stage_counts[self._stage] = max(0, _stage_counts[self._stage] - 1)
                self._stage = None
            _finished_total += 1


@asynccontextmanager
async def track(stage: str):
    """`async with track('waiting_upstream'):` 单阶段跟踪，异常/取消也会回收。"""
    tk = Tracker()
    tk.enter(stage)
    try:
        yield tk
    finally:
        tk.done()


def _local_snapshot() -> dict:
    """本进程的实时状态快照。"""
    with _lock:
        stages = dict(_stage_counts)
        started = _started_total
        finished = _finished_total
    in_flight = sum(stages.values())
    return {
        "in_flight": in_flight,
        "stages": stages,
        "started_total": started,
        "finished_total": finished,
        "uptime_sec": int(time.time() - _process_start),
    }


# ---------------------------------------------------------------------------
# 跨进程聚合：多 worker 下每进程只知道自己的在途请求。让每个 worker 周期性把
# 本进程快照写到 SERVICE_LOG_DIR/inflight/{pid}.json（含心跳时间戳 hb），
# 读取端（/metrics/live）聚合所有文件、跳过心跳过期的死进程。
# 单 worker 部署时不启用心跳，snapshot() 直接返回本进程快照。
# ---------------------------------------------------------------------------

_HB_INTERVAL = 1.0   # 秒；心跳写入间隔
_HB_STALE = 5.0      # 秒；超过此时长未更新视为死进程
_hb_dir: str = ""
_hb_thread = None
_hb_stop = threading.Event()
_hb_file: str = ""


def _write_heartbeat():
    snap = _local_snapshot()
    snap["pid"] = os.getpid()
    snap["hb"] = time.time()
    tmp = _hb_file + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(snap, f, ensure_ascii=False)
        safe_replace(tmp, _hb_file)  # 原子替换，读端不会看到半截文件
    except OSError:
        pass


def _hb_loop():
    while not _hb_stop.wait(_HB_INTERVAL):
        _write_heartbeat()


def _cleanup_heartbeat():
    _hb_stop.set()
    try:
        if _hb_file and os.path.exists(_hb_file):
            os.remove(_hb_file)
    except OSError:
        pass


def enable_cross_process(service_log_dir: str):
    """启用跨进程心跳。每个 worker 启动时调用一次；单 worker 也可调用（无害）。"""
    global _hb_dir, _hb_thread, _hb_file
    if _hb_thread is not None:
        return
    _hb_dir = os.path.join(service_log_dir, "inflight")
    try:
        os.makedirs(_hb_dir, exist_ok=True)
    except OSError:
        return
    _hb_file = os.path.join(_hb_dir, f"{os.getpid()}.json")
    _write_heartbeat()
    _hb_thread = threading.Thread(target=_hb_loop, name="inflight-hb", daemon=True)
    _hb_thread.start()
    atexit.register(_cleanup_heartbeat)


def snapshot() -> dict:
    """返回实时状态快照。启用了跨进程心跳时聚合所有活着的 worker，否则返回本进程。"""
    if not _hb_dir:
        return _local_snapshot()
    stages = {"connecting": 0, "waiting_upstream": 0, "streaming": 0}
    started = finished = 0
    workers = 0
    now = time.time()
    uptime = int(now - _process_start)
    saw_self = False
    for path in glob.glob(os.path.join(_hb_dir, "*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                snap = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        hb = snap.get("hb", 0)
        if now - hb > _HB_STALE:
            # 死进程残留文件：尝试清理（当前进程有权限就删）
            try:
                os.remove(path)
            except OSError:
                pass
            continue
        workers += 1
        if snap.get("pid") == os.getpid():
            saw_self = True
        for k in stages:
            stages[k] += snap.get("stages", {}).get(k, 0)
        started += snap.get("started_total", 0)
        finished += snap.get("finished_total", 0)
        uptime = max(uptime, snap.get("uptime_sec", 0))
    # 本进程心跳可能还没落盘（刚启动），补上本进程快照保证不漏
    if not saw_self:
        local = _local_snapshot()
        workers += 1
        for k in stages:
            stages[k] += local["stages"].get(k, 0)
        started += local["started_total"]
        finished += local["finished_total"]
    return {
        "in_flight": sum(stages.values()),
        "stages": stages,
        "started_total": started,
        "finished_total": finished,
        "uptime_sec": uptime,
        "workers": workers,
    }
