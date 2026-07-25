"""
utils/leader_lock.py — 多 worker 选主（文件锁）

uvicorn --workers N 模式下各 worker 是独立进程，且没有内置 worker 序号。
用 fcntl.flock 对一个锁文件加**非阻塞独占锁**：只有一个 worker 能拿到，
即成为 leader。用于把「同一份后台工作」（metrics 扫描、导出队列调度）
收敛到单个 worker，避免 N 个进程重复/冲突地写同一批文件。

关键点：
- 锁随持有它的进程存活；进程退出（正常/崩溃）内核自动释放锁，
  下次调用会有另一个 worker 抢到 → leader 自动转移，无需心跳。
- 必须持有返回的文件对象引用（放模块全局），否则被 GC 关闭会释放锁。
"""

import fcntl
import logging
import os
from typing import Optional

# 持有锁的文件对象；置于模块全局防止被 GC 关闭而释放锁
_lock_fp = None
_is_leader: Optional[bool] = None


def try_acquire_leader(lock_path: str) -> bool:
    """尝试成为 leader。拿到锁返回 True（本进程是 leader），否则 False。
    结果在进程内缓存——一个进程的 leader 身份在其生命周期内不变。"""
    global _lock_fp, _is_leader
    if _is_leader is not None:
        return _is_leader
    try:
        os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)
        # 用 'a' 打开，不截断已有文件
        _lock_fp = open(lock_path, "a")
        fcntl.flock(_lock_fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        _is_leader = True
        logging.info("leader_lock: acquired leadership (pid=%s, lock=%s)", os.getpid(), lock_path)
    except (OSError, BlockingIOError):
        # 已被其它 worker 持有
        if _lock_fp is not None:
            try:
                _lock_fp.close()
            except Exception:
                pass
            _lock_fp = None
        _is_leader = False
        logging.info("leader_lock: another worker is leader (pid=%s)", os.getpid())
    return _is_leader


def is_leader() -> bool:
    """返回本进程是否为 leader（需先调用过 try_acquire_leader）。"""
    return bool(_is_leader)
