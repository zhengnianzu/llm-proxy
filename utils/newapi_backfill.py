"""
utils/newapi_backfill.py — new-api 会话聚合的多进程后台回填

首次接入一个 new-api root（含数万文件）时，逐个解析合并文件较慢。
本模块用 ProcessPoolExecutor 并行解析：
  - 每个 worker 处理一个小时叶子目录，调 newapi_consumer.aggregate_leaf（纯解析，不碰 DB）
  - 主进程串行入库（避免多进程写同一 SQLite）：每叶子先 delete_root 再 bulk_insert，
    再 set_progress 记录消费到文件末尾

进度状态存进程内存，经 get_backfill_status(root) 暴露给 API。
"""

import os
import threading
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

from utils.log_scan import iter_index_dirs, detect_format

# root(normpath) -> 状态 dict
_status: Dict[str, dict] = {}
_status_lock = threading.Lock()

# ── 全局串行回填队列 ───────────────────────────────────────────────
# 需求：回填串行执行——一个 root 跑完再跑下一个（不并发堆多个进程池，省内存/CPU）。
# 单调度线程消费 FIFO 队列；start_backfill 只入队并立即返回（不阻塞页面请求）。
_queue: "deque[tuple]" = deque()          # (root, workers, force)
_queued_set: set = set()                  # 已在队列/在跑的 root，去重
_queue_cond = threading.Condition()
_dispatcher: Optional[threading.Thread] = None
_current_root: Optional[str] = None       # 正在跑的 root，供 UI 展示

_DEFAULT_WORKERS = min(8, (os.cpu_count() or 4))


def _worker_aggregate(leaf_dir: str):
    """子进程入口：纯解析一个叶子，返回 (leaf_dir, sessions, end_offset)。"""
    from utils.newapi_consumer import aggregate_leaf, leaf_end_offset
    try:
        sessions = aggregate_leaf(leaf_dir)
        return leaf_dir, sessions, leaf_end_offset(leaf_dir)
    except Exception as e:  # noqa: BLE001 — worker 不能让整池崩
        return leaf_dir, None, str(e)


def _set(root: str, **kw):
    with _status_lock:
        st = _status.setdefault(root, {})
        st.update(kw)


def get_backfill_status(path: str) -> dict:
    root = os.path.normpath(path)
    with _status_lock:
        st = dict(_status.get(root, {}))
    if not st:
        # 未跑过：根据 DB 是否已有该 root 的叶子判断
        return {"status": "idle", "total_leaves": 0, "done_leaves": 0}
    return st


def _leaf_needs_backfill(leaf_dir: str, _ss) -> bool:
    """判断某叶子是否需要（重新）回填。

    非 force 场景下跳过「已完成且无新增」的叶子：
      DB 已有该叶子的会话，且 index.jsonl 当前末尾偏移 == 上次记录的进度偏移
      （说明没有新增行）→ 跳过。
    否则（首次、活跃目录有增长、进度缺失/不符）→ 需要回填。
    """
    from utils.newapi_consumer import leaf_end_offset
    try:
        if _ss.get_session_count_by_root(leaf_dir) <= 0:
            return True
        recorded_offset, _lc = _ss.get_progress(leaf_dir)
        if recorded_offset <= 0:
            return True
        return leaf_end_offset(leaf_dir) != recorded_offset
    except Exception:
        return True


def _run(root: str, workers: int, force: bool = False):
    import utils.session_store as _ss

    root_path = Path(root)
    all_leaves = [str(d) for d in iter_index_dirs(root_path)]

    # 非 force：先在主进程做便宜的 DB 检查，跳过已完成且无新增的叶子，
    # 不为它们派 worker（避免几万文件的重复重扫）。
    if force:
        leaves = all_leaves
        skipped = 0
    else:
        leaves = [lf for lf in all_leaves if _leaf_needs_backfill(lf, _ss)]
        skipped = len(all_leaves) - len(leaves)

    total = len(all_leaves)
    _set(root, status="running", total_leaves=total, done_leaves=skipped,
         skipped_leaves=skipped, force=force, started_at=time.time(), error="")

    if not leaves:
        # 全部已回填、无新增
        _set(root, status="done", finished_at=time.time())
        return

    done = skipped
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_worker_aggregate, leaf): leaf for leaf in leaves}
            for fut in as_completed(futures):
                leaf_dir, sessions, extra = fut.result()
                if sessions is None:
                    # extra 是错误信息，跳过该叶子但继续
                    done += 1
                    _set(root, done_leaves=done, last_error=f"{leaf_dir}: {extra}")
                    continue
                # 主进程串行入库：幂等重建该叶子
                _ss.delete_root(leaf_dir)
                if sessions:
                    _ss.bulk_insert(leaf_dir, sessions)
                    # 重建 chain_index（bulk_insert 不写 chain_index，补上供增量归并）
                    _rebuild_chain_index(leaf_dir, sessions, _ss)
                _ss.set_progress(leaf_dir, int(extra), 0)
                done += 1
                _set(root, done_leaves=done)
        _set(root, status="done", finished_at=time.time())
    except Exception as e:  # noqa: BLE001
        _set(root, status="error", error=str(e), finished_at=time.time())


def _rebuild_chain_index(leaf_dir: str, sessions: List[dict], _ss) -> None:
    """bulk_insert 只写 sessions/traces，这里补 chain_index（lookup_key→session_key），
    供后续增量消费把新增行归并进已回填的会话（尤其活跃小时目录会继续增长）。
    aggregate_leaf 已在每个新建 session 上记录 _lookup_key。"""
    for s in sessions:
        lk = s.get("_lookup_key")
        sk = s.get("_key")
        if lk and sk:
            _ss.set_chain_index(leaf_dir, lk, sk)


def _dispatch_loop():
    """全局调度线程：串行消费队列，一个 root 跑完再跑下一个。"""
    global _current_root
    while True:
        with _queue_cond:
            while not _queue:
                _queue_cond.wait()
            root, workers, force = _queue.popleft()
            _current_root = root
        try:
            _run(root, workers, force)          # 阻塞至该 root 完成
        except Exception as e:  # noqa: BLE001 — 单个 root 失败不拖垮调度线程
            _set(root, status="error", error=str(e), finished_at=time.time())
        finally:
            with _queue_cond:
                _current_root = None
                _queued_set.discard(root)


def _ensure_dispatcher():
    global _dispatcher
    with _queue_cond:
        if _dispatcher is None or not _dispatcher.is_alive():
            _dispatcher = threading.Thread(target=_dispatch_loop, daemon=True,
                                           name="newapi-backfill-dispatch")
            _dispatcher.start()


def get_queue_snapshot() -> dict:
    """供 UI 展示：当前正在回填的 root、排队中的 root、以及各自进度。"""
    with _queue_cond:
        current = _current_root
        queued = [item[0] for item in _queue]
    with _status_lock:
        cur_st = dict(_status.get(current, {})) if current else None
    return {
        "current": current,
        "current_status": cur_st,
        "queued": queued,
        "queue_len": len(queued),
        "running": current is not None,
    }


def start_backfill(path: str, workers: Optional[int] = None, force: bool = False) -> dict:
    """把某 root 的回填加入全局串行队列。幂等、立即返回（不阻塞调用方）。

    调度线程按 FIFO 逐个执行：一个 root 跑完再跑下一个。
    同一 root 已在队列或正在跑 → 不重复入队。

    force=False（默认）：跳过已完成且无新增的小时叶子，只处理首次/活跃/有增量的目录。
    force=True：全量重建该 root 下所有叶子（用于修复数据或口径变更）。
    """
    root = os.path.normpath(path)
    if detect_format(root) != "newapi":
        return {"status": "skipped", "reason": "not a new-api root"}

    w = workers or _DEFAULT_WORKERS
    _ensure_dispatcher()
    with _queue_cond:
        if root in _queued_set:
            # 已排队或正在跑
            pos = _current_root == root
            return {"status": "running" if pos else "queued",
                    "root": root, "queue_len": len(_queue)}
        _queued_set.add(root)
        _queue.append((root, w, force))
        _set(root, status="queued", queued_at=time.time())
        _queue_cond.notify()
    return {"status": "queued", "root": root, "workers": w, "force": force,
            "queue_len": len(_queue)}
