"""
utils/newapi_backfill.py — new-api 富 index（index.db）的批量构建

给每个 new-api 小时叶子构建/补齐 `{leaf}/index.db`（见 utils/newapi_index_db）：
  - ingest：从 index.jsonl 按 offset 增量摄取 request 行（cheap）
  - enrich：解析合并文件补 q1_hash/msg_count/... 元信息（贵，叶子内多进程分片）
  - rebuild_sessions：从富 index 内存聚合出 sessions/traces（cheap）

叶子间由全局 FIFO 调度线程「串行」执行（一个 root 跑完再跑下一个，root 内逐叶子），
叶子内部由 index_db.enrich 开进程池并行——摆脱旧版「1 叶子=1 worker」。
进度状态存进程内存，经 get_backfill_status(root) 暴露给 API。
"""

import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Dict, Optional

from utils.log_scan import iter_index_dirs, detect_format, dir_key_for
import utils.newapi_index_db as nidb

# root(normpath) -> 状态 dict
_status: Dict[str, dict] = {}
_status_lock = threading.Lock()

# 磁盘回退探测的短 TTL 缓存：内存无运行态时，据叶子 index.db 数判断是否「已就绪」。
# 状态非持久化（存内存，重启即丢），但 index.db 在磁盘上——重启后据磁盘反推构建情况。
# sfs 等网络盘 stat 有延迟，TTL 内复用结果，避免每次轮询全量 stat。
_disk_probe_cache: Dict[str, tuple] = {}   # root -> (ts, status_dict)
_disk_probe_lock = threading.Lock()
_DISK_PROBE_TTL = 30.0

# ── 全局串行回填队列 ───────────────────────────────────────────────
# 需求：回填串行执行——一个 root 跑完再跑下一个（不并发堆多个进程池，省内存/CPU）。
# 单调度线程消费 FIFO 队列；start_backfill 只入队并立即返回（不阻塞页面请求）。
_queue: "deque[tuple]" = deque()          # (root, workers, force)
_queued_set: set = set()                  # 已在队列/在跑的 root，去重
_queue_cond = threading.Condition()
_dispatcher: Optional[threading.Thread] = None
_current_root: Optional[str] = None       # 正在跑的 root，供 UI 展示

_DEFAULT_WORKERS = min(8, (os.cpu_count() or 4))


def _set(root: str, **kw):
    with _status_lock:
        st = _status.setdefault(root, {})
        st.update(kw)


def _disk_probe_status(root: str) -> dict:
    """内存无运行态时，据磁盘上叶子的 index.db 反推构建情况（带 TTL 缓存）。

    只用轻量的 _db_exists（os.path.isfile），不开 db 连接，避免网络盘每叶子多次 IO。
    - 全部叶子已有 index.db → done（"已就绪"）
    - 部分 → done 且带 done/total（UI 显示已就绪 N/M）
    - 无叶子或全未建 → idle（保持"未构建"）
    """
    with _disk_probe_lock:
        cached = _disk_probe_cache.get(root)
        if cached and (time.time() - cached[0]) < _DISK_PROBE_TTL:
            return dict(cached[1])
    try:
        leaves = [str(d) for d in iter_index_dirs(Path(root))]
        total = len(leaves)
        built = sum(1 for lf in leaves if nidb._db_exists(lf))
    except Exception:
        total = built = 0
    if total and built:
        st = {"status": "done", "total_leaves": total, "done_leaves": built,
              "skipped_leaves": built, "from_disk": True}
    else:
        st = {"status": "idle", "total_leaves": total, "done_leaves": 0}
    with _disk_probe_lock:
        _disk_probe_cache[root] = (time.time(), dict(st))
    return st


def get_backfill_status(path: str) -> dict:
    root = os.path.normpath(path)
    with _status_lock:
        st = dict(_status.get(root, {}))
    if not st:
        # 本进程未跑过：回退到磁盘探测（index.db 是否已就绪），反映重启前的构建结果
        return _disk_probe_status(root)
    return st


def count_built_leaves(path: str) -> tuple:
    """(已回填叶子数, 总叶子数)。供列表页展示"已回填/总数"。

    优先取本进程运行态（正在跑时的实时计数）；无运行态则据磁盘 index.db 探测。
    两者都复用 _disk_probe_status 的 TTL 缓存路径，避免网络盘频繁全量 stat。
    """
    root = os.path.normpath(path)
    with _status_lock:
        st = dict(_status.get(root, {}))
    if st and st.get("status") in ("running", "queued", "done", "error"):
        total = int(st.get("total_leaves") or 0)
        built = int(st.get("done_leaves") or 0)
        if total:
            return built, total
    probe = _disk_probe_status(root)
    return int(probe.get("done_leaves") or 0), int(probe.get("total_leaves") or 0)


def _root_id(root: str) -> str:
    """源标识；用于 logdir_store 的 (root_id, dir_key) 主键。"""
    try:
        from utils.logs_config import get_root_id
        return get_root_id(root)
    except Exception:
        return ""


def sync_leaves(path: str) -> dict:
    """扫描某源的叶子目录，把节点清单同步进 logdir_store（不触发回填）。

    - 新增节点写入（state=pending 或 done，据当前 index.db 状态）
    - 已存在节点更新其构建状态
    返回 {total, built, added, updated}。
    """
    import utils.logdir_store as lds
    root = os.path.normpath(path)
    rid = _root_id(root)
    root_path = Path(root)
    total = built = added = updated = 0
    for leaf in iter_index_dirs(root_path):
        dir_key = dir_key_for(root_path, leaf)
        total += 1
        try:
            st = nidb.status(str(leaf))
        except Exception:
            st = {"built": False, "ingested": 0, "pending": 0, "sessions": 0}
        is_built = bool(st.get("built"))
        if is_built:
            built += 1
        res = lds.upsert_leaf(
            rid, dir_key, root_path=root,
            built=is_built, ingested=st.get("ingested", 0),
            pending=st.get("pending", 0), sessions=st.get("sessions", 0),
            state="done" if is_built else "pending",
        )
        if res == "added":
            added += 1
        elif res == "updated":
            updated += 1
    # 探测缓存失效，让列表页下次读到最新（虽然现在改读 DB，仍保守清一下）
    with _disk_probe_lock:
        _disk_probe_cache.pop(root, None)
    return {"total": total, "built": built, "added": added, "updated": updated}


def _run(root: str, workers: int, force: bool = False):
    import utils.logdir_store as lds
    root_path = Path(root)
    rid = _root_id(root)
    all_leaves = [str(d) for d in iter_index_dirs(root_path)]

    # 非 force：只处理需要构建的叶子（无 db / index.jsonl 有新增 / 有待补 meta / sessions 脏）。
    if force:
        leaves = all_leaves
        skipped = 0
    else:
        leaves = [lf for lf in all_leaves if nidb.needs_build(lf)]
        skipped = len(all_leaves) - len(leaves)

    total = len(all_leaves)
    _set(root, status="running", total_leaves=total, done_leaves=skipped,
         skipped_leaves=skipped, force=force, started_at=time.time(), error="",
         current_leaf="", leaf_total=0, leaf_done=0)

    if not leaves:
        _set(root, status="done", finished_at=time.time())
        return

    done = skipped
    try:
        for leaf in leaves:
            _set(root, current_leaf=os.path.basename(leaf.rstrip("/")),
                 leaf_total=0, leaf_done=0)
            dir_key = dir_key_for(root_path, Path(leaf))
            lds.set_leaf_state(rid, dir_key, "building")

            def _progress(d, t, _root=root):
                _set(_root, leaf_done=d, leaf_total=t)

            try:
                if force:
                    nidb.remove_db(leaf)
                nidb.build_leaf(leaf, workers=workers, progress_cb=_progress)
                # 构建后读事实源，逐叶落盘（重启后据此显示，无需再全盘 stat）
                try:
                    st = nidb.status(leaf)
                except Exception:
                    st = {"built": False, "ingested": 0, "pending": 0, "sessions": 0}
                is_built = bool(st.get("built"))
                lds.set_leaf_state(
                    rid, dir_key, "done" if is_built else "pending",
                    built=is_built, ingested=st.get("ingested", 0),
                    pending=st.get("pending", 0), sessions=st.get("sessions", 0),
                )
            except Exception as e:  # noqa: BLE001 — 单叶子失败不拖垮整个 root
                _set(root, last_error=f"{leaf}: {e}")
                lds.set_leaf_state(rid, dir_key, "error", last_error=str(e))
            done += 1
            _set(root, done_leaves=done, current_leaf="", leaf_total=0, leaf_done=0)
        _set(root, status="done", finished_at=time.time())
    except Exception as e:  # noqa: BLE001
        _set(root, status="error", error=str(e), finished_at=time.time())


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
