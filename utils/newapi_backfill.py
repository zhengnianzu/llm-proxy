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
import logging
import logging.handlers
import threading
import time
from collections import deque
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Dict, Optional

from utils.log_scan import iter_index_dirs, detect_format, dir_key_for
import utils.newapi_index_db as nidb

logger = logging.getLogger(__name__)


def _is_broken_pool_err(e: BaseException) -> bool:
    """是否进程池 worker 被 OS 终止（BrokenProcessPool 或其链上包裹）。"""
    while e is not None:
        if isinstance(e, BrokenProcessPool):
            return True
        e = getattr(e, "__cause__", None)
    return False


# 专用构建日志：new-api index 回填的生命周期事件（入队/开始/单叶成功失败/卡死/root 崩溃/完成）
# 单独落一份 {SERVICE_LOG_DIR}/backfill.log，便于事后单独排查构建历史——
# 此前整条链路几乎无日志，root 级崩溃只写进程内存且被前端 DB 权威状态盖掉，异常静默丢失。
# 该 logger 不向 root 传播（propagate=False），避免把逐叶细节灌进 app.log；
# 真正需要 app.log 也看到的（root 崩溃）仍另经模块级 logger.error 打一份。
bf_logger = logging.getLogger("newapi_backfill_events")
bf_logger.setLevel(logging.INFO)
bf_logger.propagate = False


def init_backfill_logger(log_dir: str) -> None:
    """挂 RotatingFileHandler 到 backfill.log。幂等：重复调用不叠加 handler。"""
    for h in bf_logger.handlers:
        if isinstance(h, logging.handlers.RotatingFileHandler):
            return
    try:
        os.makedirs(log_dir, exist_ok=True)
        path = os.path.join(log_dir, "backfill.log")
        handler = logging.handlers.RotatingFileHandler(
            path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8")
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
        bf_logger.addHandler(handler)
        bf_logger.info("=== backfill logger initialized -> %s ===", path)
    except Exception:  # noqa: BLE001 — 日志初始化失败不应拖垮服务启动
        logger.exception("init_backfill_logger failed")


def _backfill_log_path() -> Optional[str]:
    """从已挂的 handler 拿 backfill.log 绝对路径（未初始化则 None）。"""
    for h in bf_logger.handlers:
        if isinstance(h, logging.handlers.RotatingFileHandler):
            return h.baseFilename
    return None


def read_backfill_log(root_filter: str = "", limit: int = 400) -> dict:
    """读 backfill.log 尾部；给了 root_filter 则只返回含该源 root 的行。

    每条构建日志都带 `root=<normpath>`，据此按源过滤。返回最后 limit 条匹配行。
    读文件尾部即可（回填日志按时间追加），不做全量读，避免大文件卡住请求。
    """
    path = _backfill_log_path()
    if not path or not os.path.isfile(path):
        return {"ok": True, "path": path or "", "lines": [], "note": "日志文件尚未生成"}
    norm = os.path.normpath(root_filter) if root_filter else ""
    try:
        # 只读尾部一段（约 2MB 足够覆盖近期若干次构建），逐行匹配。
        size = os.path.getsize(path)
        tail_bytes = min(size, 2 * 1024 * 1024)
        with open(path, "rb") as f:
            if size > tail_bytes:
                f.seek(size - tail_bytes)
                f.readline()  # 丢弃可能被截断的半行
            raw = f.read().decode("utf-8", errors="replace")
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "path": path, "lines": [], "note": f"读取失败: {e}"}
    lines = raw.splitlines()
    if norm:
        # 行内以 `root=<path> ` 形式出现；用带空格/行尾的边界避免前缀误匹配
        needle = f"root={norm}"
        lines = [ln for ln in lines
                 if f"{needle} " in ln or ln.rstrip().endswith(needle)]
    return {"ok": True, "path": path, "lines": lines[-limit:]}

# 单叶子 enrich「距上次分片完成」超过此秒数无进展即判卡住（多为 NFS 读死等），
# 跳过该叶子 + 预警。可经环境变量覆盖，便于按网络盘抖动实际调。
try:
    _STALL_TIMEOUT = float(os.getenv("NEWAPI_BACKFILL_STALL_TIMEOUT", "300"))
except ValueError:
    _STALL_TIMEOUT = 300.0

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

    只做「清单 + 是否已建」的轻量同步：
    - built 用 _db_exists（一次 stat）判定，不打开 SQLite、不跑统计查询——
      网络盘上每叶 open+多查询会让同步卡很久，故此处只判是否已建。
    - ingested/pending/sessions 精确数不在此写：这些留给「构建」阶段填。
      对已存在叶子不传这几个字段 → upsert 保留其上次构建写入的值（不清零）。
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
        is_built = nidb._db_exists(str(leaf))   # 一次 stat，轻量
        if is_built:
            built += 1
        res = lds.upsert_leaf(
            rid, dir_key, root_path=root,
            built=is_built,
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
         current_leaf="", leaf_total=0, leaf_done=0,
         stuck_leaves=0, last_stuck="")

    bf_logger.info("ROOT START root=%s force=%s total=%d to_build=%d skipped=%d workers=%d",
                   root, force, total, len(leaves), skipped, workers)

    if not leaves:
        _set(root, status="done", finished_at=time.time())
        bf_logger.info("ROOT DONE root=%s nothing to build (all %d up-to-date)", root, total)
        return

    done = skipped
    stuck = 0
    failed = 0
    ok = 0
    try:
        for leaf in leaves:
            _set(root, current_leaf=os.path.basename(leaf.rstrip("/")),
                 leaf_total=0, leaf_done=0)
            dir_key = dir_key_for(root_path, Path(leaf))
            lds.set_leaf_state(rid, dir_key, "building")

            def _progress(d, t, _root=root):
                _set(_root, leaf_done=d, leaf_total=t)

            t0 = time.time()
            try:
                if force:
                    nidb.remove_db(leaf)
                nidb.build_leaf(leaf, workers=workers, progress_cb=_progress,
                                stall_timeout=_STALL_TIMEOUT)
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
                ok += 1
                bf_logger.info(
                    "LEAF OK root=%s leaf=%s built=%s ingested=%d pending=%d sessions=%d took=%.1fs",
                    root, dir_key, is_built, st.get("ingested", 0),
                    st.get("pending", 0), st.get("sessions", 0), time.time() - t0)
            except nidb.LeafStalledError as e:
                # 疑似 NFS 读死等：跳过该叶子 + 预警。标 error → needs_build 下次重扫，
                # 待人在日志管理页手动「增量构建」补跑（NFS 恢复后）。
                leaf_name = os.path.basename(leaf.rstrip("/"))
                stuck += 1
                _set(root, last_error=f"{leaf}: {e}",
                     stuck_leaves=stuck,
                     last_stuck=f"{leaf_name}（{e.elapsed:.0f}s 无进展）")
                lds.set_leaf_state(rid, dir_key, "error",
                                   last_error=f"stalled: NFS read timeout ({e.elapsed:.0f}s)")
                bf_logger.warning(
                    "LEAF STALLED root=%s leaf=%s done=%d/%d elapsed=%.0fs (疑似 NFS 阻塞，已跳过待补跑)",
                    root, dir_key, e.done, e.total, e.elapsed)
                logger.warning(
                    "new-api 回填叶子卡住（疑似 NFS 阻塞），已跳过并待补跑: %s "
                    "done=%d/%d elapsed=%.0fs", leaf, e.done, e.total, e.elapsed)
            except Exception as e:  # noqa: BLE001 — 单叶子失败不拖垮整个 root
                failed += 1
                _set(root, last_error=f"{leaf}: {e}")
                lds.set_leaf_state(rid, dir_key, "error", last_error=str(e))
                if _is_broken_pool_err(e):
                    # worker 进程被 OS 终止（enrich 内部已自动重建池续跑，这是重试仍失败）：
                    # 单独提示，方便与普通解析失败区分。
                    bf_logger.exception(
                        "LEAF FAIL root=%s leaf=%s took=%.1fs err=%s "
                        "(worker 进程被终止，enrich 已自动重建进程池续跑，仍失败)",
                        root, dir_key, time.time() - t0, e)
                    logger.exception(
                        "new-api 回填叶子 worker 崩溃: %s err=%s", leaf, e)
                else:
                    bf_logger.exception(
                        "LEAF FAIL root=%s leaf=%s took=%.1fs err=%s",
                        root, dir_key, time.time() - t0, e)
            done += 1
            _set(root, done_leaves=done, current_leaf="", leaf_total=0, leaf_done=0)
        _set(root, status="done", finished_at=time.time())
        bf_logger.info(
            "ROOT DONE root=%s ok=%d failed=%d stalled=%d skipped=%d total=%d",
            root, ok, failed, stuck, skipped, total)
    except Exception as e:  # noqa: BLE001
        _set(root, status="error", error=str(e), finished_at=time.time())
        # root 级崩溃（如 needs_build 遍历撞上损坏 db）：既落专用构建日志，也打一份到 app.log，
        # 避免像这次那样静默——error 只进内存又被前端 DB 权威状态盖掉，全程无痕。
        bf_logger.exception("ROOT CRASH root=%s err=%s", root, e)
        logger.exception("new-api 回填 root 崩溃: %s", root)


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
        bf_logger.info("ENQUEUE SKIP root=%s not a new-api root", root)
        return {"status": "skipped", "reason": "not a new-api root"}

    w = workers or _DEFAULT_WORKERS
    _ensure_dispatcher()
    with _queue_cond:
        if root in _queued_set:
            # 已排队或正在跑
            pos = _current_root == root
            bf_logger.info("ENQUEUE DEDUP root=%s already %s",
                           root, "running" if pos else "queued")
            return {"status": "running" if pos else "queued",
                    "root": root, "queue_len": len(_queue)}
        _queued_set.add(root)
        _queue.append((root, w, force))
        _set(root, status="queued", queued_at=time.time())
        _queue_cond.notify()
    bf_logger.info("ENQUEUE root=%s workers=%d force=%s queue_len=%d",
                   root, w, force, len(_queue))
    return {"status": "queued", "root": root, "workers": w, "force": force,
            "queue_len": len(_queue)}
