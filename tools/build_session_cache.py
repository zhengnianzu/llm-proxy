#!/usr/bin/env python3
"""
tools/build_session_cache.py — 多进程离线推进 session_cache.db（native 目录）

只做一件事：把 native（三元组）mtime 目录的 session 数据**增量聚合进
session_cache.db**（sessions / traces / chain_index / index_progress 四表），
等价于 Web/导出流程里 export_session_index 前置的 _refresh_session_cache →
log_routes._refresh_state，但用多进程按目录并行，加速首次大批量构建。

为什么这样并行（务必先读）：
  * session_store 是「单库 + 模块级全局连接 + threading.Lock」。写入无法在同一进程
    内跨线程真正并发，也不该让多进程各开连接狂写同一文件抢锁。
  * 但 native 的写入**天然按 root_dir 分区**：不同 mtime 目录各自维护 index.jsonl
    消费进度（index_progress.root_dir）与会话归并（per-root chain_map），互不干扰。
  * 因此并行单元 = 单个 mtime 目录：**每个子进程整包处理一个目录**（自己 init_db、
    自己跑 _refresh_state 写自己那部分 root_dir 行）。写同一个 session_cache.db 的
    不同分区，SQLite WAL + busy_timeout=5000 下是安全的；真正的瓶颈是远端 /mnt 上
    海量小文件的读 IO，写锁等待占比极小。
  * 主进程**不** init session_store（避免 fork 后继承已打开的连接），只做调度 + 汇总。

用法（项目根目录下运行）：
  python3 tools/build_session_cache.py                     # 所有 key 的全部 native mtime 目录
  python3 tools/build_session_cache.py --key 4ab5          # 只该 key（完整/slot/后4位）
  python3 tools/build_session_cache.py --mtime 260716      # 只 mtime 含此子串的目录（可多次）
  python3 tools/build_session_cache.py --workers 8         # 并行进程数（默认 = min(8, CPU)）
  python3 tools/build_session_cache.py --force             # 忽略已有进度，全量重刷该目录
  python3 tools/build_session_cache.py --dry-run           # 只打印计划，不执行
  python3 tools/build_session_cache.py \
      --service-log-dir logs/port8084/env-99oR \
      --env-dir logs_all/env-99oR                          # 手动指定（一般自动探测）

自动探测逻辑与 tools/offline_reformat_export.py 完全一致（复用其函数）。
"""

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_session_cache")


# ---------------------------------------------------------------------------
# 子进程 worker：整包处理一个 mtime 目录
# ---------------------------------------------------------------------------
def _worker(args):
    """在子进程内推进单个 root_dir 的 session_cache。

    每个子进程独立 init_db（指向同一 session_cache.db），跑 _refresh_state
    增量消费 index.jsonl 写入自己那部分 root_dir 行，返回 (root_dir, 该目录
    session 数, 耗时秒, 错误信息或 None)。
    """
    svc_dir, root_dir, force = args
    t0 = time.time()
    try:
        # 子进程各自初始化 session_store（模块级全局连接，进程内唯一）。
        import utils.session_store as _ss
        _ss.init_db(svc_dir)
        # native 增量聚合的现成实现，直接复用，不重造解析。
        from utils.log_routes import _refresh_state
        _refresh_state("export", root_dir, force=force)
        n = _ss.get_session_count_by_root(root_dir)
        return (root_dir, n, time.time() - t0, None)
    except Exception as e:  # noqa: BLE001
        import traceback
        return (root_dir, 0, time.time() - t0,
                f"{e}\n{traceback.format_exc()}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="多进程离线推进 session_cache.db（native 目录）")
    ap.add_argument("--service-log-dir", default="",
                    help="服务日志目录（含 session_cache.db）。默认自动探测")
    ap.add_argument("--env-dir", default="",
                    help="ENV_DIR，形如 logs_all/env-99oR。默认自动探测")
    ap.add_argument("--source", action="append", default=[],
                    help="直接指定 source 根目录，扫描其下所有 mtime 叶子节点全部构建"
                         "（可多次）。给了 --source 则跳过 get_stats_roots 自动探测，"
                         "只处理这些根目录下的叶子。仍受 --mtime 过滤。")
    ap.add_argument("--key", action="append", default=[],
                    help="只处理指定 key 的目录（可多次）；完整/key-XXXX/后4位")
    ap.add_argument("--mtime", action="append", default=[],
                    help="只处理 mtime 目录 key 含此子串的（可多次，子串匹配）")
    ap.add_argument("--workers", type=int, default=0,
                    help="并行进程数（默认 min(8, CPU)）")
    ap.add_argument("--force", action="store_true",
                    help="忽略已有进度，全量重刷匹配目录")
    ap.add_argument("--dry-run", action="store_true",
                    help="只打印计划，不执行")
    args = ap.parse_args()

    # 复用 offline_reformat_export 的探测与 mtime 解析，保证与 Web/导出同源
    from tools.offline_reformat_export import (
        _autodetect_service_log_dir, _autodetect_env_dir,
    )

    svc_dir = args.service_log_dir or _autodetect_service_log_dir()
    if not svc_dir or not Path(svc_dir).is_dir():
        logger.error("无法定位 SERVICE_LOG_DIR（含 session_cache.db 的目录）；"
                     "请用 --service-log-dir 指定")
        return 2
    env_key_name = os.path.basename(os.path.normpath(svc_dir))
    env_dir = args.env_dir or _autodetect_env_dir(env_key_name)
    if not env_dir:
        logger.error("无法自动探测 ENV_DIR；请用 --env-dir 指定")
        return 2
    env_dir_p = Path(env_dir).resolve()

    logger.info("SERVICE_LOG_DIR = %s", svc_dir)
    logger.info("ENV_DIR         = %s", env_dir_p)

    # 枚举候选 root（多数据源前缀感知），只保留 native 且存在 index.jsonl 的目录。
    # native 判据：detect_format != 'newapi'；new-api 走 index.db 另一套，不在此脚本。
    import utils.logdir_store as lds
    lds.init_db(svc_dir)  # get_stats_roots / mtime 解析要读 sources 表
    from utils.logs_config import get_stats_roots
    from utils.log_scan import detect_format

    roots = [r for r in get_stats_roots(str(env_dir_p)) if Path(r).is_dir()]
    # 显式 --source 优先：直接用指定根目录，跳过自动探测（扫其下全部叶子）。
    if args.source:
        roots = []
        for s in args.source:
            sp = Path(s)
            if not sp.is_dir():
                logger.warning("--source 目录不存在，跳过: %s", s)
                continue
            roots.append(str(sp.resolve()))
        if not roots:
            logger.error("所有 --source 目录都不存在")
            return 2
    key_filters = [k[-4:] if k.startswith("key-") else k for k in args.key]

    targets = []  # list[str]，每个是一个 mtime 叶子目录绝对路径
    for root in roots:
        rp = Path(root)
        for child in sorted(rp.iterdir()) if rp.is_dir() else []:
            if not child.is_dir():
                continue
            mt_name = child.name
            # mtime 子串过滤
            if args.mtime and not any(m in mt_name for m in args.mtime):
                continue
            # key 过滤：native 目录本身不按 key 分，key 过滤在 session 层，
            # 这里 --key 仅作用于「目录名含该子串」的宽松匹配（多数无效，
            # 但保留与 offline 一致的入口）；不匹配则不排除，交给 session 层。
            leaf = str(child)
            if not (Path(leaf) / "index.jsonl").is_file():
                continue  # 无 index.jsonl = 无原料，跳过（空目录/未产生索引）
            try:
                if detect_format(leaf) == "newapi":
                    continue  # new-api 不归本脚本
            except Exception:
                pass
            targets.append(leaf)

    # 去重、稳定排序
    targets = sorted(set(targets))
    if not targets:
        logger.warning("没有匹配的 native 目录（检查 --key/--mtime 过滤）")
        return 0

    workers = args.workers or min(8, os.cpu_count() or 4)
    logger.info("计划推进 %d 个 native 目录，并行进程数 %d，force=%s",
                len(targets), workers, args.force)
    for t in targets:
        logger.info("  - %s", t)

    if args.dry_run:
        logger.info("dry-run，结束。")
        return 0

    # 主进程不 init session_store：避免把已打开的连接 fork 给子进程。
    job_args = [(svc_dir, root_dir, args.force) for root_dir in targets]
    t_start = time.time()
    ok = fail = 0
    total_sessions = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_worker, a): a[1] for a in job_args}
        for fut in as_completed(futs):
            root_dir = futs[fut]
            rd, n, dt, err = fut.result()
            if err:
                fail += 1
                logger.error("[FAIL] %s (%.1fs)\n%s", rd, dt, err)
            else:
                ok += 1
                total_sessions += n
                logger.info("[OK]   %s  sessions=%d  (%.1fs)", rd, n, dt)

    logger.info("完成：成功 %d，失败 %d，共 %d 目录；累计 sessions=%d；总耗时 %.1fs",
                ok, fail, len(targets), total_sessions, time.time() - t_start)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
