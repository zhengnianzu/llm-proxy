#!/usr/bin/env python3
"""
src/backfill/auto_sync.py — 自动回填机制

每次运行：先对每个源 sync_leaves 把叶子清单/状态刷成磁盘现状，再对满足
「built < total 且 building = 0 且无 pending/running 请求」的 newapi 源自动入队回填，
避免半途停滞或磁盘新增未建的源长期挂在"部分构建"状态。native 源只同步不回填。

由常驻 backfill worker 每小时定期调用（见 worker.py），也可独立运行(cron)。
"""

import logging
import os
import sys

# 允许独立运行
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.log_paths import get_service_log_dir
import utils.logdir_store as logdir_store
import utils.newapi_backfill as newapi_backfill

logger = logging.getLogger("backfill_auto_sync")


def check_and_enqueue_stale(svc_dir: str) -> int:
    """扫描所有 source：先 sync 刷新叶子清单到磁盘现状，再对满足条件的 newapi 源入队回填。

    每个源先跑 sync_leaves（按其登记模板扫盘）把 leaf_status 的 total/built/building
    刷成磁盘真实值——否则读到的是上次同步的陈旧快照，磁盘新落的叶子看不见、永不回填。
    sync 后判入队条件（仅 newapi，native 无 index.db 可建，只同步不回填）：
        built < total 且 building = 0 且当前无 pending/running 请求。
    （不再用 synced_at>1h 阀值：sync 后 built/total 已是磁盘真值，有未建叶即应入队；
      has_active_backfill 负责防重复入队。）

    返回入队数量。
    """
    logdir_store.init_db(svc_dir)
    # native 源 sync 判 built 要读 session 聚合进度（get_all_progress），需先 init。
    # 软失败：未 init 时 native 叶一律判未 built，不影响 newapi 入队主流程。
    try:
        import utils.session_store as session_store
        session_store.init_db(svc_dir)
    except Exception:
        logger.exception("session_store 初始化失败（native 源 built 判定可能不准）")
    from utils.logs_config import get_root_id
    from utils.log_scan import default_templates
    import utils.newapi_backfill as _nb

    sources = logdir_store.list_sources()
    enqueued = 0

    for src in sources:
        root = src.get("root_path", "").strip()
        if not root or not os.path.isdir(root):
            continue
        fmt = src.get("format") or ""

        # 先 sync 刷新该源叶子清单/状态到磁盘现状（全源都刷，让页面数跟磁盘一致）。
        # 模板取该源登记值，空则回退该格式默认模板（等价旧硬规则）。单源 sync 失败
        # 不拖垮整轮，记日志后继续下一个源。
        tpls = src.get("templates") or default_templates(fmt)
        try:
            _nb.sync_leaves(root, tpls)
        except Exception:
            logger.exception("自动 sync 失败 root=%s，跳过该源", root)
            continue

        if fmt != "newapi":  # native 只同步、不回填（无 index.db 可建）
            continue

        rid = get_root_id(root)
        summ = logdir_store.count_summary(rid)
        total = summ.get("total", 0)
        built = summ.get("built", 0)
        building = summ.get("building", 0)

        # 条件:built < total 且 building = 0
        if not (built < total and building == 0):
            continue

        # 该 root 当前无 pending/running 请求
        if logdir_store.has_active_backfill(root):
            continue

        # 满足条件,自动入队（sync 已把 built/total 刷成磁盘真值，有未建叶即入队；
        # 不再用 synced_at>1h 阀值，has_active_backfill 已防重复）。
        workers = newapi_backfill._DEFAULT_WORKERS
        logger.info("自动入队回填 root=%s (built=%d total=%d building=%d)",
                    root, built, total, building)
        try:
            logdir_store.enqueue_backfill(root, workers=workers, force=False)
            enqueued += 1
        except Exception:
            logger.exception("自动入队失败 root=%s", root)

    return enqueued


def main() -> int:
    """独立运行入口(cron 用)。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    svc_dir = get_service_log_dir()
    logger.info("backfill auto_sync 启动 SERVICE_LOG_DIR=%s", svc_dir)
    n = check_and_enqueue_stale(svc_dir)
    logger.info("自动入队完成,共 %d 个任务", n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
