#!/usr/bin/env python3
"""
src/backfill/auto_sync.py — 自动回填机制

触发条件:built < total 且 building = 0(没有叶子正在构建)且距上次更新超过 1 小时
→ 自动提交回填任务,避免半途停滞的源长期挂在"部分构建"状态。

可由 worker 定期调用,也可独立运行(cron)。
"""

import logging
import os
import sys
import time
from datetime import datetime, timedelta

# 允许独立运行
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.log_paths import get_service_log_dir
import utils.logdir_store as logdir_store
import utils.newapi_backfill as newapi_backfill

logger = logging.getLogger("backfill_auto_sync")

STALE_THRESHOLD_SECONDS = 3600  # 1 小时


def check_and_enqueue_stale(svc_dir: str) -> int:
    """扫描所有 source,对满足条件的自动入队回填任务。

    条件:built < total 且 building = 0 且该 source 的 synced_at 距今 >1h 且当前无 pending/running 请求。

    返回入队数量。
    """
    logdir_store.init_db(svc_dir)
    sources = logdir_store.list_sources()
    enqueued = 0
    now = datetime.now()
    threshold = timedelta(seconds=STALE_THRESHOLD_SECONDS)

    for src in sources:
        root = src.get("root_path", "").strip()
        if not root or not os.path.isdir(root):
            continue
        fmt = src.get("format") or ""
        if fmt != "newapi":  # 只对 newapi 源回填
            continue

        from utils.logs_config import get_root_id
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

        # 检查 source 的 synced_at(该源最后一次同步/活动时间)
        synced_at = src.get("synced_at", "").strip()
        if not synced_at:
            continue  # 无时间戳,跳过
        try:
            last_dt = datetime.fromisoformat(synced_at)
        except (ValueError, TypeError):
            continue
        if now - last_dt < threshold:
            continue  # 未超 1h

        # 满足条件,自动入队
        workers = newapi_backfill._DEFAULT_WORKERS
        logger.info("自动入队回填 root=%s (built=%d total=%d building=%d synced_at=%s)",
                    root, built, total, building, synced_at)
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
