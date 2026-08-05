#!/usr/bin/env python3
"""
scripts/sync_all.py — 离线同步叶子清单（数据管理「同步」的命令行版）

只扫盘、写 DB，不构建 index.db。等价于在「数据管理」页对每个 new-api 源点
「同步」按钮，把叶子清单落进 log_dir.db：

  - sync_leaves(root)：把源下的叶子清单写进 leaf_status（新增 pending / 更新构建态），
    并 upsert 一行 sources（格式/模板/叶子数/已建数）

与 scripts/backfill_all.py 的区别：本脚本**只同步、不回填**（不跑 _run、不构建
index.db），因此很快、只读磁盘不重算。适合离线/后台构建后，让平台快速读到最新的
叶子清单，无需页面逐个点「同步」。

    # 同步所有 new-api 源
    python -m scripts.sync_all

    # 只同步指定源（可多次 --root；未给则同步全部登记源）
    python -m scripts.sync_all --root /data/logs_all/xxx

    # 按「标识」筛选（匹配 name / root_id / 路径子串，可多次，取并集）
    python -m scripts.sync_all --source jumper-003

    # 预演：只打印将同步哪些源，不写 DB
    python -m scripts.sync_all --dry-run

    # 显式指定写 DB 的目录（覆盖环境变量推导；含 log_dir.db 的那个目录）
    python -m scripts.sync_all --db-dir logs/port8084/env-99oR

环境变量与服务进程保持一致（决定 SERVICE_LOG_DIR / 活跃 base 的解析）：
    PROXY_PORT, LOG_TASK_TAG, UPSTREAM_API_KEY, LOGS_DIR, LOGS_DIRS_CONFIG
务必在与服务相同的环境变量下运行，否则会写进另一个 log_dir.db（如 PROXY_PORT
未设 → logs/port0/... ），平台读不到。建议：`set -a; source .env; set +a`。
不想依赖环境变量时，直接用 `--db-dir <含 log_dir.db 的目录>` 指定目标 DB。

退出码: 0 全部成功；1 有源同步失败。
"""

import argparse
import logging
import os
import sys

# 允许 `python scripts/sync_all.py` 直接运行（把项目根加入 import 路径）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.log_paths import get_service_log_dir, get_log_dir
from utils.logs_config import get_stats_roots, get_path_name, get_root_id
import utils.logdir_store as lds
import utils.newapi_backfill as bf

# 复用 backfill_all 已验证的源枚举 / 标识筛选逻辑，避免两处口径漂移。
from scripts.backfill_all import _resolve_roots, _classify, _filter_by_sources


logger = logging.getLogger("sync_all")


def _sync_one(root: str) -> bool:
    """同步单个源：sync_leaves 写清单 + upsert sources。成功返回 True。"""
    try:
        res = bf.sync_leaves(root)
        logger.info("    同步：total=%d added=%d updated=%d built=%d",
                    res["total"], res["added"], res["updated"], res["built"])
    except Exception:
        logger.exception("    同步叶子清单失败：%s", root)
        return False
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="离线同步所有 new-api 源的叶子清单（数据管理「同步」命令行版）")
    parser.add_argument("--root", action="append", default=[],
                        help="只同步指定源目录（可多次）。未给则同步所有登记源。")
    parser.add_argument("--source", action="append", default=[],
                        help="按「标识」筛选源（可多次，取并集）：匹配 name / root_id / 路径子串。")
    parser.add_argument("--db-dir", default="",
                        help="显式指定写 log_dir.db 的目录（含该 DB 的目录）。默认由环境变量推导"
                             "（logs/port{PROXY_PORT}/{LOG_TASK_TAG}-{UPSTREAM_API_KEY 后4位}），"
                             "服务进程用同一套推导，环境变量一致即命中同一 DB。")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印将同步哪些源、不写 DB。")
    parser.add_argument("--verbose", "-v", action="store_true", help="更详细日志。")
    args = parser.parse_args(argv)

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")

    service_log_dir = os.path.normpath(args.db_dir) if args.db_dir else get_service_log_dir()
    if args.dry_run:
        logger.info("[dry-run] DB_DIR=%s（不写 DB）", service_log_dir)
    else:
        lds.init_db(service_log_dir)
        logger.info("DB_DIR=%s（log_dir.db 写入该目录）", service_log_dir)

    all_roots = _resolve_roots(args.root)
    if not all_roots:
        logger.error("未找到任何源。检查工作目录与环境变量（LOGS_DIR / 历史路径配置）。")
        return 1

    all_roots, unmatched = _filter_by_sources(all_roots, args.source)
    if unmatched:
        logger.error("以下 --source 标识未匹配到任何源：%s", "，".join(unmatched))
        return 1
    if not all_roots:
        logger.error("按 --source 筛选后无源可同步。")
        return 1

    logger.info("共发现 %d 个源，逐个检查是否可同步（new-api / 本项目 native）：", len(all_roots))
    targets = []
    for r in all_roots:
        exists, fmt, _ = _classify(r)
        # 同步只扫盘写 DB（sync_leaves 按 fmt 默认模板枚举叶子），newapi 与 native 都支持——
        # 与「数据管理」页「同步」按钮口径一致（route 允许 newapi/native）。
        syncable = fmt in ("newapi", "native")
        tag = "同步" if syncable else ("跳过-不支持" if exists else "跳过-不存在")
        logger.info("  [%s] %s  name=%s root_id=%s  fmt=%s",
                    tag, r, get_path_name(r), get_root_id(r), fmt)
        if syncable:
            targets.append(r)

    if not targets:
        logger.warning("没有可同步的 new-api / native 源，结束。")
        return 0

    if args.dry_run:
        logger.info("[dry-run] 将对以下 %d 个源执行「同步清单」（实际未执行）：",
                    len(targets))
        for i, r in enumerate(targets, 1):
            logger.info("  [dry-run] [%d/%d] %s (name=%s root_id=%s)",
                        i, len(targets), r, get_path_name(r), get_root_id(r))
        return 0

    logger.info("=" * 70)
    logger.info("开始逐个同步 %d 个源（new-api / native）", len(targets))
    logger.info("=" * 70)

    failures = []
    for idx, root in enumerate(targets, 1):
        logger.info("")
        logger.info(">>> [%d/%d] 源 %s (name=%s)", idx, len(targets), root, get_path_name(root))
        if not _sync_one(root):
            failures.append(root)

    logger.info("")
    logger.info("=" * 70)
    if failures:
        logger.error("同步结束：%d/%d 个源失败", len(failures), len(targets))
        for r in failures:
            logger.error("    失败源 %s", r)
        return 1
    logger.info("同步结束：全部 %d 个源成功，状态已写入 log_dir.db（%s）",
                len(targets), os.path.join(service_log_dir, "log_dir.db"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
