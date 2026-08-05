#!/usr/bin/env python3
"""
scripts/backfill_templates.py — 给 sources 表回填默认层级模板（一次性迁移）

背景：`sources` 表的 templates 字段存 JSON 数组，但历史上多数源从未点过「同步」
（只有同步才 upsert_source），因此要么根本没进 sources 表，要么 templates 为空。
本脚本按「数据管理」页同一份源集合（活跃 env_dir + 历史路径）逐个：

  - 若该源在 sources 表**无行**或其 **templates 为空** → 按 format 回填默认模板
    （newapi=["{日6}/{时8}", "details/{日6}/{时8}"]，native=["{时8}"]，见
    log_scan.default_templates），并 upsert 一行 sources（含 format）。
  - 已有非空 templates 的源（用户显式登记 / 已同步过）→ 跳过，不覆盖。

只写模板/格式，**不扫盘、不数叶子**：leaf_count/built_count 不传，upsert_source
保留其已有值（新建行则默认 0，待日后「同步」刷新）。幂等：重复跑不改已填的源。

    # 回填所有源（活跃 + 历史）
    python -m scripts.backfill_templates

    # 只回填指定源（可多次 --root；未给则全部登记源）
    python -m scripts.backfill_templates --root /data/logs_all/xxx

    # 预演：只打印将回填哪些源、填什么模板，不写 DB
    python -m scripts.backfill_templates --dry-run

    # 显式指定写 DB 的目录（含 log_dir.db 的那个目录）
    python -m scripts.backfill_templates --db-dir logs/port8084/env-99oR

环境变量与服务进程保持一致（决定 SERVICE_LOG_DIR / 活跃 base 的解析）：
    PROXY_PORT, LOG_TASK_TAG, UPSTREAM_API_KEY, LOGS_DIR, LOGS_DIRS_CONFIG
务必在与服务相同的环境变量下运行，否则会写进另一个 log_dir.db。
建议：`set -a; source .env; set +a`，或直接 `--db-dir <含 log_dir.db 的目录>`。

退出码: 0 正常结束（含无源可填）。
"""

import argparse
import logging
import os
import sys

# 允许 `python scripts/backfill_templates.py` 直接运行（把项目根加入 import 路径）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.log_paths import get_service_log_dir
from utils.logs_config import get_path_name, get_root_id
from utils.log_scan import default_templates
import utils.logdir_store as lds

# 复用 backfill_all 已验证的源枚举 / 分类逻辑，避免口径漂移。
from scripts.backfill_all import _resolve_roots, _classify


logger = logging.getLogger("backfill_templates")


def _fill_one(root: str, dry_run: bool) -> str:
    """回填单个源的模板。返回 'filled' | 'skipped-has' | 'skipped-nofmt' | 'skipped-missing'。"""
    exists, fmt, _ = _classify(root)
    if not exists:
        return "skipped-missing"
    tpls = default_templates(fmt)
    if not tpls:
        # unknown/empty 等无默认模板的格式：无从回填
        return "skipped-nofmt"

    rid = get_root_id(root)
    src = lds.get_source(rid)
    if src and (src.get("templates") or []):
        # 已有非空模板（用户登记 / 同步过）→ 不覆盖
        return "skipped-has"

    if dry_run:
        return "filled"

    # 只写 format/templates；leaf_count/built_count 不传 → upsert 保留原值（新行默认 0）。
    # name 也不传 → 保留原值（新行默认 'default'，由 add/rename 维护，不在此动）。
    lds.upsert_source(rid, root_path=root, format=fmt, templates=tpls)
    return "filled"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="给 sources 表回填默认层级模板（一次性迁移，幂等）")
    parser.add_argument("--root", action="append", default=[],
                        help="只回填指定源目录（可多次）。未给则回填所有登记源。")
    parser.add_argument("--db-dir", default="",
                        help="显式指定写 log_dir.db 的目录（含该 DB 的目录）。默认由环境变量推导。")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印将回填哪些源、填什么模板，不写 DB。")
    parser.add_argument("--verbose", "-v", action="store_true", help="更详细日志。")
    args = parser.parse_args(argv)

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")

    service_log_dir = os.path.normpath(args.db_dir) if args.db_dir else get_service_log_dir()
    # dry-run 也 init_db：只为让 get_source 能读到现有 templates 从而准确判「已有→跳过」，
    # 本身不写任何行（写只发生在 _fill_one 的 upsert，dry-run 分支已提前 return）。
    lds.init_db(service_log_dir)
    if args.dry_run:
        logger.info("[dry-run] DB_DIR=%s（只读判重，不写 DB）", service_log_dir)
    else:
        logger.info("DB_DIR=%s（log_dir.db 写入该目录）", service_log_dir)

    roots = _resolve_roots(args.root)
    if not roots:
        logger.error("未找到任何源。检查工作目录与环境变量（LOGS_DIR / 历史路径配置）。")
        return 0

    logger.info("共发现 %d 个源，逐个检查是否需回填模板：", len(roots))
    counts = {"filled": 0, "skipped-has": 0, "skipped-nofmt": 0, "skipped-missing": 0}
    for root in roots:
        res = _fill_one(root, args.dry_run)
        counts[res] += 1
        exists, fmt, _ = _classify(root)
        tpls = default_templates(fmt)
        if res == "filled":
            tag = "回填" if not args.dry_run else "将回填"
            logger.info("  [%s] %s  name=%s root_id=%s fmt=%s → templates=%s",
                        tag, root, get_path_name(root), get_root_id(root), fmt, tpls)
        else:
            reason = {"skipped-has": "已有模板", "skipped-nofmt": f"格式无默认模板({fmt})",
                      "skipped-missing": "目录不存在"}[res]
            logger.info("  [跳过-%s] %s  root_id=%s", reason, root, get_root_id(root))

    logger.info("=" * 60)
    verb = "将回填" if args.dry_run else "已回填"
    logger.info("%s %d 个源；跳过：已有模板 %d、无默认模板 %d、目录不存在 %d",
                verb, counts["filled"], counts["skipped-has"],
                counts["skipped-nofmt"], counts["skipped-missing"])
    if not args.dry_run:
        logger.info("模板已写入 %s", os.path.join(service_log_dir, "log_dir.db"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
