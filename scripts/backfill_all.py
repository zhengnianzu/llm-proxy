#!/usr/bin/env python3
"""
scripts/backfill_all.py — 离线批量回填（数据管理「回填」的命令行版）

对「数据管理」页登记的所有源（活跃 env_dir + 历史路径）逐个执行 new-api 富
index（index.db）回填，等价于在页面上对每个 new-api 源点「同步」+「构建索引」。

与 Web 版共用同一套底层，因此回填状态、进度、日志与页面完全同源：
  - 叶子构建状态 / 计数写进 log_dir.db（数据管理页、导出页读同一张表）
  - 生命周期事件写进 {SERVICE_LOG_DIR}/backfill.log（页面「构建日志」读同一文件）

与 Web 版的关键差异：Web 版由一个全局调度线程异步串行执行（start_backfill 只
入队即返回）；本脚本在当前进程内**同步**串行执行 utils.newapi_backfill._run，
一个源跑完再跑下一个，跑完即退出——适合 cron / 手动补跑，无需服务在跑。

用法:
    # 增量回填所有 new-api 源（跳过已完成且无新增的叶子）
    python -m scripts.backfill_all

    # 全量重建（清掉旧 index.db 重建，用于口径变更 / 修数据）
    python -m scripts.backfill_all --force

    # 只回填指定源（可多次 --root；未给则回填全部登记源）
    python -m scripts.backfill_all --root /data/logs_all/xxx --root /data/hist/yyy

    # 按「标识」筛选源（匹配 name / root_id / 路径子串，可多次，取并集）
    python -m scripts.backfill_all --source jumper-003
    python -m scripts.backfill_all --source 438181a9 --source proxy-004

    # 预演：只打印将回填哪些源，不写 DB、不构建
    python -m scripts.backfill_all --dry-run
    python -m scripts.backfill_all --source jumper --dry-run

    # 覆盖叶子内 enrich 进程池并行度（默认 min(8, CPU)）
    python -m scripts.backfill_all --workers 4

    # 只同步叶子清单到 DB、不真正构建（等价页面「同步」按钮）
    python -m scripts.backfill_all --sync-only

环境变量与服务进程保持一致（决定 SERVICE_LOG_DIR / 活跃 base 的解析）：
    PROXY_PORT, LOG_TASK_TAG, UPSTREAM_API_KEY, LOGS_DIR, LOGS_DIRS_CONFIG
建议在与服务相同的工作目录、相同环境变量下运行，以命中同一 log_dir.db 与源列表。

退出码: 0 全部成功；1 有源在回填中报错（root 级崩溃或叶子失败）。
"""

import argparse
import logging
import os
import sys
import threading
import time
from pathlib import Path

# 允许 `python scripts/backfill_all.py` 直接运行（把项目根加入 import 路径）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.log_paths import get_service_log_dir, get_log_dir
from utils.logs_config import get_stats_roots, get_path_name, get_root_id
from utils.log_scan import detect_format
import utils.logdir_store as lds
import utils.newapi_backfill as bf


logger = logging.getLogger("backfill_all")


def _setup_console_logging(verbose: bool) -> None:
    """把进度打到 stdout。同时让 backfill 事件 logger 也回显一份到控制台，
    这样命令行能实时看到 ROOT START / LEAF OK 等（文件里照常写 backfill.log）。"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter("  bf | %(message)s"))
    console.setLevel(level)
    bf.bf_logger.addHandler(console)


def _resolve_roots(explicit_roots):
    """确定要回填的源列表。

    未显式指定 --root 时，取「数据管理」页同一份源集合：活跃 env_dir + 历史路径。
    活跃 env_dir 用与 app.py 相同的方式解析（get_log_dir("logs_all") 的父目录）。
    """
    if explicit_roots:
        roots = [os.path.normpath(r) for r in explicit_roots]
    else:
        logs_dir = get_log_dir("logs_all")
        env_dir = os.path.dirname(logs_dir)
        roots = [os.path.normpath(r) for r in get_stats_roots(env_dir)]
    # 去重保序
    seen = set()
    uniq = []
    for r in roots:
        if r in seen:
            continue
        seen.add(r)
        uniq.append(r)
    return uniq


def _classify(root):
    """返回 (存在?, 格式, 是否 new-api)。回填只对 new-api 源有意义。"""
    exists = os.path.isdir(root)
    fmt = detect_format(root) if exists else "missing"
    return exists, fmt, (fmt == "newapi")


def _match_source(root, sel):
    """判断某源是否匹配一个 --source 选择器 sel。

    「标识」口径宽松，命中任一即可（不区分大小写）：
      - name 精确相等（get_path_name，页面上展示的名称）
      - root_id 精确相等或前缀（get_root_id，路径 md5 前 8 位，唯一）
      - 路径子串命中（方便只记得一段路径时用）
    name 可能重复（如多个源同名），root_id 唯一——想精确定位单一源用 root_id。
    """
    s = sel.strip().lower()
    if not s:
        return False
    name = get_path_name(root).lower()
    rid = get_root_id(root).lower()
    if s == name:
        return True
    if rid and (s == rid or rid.startswith(s)):
        return True
    if s in root.lower():
        return True
    return False


def _filter_by_sources(roots, selectors):
    """按 --source 选择器过滤源列表；返回 (命中的源, 未命中的选择器)。

    多个 --source 取并集：任一选择器命中某源即保留该源，保持原顺序、去重。
    """
    if not selectors:
        return roots, []
    kept = []
    unmatched = []
    for sel in selectors:
        hits = [r for r in roots if _match_source(r, sel)]
        if not hits:
            unmatched.append(sel)
        for r in hits:
            if r not in kept:
                kept.append(r)
    return kept, unmatched


def _run_with_progress(root, workers, force, interval):
    """在后台线程同步跑 bf._run，同时轮询内存运行态打印「当前叶子 + 处理条数」。

    bf._run 逐叶把 current_leaf / leaf_done / leaf_total 写进程内存状态（Web 页面靠
    轮询 get_backfill_status 展示同一份）；命令行无轮询，故这里起一个 poller 线程
    每 interval 秒读一次，打印当前正在构建哪个叶子、处理到多少条、总叶子进度。

    _run 本身在**本函数调用线程内同步执行**（不是后台进程），poller 只是旁路观察，
    _run 一返回就停。
    """
    stop = threading.Event()

    def _poll():
        last_line = ""
        while not stop.is_set():
            st = bf.get_backfill_status(root)
            leaf = st.get("current_leaf") or ""
            ld = st.get("leaf_done") or 0
            lt = st.get("leaf_total") or 0
            done = st.get("done_leaves") or 0
            total = st.get("total_leaves") or 0
            if leaf:
                if lt:
                    line = f"    构建中 [叶子 {done}/{total}] 当前={leaf} 已处理 {ld}/{lt} 条"
                else:
                    line = f"    构建中 [叶子 {done}/{total}] 当前={leaf}（计数中…）"
            else:
                line = f"    构建中 [叶子 {done}/{total}]"
            # 只在有变化时打印，避免刷屏；无叶子信息（收尾/切叶子间隙）跳过
            if leaf and line != last_line:
                logger.info(line)
                last_line = line
            stop.wait(interval)

    poller = threading.Thread(target=_poll, daemon=True, name=f"bf-progress-{get_root_id(root)}")
    poller.start()
    try:
        bf._run(root, workers=workers, force=force)
    finally:
        stop.set()
        poller.join(timeout=2)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="离线批量回填所有源的 new-api index.db（数据管理「回填」命令行版）")
    parser.add_argument("--root", action="append", default=[],
                        help="只回填指定源目录（可多次）。未给则回填所有登记源。")
    parser.add_argument("--source", action="append", default=[],
                        help="按「标识」筛选源（可多次，取并集）：匹配 name / root_id / 路径子串。"
                             "name 可能重复；要精确定位单一源用 root_id。")
    parser.add_argument("--force", action="store_true",
                        help="全量重建（清掉旧 index.db 重建），否则仅增量。")
    parser.add_argument("--workers", type=int, default=None,
                        help="叶子内 enrich 进程池并行度（默认 min(8, CPU)）。")
    parser.add_argument("--sync-only", action="store_true",
                        help="只同步叶子清单到 DB、不真正构建（等价页面「同步」）。")
    parser.add_argument("--progress-interval", type=float, default=5.0,
                        help="构建时打印「当前叶子/处理条数」进度的间隔秒数（默认 5，0 关闭）。")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印将回填哪些源、不写 DB、不构建。")
    parser.add_argument("--verbose", "-v", action="store_true", help="更详细日志。")
    args = parser.parse_args(argv)

    _setup_console_logging(args.verbose)

    service_log_dir = get_service_log_dir()
    if args.dry_run:
        # dry-run 不落任何盘：不初始化 log_dir.db、不挂 backfill.log handler。
        logger.info("[dry-run] SERVICE_LOG_DIR=%s（不写 DB / 不写日志文件）", service_log_dir)
    else:
        # 与服务同源：状态写 log_dir.db，日志写 backfill.log
        lds.init_db(service_log_dir)
        bf.init_backfill_logger(service_log_dir)
        logger.info("SERVICE_LOG_DIR=%s", service_log_dir)
        logger.info("log_dir.db / backfill.log 与服务进程同源")

    all_roots = _resolve_roots(args.root)
    if not all_roots:
        logger.error("未找到任何源。检查工作目录与环境变量（LOGS_DIR / 历史路径配置）。")
        return 1

    # --source 「标识」筛选（在 new-api 判定之前，先按标识收窄源集合）
    all_roots, unmatched = _filter_by_sources(all_roots, args.source)
    if unmatched:
        logger.error("以下 --source 标识未匹配到任何源：%s", "，".join(unmatched))
        return 1
    if not all_roots:
        logger.error("按 --source 筛选后无源可回填。")
        return 1

    logger.info("共发现 %d 个源，逐个检查是否为 new-api：", len(all_roots))
    targets = []
    for r in all_roots:
        exists, fmt, is_newapi = _classify(r)
        name = get_path_name(r)
        rid = get_root_id(r)
        tag = "回填" if is_newapi else ("跳过-非newapi" if exists else "跳过-不存在")
        logger.info("  [%s] %s  name=%s root_id=%s  fmt=%s  -> %s",
                    tag, r, name, rid, fmt, tag)
        if is_newapi:
            targets.append(r)

    if not targets:
        logger.warning("没有可回填的 new-api 源，结束。")
        return 0

    if args.dry_run:
        mode = "同步清单（不构建）" if args.sync_only else ("全量重建" if args.force else "增量构建")
        logger.info("[dry-run] 将对以下 %d 个 new-api 源执行「%s」（实际未执行）：",
                    len(targets), mode)
        for i, r in enumerate(targets, 1):
            logger.info("  [dry-run] [%d/%d] %s (name=%s root_id=%s)",
                        i, len(targets), r, get_path_name(r), get_root_id(r))
        return 0

    mode = "同步清单（不构建）" if args.sync_only else ("全量重建" if args.force else "增量构建")
    logger.info("=" * 70)
    logger.info("开始逐个回填 %d 个 new-api 源，模式：%s", len(targets), mode)
    logger.info("=" * 70)

    failures = []
    t_all = time.time()

    for idx, root in enumerate(targets, 1):
        name = get_path_name(root)
        logger.info("")
        logger.info(">>> [%d/%d] 源 %s (name=%s)", idx, len(targets), root, name)
        t0 = time.time()

        # 步骤 1：同步叶子清单到 log_dir.db（等价页面「同步」）。
        # 让 DB 先有权威的叶子总数/已建数，页面计数与本次构建同源、不靠磁盘探测。
        try:
            res = bf.sync_leaves(root)
            logger.info("    同步：total=%d added=%d updated=%d built=%d",
                        res["total"], res["added"], res["updated"], res["built"])
        except Exception:
            logger.exception("    同步叶子清单失败：%s", root)
            failures.append((root, "sync_leaves failed"))
            continue

        if args.sync_only:
            # sync-only 也做一次主动核对并落库：DB 里既有清单，又能如实反映磁盘
            # 是否追上 index.jsonl（离线也能写 verify_log，页面可直接读）。
            try:
                from utils.newapi_index_db import verify_root
                v = verify_root(root, ttl=0)
                lds.save_verify(rid, {k: x for k, x in v.items() if not k.startswith("_")},
                                leaf_dir_keys={
                                    p: __import__("utils.log_scan", fromlist=["dir_key_for"]).dir_key_for(
                                        Path(root), Path(p))
                                    for p in v.get("_leaf_map", {})
                                },
                                completed_paths=v.get("_completed_paths"))
                logger.info("    核对：total=%d completed=%d pending=%d（已写入 DB）",
                            v["total"], v["completed"], v["pending"])
            except Exception:
                logger.exception("    核对落库失败：%s", root)
            continue

        # 步骤 2：同步执行回填（等价页面「构建索引」，但不入全局队列、当前进程内跑）。
        # _run 内部：逐叶 build_leaf；状态/计数写 log_dir.db；事件写 backfill.log。
        # progress-interval > 0 时另起 poller 线程打印「当前叶子/处理条数」实时进度。
        workers = args.workers or bf._DEFAULT_WORKERS
        try:
            if args.progress_interval and args.progress_interval > 0:
                _run_with_progress(root, workers, args.force, args.progress_interval)
            else:
                bf._run(root, workers=workers, force=args.force)
        except Exception:
            logger.exception("    回填 _run 崩溃：%s", root)
            failures.append((root, "_run crashed"))
            continue

        # 读回本源最终状态（内存运行态 + DB 汇总），打一行小结。
        st = bf.get_backfill_status(root)
        rid = get_root_id(root)
        db_summ = lds.count_summary(rid) if lds.has_any(rid) else {}
        took = time.time() - t0
        logger.info("    完成：status=%s done_leaves=%s total_leaves=%s "
                    "db(built=%s/total=%s error=%s) took=%.1fs",
                    st.get("status"), st.get("done_leaves"), st.get("total_leaves"),
                    db_summ.get("built"), db_summ.get("total"),
                    db_summ.get("error"), took)

        # 步骤 3：回填后主动核对（逐叶实测 index.db 是否追上 index.jsonl），并落库。
        # 这样离线回填的成果不止写在构建状态里，也写进 verify_log + leaf_status.verified，
        # 页面刷新即可读 DB 展示，无需重新扫盘。
        try:
            from utils.newapi_index_db import verify_root
            v = verify_root(root, ttl=0)
            lds.save_verify(rid, {k: x for k, x in v.items() if not k.startswith("_")},
                            leaf_dir_keys={
                                p: __import__("utils.log_scan", fromlist=["dir_key_for"]).dir_key_for(
                                    Path(root), Path(p))
                                for p in v.get("_leaf_map", {})
                            },
                            completed_paths=v.get("_completed_paths"))
            logger.info("    核对：total=%d completed=%d pending=%d（已写入 DB）",
                        v["total"], v["completed"], v["pending"])
        except Exception:
            logger.exception("    核对落库失败：%s", root)

        # 判定该源是否有失败/卡住的叶子（DB error 计数或内存 error 状态）。
        if st.get("status") == "error" or db_summ.get("error", 0):
            failures.append((root, f"error_leaves={db_summ.get('error', 0)} "
                                   f"status={st.get('status')}"))

    logger.info("")
    logger.info("=" * 70)
    total_took = time.time() - t_all
    if failures:
        logger.error("回填结束：%d/%d 个源有问题，总耗时 %.1fs",
                     len(failures), len(targets), total_took)
        for root, why in failures:
            logger.error("    失败源 %s：%s", root, why)
        logger.error("详情见构建日志：%s", os.path.join(service_log_dir, "backfill.log"))
        return 1

    logger.info("回填结束：全部 %d 个源成功，总耗时 %.1fs", len(targets), total_took)
    logger.info("状态已写入 log_dir.db，日志见 %s",
                os.path.join(service_log_dir, "backfill.log"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
