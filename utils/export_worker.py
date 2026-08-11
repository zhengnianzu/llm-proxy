#!/usr/bin/env python3
"""
utils/export_worker.py — 导出独立 worker 进程

把导出/质检任务的**执行**从 app 主进程（uvicorn）剥离到独立长驻进程，与 web
事件循环彻底隔离 CPU/IO —— 大导出任务打满磁盘 IO 时不再拖垮代理请求。

职责：轮询 export_session_record.db，领取 status='queued' 的记录，按其 task_json
参数重建执行上下文并跑任务（复用 export_routes 里已提到模块级的
_run_task_from_record / _run_upload_only）。跨进程互斥靠 export_queue.slot*.lock
文件锁（见 export_routes.init_export_lock_dir 的绝对路径修复）。

与 obs_sync.py / scripts/backfill_all.py 同款守护进程骨架：
  - sys.path 注入，保证 `python -m utils.export_worker` 直接可 import
  - SIGTERM/SIGINT 优雅退出：置停机标志，正在跑的任务在目录检查点自然收尾
  - 启动时 cancel_interrupted()：重启即清空上一轮遗留队列（原 app 侧语义搬到这里）

用法：
    python -m utils.export_worker            # 长驻
    python -m utils.export_worker --once     # 把当前队列跑空后退出（调试/cron）

环境变量与服务进程保持一致（决定 SERVICE_LOG_DIR / DB 路径 / 源解析）：
    PROXY_PORT, LOG_TASK_TAG, UPSTREAM_API_KEY, LOGS_DIR
必须在与 app 相同的工作目录、相同环境变量下运行，才能命中同一
export_session_record.db 与同一槽位锁目录。
"""

import argparse
import logging
import os
import signal
import sys
import time

# 允许 `python utils/export_worker.py` 直接运行（把项目根加入 import 路径）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.log_paths import get_service_log_dir
import utils.export_store as export_store
import utils.export_jobs as export_jobs
import utils.export_routes as export_routes
import utils.logdir_store as logdir_store
import utils.session_store as session_store

logger = logging.getLogger("export_worker")

_shutdown_requested = False


def _handle_signal(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    logger.info("收到信号 %s，准备优雅退出（正在跑的任务将在检查点收尾）", signum)


def _interruptible_sleep(seconds: float) -> None:
    """分片 sleep，收到停机信号立即醒来。"""
    end = seconds
    step = 0.5
    while end > 0:
        if _shutdown_requested:
            break
        time.sleep(min(step, end))
        end -= step


def _setup_logging() -> None:
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(sh)


def _init_env(svc_dir: str) -> None:
    """worker 进程独立初始化：DB 连接 + 槽位锁目录 + 清理遗留队列。

    与 app 各自持有自己的 sqlite 连接（export_session_record.db 为 WAL +
    busy_timeout，多进程读写安全）。init_export_lock_dir 用绝对路径，与 app 的
    register_export_routes 对齐 → 槽位文件锁跨进程互斥成立。
    """
    export_store.init_db(svc_dir)
    # session_store 也必须初始化：native 目录导出走 export_session_index →
    # _refresh_session_cache → log_routes._refresh_state，后者逐行读 index.jsonl 并
    # 写 session_store（_ss.create_session/set_progress 等）。未 init 时该模块级 _conn
    # 为 None，凡「有 index.jsonl、需增量刷新写库」的 native 目录都会抛
    # "session_store not initialized"（空/无 index 目录因提前 return 反而躲过）。
    # 与 app.py init_session_db(SERVICE_LOG_DIR) 同源（svc_dir 与之同源），
    # 指向同一个 session_cache.db。
    session_store.init_db(svc_dir)
    # logdir_store 也必须初始化：reformat/eval 解析 mtime key 里的 <root_id>/ 前缀
    # 要读 sources 表（get_stats_roots → _list_sources → lds.list_sources）。
    # 未 init 时 _list_sources 遇 _ready()=False 返回 []，历史 new-api 源的
    # root_id（如 240fa79b）解析不出真实根路径，fallback 成 env_dir/<mt> 拼出
    # 不存在的目录 → “未找到 .session_cache.jsonl 且 DB 无数据”。app 侧同款
    # init_logdir_db(SERVICE_LOG_DIR)，svc_dir 与之同源。
    logdir_store.init_db(svc_dir)
    export_routes.init_export_lock_dir(svc_dir)
    # 重启即清空上一轮遗留 running/queued/pending（draft 保留）。原为 app 启动时执行，
    # 剥离后归 worker：app 重启不再误杀 worker 正在跑的任务。
    try:
        n = export_store.cancel_interrupted()
        if n:
            logger.info("启动清理：取消 %d 条遗留导出记录（队列重置）", n)
    except Exception:
        logger.exception("启动清理 cancel_interrupted 失败")


def _run_one(rec: dict) -> None:
    """执行一条已领取（status=pending）的记录。参数来自 task_json。"""
    record_id = rec["id"]
    params = export_jobs.load_params(record_id)
    if params is None:
        # 老记录 / 升级瞬间在途：无执行参数，无法重建，标 failed 提示重建。
        export_store.update_status(
            record_id, "failed",
            error_message="缺少任务参数（旧记录或升级前遗留），请在页面重新发起")
        logger.warning("record %s 无 task_json，标 failed", record_id)
        return

    if params.get("kind") == export_jobs.KIND_UPLOAD_RETRY:
        export_store.update_status(record_id, "running", error_message="")
        export_routes._run_upload_only(
            record_id, rec.get("local_copy_dir", ""), rec.get("obs_dst", ""))
        return

    export_routes._run_task_from_record(record_id, params)


def _drain(once: bool) -> None:
    """主循环：抢槽 → 领取 queued 记录 → 执行 → 释放槽。空闲时分片轮询。

    槽位文件锁保证全局并发 ≤ EXPORT_CONCURRENCY（默认 1，全局串行）。先抢槽再领取，
    使多 worker / 多槽场景下每个执行位都对应一个真实的空槽。
    """
    while not _shutdown_requested:
        lock_fp = export_routes._acquire_slot()
        if lock_fp is None:
            # 槽满（其他 worker/槽在跑）→ 稍后再试
            _interruptible_sleep(1.0)
            continue
        try:
            rec = export_store.claim_next_queued()
            if rec is None:
                # 无待执行任务：释放槽后小睡
                export_routes._release_slot(lock_fp)
                lock_fp = None
                if once:
                    return
                _interruptible_sleep(1.0)
                continue
            logger.info("领取导出任务 record=%s mode=%s", rec["id"], rec.get("mode"))
            try:
                _run_one(rec)
            except Exception:
                logger.exception("执行 record=%s 崩溃", rec.get("id"))
                try:
                    export_store.update_status(rec["id"], "failed",
                                               error_message="worker 执行异常")
                except Exception:
                    logger.exception("回写 failed 也失败 record=%s", rec.get("id"))
        finally:
            export_routes._release_slot(lock_fp)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出独立 worker 进程")
    parser.add_argument("--once", action="store_true",
                        help="把当前队列跑空后退出（调试 / cron），否则长驻。")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _setup_logging()
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    svc_dir = get_service_log_dir()
    logger.info("export_worker 启动 SERVICE_LOG_DIR=%s once=%s", svc_dir, args.once)
    _init_env(svc_dir)

    try:
        _drain(once=args.once)
    except Exception:
        logger.exception("export_worker 主循环异常退出")
        return 1
    logger.info("export_worker 退出")
    return 0


if __name__ == "__main__":
    sys.exit(main())
