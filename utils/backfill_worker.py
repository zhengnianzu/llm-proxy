#!/usr/bin/env python3
"""
utils/backfill_worker.py — 回填独立 worker 进程

把 new-api index.db 回填的**执行**从 app 主进程（uvicorn）剥离到独立长驻进程，
与 web 事件循环彻底隔离 CPU/IO —— 7GB 级构建打满磁盘 IO / 进程池吃满 CPU 时，
不再拖垮代理请求。

职责：轮询 log_dir.db 的 backfill_requests 表，领取 status='pending' 的请求，
按其 (root, workers, force) 调 newapi_backfill._run 执行，完成后回写终态。
单进程单线程 + root 文件锁 ⇒ 全局串行（7GB IO 场景正需序列化；pending 列表向
用户透明展示真实排队）。叶子内部 _run 仍开 ProcessPoolExecutor 并行。

与 export_worker.py / obs_sync.py 同款守护进程骨架：
  - sys.path 注入，保证 `python -m utils.backfill_worker` 直接可 import
  - SIGTERM/SIGINT 优雅退出：置停机标志，正在跑的叶子在检查点自然收尾
  - 启动时清理上一轮遗留：reset_building_on_startup（叶子 building→pending）
    + reset_running_backfill_on_startup（请求 running/pending→failed）

用法：
    python -m utils.backfill_worker            # 长驻
    python -m utils.backfill_worker --once     # 把当前队列跑空后退出（调试/cron）

环境变量与服务进程保持一致（决定 SERVICE_LOG_DIR / DB 路径 / 源解析）：
    PROXY_PORT, LOG_TASK_TAG, UPSTREAM_API_KEY, LOGS_DIR
必须在与 app 相同的工作目录、相同环境变量下运行，才能命中同一 log_dir.db。
"""

import argparse
import logging
import os
import signal
import sys
import time

# 允许 `python utils/backfill_worker.py` 直接运行（把项目根加入 import 路径）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.log_paths import get_service_log_dir
import utils.logdir_store as logdir_store
import utils.newapi_backfill as newapi_backfill

logger = logging.getLogger("backfill_worker")

_shutdown_requested = False
_SVC_DIR = ""


def _handle_signal(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    logger.info("收到信号 %s，准备优雅退出（正在跑的叶子将在检查点收尾）", signum)


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
    """worker 进程独立初始化：DB 连接 + 回填专用日志 + 清理遗留队列。

    与 app 各自持有自己的 sqlite 连接（log_dir.db 为 WAL + busy_timeout，多进程
    读写安全）。启动即清空上一轮在途：请求行 running/pending→failed、叶子
    building→pending。原为 app 启动时执行，剥离后归 worker：app 重启不再误清
    worker 正在跑的请求。
    """
    global _SVC_DIR
    _SVC_DIR = svc_dir
    logdir_store.init_db(svc_dir)
    newapi_backfill.init_backfill_logger(svc_dir)
    try:
        n_leaf = logdir_store.reset_building_on_startup()
    except Exception:
        logger.exception("启动清理 reset_building_on_startup 失败")
    try:
        n_req = logdir_store.reset_running_backfill_on_startup()
        if n_req:
            logger.info("启动清理：取消 %d 条遗留回填请求（队列重置）", n_req)
    except Exception:
        logger.exception("启动清理 reset_running_backfill_on_startup 失败")


def _root_lock_path(req_id: int) -> str:
    return os.path.join(_SVC_DIR, f"backfill.root.{req_id}.lock")


def _run_one(rec: dict) -> None:
    """执行一条已领取（status=running）的回填请求。参数来自请求行。"""
    import fcntl

    req_id = rec["id"]
    root = rec["root"]
    workers = int(rec.get("workers") or newapi_backfill._DEFAULT_WORKERS)
    force = bool(rec.get("force"))

    # root 级文件锁：belt-and-suspenders，防第二个 worker 实例并发同源。单进程单线程
    # 本已串行；多实例误启动时靠此锁兜底（拿不到即让给持锁者，标 failed 由其重跑）。
    lock_path = _root_lock_path(req_id)
    lock_fp = open(lock_path, "w")
    try:
        try:
            fcntl.flock(lock_fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            logger.warning("root 锁被占用，跳过 req=%s root=%s", req_id, root)
            logdir_store.complete_backfill(req_id, ok=False, error="root 锁被占用（疑似多 worker 实例）")
            return
        logger.info("开始回填 req=%s root=%s workers=%d force=%s", req_id, root, workers, force)
        try:
            newapi_backfill._run(root, workers=workers, force=force)
            logdir_store.complete_backfill(req_id, ok=True)
            logger.info("回填完成 req=%s root=%s", req_id, root)
        except Exception as e:
            logger.exception("回填执行崩溃 req=%s root=%s", req_id, root)
            logdir_store.complete_backfill(req_id, ok=False, error=str(e))
    finally:
        try:
            fcntl.flock(lock_fp, fcntl.LOCK_UN)
            lock_fp.close()
            os.unlink(lock_path)
        except OSError:
            pass


def _drain(once: bool) -> None:
    """主循环：领取 pending 请求 → 执行 → 回写终态。空闲时分片轮询。

    单进程单线程逐条领取执行 ⇒ 全局串行（同一时刻至多一个 root 在回填），
    正是 7GB IO 场景所需；其余 pending 请求即真实排队，向用户透明展示。
    """
    while not _shutdown_requested:
        rec = logdir_store.claim_next_backfill()
        if rec is None:
            if once:
                return
            _interruptible_sleep(1.0)
            continue
        try:
            _run_one(rec)
        except Exception:
            logger.exception("处理回填请求崩溃 req=%s", rec.get("id"))
            try:
                logdir_store.complete_backfill(rec["id"], ok=False, error="worker 执行异常")
            except Exception:
                logger.exception("回写 failed 也失败 req=%s", rec.get("id"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="回填独立 worker 进程")
    parser.add_argument("--once", action="store_true",
                        help="把当前队列跑空后退出（调试 / cron），否则长驻。")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _setup_logging()
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    svc_dir = get_service_log_dir()
    logger.info("backfill_worker 启动 SERVICE_LOG_DIR=%s once=%s", svc_dir, args.once)
    _init_env(svc_dir)

    try:
        _drain(once=args.once)
    except Exception:
        logger.exception("backfill_worker 主循环异常退出")
        return 1
    logger.info("backfill_worker 退出")
    return 0


if __name__ == "__main__":
    sys.exit(main())
