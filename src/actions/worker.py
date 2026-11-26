"""
src/actions/worker.py — 动作决策分析 worker

职责：消费 `action_analysis` 表里的 pending 请求，把某个会话的全部 trace 抽成
**一条**有序动作链，结果写回同一行。**纯计算，不落盘产物**。

与 keyframe worker 的分工（两者互不影响，可同时运行）：
  - keyframe worker：判上下文连不连续（compaction / restart / tools_changed 边界）
  - actions  worker：判每个动作成没成（成功 / 失败 / 失败原因 / 是否已修复）

为什么独立进程：分析一个会话要读它全部的 trace 文件（NFS，单个会话最多上百个
文件、几百 MB），放在 Web 请求里会让页面卡十几秒。前端只写一行 pending 记录、
立刻返回，然后轮询拿结果。

关键设计——**跨 trace 合并**：同一个会话里每条 trace 都携带完整历史，后一条是
前一条的超集（实测 msg_count 100→102→…→134）。逐条抽完直接拼接会得到 19 倍
的重复动作（实测 1064 vs 真实 65）。故用稳定的 call_id 合并成一条全局链，
见 ``utils.eval.actions.merge_action_chains``。
"""

import argparse
import json
import logging
import os
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.log_paths import get_service_log_dir
import utils.logdir_store as logdir_store

logger = logging.getLogger("actions_worker")

_shutdown_requested = False

# 空闲时的轮询间隔（秒）。分析是用户触发的低频操作，不需要高频空转。
INTERVAL = float(os.getenv("ACTIONS_WORKER_INTERVAL", "3") or 3)
# 崩溃遗留的 running 请求超过这个时长就退回 pending。
STALE_MINUTES = int(os.getenv("ACTIONS_WORKER_STALE_MIN", "30") or 30)
# 读 trace 文件的并发度。NFS 上单文件可达数 MB，IO 密集型，8 是稳值。
READ_WORKERS = max(1, min(int(os.getenv("ACTIONS_WORKER_READ_CONCURRENCY", "8") or 8), 32))


def _handle_signal(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    logger.info("收到信号 %s，准备优雅退出（当前任务做完后退出）", signum)


def _sleep(seconds: float) -> None:
    end = seconds
    while end > 0:
        if _shutdown_requested:
            return
        time.sleep(min(0.5, end))
        end -= 0.5


def _setup_logging() -> None:
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    log_file = os.path.join(get_service_log_dir(), "actions_worker.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    # 只挂 FileHandler：launcher 已把 stdout/stderr 重定向到同一文件。
    logging.basicConfig(level=logging.INFO, format=fmt,
                        handlers=[logging.FileHandler(log_file, encoding="utf-8")])


def _init_env() -> None:
    svc_dir = get_service_log_dir()
    logdir_store.init_db(svc_dir)
    logger.info("actions_worker 启动 svc_dir=%s interval=%.1fs read_workers=%d",
                svc_dir, INTERVAL, READ_WORKERS)


def _trace_messages(leaf_path: str, filename: str):
    """读一个 trace 文件，返回 messages（读不到返回 None）。"""
    from utils.newapi_format import parse_combined_file
    parsed = parse_combined_file(Path(leaf_path) / filename)
    if parsed is None:
        return None
    return parsed.get("messages")


def analyze_session(leaf_path: str, session_key: str) -> dict:
    """分析一个会话的动作链，返回可直接入库的汇总。

    流程：取会话的 trace 列表（按 ts 升序）→ 并发读文件 → 逐条抽动作链
    → 跨 trace 按 call_id 合并 → 统计。
    """
    from utils.eval.actions import extract_actions, merge_action_chains, summarize
    import utils.newapi_index_db as nidb

    traces = nidb.get_traces_for_session(leaf_path, session_key)
    filenames = [t.get("filename") for t in traces if t.get("filename")]
    if not filenames:
        raise ValueError("session has no trace files")

    errors = []

    def _one(filename):
        try:
            return filename, _trace_messages(leaf_path, filename)
        except Exception as ex:  # noqa: BLE001  单个文件坏掉不该让整场分析失败
            return filename, ("__error__", str(ex))

    with ThreadPoolExecutor(max_workers=min(READ_WORKERS, len(filenames))) as ex:
        loaded = list(ex.map(_one, filenames))

    chains = []
    per_trace = []
    for filename, msgs in loaded:
        if isinstance(msgs, tuple) and msgs and msgs[0] == "__error__":
            errors.append(f"{filename}: {msgs[1]}")
            continue
        if not msgs:
            errors.append(f"{filename}: missing or damaged file")
            continue
        actions = extract_actions(msgs)
        chains.append(actions)
        per_trace.append({
            "filename": filename,
            "messages": len(msgs),
            "actions": len(actions),
            "fail": sum(1 for a in actions if not a.ok),
        })

    if not chains:
        raise ValueError("; ".join(errors[:3]) or "no usable trace")

    merged = merge_action_chains(chains)
    stats = summarize(merged)

    # 轮次口径与关键帧/统计模块一致：取最长那条 trace 的真实用户轮次
    # （每条 trace 都带完整历史，最长的那条轮次最全）。
    turns = 0
    try:
        from utils.message_common import count_real_user_turns
        for filename, msgs in loaded:
            if isinstance(msgs, list) and msgs:
                turns = max(turns, int(count_real_user_turns(msgs) or 0))
    except Exception:  # noqa: BLE001  统计失败不该让整场分析失败
        turns = 0

    return {
        "actions": [a.as_dict() for a in merged],
        "traces": per_trace,
        "trace_total": len(per_trace),
        "action_total": stats["total"],
        "fail_total": stats["fail"],
        "repaired_total": stats["repaired"],
        "turns": turns,
        "errors": errors,
    }


def run_once() -> int:
    """领取并处理一条请求，返回处理条数（0 或 1）。"""
    row = logdir_store.claim_action_request()
    if row is None:
        return 0

    leaf_path = row.get("leaf_path") or ""
    session_key = row.get("session_key") or ""
    logger.info("开始分析 leaf=%s session=%s", leaf_path, session_key)

    t0 = time.monotonic()
    try:
        result = analyze_session(leaf_path, session_key)
        logdir_store.finish_action_request(
            leaf_path, session_key,
            status="done",
            actions_json=json.dumps(result["actions"], ensure_ascii=False),
            traces_json=json.dumps(result["traces"], ensure_ascii=False),
            trace_total=result["trace_total"],
            action_total=result["action_total"],
            fail_total=result["fail_total"],
            repaired_total=result["repaired_total"],
            turns=result["turns"],
        )
        logger.info("分析完成 session=%s 动作=%d 失败=%d 已修复=%d 用时 %.2fs (errors=%d)",
                    session_key, result["action_total"], result["fail_total"],
                    result["repaired_total"], time.monotonic() - t0, len(result["errors"]))
    except Exception as ex:  # noqa: BLE001
        logger.warning("分析失败 session=%s: %s", session_key, ex)
        logdir_store.finish_action_request(
            leaf_path, session_key, status="failed", error=str(ex)[:500],
        )
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="动作决策分析 worker")
    parser.add_argument("--once", action="store_true",
                        help="把当前待分析请求跑空后退出（调试 / cron）")
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    _setup_logging()
    _init_env()

    try:
        n = logdir_store.reset_stale_action_requests()
        if n:
            logger.info("重置 %d 条超时未完成的分析请求", n)
    except Exception as ex:  # noqa: BLE001
        logger.warning("重置超时请求失败: %s", ex)

    try:
        while not _shutdown_requested:
            n = run_once()
            if args.once and n == 0:
                break
            _sleep(0.1 if n else INTERVAL)
    except Exception:
        logger.exception("actions_worker 主循环异常退出")
        return 1

    logger.info("actions_worker 退出")
    return 0


if __name__ == "__main__":
    sys.exit(main())
