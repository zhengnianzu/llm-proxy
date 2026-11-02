"""
src/keyframe/worker.py — 关键帧分析 worker

职责：消费 `keyframe_analysis` 表里的 pending 请求，把某个会话切成关键帧，
         结果写回同一行。**纯计算，不落盘产物**（批量导出走 `key_frame_export` 模式）。

与其它 worker 的分工：
  - session worker  ：把 index.db 的数据聚合成统计快照
  - export worker   ：执行导出任务，产物写磁盘 / OBS
  - keyframe worker ：按需分析单个会话的分段结构（本进程）

为什么独立进程：
  分析一个会话要逐条读 trace 文件（NFS，单会话最多上百个文件、几百 MB），
  放在 Web 请求里会让页面卡住十几秒。做成 worker 后，前端点「分析」只写一行
  pending 记录、立刻返回，轮询拿结果。

主循环从 logdir_store.claim_keyframe_request() 领取任务（pending → running），
完成后 finish_keyframe_request() 写回 done/failed。领取用原子的
`UPDATE ... WHERE status='pending'`，多实例并发安全。
"""

import argparse
import logging
import os
import signal
import sys
import time
from pathlib import Path

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.log_paths import get_service_log_dir
import utils.logdir_store as logdir_store

logger = logging.getLogger("keyframe_worker")

_shutdown_requested = False

# 空闲时的轮询间隔（秒）。分析是用户触发的低频操作，不需要高频空转。
INTERVAL = float(os.getenv("KEYFRAME_WORKER_INTERVAL", "3") or 3)
# 崩溃遗留的 running 请求超过这个时长就退回 pending。
STALE_MINUTES = int(os.getenv("KEYFRAME_WORKER_STALE_MIN", "30") or 30)


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
    log_file = os.path.join(get_service_log_dir(), "keyframe_worker.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    # 只挂 FileHandler：launcher 已把 stdout/stderr 重定向到同一文件。
    logging.basicConfig(level=logging.INFO, format=fmt,
                        handlers=[logging.FileHandler(log_file, encoding="utf-8")])


def _init_env() -> None:
    svc_dir = get_service_log_dir()
    logdir_store.init_db(svc_dir)
    logger.info("keyframe_worker 启动 svc_dir=%s interval=%.1fs", svc_dir, INTERVAL)


def analyze_session(leaf_path: str, session_key: str) -> dict:
    """分析一个会话，返回可直接入库的汇总。

    复用导出链路的加载与切分：`keyframe_export.analyze_traces` 只算不写盘，
    `keyframes.split_keyframes` 出帧。会话的 trace 列表从叶子 index.db 取。
    """
    from utils.eval.keyframe_export import analyze_traces
    from utils.eval.keyframes import split_keyframes
    import utils.newapi_index_db as nidb

    traces = nidb.get_traces_for_session(leaf_path, session_key)
    filenames = [t.get("filename") for t in traces if t.get("filename")]
    if not filenames:
        raise ValueError("session has no trace files")

    nodes, _raws, errors = analyze_traces(Path(leaf_path), filenames)
    if not nodes:
        raise ValueError("; ".join(errors[:3]) or "no usable trace")

    frames = split_keyframes(nodes)
    frame_dicts = [f.as_dict() for f in frames]

    # 逐 trace 的展示信息：关联列表每条都要显示轮次与所属段。
    # 用 filename → frame 的映射反查，frames 里的 first_file/last_file 是段边界。
    seg_of: dict[str, int] = {}
    for frame in frames:
        start = next((i for i, n in enumerate(nodes) if n.filename == frame.first_file), None)
        end = next((i for i, n in enumerate(nodes) if n.filename == frame.last_file), None)
        if start is None or end is None:
            continue
        for node in nodes[start:end + 1]:
            seg_of[node.filename] = frame.index

    traces_info = []
    for node in nodes:
        traces_info.append({
            "filename": node.filename,
            "messages": node.messages,
            "turns": node.turns,
            "segment": seg_of.get(node.filename, 0),
        })

    breaks = sum(1 for f in frames if f.is_mutation)
    turns = max((f.turns for f in frames), default=0)

    return {
        "frames": frame_dicts,
        "traces": traces_info,
        "trace_total": len(nodes),
        "frame_count": len(frames),
        "break_count": breaks,
        "turns": turns,
        "has_mutation": breaks > 0,
        "errors": errors,
    }


def run_once() -> int:
    """领取并处理一条请求，返回处理条数（0 或 1）。"""
    row = logdir_store.claim_keyframe_request()
    if row is None:
        return 0

    leaf_path = row.get("leaf_path") or ""
    session_key = row.get("session_key") or ""
    logger.info("开始分析 leaf=%s session=%s", leaf_path, session_key)

    import json
    t0 = time.monotonic()
    try:
        result = analyze_session(leaf_path, session_key)
        logdir_store.finish_keyframe_request(
            leaf_path, session_key,
            status="done",
            frames_json=json.dumps(result["frames"], ensure_ascii=False),
            traces_json=json.dumps(result.get("traces") or [], ensure_ascii=False),
            trace_total=result["trace_total"],
            frame_count=result["frame_count"],
            break_count=result["break_count"],
            turns=result["turns"],
            has_mutation=result["has_mutation"],
        )
        logger.info("分析完成 session=%s 帧数=%d 断点=%d 轮次=%d 用时 %.2fs (errors=%d)",
                    session_key, result["frame_count"], result["break_count"],
                    result["turns"], time.monotonic() - t0, len(result["errors"]))
    except Exception as ex:  # noqa: BLE001
        logger.warning("分析失败 session=%s: %s", session_key, ex)
        logdir_store.finish_keyframe_request(
            leaf_path, session_key, status="failed", error=str(ex)[:500],
        )
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="关键帧分析 worker")
    parser.add_argument("--once", action="store_true",
                        help="把当前待分析请求跑空后退出（调试 / cron）")
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    _setup_logging()
    _init_env()

    try:
        n = logdir_store.reset_stale_keyframe_requests()
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
        logger.exception("keyframe_worker 主循环异常退出")
        return 1

    logger.info("keyframe_worker 退出")
    return 0


if __name__ == "__main__":
    sys.exit(main())
