from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from utils.key_store import get_key_full

from . import db
from .consumer import reflect
from .prompt_loader import load_prompt

# 全局最大存活 worker 数默认值（无额外限制单机一个满额 run）。
DEFAULT_MAX_GLOBAL_WORKERS = 32


class WorkerManager:
    """反思 worker 调度。

    全局并发闸门 / 排队 / RPM-TPM 全部落 SQLite，跨 uvicorn 多进程一致：
    - 名额口径 = SUM(worker_count) WHERE status='running'（try_claim_slot 事务判定）。
    - 排队 = status='queued'，任一进程有 run 结束就 claim_next_queued 接力拉起。
    - RPM/TPM = reflection_req_log 表窗口聚合。
    仅 `_stops`（本进程线程的停止事件）和本进程存活计数留在内存——它们天然是
    进程本地的（run 的线程就在启动它的进程里）。
    """

    def __init__(self, db_path: Path, prompt_dir: Path):
        self.db_path, self.prompt_dir = db_path, prompt_dir
        self._stops: dict[str, threading.Event] = {}
        self._local_alive: dict[str, int] = {}  # 本进程内 run 的存活 worker 线程数
        self._lock = threading.Lock()

    def _max_global_workers(self) -> int:
        raw = db.get_setting(self.db_path, "max_global_workers",
                             str(DEFAULT_MAX_GLOBAL_WORKERS))
        try:
            return max(1, int(raw))
        except (TypeError, ValueError):
            return DEFAULT_MAX_GLOBAL_WORKERS

    def start(self, run_id: str) -> None:
        run = db.get_run(self.db_path, run_id)
        if not run or run["status"] not in {"draft", "paused", "queued"}:
            raise ValueError("Run 当前状态不能启动")
        key = get_key_full(run["reflection_api_key_id"])
        if not key or key["status"] != "active":
            raise ValueError("Reflection Key 已禁用或不存在")
        # 预加载（失败即抛，不占名额）
        prompt = load_prompt(self.prompt_dir, run["method"])
        snapshot = db.get_run_snapshot(self.db_path, run_id)
        limit = self._max_global_workers()
        # 原子决定 running / queued / blocked（跨进程互斥）
        decision = db.try_claim_slot(self.db_path, run_id, limit)
        if decision == "blocked":
            raise ValueError(f"数据集 {run['source_export_id']} 已有活跃 Run")
        if decision == "invalid":
            raise ValueError("Run 当前状态不能启动")
        if decision == "queued":
            db.append_run_log(self.db_path, run_id,
                              f"[run] 全局 worker 已满(limit={limit})，"
                              f"需要 {run['worker_count']} 个 → 排队等待")
            return
        # decision == "running"：名额已在事务里占好（status=running），起线程
        self._launch(run, snapshot, key["key"], prompt, int(run["source_export_id"]))

    def _launch(self, run: dict, snapshot: dict, api_key: str, prompt,
                export_id: int) -> None:
        """起 worker 线程（status 已在 try_claim_slot/claim_next_queued 里落成 running）。"""
        run_id = run["run_id"]
        stop = threading.Event()
        with self._lock:
            self._stops[run_id] = stop
        db.append_run_log(self.db_path, run_id,
                          f"[run] 启动 {run['worker_count']} 个 worker, "
                          f"model={run['reflection_model']}, method={run['method']}, "
                          f"export_id={export_id}")
        prompt_obj, snap = prompt, snapshot
        for index in range(run["worker_count"]):
            threading.Thread(
                target=self._loop,
                args=(run, snap, api_key, prompt_obj, stop, index, export_id),
                daemon=True, name=f"reflection-{run_id}-{index}",
            ).start()

    def stop(self, run_id: str, *, cancel: bool = False) -> None:
        event = self._stops.get(run_id)
        if event:
            event.set()
        # 落终态 → 释放全局名额（status 离开 running）
        db.set_run_status(self.db_path, run_id, "cancelled" if cancel else "paused")
        with self._lock:
            self._stops.pop(run_id, None)
        self._pump_queue()

    # ---- 本进程存活计数：仅用于触发"最后一个 worker 退出→泵队列" ----

    def _inc_alive(self, run_id: str) -> None:
        with self._lock:
            self._local_alive[run_id] = self._local_alive.get(run_id, 0) + 1

    def _dec_alive(self, run_id: str) -> bool:
        """减一；返回本进程该 run 是否已无存活 worker。"""
        with self._lock:
            remaining = self._local_alive.get(run_id, 0) - 1
            if remaining > 0:
                self._local_alive[run_id] = remaining
                return False
            self._local_alive.pop(run_id, None)
            self._stops.pop(run_id, None)
            return True

    def _pump_queue(self) -> None:
        """名额释放后，尽可能多地把 queued run 翻成 running 并在本进程起线程。

        claim_next_queued 在 DB 事务里原子占名额，跨进程安全——每个腾出名额的
        进程都会泵一次，自然接力。被本进程 claim 到的 run 就在本进程起线程。
        """
        while True:
            limit = self._max_global_workers()
            row = db.claim_next_queued(self.db_path, limit)
            if not row:
                return
            run_id = row["run_id"]
            export_id = int(row["source_export_id"])
            key = get_key_full(row["reflection_api_key_id"])
            if not key or key["status"] != "active":
                # key 失效：退回 paused，别占着 running 名额，继续泵下一个
                db.set_run_status(self.db_path, run_id, "paused")
                db.append_run_log(self.db_path, run_id,
                                  "[run] 出队启动失败: Reflection Key 已禁用或不存在",
                                  level="error")
                continue
            try:
                prompt = load_prompt(self.prompt_dir, row["method"])
                snapshot = db.get_run_snapshot(self.db_path, run_id)
            except Exception as exc:
                db.set_run_status(self.db_path, run_id, "paused")
                db.append_run_log(self.db_path, run_id,
                                  f"[run] 出队启动失败: {type(exc).__name__}: {exc}",
                                  level="error")
                continue
            self._launch(dict(row), snapshot, key["key"], prompt, export_id)

    def _record_request(self, in_tok: int, out_tok: int) -> None:
        db.record_reflection_request(self.db_path, in_tok + out_tok)

    def live_stats(self) -> dict:
        limit = self._max_global_workers()
        with db.connect(self.db_path) as conn:
            active = db.global_running_workers(conn)
            rpm, tpm = db.reflection_rpm_tpm(conn)
            queued = conn.execute(
                "SELECT COUNT(*) n FROM reflection_runs WHERE status='queued'"
            ).fetchone()["n"]
        return {
            "active_workers": active,
            "rpm": rpm,
            "tpm": tpm,
            "global_limit": limit,
            "queued_runs": int(queued),
        }

    def _loop(self, run: dict, snapshot: dict, api_key: str, prompt,
              stop: threading.Event, worker_index: int, export_id: int) -> None:
        run_id = run["run_id"]
        tag = f"[worker-{worker_index}]"
        self._inc_alive(run_id)
        try:
            while not stop.is_set():
                with db.connect(self.db_path) as conn:
                    conn.execute("BEGIN IMMEDIATE")
                    task = conn.execute(
                        "SELECT * FROM dataset_tasks "
                        "WHERE export_id=? AND latest_status='pending' "
                        "ORDER BY updated_at LIMIT 1",
                        (export_id,),
                    ).fetchone()
                    if not task:
                        conn.commit()
                        break
                    now = time.time()
                    conn.execute(
                        "UPDATE dataset_tasks SET latest_status='processing',"
                        "last_attempt_at=?,updated_at=? WHERE uuid=?",
                        (now, now, task["uuid"]),
                    )
                    attempt_no = task["retry_count"] + 1
                    cursor = conn.execute(
                        "INSERT INTO task_attempts"
                        "(run_id,task_uuid,export_id,session_id,trajectory_path,block_path,"
                        "attempt_no,status,started_at) VALUES(?,?,?,?,?,?,?, 'running',?)",
                        (run_id, task["uuid"], export_id, task["session_id"],
                         task["trajectory_path"], task["block_path"], attempt_no, now),
                    )
                    attempt_id = cursor.lastrowid
                detail_path = task["detail_path"] if task["detail_path"] else None
                if detail_path and Path(detail_path).is_file():
                    detail = json.loads(Path(detail_path).read_text(encoding="utf-8"))
                    task_signature = detail.get("signature", task["signature"])
                    task_thinking = detail.get("original_thinking", task["original_thinking"])
                else:
                    task_signature = task["signature"]
                    task_thinking = task["original_thinking"]
                db.append_run_log(self.db_path, run_id,
                                  f"{tag} 认领 {task['session_id']}/{task['block_path']}")
                try:
                    result = reflect(endpoint=run["reflection_endpoint"], api_key=api_key,
                                     model=run["reflection_model"], instruction=prompt.instruction,
                                     tool=prompt.tool, unrelated_thinking=prompt.unrelated_thinking,
                                     signature=task_signature, thinking=task_thinking,
                                     method=run["method"],
                                     stream=bool(snapshot.get("stream", False)),
                                     max_tokens=int(snapshot.get("max_tokens", 16384)))
                    with db.connect(self.db_path) as conn:
                        now = time.time()
                        conn.execute(
                            "UPDATE dataset_tasks SET latest_status='done',"
                            "latest_run_id=?,latest_sentence_count=?,latest_model=?,"
                            "latest_response_id=?,latest_stop_reason=?,latest_usage_json=?,"
                            "last_error=NULL,updated_at=? WHERE uuid=?",
                            (run_id, result.sentence_count, result.model, result.response_id,
                             result.stop_reason,
                             json.dumps(result.usage) if result.usage else None,
                             now, task["uuid"]),
                        )
                        conn.execute(
                            "UPDATE task_attempts SET status='done',finished_at=?,"
                            "response_id=?,usage_json=? WHERE attempt_id=?",
                            (now, result.response_id,
                             json.dumps(result.usage) if result.usage else None, attempt_id),
                        )
                    db.increment_run_stat(self.db_path, run_id, "snapshot_done")
                    if result.usage:
                        self._record_request(
                            int(result.usage.get("input_tokens", 0) or 0),
                            int(result.usage.get("output_tokens", 0) or 0))
                    if detail_path:
                        try:
                            detail = json.loads(Path(detail_path).read_text(encoding="utf-8"))
                        except (OSError, json.JSONDecodeError):
                            detail = {}
                        detail["processed_text"] = result.text
                        detail["tool_input"] = result.tool_input
                        detail["raw_response"] = result.raw_response
                        Path(detail_path).write_text(
                            json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8")
                    usage_str = ""
                    if result.usage:
                        usage_str = (f" in={result.usage.get('input_tokens', 0)}"
                                     f" out={result.usage.get('output_tokens', 0)}")
                    db.append_run_log(self.db_path, run_id,
                                      f"{tag} ✓ {task['session_id']}/{task['block_path']} "
                                      f"({result.sentence_count} 句{usage_str})")
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"[:4000]
                    retry = attempt_no
                    final_failed = retry >= task["max_retries"]
                    new_status = "failed" if final_failed else "pending"
                    with db.connect(self.db_path) as conn:
                        now = time.time()
                        if final_failed:
                            conn.execute(
                                "UPDATE dataset_tasks SET latest_status=?,latest_run_id=?,"
                                "retry_count=?,last_error=?,updated_at=? WHERE uuid=?",
                                (new_status, run_id, retry, error, now, task["uuid"]),
                            )
                        else:
                            conn.execute(
                                "UPDATE dataset_tasks SET latest_status=?,"
                                "retry_count=?,last_error=?,updated_at=? WHERE uuid=?",
                                (new_status, retry, error, now, task["uuid"]),
                            )
                        conn.execute(
                            "UPDATE task_attempts SET status='failed',finished_at=?,"
                            "error=? WHERE attempt_id=?",
                            (now, error, attempt_id),
                        )
                    if final_failed:
                        db.increment_run_stat(self.db_path, run_id, "snapshot_failed")
                    db.append_run_log(self.db_path, run_id,
                                      f"{tag} ✗ {task['session_id']}/{task['block_path']} "
                                      f"retry={retry}/{task['max_retries']} {error[:200]}",
                                      level="error")
            self._finish_if_empty(run_id, export_id)
        finally:
            # 本进程该 run 最后一个 worker 退出：run 已落终态(completed/paused/…)，
            # 全局名额随之释放 → 泵一次队列，把排队中的 run 接力拉起（跨进程亦然）。
            if self._dec_alive(run_id):
                self._pump_queue()

    def _finish_if_empty(self, run_id: str, export_id: int) -> None:
        with db.connect(self.db_path) as conn:
            counts = db.dataset_counts(conn, export_id)
        if counts["pending"] == counts["processing"] == 0:
            status = "completed_with_failures" if counts["failed"] else "completed"
            db.set_run_status(self.db_path, run_id, status)
            db.append_run_log(self.db_path, run_id,
                              f"[run] 全部完成 done={counts['done']} failed={counts['failed']}")
