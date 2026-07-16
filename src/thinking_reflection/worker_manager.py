from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from utils.key_store import get_key_full

from . import db
from .consumer import reflect
from .prompt_loader import load_prompt


class WorkerManager:
    def __init__(self, db_path: Path, prompt_dir: Path):
        self.db_path, self.prompt_dir = db_path, prompt_dir
        self._stops: dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    def start(self, run_id: str, task_group_id: str) -> None:
        run = db.get_run(self.db_path, run_id)
        if not run or run["status"] not in {"draft", "paused", "queued"}:
            raise ValueError("Run 当前状态不能启动")
        key = get_key_full(run["reflection_api_key_id"])
        if not key or key["status"] != "active": raise ValueError("Reflection Key 已禁用或不存在")
        prompt = load_prompt(self.prompt_dir, run["method"])
        snapshot = db.get_run_snapshot(self.db_path, run_id)
        stop = threading.Event()
        with self._lock:
            if run_id in self._stops: raise ValueError("Run 已在运行")
            self._stops[run_id] = stop
        db.set_run_status(self.db_path, run_id, "running")
        db.append_run_log(self.db_path, run_id, f"[run] 启动 {run['worker_count']} 个 worker, model={run['reflection_model']}, method={run['method']}, task_group={task_group_id}")
        for index in range(run["worker_count"]):
            threading.Thread(target=self._loop, args=(run, snapshot, key["key"], prompt, stop, index, task_group_id),
                             daemon=True, name=f"reflection-{run_id}-{index}").start()

    def stop(self, run_id: str, *, cancel: bool = False) -> None:
        event = self._stops.get(run_id)
        if event: event.set()
        db.set_run_status(self.db_path, run_id, "cancelled" if cancel else "paused")

    def _loop(self, run: dict, snapshot: dict, api_key: str, prompt, stop: threading.Event, worker_index: int, task_group_id: str) -> None:
        run_id = run["run_id"]
        tag = f"[worker-{worker_index}]"
        try:
            while not stop.is_set():
                with db.connect(self.db_path) as conn:
                    conn.execute("BEGIN IMMEDIATE")
                    task = conn.execute("SELECT * FROM thinking_tasks WHERE run_id=? AND status='pending' ORDER BY updated_at LIMIT 1", (task_group_id,)).fetchone()
                    if not task:
                        conn.commit(); break
                    now = time.time()
                    conn.execute("UPDATE thinking_tasks SET status='processing',last_attempt_at=?,updated_at=? WHERE uuid=?", (now, now, task["uuid"]))
                    attempt_no = task["retry_count"] + 1
                    cursor = conn.execute("INSERT INTO task_attempts(run_id,task_uuid,attempt_no,status,started_at) VALUES(?,?,?,'running',?)", (run_id, task["uuid"], attempt_no, now))
                    attempt_id = cursor.lastrowid
                detail_path = task["detail_path"] if task["detail_path"] else None
                if detail_path and Path(detail_path).is_file():
                    detail = json.loads(Path(detail_path).read_text(encoding="utf-8"))
                    task_signature = detail.get("signature", task["signature"])
                    task_thinking = detail.get("original_thinking", task["original_thinking"])
                else:
                    task_signature = task["signature"]
                    task_thinking = task["original_thinking"]
                db.append_run_log(self.db_path, run_id, f"{tag} 认领 {task['session_id']}/{task['block_path']}")
                try:
                    result = reflect(endpoint=run["reflection_endpoint"], api_key=api_key,
                                     model=run["reflection_model"], instruction=prompt.instruction,
                                     tool=prompt.tool, unrelated_thinking=prompt.unrelated_thinking,
                                     signature=task_signature, thinking=task_thinking, method=run["method"],
                                     stream=bool(snapshot.get("stream", False)),
                                     max_tokens=int(snapshot.get("max_tokens", 16384)))
                    with db.connect(self.db_path) as conn:
                        now = time.time()
                        conn.execute("""UPDATE thinking_tasks SET status='done',sentence_count=?,
                          model=?,response_id=?,stop_reason=?,usage_json=?,last_error=NULL,updated_at=?
                          WHERE uuid=?""", (result.sentence_count,
                          result.model, result.response_id,
                          result.stop_reason, json.dumps(result.usage) if result.usage else None, now, task["uuid"]))
                        conn.execute("UPDATE task_attempts SET status='done',finished_at=?,response_id=?,usage_json=? WHERE attempt_id=?", (now, result.response_id, json.dumps(result.usage) if result.usage else None, attempt_id))
                    db.increment_run_stat(self.db_path, run_id, "snapshot_done")
                    if detail_path:
                        try:
                            detail = json.loads(Path(detail_path).read_text(encoding="utf-8"))
                        except (OSError, json.JSONDecodeError):
                            detail = {}
                        detail["processed_text"] = result.text
                        detail["tool_input"] = result.tool_input
                        detail["raw_response"] = result.raw_response
                        Path(detail_path).write_text(json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8")
                    usage_str = ""
                    if result.usage:
                        usage_str = f" in={result.usage.get('input_tokens',0)} out={result.usage.get('output_tokens',0)}"
                    db.append_run_log(self.db_path, run_id, f"{tag} ✓ {task['session_id']}/{task['block_path']} ({result.sentence_count} 句{usage_str})")
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"[:4000]
                    retry = attempt_no
                    status = "failed" if retry >= task["max_retries"] else "pending"
                    with db.connect(self.db_path) as conn:
                        now = time.time()
                        conn.execute("UPDATE thinking_tasks SET status=?,retry_count=?,last_error=?,updated_at=? WHERE uuid=?", (status, retry, error, now, task["uuid"]))
                        conn.execute("UPDATE task_attempts SET status='failed',finished_at=?,error=? WHERE attempt_id=?", (now, error, attempt_id))
                    if status == "failed":
                        db.increment_run_stat(self.db_path, run_id, "snapshot_failed")
                    db.append_run_log(self.db_path, run_id, f"{tag} ✗ {task['session_id']}/{task['block_path']} retry={retry}/{task['max_retries']} {error[:200]}", level="error")
            self._finish_if_empty(run_id, task_group_id)
        finally:
            with self._lock:
                self._stops.pop(run_id, None)

    def _finish_if_empty(self, run_id: str, task_group_id: str) -> None:
        with db.connect(self.db_path) as conn:
            counts = db.run_counts(conn, task_group_id)
        if counts["pending"] == counts["processing"] == 0:
            status = "completed_with_failures" if counts["failed"] else "completed"
            db.set_run_status(self.db_path, run_id, status)
            db.append_run_log(self.db_path, run_id, f"[run] 全部完成 done={counts['done']} failed={counts['failed']}")
