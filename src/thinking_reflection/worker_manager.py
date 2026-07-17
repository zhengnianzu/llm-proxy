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
        self._active_by_export: dict[int, str] = {}
        self._lock = threading.Lock()

    def start(self, run_id: str) -> None:
        run = db.get_run(self.db_path, run_id)
        if not run or run["status"] not in {"draft", "paused", "queued"}:
            raise ValueError("Run 当前状态不能启动")
        export_id = int(run["source_export_id"])
        key = get_key_full(run["reflection_api_key_id"])
        if not key or key["status"] != "active":
            raise ValueError("Reflection Key 已禁用或不存在")
        prompt = load_prompt(self.prompt_dir, run["method"])
        snapshot = db.get_run_snapshot(self.db_path, run_id)
        stop = threading.Event()
        with self._lock:
            active_rid = self._active_by_export.get(export_id)
            if active_rid and active_rid != run_id:
                raise ValueError(f"数据集 {export_id} 已有活跃 Run {active_rid}")
            if run_id in self._stops:
                raise ValueError("Run 已在运行")
            self._stops[run_id] = stop
            self._active_by_export[export_id] = run_id
        db.set_run_status(self.db_path, run_id, "running")
        db.append_run_log(self.db_path, run_id,
                          f"[run] 启动 {run['worker_count']} 个 worker, "
                          f"model={run['reflection_model']}, method={run['method']}, "
                          f"export_id={export_id}")
        for index in range(run["worker_count"]):
            threading.Thread(
                target=self._loop,
                args=(run, snapshot, key["key"], prompt, stop, index, export_id),
                daemon=True, name=f"reflection-{run_id}-{index}",
            ).start()

    def stop(self, run_id: str, *, cancel: bool = False) -> None:
        event = self._stops.get(run_id)
        if event:
            event.set()
        db.set_run_status(self.db_path, run_id, "cancelled" if cancel else "paused")
        with self._lock:
            self._stops.pop(run_id, None)
            for eid, rid in list(self._active_by_export.items()):
                if rid == run_id:
                    del self._active_by_export[eid]

    def _loop(self, run: dict, snapshot: dict, api_key: str, prompt,
              stop: threading.Event, worker_index: int, export_id: int) -> None:
        run_id = run["run_id"]
        tag = f"[worker-{worker_index}]"
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
            with self._lock:
                self._stops.pop(run_id, None)
                for eid, rid in list(self._active_by_export.items()):
                    if rid == run_id:
                        del self._active_by_export[eid]

    def _finish_if_empty(self, run_id: str, export_id: int) -> None:
        with db.connect(self.db_path) as conn:
            counts = db.dataset_counts(conn, export_id)
        if counts["pending"] == counts["processing"] == 0:
            status = "completed_with_failures" if counts["failed"] else "completed"
            db.set_run_status(self.db_path, run_id, status)
            db.append_run_log(self.db_path, run_id,
                              f"[run] 全部完成 done={counts['done']} failed={counts['failed']}")
