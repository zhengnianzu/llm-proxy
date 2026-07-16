from __future__ import annotations

import json
import re
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def init(path: Path) -> None:
    with connect(path) as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS reflection_runs (
          run_id TEXT PRIMARY KEY, source_key TEXT NOT NULL, source_export_id INTEGER NOT NULL,
          quality_record_id INTEGER NOT NULL, reflection_endpoint TEXT NOT NULL,
          reflection_api_key_id INTEGER NOT NULL, reflection_key_mask TEXT NOT NULL,
          reflection_model TEXT NOT NULL, method TEXT NOT NULL, worker_count INTEGER NOT NULL,
          max_retries INTEGER NOT NULL, export_root TEXT NOT NULL, obs_root TEXT NOT NULL DEFAULT '',
          status TEXT NOT NULL, config_snapshot TEXT NOT NULL, prompt_name TEXT NOT NULL,
          prompt_sha256 TEXT NOT NULL, prompt_loaded_at TEXT NOT NULL,
          created_at REAL NOT NULL, started_at REAL, finished_at REAL, updated_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS run_trajectories (
          id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, trajectory_id TEXT NOT NULL,
          session_id TEXT NOT NULL, trajectory_path TEXT NOT NULL, raw_json TEXT NOT NULL,
          output_path TEXT, exported_at REAL, UNIQUE(run_id, trajectory_path)
        );
        CREATE TABLE IF NOT EXISTS thinking_tasks (
          uuid TEXT PRIMARY KEY, run_id TEXT NOT NULL, trajectory_id TEXT NOT NULL,
          session_id TEXT NOT NULL, trajectory_path TEXT NOT NULL, block_path TEXT NOT NULL,
          message_index INTEGER, original_thinking TEXT, signature TEXT NOT NULL,
          signature_len INTEGER NOT NULL, processed_text TEXT, status TEXT NOT NULL,
          retry_count INTEGER NOT NULL DEFAULT 0, max_retries INTEGER NOT NULL,
          last_error TEXT, sentence_count INTEGER, tool_input_json TEXT, model TEXT,
          response_id TEXT, stop_reason TEXT, usage_json TEXT, last_attempt_at REAL,
          created_at REAL NOT NULL, updated_at REAL NOT NULL,
          UNIQUE(run_id, trajectory_id, block_path)
        );
        CREATE TABLE IF NOT EXISTS task_attempts (
          attempt_id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, task_uuid TEXT NOT NULL,
          attempt_no INTEGER NOT NULL, status TEXT NOT NULL, started_at REAL NOT NULL,
          finished_at REAL, error TEXT, response_id TEXT, usage_json TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_reflection_tasks_claim ON thinking_tasks(run_id,status,updated_at);
        CREATE TABLE IF NOT EXISTS run_logs (
          id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL,
          level TEXT NOT NULL DEFAULT 'info', message TEXT NOT NULL,
          created_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_run_logs_run ON run_logs(run_id,id);
        """)
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(thinking_tasks)")}
        for definition in (
            "sentence_count INTEGER",
            "tool_input_json TEXT",
            "model TEXT",
            "stop_reason TEXT",
            "detail_path TEXT",
        ):
            if definition.split()[0] not in columns:
                conn.execute(f"ALTER TABLE thinking_tasks ADD COLUMN {definition}")
        run_columns = {row["name"] for row in conn.execute("PRAGMA table_info(reflection_runs)")}
        for definition in (
            "launch_type TEXT NOT NULL DEFAULT 'start'",
            "parent_run_id TEXT",
            "task_group_id TEXT",
            "snapshot_total INTEGER DEFAULT 0",
            "snapshot_pending INTEGER DEFAULT 0",
            "snapshot_done INTEGER DEFAULT 0",
            "snapshot_failed INTEGER DEFAULT 0",
        ):
            if definition.split()[0] not in run_columns:
                conn.execute(f"ALTER TABLE reflection_runs ADD COLUMN {definition}")


def task_detail_dir(source_key: str, export_created_at: str) -> Path:
    ts = re.sub(r"[^0-9]", "", export_created_at)[:14]
    if len(ts) < 14:
        ts = ts.ljust(14, "0")
    formatted = ts[:8] + "_" + ts[8:]
    return Path("logs_thinking") / source_key / formatted


def create_run(path: Path, values: dict[str, Any]) -> str:
    run_id = "run_" + uuid.uuid4().hex[:12]
    now = time.time()
    launch_type = values.get("launch_type", "start")
    parent_run_id = values.get("parent_run_id")
    task_group_id = values.get("task_group_id")
    snapshot_total = values.get("snapshot_total", 0)
    snapshot_pending = values.get("snapshot_pending", 0)
    with connect(path) as conn:
        conn.execute("""INSERT INTO reflection_runs(
          run_id,source_key,source_export_id,quality_record_id,reflection_endpoint,
          reflection_api_key_id,reflection_key_mask,reflection_model,method,worker_count,max_retries,
          export_root,obs_root,status,config_snapshot,prompt_name,prompt_sha256,prompt_loaded_at,
          launch_type,parent_run_id,task_group_id,snapshot_total,snapshot_pending,
          created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,'draft',?,?,?,?,?,?,?,?,?,?,?)""",
          (run_id, values["source_key"], values["source_export_id"], values["quality_record_id"],
           values["reflection_endpoint"], values["reflection_api_key_id"], values["reflection_key_mask"],
           values["reflection_model"], values["method"], values["worker_count"], values["max_retries"],
           values["export_root"], values.get("obs_root", ""), json.dumps(values["snapshot"], ensure_ascii=False),
           values["prompt_name"], values["prompt_sha256"], values["prompt_loaded_at"],
           launch_type, parent_run_id, task_group_id, snapshot_total, snapshot_pending, now, now))
    return run_id


def run_counts(conn: sqlite3.Connection, run_id: str) -> dict[str, int]:
    counts = {"pending": 0, "processing": 0, "done": 0, "failed": 0}
    for row in conn.execute("SELECT status,COUNT(*) n FROM thinking_tasks WHERE run_id=? GROUP BY status", (run_id,)):
        counts[row["status"]] = row["n"]
    return counts


def run_dict(conn: sqlite3.Connection, row: sqlite3.Row) -> dict:
    result = dict(row)
    result.pop("config_snapshot", None)
    tg = result.get("task_group_id") or result["run_id"]
    result.update(run_counts(conn, tg))
    result["total_count"] = sum(result[k] for k in ("pending", "processing", "done", "failed"))
    return result


def list_runs(path: Path, source_export_id: int | None = None) -> list[dict]:
    with connect(path) as conn:
        if source_export_id is None:
            rows = conn.execute("SELECT * FROM reflection_runs ORDER BY created_at DESC").fetchall()
        else:
            rows = conn.execute("SELECT * FROM reflection_runs WHERE source_export_id=? ORDER BY created_at DESC", (source_export_id,)).fetchall()
        return [run_dict(conn, row) for row in rows]


def get_run(path: Path, run_id: str) -> dict | None:
    with connect(path) as conn:
        row = conn.execute("SELECT * FROM reflection_runs WHERE run_id=?", (run_id,)).fetchone()
        return run_dict(conn, row) if row else None


def get_run_snapshot(path: Path, run_id: str) -> dict[str, Any]:
    with connect(path) as conn:
        row = conn.execute(
            "SELECT config_snapshot FROM reflection_runs WHERE run_id=?", (run_id,)
        ).fetchone()
    if not row:
        raise KeyError(run_id)
    value = json.loads(row["config_snapshot"])
    if not isinstance(value, dict):
        raise ValueError("invalid Run config snapshot")
    return value


def set_run_status(path: Path, run_id: str, status: str) -> None:
    now = time.time()
    started = ", started_at=COALESCE(started_at,?)" if status == "running" else ""
    finished = ", finished_at=?" if status in {"completed", "completed_with_failures", "cancelled", "failed"} else ""
    params: list[Any] = [status, now]
    if started: params.append(now)
    if finished: params.append(now)
    params.append(run_id)
    with connect(path) as conn:
        conn.execute(f"UPDATE reflection_runs SET status=?,updated_at=?{started}{finished} WHERE run_id=?", params)


def reset_processing(path: Path) -> None:
    with connect(path) as conn:
        conn.execute("UPDATE thinking_tasks SET status='pending' WHERE status='processing'")
        conn.execute("UPDATE reflection_runs SET status='paused' WHERE status IN ('running','queued')")


def get_task(path: Path, task_uuid: str) -> dict | None:
    with connect(path) as conn:
        row = conn.execute("SELECT * FROM thinking_tasks WHERE uuid=?", (task_uuid,)).fetchone()
        return dict(row) if row else None


def list_attempts(path: Path, task_uuid: str) -> list[dict]:
    with connect(path) as conn:
        rows = conn.execute(
            "SELECT * FROM task_attempts WHERE task_uuid=? ORDER BY attempt_no",
            (task_uuid,),
        ).fetchall()
        return [dict(r) for r in rows]


def retry_all_failed(path: Path, task_group_id: str) -> int:
    now = time.time()
    with connect(path) as conn:
        cursor = conn.execute(
            "UPDATE thinking_tasks SET status='pending',last_error=NULL,retry_count=0,updated_at=? WHERE run_id=? AND status='failed'",
            (now, task_group_id),
        )
        return cursor.rowcount


def get_task_group_id(path: Path, source_export_id: int) -> str | None:
    with connect(path) as conn:
        row = conn.execute(
            "SELECT run_id FROM reflection_runs WHERE source_export_id=? AND task_group_id IS NULL ORDER BY created_at LIMIT 1",
            (source_export_id,),
        ).fetchone()
        if row:
            return row["run_id"]
        row = conn.execute(
            "SELECT task_group_id FROM reflection_runs WHERE source_export_id=? AND task_group_id IS NOT NULL ORDER BY created_at LIMIT 1",
            (source_export_id,),
        ).fetchone()
        return row["task_group_id"] if row else None


def increment_run_stat(path: Path, run_id: str, field: str) -> None:
    if field not in ("snapshot_done", "snapshot_failed"):
        return
    with connect(path) as conn:
        conn.execute(
            f"UPDATE reflection_runs SET {field}={field}+1,updated_at=? WHERE run_id=?",
            (time.time(), run_id),
        )


def delete_run(path: Path, run_id: str) -> None:
    with connect(path) as conn:
        conn.execute("DELETE FROM task_attempts WHERE run_id=?", (run_id,))
        conn.execute("DELETE FROM thinking_tasks WHERE run_id=?", (run_id,))
        conn.execute("DELETE FROM run_trajectories WHERE run_id=?", (run_id,))
        conn.execute("DELETE FROM reflection_runs WHERE run_id=?", (run_id,))


def update_run_config(path: Path, run_id: str, updates: dict[str, Any]) -> None:
    allowed = {
        "reflection_endpoint", "reflection_api_key_id", "reflection_key_mask", "reflection_model",
        "method", "worker_count", "max_retries", "config_snapshot",
    }
    sets = []
    params: list[Any] = []
    for k, v in updates.items():
        if k in allowed:
            sets.append(f"{k}=?")
            params.append(v)
    if not sets:
        return
    sets.append("updated_at=?")
    params.append(time.time())
    params.append(run_id)
    with connect(path) as conn:
        conn.execute(
            f"UPDATE reflection_runs SET {','.join(sets)} WHERE run_id=?",
            params,
        )


def count_trajectories(path: Path, run_id: str) -> int:
    with connect(path) as conn:
        row = conn.execute(
            "SELECT COUNT(*) n FROM run_trajectories WHERE run_id=?", (run_id,)
        ).fetchone()
        return row["n"] if row else 0


def list_trajectories(path: Path, run_id: str, offset: int = 0, limit: int = 0) -> list[dict]:
    with connect(path) as conn:
        sql = "SELECT id,run_id,trajectory_id,session_id,trajectory_path,output_path,exported_at FROM run_trajectories WHERE run_id=? ORDER BY session_id,trajectory_path"
        params: list = [run_id]
        if limit > 0:
            sql += " LIMIT ? OFFSET ?"
            params += [limit, offset]
        rows = conn.execute(sql, params).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            counts = {"pending": 0, "processing": 0, "done": 0, "failed": 0}
            for cr in conn.execute(
                "SELECT status,COUNT(*) n FROM thinking_tasks WHERE run_id=? AND trajectory_id=? GROUP BY status",
                (run_id, row["trajectory_id"]),
            ):
                counts[cr["status"]] = cr["n"]
            item.update(counts)
            item["total_tasks"] = sum(counts.values())
            result.append(item)
        return result


def append_run_log(path: Path, run_id: str, message: str, level: str = "info") -> None:
    with connect(path) as conn:
        conn.execute(
            "INSERT INTO run_logs(run_id,level,message,created_at) VALUES(?,?,?,?)",
            (run_id, level, message, time.time()),
        )


def get_run_logs(path: Path, run_id: str, since_id: int = 0, limit: int = 200) -> list[dict]:
    with connect(path) as conn:
        rows = conn.execute(
            "SELECT id,run_id,level,message,created_at FROM run_logs WHERE run_id=? AND id>? ORDER BY id LIMIT ?",
            (run_id, since_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]
