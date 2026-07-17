from __future__ import annotations

import json
import re
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}


def _needs_reset(conn: sqlite3.Connection) -> bool:
    if _table_exists(conn, "run_trajectories"):
        return True
    if _table_exists(conn, "thinking_tasks"):
        return True
    if _table_exists(conn, "reflection_runs") and "task_group_id" in _columns(conn, "reflection_runs"):
        return True
    if _table_exists(conn, "task_attempts") and "trajectory_path" not in _columns(conn, "task_attempts"):
        return True
    if _table_exists(conn, "dataset_tasks"):
        cols = _columns(conn, "dataset_tasks")
        if "latest_status" not in cols:
            return True
    return False


def _drop_legacy(conn: sqlite3.Connection) -> None:
    for name in ("run_trajectories", "thinking_tasks", "reflection_runs",
                 "task_attempts", "dataset_trajectories", "dataset_tasks",
                 "run_trajectory_outputs", "run_logs"):
        conn.execute(f"DROP TABLE IF EXISTS {name}")


def init(path: Path) -> None:
    with connect(path) as conn:
        current = conn.execute("PRAGMA user_version").fetchone()[0]
        if current < SCHEMA_VERSION and _needs_reset(conn):
            _drop_legacy(conn)
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS reflection_runs (
          run_id TEXT PRIMARY KEY,
          source_key TEXT NOT NULL,
          source_export_id INTEGER NOT NULL,
          quality_record_id INTEGER NOT NULL,
          reflection_endpoint TEXT NOT NULL,
          reflection_api_key_id INTEGER NOT NULL,
          reflection_key_mask TEXT NOT NULL,
          reflection_model TEXT NOT NULL,
          method TEXT NOT NULL,
          worker_count INTEGER NOT NULL,
          max_retries INTEGER NOT NULL,
          export_root TEXT NOT NULL,
          obs_root TEXT NOT NULL DEFAULT '',
          status TEXT NOT NULL,
          config_snapshot TEXT NOT NULL,
          prompt_name TEXT NOT NULL,
          prompt_sha256 TEXT NOT NULL,
          prompt_loaded_at TEXT NOT NULL,
          launch_type TEXT NOT NULL DEFAULT 'start',
          parent_run_id TEXT,
          snapshot_total INTEGER NOT NULL DEFAULT 0,
          snapshot_pending INTEGER NOT NULL DEFAULT 0,
          snapshot_done INTEGER NOT NULL DEFAULT 0,
          snapshot_failed INTEGER NOT NULL DEFAULT 0,
          created_at REAL NOT NULL,
          started_at REAL,
          finished_at REAL,
          updated_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_reflection_runs_export
          ON reflection_runs(source_export_id, created_at DESC);

        CREATE TABLE IF NOT EXISTS dataset_trajectories (
          export_id INTEGER NOT NULL,
          trajectory_path TEXT NOT NULL,
          trajectory_id TEXT NOT NULL,
          session_id TEXT NOT NULL,
          source_root TEXT NOT NULL DEFAULT '',
          created_at REAL NOT NULL,
          updated_at REAL NOT NULL,
          PRIMARY KEY (export_id, trajectory_path)
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_dataset_trajectories_tid
          ON dataset_trajectories(trajectory_id);
        CREATE INDEX IF NOT EXISTS idx_dataset_trajectories_export_session
          ON dataset_trajectories(export_id, session_id);

        CREATE TABLE IF NOT EXISTS dataset_tasks (
          uuid TEXT PRIMARY KEY,
          export_id INTEGER NOT NULL,
          session_id TEXT NOT NULL,
          trajectory_id TEXT NOT NULL,
          trajectory_path TEXT NOT NULL,
          block_path TEXT NOT NULL,
          message_index INTEGER,
          original_thinking TEXT,
          signature TEXT NOT NULL,
          signature_len INTEGER NOT NULL,
          detail_path TEXT,
          latest_run_id TEXT,
          latest_status TEXT NOT NULL DEFAULT 'pending',
          latest_processed_text TEXT,
          latest_model TEXT,
          latest_response_id TEXT,
          latest_stop_reason TEXT,
          latest_usage_json TEXT,
          latest_sentence_count INTEGER,
          retry_count INTEGER NOT NULL DEFAULT 0,
          max_retries INTEGER NOT NULL,
          last_error TEXT,
          last_attempt_at REAL,
          created_at REAL NOT NULL,
          updated_at REAL NOT NULL,
          UNIQUE (export_id, session_id, trajectory_path, block_path)
        );
        CREATE INDEX IF NOT EXISTS idx_dataset_tasks_claim
          ON dataset_tasks(export_id, latest_status, updated_at);
        CREATE INDEX IF NOT EXISTS idx_dataset_tasks_trajectory
          ON dataset_tasks(export_id, trajectory_id);
        CREATE INDEX IF NOT EXISTS idx_dataset_tasks_latest_run
          ON dataset_tasks(latest_run_id);

        CREATE TABLE IF NOT EXISTS run_trajectory_outputs (
          run_id TEXT NOT NULL,
          export_id INTEGER NOT NULL,
          trajectory_id TEXT NOT NULL,
          output_path TEXT NOT NULL,
          exported_at REAL NOT NULL,
          PRIMARY KEY (run_id, trajectory_id)
        );
        CREATE INDEX IF NOT EXISTS idx_run_traj_outputs_run
          ON run_trajectory_outputs(run_id);

        CREATE TABLE IF NOT EXISTS task_attempts (
          attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
          run_id TEXT NOT NULL,
          task_uuid TEXT NOT NULL,
          export_id INTEGER NOT NULL,
          session_id TEXT NOT NULL,
          trajectory_path TEXT NOT NULL,
          block_path TEXT NOT NULL,
          attempt_no INTEGER NOT NULL,
          status TEXT NOT NULL,
          started_at REAL NOT NULL,
          finished_at REAL,
          error TEXT,
          response_id TEXT,
          usage_json TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_task_attempts_task
          ON task_attempts(task_uuid, attempt_no);
        CREATE INDEX IF NOT EXISTS idx_task_attempts_run
          ON task_attempts(run_id);
        CREATE INDEX IF NOT EXISTS idx_task_attempts_dataset
          ON task_attempts(export_id, session_id, trajectory_path, block_path);

        CREATE TABLE IF NOT EXISTS run_logs (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          run_id TEXT NOT NULL,
          level TEXT NOT NULL DEFAULT 'info',
          message TEXT NOT NULL,
          created_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_run_logs_run ON run_logs(run_id, id);
        """)
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")


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
    snapshot_total = values.get("snapshot_total", 0)
    snapshot_pending = values.get("snapshot_pending", 0)
    with connect(path) as conn:
        conn.execute("""INSERT INTO reflection_runs(
          run_id,source_key,source_export_id,quality_record_id,reflection_endpoint,
          reflection_api_key_id,reflection_key_mask,reflection_model,method,worker_count,max_retries,
          export_root,obs_root,status,config_snapshot,prompt_name,prompt_sha256,prompt_loaded_at,
          launch_type,parent_run_id,snapshot_total,snapshot_pending,
          created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,'draft',?,?,?,?,?,?,?,?,?,?)""",
          (run_id, values["source_key"], values["source_export_id"], values["quality_record_id"],
           values["reflection_endpoint"], values["reflection_api_key_id"], values["reflection_key_mask"],
           values["reflection_model"], values["method"], values["worker_count"], values["max_retries"],
           values["export_root"], values.get("obs_root", ""), json.dumps(values["snapshot"], ensure_ascii=False),
           values["prompt_name"], values["prompt_sha256"], values["prompt_loaded_at"],
           launch_type, parent_run_id, snapshot_total, snapshot_pending, now, now))
    return run_id


def dataset_counts(conn: sqlite3.Connection, export_id: int) -> dict[str, int]:
    counts = {"pending": 0, "processing": 0, "done": 0, "failed": 0}
    for row in conn.execute(
        "SELECT latest_status,COUNT(*) n FROM dataset_tasks WHERE export_id=? GROUP BY latest_status",
        (export_id,),
    ):
        counts[row["latest_status"]] = row["n"]
    return counts


def run_dict(conn: sqlite3.Connection, row: sqlite3.Row) -> dict:
    result = dict(row)
    result.pop("config_snapshot", None)
    counts = dataset_counts(conn, result["source_export_id"])
    result.update(counts)
    result["total_count"] = sum(counts.values())
    # UI backward-compat: templates fall back to `task_group_id || run_id`;
    # aliasing to run_id keeps the fallback pointing at a valid handle.
    result["task_group_id"] = result["run_id"]
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
    now = time.time()
    with connect(path) as conn:
        conn.execute(
            "UPDATE dataset_tasks SET latest_status='pending',updated_at=? WHERE latest_status='processing'",
            (now,),
        )
        conn.execute("UPDATE reflection_runs SET status='paused' WHERE status IN ('running','queued')")


def get_task(path: Path, task_uuid: str) -> dict | None:
    with connect(path) as conn:
        row = conn.execute("SELECT * FROM dataset_tasks WHERE uuid=?", (task_uuid,)).fetchone()
        return dict(row) if row else None


def list_attempts(path: Path, task_uuid: str) -> list[dict]:
    with connect(path) as conn:
        rows = conn.execute(
            "SELECT * FROM task_attempts WHERE task_uuid=? ORDER BY attempt_no",
            (task_uuid,),
        ).fetchall()
        return [dict(r) for r in rows]


def retry_all_failed(path: Path, export_id: int) -> int:
    now = time.time()
    with connect(path) as conn:
        cursor = conn.execute(
            "UPDATE dataset_tasks SET latest_status='pending',last_error=NULL,retry_count=0,updated_at=? "
            "WHERE export_id=? AND latest_status='failed'",
            (now, export_id),
        )
        return cursor.rowcount


def reset_all_done(path: Path, export_id: int) -> int:
    now = time.time()
    with connect(path) as conn:
        cursor = conn.execute(
            "UPDATE dataset_tasks SET latest_status='pending',latest_processed_text=NULL,"
            "latest_response_id=NULL,latest_stop_reason=NULL,latest_usage_json=NULL,"
            "latest_sentence_count=NULL,last_error=NULL,retry_count=0,updated_at=? "
            "WHERE export_id=? AND latest_status='done'",
            (now, export_id),
        )
        return cursor.rowcount


def resolve_export_id(path: Path, key: str | int | None) -> int | None:
    """Accept a run_id string OR a numeric string/int and return the dataset's export_id."""
    if key is None:
        return None
    if isinstance(key, int):
        return key
    s = str(key).strip()
    if not s:
        return None
    if s.isdigit():
        return int(s)
    with connect(path) as conn:
        row = conn.execute(
            "SELECT source_export_id FROM reflection_runs WHERE run_id=?", (s,)
        ).fetchone()
        return int(row["source_export_id"]) if row else None


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
        conn.execute("DELETE FROM run_trajectory_outputs WHERE run_id=?", (run_id,))
        conn.execute("DELETE FROM run_logs WHERE run_id=?", (run_id,))
        conn.execute(
            "UPDATE dataset_tasks SET latest_run_id=NULL WHERE latest_run_id=?",
            (run_id,),
        )
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


def count_trajectories(path: Path, export_id: int) -> int:
    with connect(path) as conn:
        row = conn.execute(
            "SELECT COUNT(*) n FROM dataset_trajectories WHERE export_id=?", (export_id,)
        ).fetchone()
        return row["n"] if row else 0


def list_trajectories(path: Path, export_id: int, run_id: str | None = None,
                      offset: int = 0, limit: int = 0) -> list[dict]:
    with connect(path) as conn:
        if run_id:
            sql = ("SELECT t.export_id,t.trajectory_id,t.session_id,t.trajectory_path,"
                   "o.output_path AS output_path,o.exported_at AS exported_at "
                   "FROM dataset_trajectories t "
                   "LEFT JOIN run_trajectory_outputs o "
                   "  ON o.trajectory_id=t.trajectory_id AND o.run_id=? "
                   "WHERE t.export_id=? ORDER BY t.session_id,t.trajectory_path")
            params: list = [run_id, export_id]
        else:
            sql = ("SELECT export_id,trajectory_id,session_id,trajectory_path,"
                   "NULL AS output_path,NULL AS exported_at "
                   "FROM dataset_trajectories WHERE export_id=? "
                   "ORDER BY session_id,trajectory_path")
            params = [export_id]
        if limit > 0:
            sql += " LIMIT ? OFFSET ?"
            params += [limit, offset]
        rows = conn.execute(sql, params).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            counts = {"pending": 0, "processing": 0, "done": 0, "failed": 0}
            for cr in conn.execute(
                "SELECT latest_status,COUNT(*) n FROM dataset_tasks "
                "WHERE export_id=? AND trajectory_id=? GROUP BY latest_status",
                (export_id, row["trajectory_id"]),
            ):
                counts[cr["latest_status"]] = cr["n"]
            item.update(counts)
            item["total_tasks"] = sum(counts.values())
            # UI expects `run_id` on trajectory rows; expose the querying run
            item["run_id"] = run_id or ""
            result.append(item)
        return result


def record_run_output(path: Path, run_id: str, export_id: int,
                      trajectory_id: str, output_path: str) -> None:
    with connect(path) as conn:
        conn.execute(
            "INSERT INTO run_trajectory_outputs(run_id,export_id,trajectory_id,output_path,exported_at) "
            "VALUES(?,?,?,?,?) "
            "ON CONFLICT(run_id,trajectory_id) DO UPDATE SET "
            "output_path=excluded.output_path,exported_at=excluded.exported_at",
            (run_id, export_id, trajectory_id, output_path, time.time()),
        )


def list_run_outputs(path: Path, run_id: str) -> list[dict]:
    with connect(path) as conn:
        rows = conn.execute(
            "SELECT * FROM run_trajectory_outputs WHERE run_id=? ORDER BY trajectory_id",
            (run_id,),
        ).fetchall()
        return [dict(r) for r in rows]


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
