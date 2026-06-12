"""
utils/export_store.py — Session 导出记录持久化

SQLite 后端，记录每次导出操作的状态和结果。
数据库路径：{service_log_dir}/export_session_record.db
"""

import os
import sqlite3
import threading
from typing import Optional

_lock = threading.Lock()
_conn: Optional[sqlite3.Connection] = None
_db_path: str = ""


def init_db(db_dir: str):
    global _conn, _db_path
    _db_path = os.path.join(db_dir, "export_session_record.db")
    os.makedirs(db_dir, exist_ok=True)
    _conn = sqlite3.connect(_db_path, check_same_thread=False)
    _conn.execute("PRAGMA journal_mode=WAL")
    _conn.execute("PRAGMA busy_timeout=5000")
    _conn.row_factory = sqlite3.Row
    with _lock:
        _conn.execute("""
            CREATE TABLE IF NOT EXISTS export_records (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                api_key         TEXT NOT NULL DEFAULT '',
                key_slot        TEXT NOT NULL DEFAULT 'all',
                mtime_dirs      TEXT NOT NULL DEFAULT '[]',
                status          TEXT NOT NULL DEFAULT 'pending',
                error_message   TEXT NOT NULL DEFAULT '',
                total_sessions  INTEGER NOT NULL DEFAULT 0,
                files_uploaded  INTEGER NOT NULL DEFAULT 0,
                files_skipped   INTEGER NOT NULL DEFAULT 0,
                obs_dst         TEXT NOT NULL DEFAULT '',
                local_copy_dir  TEXT NOT NULL DEFAULT '',
                progress_log    TEXT NOT NULL DEFAULT '[]',
                created_at      TEXT NOT NULL DEFAULT (datetime('now','localtime')),
                started_at      TEXT,
                finished_at     TEXT
            )
        """)
        _conn.execute("CREATE INDEX IF NOT EXISTS idx_export_key_slot ON export_records(key_slot)")
        _conn.execute("CREATE INDEX IF NOT EXISTS idx_export_status ON export_records(status)")
        try:
            _conn.execute("ALTER TABLE export_records ADD COLUMN progress_log TEXT NOT NULL DEFAULT '[]'")
        except sqlite3.OperationalError:
            pass
        _conn.commit()


def _get_conn() -> sqlite3.Connection:
    if _conn is None:
        raise RuntimeError("export_store not initialized, call init_db() first")
    return _conn


def create_record(
    api_key: str,
    key_slot: str,
    mtime_dirs: str,
    obs_dst: str = "",
    local_copy_dir: str = "",
) -> int:
    conn = _get_conn()
    with _lock:
        cur = conn.execute(
            """INSERT INTO export_records (api_key, key_slot, mtime_dirs, obs_dst, local_copy_dir)
               VALUES (?, ?, ?, ?, ?)""",
            (api_key, key_slot, mtime_dirs, obs_dst, local_copy_dir),
        )
        conn.commit()
    return cur.lastrowid


def update_status(record_id: int, status: str, **kwargs):
    conn = _get_conn()
    fields = ["status = ?"]
    values = [status]
    if status == "running":
        fields.append("started_at = datetime('now','localtime')")
    elif status in ("success", "failed"):
        fields.append("finished_at = datetime('now','localtime')")
    for k, v in kwargs.items():
        if k in ("error_message", "total_sessions", "files_uploaded", "files_skipped", "obs_dst", "local_copy_dir"):
            fields.append(f"{k} = ?")
            values.append(v)
    values.append(record_id)
    with _lock:
        conn.execute(f"UPDATE export_records SET {', '.join(fields)} WHERE id = ?", values)
        conn.commit()


def get_record(record_id: int) -> Optional[dict]:
    conn = _get_conn()
    row = conn.execute("SELECT * FROM export_records WHERE id = ?", (record_id,)).fetchone()
    return dict(row) if row else None


def list_records(limit: int = 100) -> list:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM export_records ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
    return [dict(r) for r in rows]


def list_records_by_key(key_slot: str, limit: int = 50) -> list:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM export_records WHERE key_slot = ? ORDER BY created_at DESC LIMIT ?",
        (key_slot, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def mark_interrupted():
    """启动时将遗留的 running 记录标记为 failed。"""
    conn = _get_conn()
    with _lock:
        conn.execute(
            "UPDATE export_records SET status = 'failed', error_message = '服务重启中断', "
            "finished_at = datetime('now','localtime') WHERE status = 'running'"
        )
        conn.commit()


def append_log(record_id: int, message: str):
    """追加一条进度日志到 progress_log JSON 数组。"""
    import json
    from datetime import datetime as _dt

    conn = _get_conn()
    entry = {"ts": _dt.now().strftime("%H:%M:%S"), "msg": message}
    with _lock:
        row = conn.execute("SELECT progress_log FROM export_records WHERE id = ?", (record_id,)).fetchone()
        if row:
            try:
                logs = json.loads(row[0] or "[]")
            except (json.JSONDecodeError, TypeError):
                logs = []
            logs.append(entry)
            conn.execute("UPDATE export_records SET progress_log = ? WHERE id = ?",
                         (json.dumps(logs, ensure_ascii=False), record_id))
            conn.commit()
