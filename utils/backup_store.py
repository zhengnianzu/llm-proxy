"""
utils/backup_store.py — 备份同步状态持久化

SQLite 后端，记录每个 mtime 目录的 OBS 同步状态和同步日志。
数据库路径：{service_log_dir}/backup_sync.db
"""

import os
import sqlite3
import threading
from datetime import datetime
from typing import Dict, List, Optional

_lock = threading.Lock()
_conn: Optional[sqlite3.Connection] = None
_db_path: str = ""


def init_db(db_dir: str):
    global _conn, _db_path
    _db_path = os.path.join(db_dir, "backup_sync.db")
    os.makedirs(db_dir, exist_ok=True)
    _conn = sqlite3.connect(_db_path, check_same_thread=False)
    _conn.execute("PRAGMA journal_mode=WAL")
    _conn.execute("PRAGMA busy_timeout=5000")
    _conn.row_factory = sqlite3.Row
    with _lock:
        _conn.execute("""
            CREATE TABLE IF NOT EXISTS backup_dirs (
                dir_path    TEXT PRIMARY KEY,
                env_name    TEXT NOT NULL DEFAULT '',
                mtime_tag   TEXT NOT NULL DEFAULT '',
                file_count  INTEGER NOT NULL DEFAULT 0,
                synced      INTEGER NOT NULL DEFAULT 0,
                status      TEXT NOT NULL DEFAULT 'pending',
                sync_time   TEXT,
                obs_path    TEXT NOT NULL DEFAULT '',
                error_msg   TEXT NOT NULL DEFAULT '',
                created_at  TEXT NOT NULL DEFAULT (datetime('now','localtime')),
                updated_at  TEXT NOT NULL DEFAULT (datetime('now','localtime'))
            )
        """)
        _conn.execute("""
            CREATE TABLE IF NOT EXISTS backup_logs (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                dir_path    TEXT NOT NULL,
                level       TEXT NOT NULL DEFAULT 'info',
                message     TEXT NOT NULL,
                created_at  TEXT NOT NULL DEFAULT (datetime('now','localtime'))
            )
        """)
        _conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_backup_logs_dir ON backup_logs(dir_path)"
        )
        try:
            _conn.execute("ALTER TABLE backup_dirs ADD COLUMN sync_pid INTEGER DEFAULT NULL")
        except sqlite3.OperationalError:
            pass
        try:
            _conn.execute("ALTER TABLE backup_dirs ADD COLUMN port TEXT NOT NULL DEFAULT ''")
        except sqlite3.OperationalError:
            pass
        try:
            _conn.execute("ALTER TABLE backup_dirs ADD COLUMN source TEXT NOT NULL DEFAULT 'local'")
        except sqlite3.OperationalError:
            pass
        _conn.commit()


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def upsert_dir(dir_path: str, env_name: str, mtime_tag: str, file_count: int = 0, port: str = "", source: str = "local"):
    with _lock:
        existing = _conn.execute(
            "SELECT source, port FROM backup_dirs WHERE dir_path = ?", (dir_path,)
        ).fetchone()
        if existing:
            old_source = existing["source"] or "local"
            old_port = existing["port"] or ""
            if source and old_source and source != old_source:
                merged_source = "both"
            else:
                merged_source = source or old_source
            merged_port = old_port or port
            _conn.execute("""
                UPDATE backup_dirs SET
                    file_count = ?, source = ?, port = ?, updated_at = ?
                WHERE dir_path = ?
            """, (file_count, merged_source, merged_port, _now(), dir_path))
        else:
            _conn.execute("""
                INSERT INTO backup_dirs (dir_path, env_name, mtime_tag, file_count, port, source, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (dir_path, env_name, mtime_tag, file_count, port, source, _now(), _now()))
        _conn.commit()


def update_sync_status(
    dir_path: str,
    status: str,
    obs_path: str = "",
    error_msg: str = "",
):
    fields = ["status = ?", "updated_at = ?"]
    params: list = [status, _now()]
    if obs_path:
        fields.append("obs_path = ?")
        params.append(obs_path)
    if error_msg:
        fields.append("error_msg = ?")
        params.append(error_msg)
    if status == "done":
        fields.append("synced = 1")
        fields.append("sync_time = ?")
        params.append(_now())
    if status in ("done", "error", "pending"):
        fields.append("sync_pid = NULL")
    params.append(dir_path)
    with _lock:
        _conn.execute(
            f"UPDATE backup_dirs SET {', '.join(fields)} WHERE dir_path = ?",
            params,
        )
        _conn.commit()


def list_dirs(env_name: Optional[str] = None) -> List[dict]:
    with _lock:
        if env_name:
            rows = _conn.execute(
                "SELECT * FROM backup_dirs WHERE env_name = ? ORDER BY env_name, mtime_tag",
                (env_name,),
            ).fetchall()
        else:
            rows = _conn.execute(
                "SELECT * FROM backup_dirs ORDER BY env_name, mtime_tag"
            ).fetchall()
    return [dict(r) for r in rows]


def list_env_names() -> List[str]:
    with _lock:
        rows = _conn.execute(
            "SELECT DISTINCT env_name FROM backup_dirs WHERE env_name != '' ORDER BY env_name"
        ).fetchall()
    return [r["env_name"] for r in rows]


def get_dir(dir_path: str) -> Optional[dict]:
    with _lock:
        row = _conn.execute(
            "SELECT * FROM backup_dirs WHERE dir_path = ?", (dir_path,)
        ).fetchone()
    return dict(row) if row else None


def mark_backed_up(dir_path: str):
    with _lock:
        _conn.execute(
            "UPDATE backup_dirs SET status = 'backed_up', source = 'obs', updated_at = ? WHERE dir_path = ?",
            (_now(), dir_path),
        )
        _conn.commit()


def update_sync_pid(dir_path: str, pid: Optional[int]):
    with _lock:
        _conn.execute(
            "UPDATE backup_dirs SET sync_pid = ?, updated_at = ? WHERE dir_path = ?",
            (pid, _now(), dir_path),
        )
        _conn.commit()


def get_live_syncing_dirs() -> List[dict]:
    with _lock:
        rows = _conn.execute(
            "SELECT * FROM backup_dirs WHERE status = 'live_syncing'"
        ).fetchall()
    return [dict(r) for r in rows]


def remove_dir(dir_path: str):
    with _lock:
        _conn.execute("DELETE FROM backup_dirs WHERE dir_path = ?", (dir_path,))
        _conn.execute("DELETE FROM backup_logs WHERE dir_path = ?", (dir_path,))
        _conn.commit()


def append_log(dir_path: str, message: str, level: str = "info"):
    with _lock:
        _conn.execute(
            "INSERT INTO backup_logs (dir_path, level, message, created_at) VALUES (?, ?, ?, ?)",
            (dir_path, level, message, _now()),
        )
        _conn.commit()


def get_logs(dir_path: str, limit: int = 200) -> List[dict]:
    with _lock:
        rows = _conn.execute(
            "SELECT * FROM backup_logs WHERE dir_path = ? ORDER BY id DESC LIMIT ?",
            (dir_path, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def clear_logs(dir_path: str):
    with _lock:
        _conn.execute("DELETE FROM backup_logs WHERE dir_path = ?", (dir_path,))
        _conn.commit()
