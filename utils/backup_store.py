"""
utils/backup_store.py — 备份同步状态持久化

SQLite 后端，记录每个 mtime 目录的 OBS 同步状态和同步日志。
数据库路径：{service_log_dir}/backup_sync.db
"""

import os
import sqlite3
import threading
from datetime import datetime
from typing import List, Optional

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
        _conn.execute("""
            CREATE TABLE IF NOT EXISTS backup_failed_files (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                dir_path   TEXT NOT NULL,
                local_path TEXT NOT NULL,
                obs_path   TEXT NOT NULL,
                error      TEXT NOT NULL DEFAULT '',
                resolved   INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT (datetime('now','localtime'))
            )
        """)
        _conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_failed_files_dir ON backup_failed_files(dir_path)"
        )
        for col, definition in [
            ("sync_pid", "INTEGER DEFAULT NULL"),
            ("port",     "TEXT NOT NULL DEFAULT ''"),
            ("has_local","INTEGER NOT NULL DEFAULT 1"),
            ("has_obs",  "INTEGER NOT NULL DEFAULT 0"),
        ]:
            try:
                _conn.execute(f"ALTER TABLE backup_dirs ADD COLUMN {col} {definition}")
            except sqlite3.OperationalError:
                pass

        # 迁移旧 source 字段 -> has_local / has_obs
        cols = {row[1] for row in _conn.execute("PRAGMA table_info(backup_dirs)")}
        if "source" in cols:
            _conn.execute("""
                UPDATE backup_dirs SET
                    has_local = CASE WHEN source IN ('local','both') THEN 1 ELSE 0 END,
                    has_obs   = CASE WHEN source IN ('obs','both') OR obs_path != '' THEN 1 ELSE 0 END
                WHERE has_local = 1 AND has_obs = 0 AND source IS NOT NULL
            """)
            # 对于 backed_up 状态（本地已删）本地标记清零
            _conn.execute("""
                UPDATE backup_dirs SET has_local = 0
                WHERE status = 'backed_up'
            """)

        _conn.commit()


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def upsert_dir(dir_path: str, env_name: str, mtime_tag: str, file_count: int = 0,
               port: str = "", has_local: bool = True, has_obs: bool = False):
    with _lock:
        existing = _conn.execute(
            "SELECT has_local, has_obs, port FROM backup_dirs WHERE dir_path = ?", (dir_path,)
        ).fetchone()
        if existing:
            merged_local = existing["has_local"] or int(has_local)
            merged_obs   = existing["has_obs"]   or int(has_obs)
            merged_port  = existing["port"] or port
            fields = ["has_local = ?", "has_obs = ?", "port = ?", "updated_at = ?"]
            params = [merged_local, merged_obs, merged_port, _now()]
            if file_count > 0:
                fields.insert(0, "file_count = ?")
                params.insert(0, file_count)
            params.append(dir_path)
            _conn.execute(
                f"UPDATE backup_dirs SET {', '.join(fields)} WHERE dir_path = ?",
                params,
            )
        else:
            _conn.execute("""
                INSERT INTO backup_dirs
                    (dir_path, env_name, mtime_tag, file_count, port, has_local, has_obs, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (dir_path, env_name, mtime_tag, file_count, port,
                  int(has_local), int(has_obs), _now(), _now()))
        _conn.commit()


def update_sync_status(dir_path: str, status: str, obs_path: str = "", error_msg: str = ""):
    fields = ["status = ?", "updated_at = ?"]
    params: list = [status, _now()]
    if obs_path:
        fields.append("obs_path = ?")
        fields.append("has_obs = 1")
        params.append(obs_path)
    if error_msg:
        fields.append("error_msg = ?")
        params.append(error_msg)
    else:
        fields.append("error_msg = ''")
    if status == "done":
        fields.append("synced = 1")
        fields.append("sync_time = ?")
        params.append(_now())
    if status in ("done", "error", "pending", "partial"):
        fields.append("sync_pid = NULL")
    if status == "backed_up":
        fields.append("has_local = 0")
        fields.append("has_obs = 1")
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
            "UPDATE backup_dirs SET status='backed_up', has_local=0, has_obs=1, updated_at=? WHERE dir_path=?",
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


def has_any_dirs(env_name: str = None) -> bool:
    with _lock:
        if env_name:
            row = _conn.execute(
                "SELECT 1 FROM backup_dirs WHERE env_name = ? LIMIT 1", (env_name,)
            ).fetchone()
        else:
            row = _conn.execute("SELECT 1 FROM backup_dirs LIMIT 1").fetchone()
    return row is not None


def remove_dir(dir_path: str):
    with _lock:
        _conn.execute("DELETE FROM backup_dirs WHERE dir_path = ?", (dir_path,))
        _conn.execute("DELETE FROM backup_logs WHERE dir_path = ?", (dir_path,))
        _conn.execute("DELETE FROM backup_failed_files WHERE dir_path = ?", (dir_path,))
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


# ---------------------------------------------------------------------------
# backup_failed_files — obsutil 上传部分失败的文件清单，供"补同步"逐个重传
# ---------------------------------------------------------------------------

def record_failed_files(dir_path: str, items: List[tuple]):
    """记录失败文件清单。items = [(local_path, obs_path, error), ...]。
    先清除该目录已有的未 resolved 记录，避免重复。"""
    with _lock:
        _conn.execute(
            "DELETE FROM backup_failed_files WHERE dir_path = ? AND resolved = 0",
            (dir_path,),
        )
        _conn.executemany(
            """INSERT INTO backup_failed_files (dir_path, local_path, obs_path, error, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            [(dir_path, local, obs, error or "", _now()) for local, obs, error in items],
        )
        _conn.commit()


def list_failed_files(dir_path: str, only_unresolved: bool = True) -> List[dict]:
    with _lock:
        if only_unresolved:
            rows = _conn.execute(
                "SELECT * FROM backup_failed_files WHERE dir_path = ? AND resolved = 0 ORDER BY id",
                (dir_path,),
            ).fetchall()
        else:
            rows = _conn.execute(
                "SELECT * FROM backup_failed_files WHERE dir_path = ? ORDER BY id",
                (dir_path,),
            ).fetchall()
    return [dict(r) for r in rows]


def mark_file_resolved(dir_path: str, obs_path: str):
    with _lock:
        _conn.execute(
            "UPDATE backup_failed_files SET resolved = 1 WHERE dir_path = ? AND obs_path = ?",
            (dir_path, obs_path),
        )
        _conn.commit()


def count_unresolved_failed(dir_path: str) -> int:
    with _lock:
        row = _conn.execute(
            "SELECT COUNT(*) FROM backup_failed_files WHERE dir_path = ? AND resolved = 0",
            (dir_path,),
        ).fetchone()
    return row[0] if row else 0


def clear_failed_files(dir_path: str):
    """清空该目录的失败文件记录（用于目录整体同步成功后）。"""
    with _lock:
        _conn.execute("DELETE FROM backup_failed_files WHERE dir_path = ?", (dir_path,))
        _conn.commit()
