"""
utils/logdir_store.py — 日志路径叶子（子节点）构建状态持久化

SQLite 后端，记录每个 new-api 源下每个叶子目录（小时节点）的索引构建状态。
数据库路径：{service_log_dir}/log_dir.db

替代旧的「进程内存 + 前端轮询」方案：状态落盘、重启不丢、粒度到叶子。
- 「同步」扫描源的叶子目录，把新增节点写入（state=pending），已存在的更新构建状态。
- 回填过程逐叶写入 building/done/error。
- 列表页读汇总（count_summary），不再实时全盘 stat。

主键 (root_id, dir_key)：root_id 来自 logs_config.get_root_id（消除同 basename 冲突），
dir_key 来自 log_scan.dir_key_for（稳定、已折叠冗余层、可逆）。
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
    _db_path = os.path.join(db_dir, "log_dir.db")
    os.makedirs(db_dir, exist_ok=True)
    _conn = sqlite3.connect(_db_path, check_same_thread=False)
    _conn.execute("PRAGMA journal_mode=WAL")
    _conn.execute("PRAGMA busy_timeout=5000")
    _conn.row_factory = sqlite3.Row
    with _lock:
        _conn.execute("""
            CREATE TABLE IF NOT EXISTS leaf_status (
                root_id     TEXT NOT NULL,
                dir_key     TEXT NOT NULL,
                root_path   TEXT NOT NULL DEFAULT '',
                built       INTEGER NOT NULL DEFAULT 0,
                ingested    INTEGER NOT NULL DEFAULT 0,
                pending     INTEGER NOT NULL DEFAULT 0,
                sessions    INTEGER NOT NULL DEFAULT 0,
                state       TEXT NOT NULL DEFAULT 'pending',  -- pending|building|done|error
                last_error  TEXT NOT NULL DEFAULT '',
                synced_at   TEXT NOT NULL DEFAULT (datetime('now','localtime')),
                created_at  TEXT NOT NULL DEFAULT (datetime('now','localtime')),
                PRIMARY KEY (root_id, dir_key)
            )
        """)
        _conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_leaf_status_root ON leaf_status(root_id)"
        )
        _conn.commit()


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _ready() -> bool:
    return _conn is not None


def upsert_leaf(root_id: str, dir_key: str, root_path: str = "", *,
                built: Optional[bool] = None, ingested: Optional[int] = None,
                pending: Optional[int] = None, sessions: Optional[int] = None,
                state: Optional[str] = None, last_error: str = "") -> str:
    """新增或更新一个叶子。返回 'added' | 'updated'。仅传入的字段被更新。"""
    if not _ready():
        return "skipped"
    with _lock:
        existing = _conn.execute(
            "SELECT dir_key FROM leaf_status WHERE root_id = ? AND dir_key = ?",
            (root_id, dir_key),
        ).fetchone()
        if existing:
            fields = ["synced_at = ?"]
            params: list = [_now()]
            if root_path:
                fields.append("root_path = ?"); params.append(root_path)
            if built is not None:
                fields.append("built = ?"); params.append(int(built))
            if ingested is not None:
                fields.append("ingested = ?"); params.append(ingested)
            if pending is not None:
                fields.append("pending = ?"); params.append(pending)
            if sessions is not None:
                fields.append("sessions = ?"); params.append(sessions)
            if state is not None:
                fields.append("state = ?"); params.append(state)
            fields.append("last_error = ?"); params.append(last_error)
            params.extend([root_id, dir_key])
            _conn.execute(
                f"UPDATE leaf_status SET {', '.join(fields)} WHERE root_id = ? AND dir_key = ?",
                params,
            )
            _conn.commit()
            return "updated"
        else:
            _conn.execute("""
                INSERT INTO leaf_status
                    (root_id, dir_key, root_path, built, ingested, pending, sessions,
                     state, last_error, synced_at, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (root_id, dir_key, root_path, int(bool(built)),
                  ingested or 0, pending or 0, sessions or 0,
                  state or "pending", last_error, _now(), _now()))
            _conn.commit()
            return "added"


def set_leaf_state(root_id: str, dir_key: str, state: str, *,
                   built: Optional[bool] = None, ingested: Optional[int] = None,
                   pending: Optional[int] = None, sessions: Optional[int] = None,
                   last_error: str = ""):
    """回填过程写单叶状态（building/done/error）。叶子不存在则插入。"""
    if not _ready():
        return
    upsert_leaf(root_id, dir_key, built=built, ingested=ingested,
                pending=pending, sessions=sessions, state=state, last_error=last_error)


def bulk_get(root_id: str) -> List[dict]:
    if not _ready():
        return []
    with _lock:
        rows = _conn.execute(
            "SELECT * FROM leaf_status WHERE root_id = ? ORDER BY dir_key", (root_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def count_summary(root_id: str) -> dict:
    """{total, built, pending, building, error}；供列表页一次性读，无需全盘 stat。"""
    if not _ready():
        return {"total": 0, "built": 0, "pending": 0, "building": 0, "error": 0}
    with _lock:
        row = _conn.execute("""
            SELECT
                COUNT(*)                                          AS total,
                COALESCE(SUM(built), 0)                           AS built,
                COALESCE(SUM(state = 'pending'), 0)               AS pending,
                COALESCE(SUM(state = 'building'), 0)              AS building,
                COALESCE(SUM(state = 'error'), 0)                 AS error
            FROM leaf_status WHERE root_id = ?
        """, (root_id,)).fetchone()
    return {k: (row[k] or 0) for k in ("total", "built", "pending", "building", "error")}


def has_any(root_id: str) -> bool:
    """该源是否已同步过（DB 里是否有其叶子记录）。"""
    if not _ready():
        return False
    with _lock:
        row = _conn.execute(
            "SELECT 1 FROM leaf_status WHERE root_id = ? LIMIT 1", (root_id,)
        ).fetchone()
    return row is not None


def delete_root(root_id: str):
    """移除某源时清掉其所有叶子记录。"""
    if not _ready():
        return
    with _lock:
        _conn.execute("DELETE FROM leaf_status WHERE root_id = ?", (root_id,))
        _conn.commit()


def reset_building_on_startup():
    """进程启动时把上次中断的 building 降回 pending（回填被打断，需重新构建）。"""
    if not _ready():
        return
    with _lock:
        _conn.execute(
            "UPDATE leaf_status SET state = 'pending' WHERE state = 'building'"
        )
        _conn.commit()
