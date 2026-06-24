"""
渠道管理存储模块。
SQLite 后端，支持渠道的 CRUD、key-channel 绑定、随机路由。
数据库路径由 init_db() 调用时指定（通常放在对应 port 目录下）。
"""

import os
import random
import sqlite3
import threading
from typing import Optional

_lock = threading.Lock()
_conn: Optional[sqlite3.Connection] = None
_db_path: str = ""


def init_db(db_dir: str = "data"):
    """初始化渠道数据库。db_dir 为存放 channels.db 的目录。"""
    global _conn, _db_path
    _db_path = os.path.join(db_dir, "channels.db")
    os.makedirs(db_dir, exist_ok=True)
    _conn = sqlite3.connect(_db_path, check_same_thread=False)
    _conn.execute("PRAGMA journal_mode=WAL")
    _conn.execute("PRAGMA busy_timeout=5000")
    _conn.row_factory = sqlite3.Row
    with _lock:
        _conn.execute("""
            CREATE TABLE IF NOT EXISTS channels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL DEFAULT '',
                upstream_url TEXT NOT NULL,
                upstream_key TEXT NOT NULL,
                alive INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
            )
        """)
        _conn.execute("""
            CREATE TABLE IF NOT EXISTS key_channel_bindings (
                key_id INTEGER NOT NULL,
                channel_id INTEGER NOT NULL,
                PRIMARY KEY (key_id, channel_id)
            )
        """)
        _conn.commit()


def _get_conn() -> sqlite3.Connection:
    if _conn is None:
        raise RuntimeError("channel_store not initialized, call init_db() first")
    return _conn


def mask_upstream_key(key: str) -> str:
    if len(key) <= 4:
        return key
    return "***" + key[-4:]


def key_suffix(key: str) -> str:
    return key[-4:] if len(key) >= 4 else key


def add_channel(name: str, upstream_url: str, upstream_key: str) -> dict:
    conn = _get_conn()
    with _lock:
        cur = conn.execute(
            "INSERT INTO channels (name, upstream_url, upstream_key) VALUES (?, ?, ?)",
            (name, upstream_url, upstream_key),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM channels WHERE id = ?", (cur.lastrowid,)).fetchone()
    d = dict(row)
    d["key_suffix"] = key_suffix(d["upstream_key"])
    d["upstream_key"] = mask_upstream_key(d["upstream_key"])
    return d


def list_channels() -> list[dict]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT id, name, upstream_url, upstream_key, alive, created_at FROM channels ORDER BY id"
    ).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        d["key_suffix"] = key_suffix(d["upstream_key"])
        d["upstream_key"] = mask_upstream_key(d["upstream_key"])
        result.append(d)
    return result


def get_channel(channel_id: int) -> Optional[dict]:
    conn = _get_conn()
    row = conn.execute("SELECT * FROM channels WHERE id = ?", (channel_id,)).fetchone()
    return dict(row) if row else None


def toggle_channel_alive(channel_id: int, alive: bool) -> bool:
    conn = _get_conn()
    with _lock:
        cur = conn.execute(
            "UPDATE channels SET alive = ? WHERE id = ?", (1 if alive else 0, channel_id)
        )
        conn.commit()
    return cur.rowcount > 0


def delete_channel(channel_id: int) -> bool:
    conn = _get_conn()
    with _lock:
        conn.execute("DELETE FROM key_channel_bindings WHERE channel_id = ?", (channel_id,))
        cur = conn.execute("DELETE FROM channels WHERE id = ?", (channel_id,))
        conn.commit()
    return cur.rowcount > 0


# ---- Key-Channel 绑定 ----

def set_key_channels(key_id: int, channel_ids: list[int]):
    conn = _get_conn()
    with _lock:
        conn.execute("DELETE FROM key_channel_bindings WHERE key_id = ?", (key_id,))
        for cid in channel_ids:
            conn.execute(
                "INSERT OR IGNORE INTO key_channel_bindings (key_id, channel_id) VALUES (?, ?)",
                (key_id, cid),
            )
        conn.commit()


def get_key_channel_ids(key_id: int) -> list[int]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT channel_id FROM key_channel_bindings WHERE key_id = ?", (key_id,)
    ).fetchall()
    return [r["channel_id"] for r in rows]


def get_key_channels(key_id: int) -> list[dict]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT c.id, c.name, c.upstream_url, c.upstream_key, c.alive, c.created_at "
        "FROM channels c JOIN key_channel_bindings b ON c.id = b.channel_id "
        "WHERE b.key_id = ? ORDER BY c.id",
        (key_id,),
    ).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        d["key_suffix"] = key_suffix(d["upstream_key"])
        d["upstream_key"] = mask_upstream_key(d["upstream_key"])
        result.append(d)
    return result


def resolve_channel_for_key(key_id: int) -> Optional[dict]:
    """为某个 key 随机选一个 alive 的绑定渠道，返回含完整 upstream_key 的 dict 或 None。"""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT c.id, c.upstream_url, c.upstream_key "
        "FROM channels c JOIN key_channel_bindings b ON c.id = b.channel_id "
        "WHERE b.key_id = ? AND c.alive = 1",
        (key_id,),
    ).fetchall()
    if not rows:
        return None
    chosen = random.choice(rows)
    return dict(chosen)


def get_channels_for_key_display(key_id: int) -> list[str]:
    """返回某个 key 绑定的渠道 key 后4位列表。"""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT c.upstream_key FROM channels c "
        "JOIN key_channel_bindings b ON c.id = b.channel_id "
        "WHERE b.key_id = ? ORDER BY c.id",
        (key_id,),
    ).fetchall()
    return [key_suffix(r["upstream_key"]) for r in rows]
