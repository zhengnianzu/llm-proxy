"""
客户端 API Key 存储模块。
SQLite 后端，支持 key 的 CRUD、验证、脱敏。
数据库路径由 init_db() 调用时指定（通常放在对应 port 目录下）。
"""

import os
import sqlite3
import secrets
import threading

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None
_db_path: str = ""


def init_db(db_dir: str = "data"):
    """初始化数据库。db_dir 为存放 keys.db 的目录。"""
    global _conn, _db_path
    _db_path = os.path.join(db_dir, "keys.db")
    os.makedirs(db_dir, exist_ok=True)
    _conn = sqlite3.connect(_db_path, check_same_thread=False)
    _conn.execute("PRAGMA journal_mode=WAL")
    _conn.execute("PRAGMA busy_timeout=5000")
    _conn.row_factory = sqlite3.Row
    with _lock:
        _conn.execute("""
            CREATE TABLE IF NOT EXISTS api_keys (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL DEFAULT '',
                key TEXT NOT NULL UNIQUE,
                status TEXT NOT NULL DEFAULT 'active',
                invite_code TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
            )
        """)
        # 兼容旧表：如果 invite_code 列不存在则添加
        try:
            _conn.execute("SELECT invite_code FROM api_keys LIMIT 0")
        except sqlite3.OperationalError:
            _conn.execute("ALTER TABLE api_keys ADD COLUMN invite_code TEXT NOT NULL DEFAULT ''")
        _conn.commit()


def _get_conn() -> sqlite3.Connection:
    if _conn is None:
        raise RuntimeError("key_store not initialized, call init_db() first")
    return _conn


def generate_key(byte_len: int = 24) -> str:
    return "sk-" + secrets.token_hex(byte_len)


def add_key(name: str = "", key: str = "", key_len: int = 24, invite_code: str = "") -> dict:
    if not key:
        key = generate_key(key_len)
    conn = _get_conn()
    with _lock:
        cur = conn.execute(
            "INSERT INTO api_keys (name, key, invite_code) VALUES (?, ?, ?)",
            (name, key, invite_code),
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM api_keys WHERE id = ?", (cur.lastrowid,)
        ).fetchone()
    return dict(row)


def list_keys() -> list[dict]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT id, name, key, status, invite_code, created_at FROM api_keys ORDER BY id"
    ).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        d["key"] = mask_key(d["key"])
        result.append(d)
    return result


def get_key_full(key_id: int) -> dict | None:
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM api_keys WHERE id = ?", (key_id,)
    ).fetchone()
    return dict(row) if row else None


def disable_key(key_id: int) -> bool:
    conn = _get_conn()
    with _lock:
        cur = conn.execute(
            "UPDATE api_keys SET status = 'disabled' WHERE id = ?", (key_id,)
        )
        conn.commit()
    return cur.rowcount > 0


def enable_key(key_id: int) -> bool:
    conn = _get_conn()
    with _lock:
        cur = conn.execute(
            "UPDATE api_keys SET status = 'active' WHERE id = ?", (key_id,)
        )
        conn.commit()
    return cur.rowcount > 0


def delete_key(key_id: int) -> bool:
    conn = _get_conn()
    with _lock:
        cur = conn.execute(
            "DELETE FROM api_keys WHERE id = ?", (key_id,)
        )
        conn.commit()
    return cur.rowcount > 0


def find_key(identifier: str) -> dict | None:
    """按 ID 或 key 值查找。identifier 可以是数字 ID 或 key 字符串。"""
    conn = _get_conn()
    if identifier.isdigit():
        row = conn.execute("SELECT * FROM api_keys WHERE id = ?", (int(identifier),)).fetchone()
    else:
        row = conn.execute("SELECT * FROM api_keys WHERE key = ?", (identifier,)).fetchone()
    return dict(row) if row else None


def validate_key(raw_key: str) -> str | None:
    if _conn is None:
        return None
    row = _conn.execute(
        "SELECT key FROM api_keys WHERE key = ? AND status = 'active'",
        (raw_key,),
    ).fetchone()
    return row["key"] if row else None


def has_active_keys() -> bool:
    if _conn is None:
        return False
    row = _conn.execute(
        "SELECT 1 FROM api_keys WHERE status = 'active' LIMIT 1"
    ).fetchone()
    return row is not None


def mask_key(key: str) -> str:
    if len(key) <= 10:
        return key[:3] + "***"
    return key[:6] + "..." + key[-4:]
