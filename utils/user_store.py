import os
import sqlite3
import secrets
import hashlib
import threading

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None
_db_path: str = ""


def init_db(db_dir: str = "data"):
    global _conn, _db_path
    _db_path = os.path.join(db_dir, "users.db")
    os.makedirs(db_dir, exist_ok=True)
    _conn = sqlite3.connect(_db_path, check_same_thread=False)
    _conn.execute("PRAGMA journal_mode=WAL")
    _conn.execute("PRAGMA busy_timeout=5000")
    _conn.row_factory = sqlite3.Row
    with _lock:
        _conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user',
                permissions TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
            )
        """)
        _conn.commit()
        _migrate(_conn)


def _migrate(conn: sqlite3.Connection):
    cols = {row[1] for row in conn.execute("PRAGMA table_info(users)").fetchall()}
    if "permissions" not in cols:
        conn.execute("ALTER TABLE users ADD COLUMN permissions TEXT NOT NULL DEFAULT ''")
        conn.commit()


def _get_conn() -> sqlite3.Connection:
    if _conn is None:
        raise RuntimeError("user_store not initialized, call init_db() first")
    return _conn


def _hash_password(password: str, salt: str) -> str:
    return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()


def create_user(username: str, password: str, role: str = "user", permissions: str = "") -> dict | None:
    conn = _get_conn()
    salt = secrets.token_hex(16)
    password_hash = _hash_password(password, salt)
    with _lock:
        try:
            cur = conn.execute(
                "INSERT INTO users (username, password_hash, salt, role, permissions) VALUES (?, ?, ?, ?, ?)",
                (username, password_hash, salt, role, permissions),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM users WHERE id = ?", (cur.lastrowid,)).fetchone()
            return dict(row)
        except sqlite3.IntegrityError:
            return None


def verify_user(username: str, password: str) -> dict | None:
    conn = _get_conn()
    row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    if not row:
        return None
    user = dict(row)
    if _hash_password(password, user["salt"]) == user["password_hash"]:
        return user
    return None


def list_users() -> list[dict]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT id, username, role, permissions, created_at FROM users ORDER BY id"
    ).fetchall()
    return [dict(r) for r in rows]


def update_user(user_id: int, role: str | None = None, permissions: str | None = None) -> bool:
    conn = _get_conn()
    fields = []
    values = []
    if role is not None:
        fields.append("role = ?")
        values.append(role)
    if permissions is not None:
        fields.append("permissions = ?")
        values.append(permissions)
    if not fields:
        return False
    values.append(user_id)
    with _lock:
        cur = conn.execute(f"UPDATE users SET {', '.join(fields)} WHERE id = ?", values)
        conn.commit()
    return cur.rowcount > 0


def delete_user(user_id: int) -> bool:
    conn = _get_conn()
    with _lock:
        cur = conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()
    return cur.rowcount > 0


def has_users() -> bool:
    if _conn is None:
        return False
    row = _conn.execute("SELECT 1 FROM users LIMIT 1").fetchone()
    return row is not None


def get_user_permissions(username: str) -> list[str]:
    conn = _get_conn()
    row = conn.execute("SELECT permissions FROM users WHERE username = ?", (username,)).fetchone()
    if not row:
        return []
    raw = row["permissions"] or ""
    return [p.strip() for p in raw.split(",") if p.strip()]
