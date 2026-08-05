"""
utils/export_store.py — Session 导出记录持久化

SQLite 后端，记录每次导出操作的状态和结果。
数据库路径：{service_log_dir}/export_session_record.db
"""

import os
import sqlite3
import threading
import json
from pathlib import Path
from typing import Optional

_lock = threading.Lock()
_conn: Optional[sqlite3.Connection] = None
_db_path: str = ""


def _get_export_log_dir() -> Path:
    """返回外部日志目录路径，与数据库同目录"""
    if _db_path:
        return Path(_db_path).parent / "export_log"
    return Path("logs") / "export_log"


def _write_external_log(record_id: int, field_name: str, content: str) -> str:
    """将字段内容写入外部文件，返回 file:// 路径标记"""
    log_dir = _get_export_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{record_id}.{field_name}.log"
    log_path.write_text(content, encoding="utf-8")
    return f"file://{log_path}"


def _read_field_content(value: str) -> str:
    """读取字段内容：如果是 file:// 路径则从文件读取，否则直接返回（兼容旧数据）"""
    if not value:
        return ""
    if value.startswith("file://"):
        file_path = Path(value[7:])  # 去掉 file:// 前缀
        if file_path.is_file():
            return file_path.read_text(encoding="utf-8")
        # 相对路径：尝试从数据库所在目录解析
        if _db_path and not file_path.is_absolute():
            abs_path = Path(_db_path).parent / file_path
            if abs_path.is_file():
                return abs_path.read_text(encoding="utf-8")
        return ""
    # 旧数据：直接存储的内容
    return value


def _externalize_field(record_id: int, field_name: str, content: str) -> str:
    """默认写入外部文件并返回路径标记"""
    return _write_external_log(record_id, field_name, content)


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
        for col_def in [
            "progress_log TEXT NOT NULL DEFAULT '[]'",
            "eval_status TEXT NOT NULL DEFAULT ''",
            "eval_report_path TEXT NOT NULL DEFAULT ''",
            "mode TEXT NOT NULL DEFAULT 'export'",
            "analysis_json TEXT NOT NULL DEFAULT ''",
            "source_export_id INTEGER",
            "in_manage INTEGER NOT NULL DEFAULT 0",
            "manage_name TEXT NOT NULL DEFAULT ''",
            "key_name TEXT NOT NULL DEFAULT ''",
            "workers INTEGER NOT NULL DEFAULT 0",
            "dir_workers INTEGER NOT NULL DEFAULT 0",
            "leaves_cache TEXT NOT NULL DEFAULT ''",
        ]:
            try:
                _conn.execute(f"ALTER TABLE export_records ADD COLUMN {col_def}")
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
    mode: str = "export",
    source_export_id: int | None = None,
    key_name: str = "",
    workers: int = 0,
    dir_workers: int = 0,
) -> int:
    conn = _get_conn()
    with _lock:
        cur = conn.execute(
            """INSERT INTO export_records
               (api_key, key_slot, mtime_dirs, obs_dst, local_copy_dir, mode, source_export_id, key_name, workers, dir_workers)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (api_key, key_slot, mtime_dirs, obs_dst, local_copy_dir, mode, source_export_id, key_name, workers, dir_workers),
        )
        conn.commit()
    return cur.lastrowid


def update_status(record_id: int, status: str, **kwargs):
    conn = _get_conn()
    fields = ["status = ?"]
    values = [status]
    if status == "running":
        fields.append("started_at = datetime('now','localtime')")
    elif status in ("success", "failed", "cancelled"):
        fields.append("finished_at = datetime('now','localtime')")
    for k, v in kwargs.items():
        if k in ("error_message", "total_sessions", "files_uploaded", "files_skipped",
                 "obs_dst", "local_copy_dir", "eval_status", "eval_report_path",
                 "analysis_json", "key_name", "leaves_cache"):
            fields.append(f"{k} = ?")
            values.append(v)
    values.append(record_id)
    with _lock:
        conn.execute(f"UPDATE export_records SET {', '.join(fields)} WHERE id = ?", values)
        conn.commit()


def save_leaves_cache(record_id: int, leaves: list, warnings: list | None = None) -> None:
    """把某记录的节点分布结果落库（JSON），供之后展开直接读、免重复扫盘。"""
    payload = json.dumps({"leaves": leaves, "warnings": warnings or []}, ensure_ascii=False)
    conn = _get_conn()
    with _lock:
        conn.execute("UPDATE export_records SET leaves_cache = ? WHERE id = ?", (payload, record_id))
        conn.commit()


def get_leaves_cache(record_id: int) -> Optional[dict]:
    """读回缓存的节点分布；无缓存或解析失败返回 None。"""
    conn = _get_conn()
    with _lock:
        row = conn.execute("SELECT leaves_cache FROM export_records WHERE id = ?", (record_id,)).fetchone()
    if not row:
        return None
    raw = row["leaves_cache"] if "leaves_cache" in row.keys() else ""
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return None


def get_record(record_id: int) -> Optional[dict]:
    conn = _get_conn()
    # 共享连接（check_same_thread=False）：读也必须持锁，否则与并发写交错会触发
    # sqlite3 "another row available"（同一连接上有未取完的语句时再 execute）。
    with _lock:
        row = conn.execute("SELECT * FROM export_records WHERE id = ?", (record_id,)).fetchone()
    return dict(row) if row else None


def get_record_resolved(record_id: int) -> Optional[dict]:
    """获取记录并自动解析外部文件内容（仅展开 progress_log，不读 analysis_json）"""
    rec = get_record(record_id)
    if not rec:
        return None
    if rec.get("progress_log"):
        rec["progress_log"] = _read_field_content(rec["progress_log"])
    return rec


def get_record_with_analysis(record_id: int) -> Optional[dict]:
    """获取记录并展开 analysis_json（体积可能极大，仅在需要重新渲染报告时使用）"""
    rec = get_record(record_id)
    if not rec:
        return None
    if rec.get("progress_log"):
        rec["progress_log"] = _read_field_content(rec["progress_log"])
    if rec.get("analysis_json"):
        rec["analysis_json"] = _read_field_content(rec["analysis_json"])
    return rec


def list_records(limit: int = 100) -> list:
    conn = _get_conn()
    with _lock:
        rows = conn.execute(
            "SELECT * FROM export_records ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def list_records_by_key(key_slot: str, limit: int = 50) -> list:
    conn = _get_conn()
    with _lock:
        rows = conn.execute(
            "SELECT * FROM export_records WHERE key_slot = ? ORDER BY created_at DESC LIMIT ?",
            (key_slot, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def latest_quality_record(source_export_id: int) -> Optional[dict]:
    with _lock:
        row = _get_conn().execute(
            """SELECT * FROM export_records
               WHERE mode='eval' AND source_export_id=?
               ORDER BY id DESC LIMIT 1""",
            (source_export_id,),
        ).fetchone()
    return dict(row) if row else None


_SLIM_COLS = "id, key_slot, api_key, key_name, mtime_dirs, status, mode, created_at, total_sessions, files_uploaded, obs_dst, error_message"

_DS_COLS = "id, key_slot, api_key, key_name, status, mode, created_at, total_sessions, obs_dst, error_message, local_copy_dir, in_manage, manage_name"


def list_records_for_datasets(limit: int = 1000, in_manage: Optional[bool] = None) -> list:
    """列出数据集候选记录。

    in_manage=None: 全部；True: 仅已加入管理列表；False: 仅未加入。
    """
    conn = _get_conn()
    sql = f"SELECT {_DS_COLS} FROM export_records"
    params: list = []
    if in_manage is not None:
        sql += " WHERE in_manage = ?"
        params.append(1 if in_manage else 0)
    sql += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    with _lock:
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def set_in_manage(record_id: int, flag: bool) -> None:
    """设置单条记录是否在 Session 管理列表中（仅解绑/加入，不删数据）。"""
    conn = _get_conn()
    with _lock:
        conn.execute("UPDATE export_records SET in_manage = ? WHERE id = ?",
                     (1 if flag else 0, record_id))
        conn.commit()


def set_manage_name(record_id: int, name: str) -> None:
    """设置该记录在管理列表中的自定义显示名（留空表示回退原 key_slot）。"""
    conn = _get_conn()
    with _lock:
        conn.execute("UPDATE export_records SET manage_name = ? WHERE id = ?",
                     (name or "", record_id))
        conn.commit()


def set_in_manage_bulk(record_ids: list, flag: bool) -> int:
    """批量设置 in_manage，返回受影响行数。"""
    ids = [int(x) for x in (record_ids or [])]
    if not ids:
        return 0
    conn = _get_conn()
    placeholders = ",".join("?" for _ in ids)
    with _lock:
        cur = conn.execute(
            f"UPDATE export_records SET in_manage = ? WHERE id IN ({placeholders})",
            [1 if flag else 0, *ids],
        )
        conn.commit()
    return cur.rowcount


def get_records_summary() -> tuple:
    """返回 (max_id, count, running_count)，用于快速判断 export_records 是否有变化。
    running_count 用于检测 status 从 running/queued → success/failed 的变化。"""
    conn = _get_conn()
    with _lock:
        row = conn.execute(
            "SELECT MAX(id), COUNT(*), SUM(CASE WHEN status IN ('running','queued') THEN 1 ELSE 0 END) FROM export_records"
        ).fetchone()
    return (row[0] or 0, row[1] or 0, row[2] or 0)


def list_records_all_slim(limit_per_key: int = 10) -> dict:
    """一次查询所有 records（排除 progress_log 等大字段），按 key_slot 分组返回。"""
    conn = _get_conn()
    with _lock:
        rows = conn.execute(
            f"SELECT {_SLIM_COLS} FROM export_records ORDER BY created_at DESC"
        ).fetchall()
    grouped: dict = {}
    for r in rows:
        d = dict(r)
        slot = d.pop("key_slot", "all")
        if slot not in grouped:
            grouped[slot] = []
        if len(grouped[slot]) < limit_per_key:
            grouped[slot].append(d)
    return grouped

def mark_interrupted():
    """启动时将遗留的 running/queued 记录标记为 failed/cancelled。

    已被 requeue_interrupted() 取代（自动重新入队而非直接判失败）；保留供无法
    重建任务的场景兜底。当前不再由启动流程调用。
    """
    conn = _get_conn()
    with _lock:
        conn.execute(
            "UPDATE export_records SET status = 'failed', error_message = '服务重启中断', "
            "finished_at = datetime('now','localtime') WHERE status = 'running'"
        )
        conn.execute(
            "UPDATE export_records SET status = 'cancelled', error_message = '服务重启取消', "
            "finished_at = datetime('now','localtime') WHERE status = 'queued'"
        )
        conn.commit()


def cancel_interrupted() -> int:
    """启动时把所有遗留的 running/queued/pending 任务直接取消，队列从零开始。

    不再重新入队（此前的 take_interrupted + _requeue_interrupted 会在频繁重启时
    反复重建任务并刷屏日志）。重启即视为放弃上一轮排队：全部标 cancelled，
    用户可在页面上按需手动重跑。返回受影响行数。

    注意：draft（草稿）状态不在此列——它是用户显式保存待启动的任务，
    重启后保留，等待手动「启动」。
    """
    conn = _get_conn()
    with _lock:
        cur = conn.execute(
            "UPDATE export_records SET status = 'cancelled', "
            "error_message = '服务重启，队列已清空', "
            "finished_at = datetime('now','localtime') "
            "WHERE status IN ('running', 'queued', 'pending')"
        )
        conn.commit()
    return cur.rowcount


def take_interrupted() -> list:
    """启动时取出被重启打断的记录（running/queued），置回 pending 待重新入队。

    返回完整记录行（dict）列表，供上层用其持久化字段重建任务再入队。
    置为 pending（而非直接 failed）：这样即使重建/入队失败，记录也停在 pending
    而非误报成功；成功重新入队后 _enqueue_task 会把它推进为 queued。

    已被 cancel_interrupted() 取代（重启即清空队列而非重建）；保留供需要
    「重启后自动续跑」的场景切回使用。当前不再由启动流程调用。
    """
    conn = _get_conn()
    with _lock:
        rows = conn.execute(
            "SELECT * FROM export_records WHERE status IN ('running', 'queued') "
            "ORDER BY id"
        ).fetchall()
        recs = [dict(r) for r in rows]
        if recs:
            conn.execute(
                "UPDATE export_records SET status = 'pending', error_message = '', "
                "started_at = NULL, finished_at = NULL "
                "WHERE status IN ('running', 'queued')"
            )
            conn.commit()
    return recs


def append_log(record_id: int, message: str):
    """追加一条进度日志到 progress_log JSON 数组。"""
    import json
    from datetime import datetime as _dt

    conn = _get_conn()
    entry = {"ts": _dt.now().strftime("%H:%M:%S"), "msg": message}
    with _lock:
        row = conn.execute("SELECT progress_log FROM export_records WHERE id = ?", (record_id,)).fetchone()
        if row:
            current_log = _read_field_content(row[0] or "[]")  # 支持外部文件和旧数据
            try:
                logs = json.loads(current_log)
            except (json.JSONDecodeError, TypeError):
                logs = []
            logs.append(entry)
            new_content = json.dumps(logs, ensure_ascii=False)
            # 默认写入外部文件
            stored_value = _externalize_field(record_id, "progress_log", new_content)
            conn.execute("UPDATE export_records SET progress_log = ? WHERE id = ?",
                         (stored_value, record_id))
            conn.commit()
