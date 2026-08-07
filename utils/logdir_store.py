"""
utils/logdir_store.py — 日志路径叶子（子节点）构建状态持久化

SQLite 后端，记录每个 new-api 源下每个叶子目录（小时节点）的索引构建状态。
数据库路径：{service_log_dir}/log_dir.db

替代旧的「进程内存 + 前端轮询」方案：状态落盘、重启不丢、粒度到叶子。
- 「同步」扫描源的叶子目录，把新增节点写入（state=pending），已存在的更新构建状态。
- 回填过程逐叶写入 building/done/error。
- 列表页读汇总（count_summary），不再实时全盘 stat。

sources 表（「数据管理」表 1）是日志路径列表的唯一数据源，无任何 YAML 导入/镜像：
增删改全部以本表为准；活跃目录行（root_id='default'）由启动时的
_refresh_active_row 刷新/补插。

主键 (root_id, dir_key)：root_id 来自 logs_config.get_root_id（消除同 basename 冲突），
dir_key 来自 log_scan.dir_key_for（稳定、已折叠冗余层、可逆）。
"""

import json as _json
import os
import sqlite3
import threading
from datetime import datetime
from typing import List, Optional

_lock = threading.RLock()
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
        # 数据源表（「数据管理」表 1）：一源一行。root_id 主键，与 leaf_status 同口径。
        # templates 存 JSON 数组（多行层级模板）；leaf_count/built_count 同步时刷新。
        _conn.execute("""
            CREATE TABLE IF NOT EXISTS sources (
                root_id     TEXT PRIMARY KEY,
                root_path   TEXT NOT NULL DEFAULT '',
                name        TEXT NOT NULL DEFAULT 'default',
                format      TEXT NOT NULL DEFAULT '',
                templates   TEXT NOT NULL DEFAULT '',
                leaf_count  INTEGER NOT NULL DEFAULT 0,
                built_count INTEGER NOT NULL DEFAULT 0,
                synced_at   TEXT NOT NULL DEFAULT (datetime('now','localtime')),
                created_at  TEXT NOT NULL DEFAULT (datetime('now','localtime'))
            )
        """)
        # 回填请求表：把「哪个 root 要回填」从 app 进程内存（newapi_backfill._running）
        # 搬到 DB，供独立 backfill_worker 进程领取执行、app 进程查询运行态。
        # 运行态真相 = status='running' 的行；逐叶进度仍以 leaf_status(count_summary) 为准。
        _conn.execute("""
            CREATE TABLE IF NOT EXISTS backfill_requests (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                root        TEXT NOT NULL,
                workers     INTEGER NOT NULL DEFAULT 8,
                force       INTEGER NOT NULL DEFAULT 0,
                status      TEXT NOT NULL DEFAULT 'pending',  -- pending|running|done|failed
                created_at  TEXT NOT NULL DEFAULT (datetime('now','localtime')),
                started_at  TEXT,
                finished_at TEXT,
                error       TEXT NOT NULL DEFAULT ''
            )
        """)
        _conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_backfill_requests_status ON backfill_requests(status)"
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


def get_last_leaf_synced_at(root_id: str) -> Optional[str]:
    """返回该 root 最后一次叶子同步时间（max(synced_at)），用于检测是否长时间无更新。"""
    if not _ready():
        return None
    with _lock:
        row = _conn.execute(
            "SELECT MAX(synced_at) FROM leaf_status WHERE root_id = ?", (root_id,)
        ).fetchone()
    return row[0] if row and row[0] else None


def has_any(root_id: str) -> bool:
    """该源是否已同步过（DB 里是否有其叶子记录）。"""
    if not _ready():
        return False
    with _lock:
        row = _conn.execute(
            "SELECT 1 FROM leaf_status WHERE root_id = ? LIMIT 1", (root_id,)
        ).fetchone()
    return row is not None


# ── 数据源表（「数据管理」表 1）───────────────────────────────────────────


def _templates_to_json(templates) -> str:
    """把模板（list 或多行字符串）规整为 JSON 数组字符串存库。"""
    if templates is None:
        return ""
    if isinstance(templates, str):
        items = [ln.strip() for ln in templates.splitlines() if ln.strip()]
    else:
        items = [str(t).strip() for t in templates if str(t).strip()]
    return _json.dumps(items, ensure_ascii=False)


def _templates_from_json(raw: str) -> list:
    """把库里的 templates 字段（JSON 数组字符串）反序列化为 list。"""
    if not raw:
        return []
    try:
        v = _json.loads(raw)
        return [str(x) for x in v] if isinstance(v, list) else []
    except (ValueError, TypeError):
        return []


def upsert_source(root_id: str, *, root_path: Optional[str] = None,
                  name: Optional[str] = None, format: Optional[str] = None,
                  templates=None, leaf_count: Optional[int] = None,
                  built_count: Optional[int] = None) -> str:
    """新增或更新一条数据源。返回 'added' | 'updated'。仅传入的字段被更新。"""
    if not _ready() or not root_id:
        return "skipped"
    with _lock:
        existing = _conn.execute(
            "SELECT root_id FROM sources WHERE root_id = ?", (root_id,)
        ).fetchone()
        if existing:
            fields = ["synced_at = ?"]
            params: list = [_now()]
            if root_path is not None:
                fields.append("root_path = ?"); params.append(root_path)
            if name is not None:
                fields.append("name = ?"); params.append(name)
            if format is not None:
                fields.append("format = ?"); params.append(format)
            if templates is not None:
                fields.append("templates = ?"); params.append(_templates_to_json(templates))
            if leaf_count is not None:
                fields.append("leaf_count = ?"); params.append(int(leaf_count))
            if built_count is not None:
                fields.append("built_count = ?"); params.append(int(built_count))
            params.append(root_id)
            _conn.execute(
                f"UPDATE sources SET {', '.join(fields)} WHERE root_id = ?", params
            )
            _conn.commit()
            return "updated"
        else:
            _conn.execute("""
                INSERT INTO sources
                    (root_id, root_path, name, format, templates,
                     leaf_count, built_count, synced_at, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (root_id, root_path or "", name or "default", format or "",
                  _templates_to_json(templates),
                  int(leaf_count or 0), int(built_count or 0), _now(), _now()))
            _conn.commit()
            return "added"


def get_source(root_id: str) -> Optional[dict]:
    """读一条数据源；templates 反序列化为 list。无则 None。"""
    if not _ready() or not root_id:
        return None
    with _lock:
        row = _conn.execute(
            "SELECT * FROM sources WHERE root_id = ?", (root_id,)
        ).fetchone()
    if not row:
        return None
    d = dict(row)
    d["templates"] = _templates_from_json(d.get("templates", ""))
    return d


def list_sources(active_env_dir: str = "") -> List[dict]:
    """数据源全量列表（「数据管理」列表页的唯一行来源）。

    活跃目录（root_id='default'）排第一，其余按 root_path 排序。
    templates 反序列化为 list；无活跃行时（空库 / 被删）返回全部。
    DB 未就绪或查询异常时返回空列表（调用方回退，不 500）。
    """
    if not _ready():
        return []
    try:
        with _lock:
            rows = _conn.execute(
                "SELECT * FROM sources ORDER BY root_id = 'default' DESC, root_path"
            ).fetchall()
    except Exception:  # noqa: BLE001
        return []
    out = [dict(r) for r in rows]
    for d in out:
        d["templates"] = _templates_from_json(d.get("templates", ""))
    return out


def get_source_by_path(path: str) -> Optional[dict]:
    """按路径取数据源行（root_id 由 get_root_id 计算）。无则 None。"""
    if not path:
        return None
    try:
        from utils.logs_config import get_root_id
    except Exception:  # noqa: BLE001
        return None
    return get_source(get_root_id(path))


def _refresh_active_row(active_env_dir: str) -> None:
    """把活跃目录行（root_id='default'）的 root_path 指向当前 env_dir。

    每次启动调用（app.py 启动流程）：env/base 变化时重指向；行不存在（新机器
    空库 / 被删）则补插 default 行——这是空库下唯一保证有活跃行的路径。
    顺带把该 root 下 leaf_status 的 root_path 批量更新，避免 /leaves 详情
    用旧路径拼接（leaf_status.root_path 存的是每叶快照）。
    """
    if not _ready() or not active_env_dir:
        return
    # import 必须在 _lock 之前（避免与另一线程的 `import utils.logdir_store`
    # 形成 importlib 可重入死锁，见旧 seed_sources_from_yaml 注释）
    from utils.log_scan import detect_format
    norm = os.path.normpath(active_env_dir)
    with _lock:
        existing = _conn.execute(
            "SELECT root_id FROM sources WHERE root_id = ?", ("default",)
        ).fetchone()
        if existing:
            _conn.execute(
                "UPDATE sources SET root_path = ?, synced_at = ? WHERE root_id = ?",
                (norm, _now(), "default"),
            )
        else:
            fmt = detect_format(norm)
            _conn.execute("""
                INSERT INTO sources
                    (root_id, root_path, name, format, templates,
                     leaf_count, built_count, synced_at, created_at)
                VALUES ('default', ?, 'default', ?, '', 0, 0, ?, ?)
            """, (norm, fmt, _now(), _now()))
        # 活跃行 root_path 变化时同步该 root 下已登记叶子的路径快照
        _conn.execute(
            "UPDATE leaf_status SET root_path = ? WHERE root_id = 'default'",
            (norm,),
        )
        _conn.commit()


def set_source_name(root_id: str, name: str) -> bool:
    """改数据源名字。行不存在返回 False。"""
    if not _ready() or not root_id:
        return False
    with _lock:
        cur = _conn.execute(
            "UPDATE sources SET name = ?, synced_at = ? WHERE root_id = ?",
            (name or "default", _now(), root_id),
        )
        _conn.commit()
        return cur.rowcount > 0


def set_source_templates(root_id: str, templates) -> bool:
    """改数据源层级模板。行不存在返回 False。"""
    if not _ready() or not root_id:
        return False
    with _lock:
        cur = _conn.execute(
            "UPDATE sources SET templates = ?, synced_at = ? WHERE root_id = ?",
            (_templates_to_json(templates), _now(), root_id),
        )
        _conn.commit()
        return cur.rowcount > 0


def delete_root(root_id: str):
    """移除某源时清掉其所有叶子记录（含数据源行）。"""
    if not _ready():
        return
    with _lock:
        _conn.execute("DELETE FROM leaf_status WHERE root_id = ?", (root_id,))
        _conn.execute("DELETE FROM sources WHERE root_id = ?", (root_id,))
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


# ── 回填请求队列（backfill_requests 表）─────────────────────────────────
# app 侧入队 / 查询运行态；backfill_worker 侧领取 / 完成。跨进程靠此表 + WAL。
# 运行态真相 = status='running' 的行；逐叶进度仍读 count_summary(leaf_status)。

_BACKFILL_ACTIVE = ("pending", "running")


def enqueue_backfill(root: str, workers: int = 8, force: bool = False) -> int:
    """入队一条回填请求，返回新记录 id。调用方应先 has_active_backfill 去重。"""
    if not _ready():
        raise RuntimeError("logdir_store not initialized")
    root = os.path.normpath(root)
    with _lock:
        cur = _conn.execute(
            "INSERT INTO backfill_requests (root, workers, force, status) "
            "VALUES (?, ?, ?, 'pending')",
            (root, int(workers or 8), 1 if force else 0),
        )
        _conn.commit()
        return cur.lastrowid


def has_active_backfill(root: str) -> bool:
    """该 root 是否已有 pending/running 请求（同源去重，避免两份进程池打架）。"""
    if not _ready():
        return False
    root = os.path.normpath(root)
    with _lock:
        row = _conn.execute(
            "SELECT 1 FROM backfill_requests WHERE root = ? AND status IN ('pending','running') LIMIT 1",
            (root,),
        ).fetchone()
    return row is not None


def get_active_backfill_request(root: str) -> Optional[dict]:
    """返回该 root 的活跃请求详情(pending/running),供构建状态判断 status。"""
    if not _ready():
        return None
    root = os.path.normpath(root)
    with _lock:
        row = _conn.execute(
            "SELECT * FROM backfill_requests WHERE root = ? AND status IN ('pending','running') ORDER BY id LIMIT 1",
            (root,),
        ).fetchone()
    return dict(row) if row else None


def list_active_backfill() -> List[dict]:
    """所有 pending/running 请求行（按 id）。供 UI/导出页展示排队与运行。"""
    if not _ready():
        return []
    with _lock:
        rows = _conn.execute(
            "SELECT * FROM backfill_requests WHERE status IN ('pending','running') ORDER BY id"
        ).fetchall()
    return [dict(r) for r in rows]


def claim_next_backfill() -> Optional[dict]:
    """worker 领取下一条 pending 请求：最老的一条原子置 running 并返回。

    UPDATE ... WHERE id=? AND status='pending' + rowcount 判定抢占；多 worker/重复
    领取会落空（该行已非 pending）。返回 None 表示暂无待执行请求。
    """
    if not _ready():
        return None
    with _lock:
        row = _conn.execute(
            "SELECT * FROM backfill_requests WHERE status = 'pending' ORDER BY id LIMIT 1"
        ).fetchone()
        if not row:
            return None
        rec = dict(row)
        cur = _conn.execute(
            "UPDATE backfill_requests SET status = 'running', "
            "started_at = datetime('now','localtime') WHERE id = ? AND status = 'pending'",
            (rec["id"],),
        )
        _conn.commit()
        if cur.rowcount == 0:
            return None  # 被其他 worker 抢先
    rec["status"] = "running"
    return rec


def complete_backfill(req_id: int, ok: bool, error: str = "") -> None:
    """标记请求终态：ok→done，否则→failed（附 error）。"""
    if not _ready():
        return
    with _lock:
        _conn.execute(
            "UPDATE backfill_requests SET status = ?, error = ?, "
            "finished_at = datetime('now','localtime') WHERE id = ?",
            ("done" if ok else "failed", error or "", req_id),
        )
        _conn.commit()


def reset_running_backfill_on_startup() -> int:
    """worker 启动时把上次中断的 running/pending 请求标 failed（队列从零开始）。

    与 export_store.cancel_interrupted 同语义：崩溃/重启视为放弃在途请求，
    用户可在页面重新发起。返回受影响行数。leaf_status 的 building→pending 由
    reset_building_on_startup 单独处理。
    """
    if not _ready():
        return 0
    with _lock:
        cur = _conn.execute(
            "UPDATE backfill_requests SET status = 'failed', "
            "error = '服务重启，队列已清空', finished_at = datetime('now','localtime') "
            "WHERE status IN ('pending', 'running')"
        )
        _conn.commit()
    return cur.rowcount
