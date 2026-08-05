"""
utils/logs_config.py — 日志根目录查询（纯 SQLite，无 YAML）

「数据管理」以 log_dir.db 的 sources 表为唯一数据源（见 logdir_store.py）。
本模块提供查询辅助：

- get_root_id / get_root_by_id — 根目录的稳定内部标识（路径 md5 前 8 位）。
- get_active_base — 进程写入的 base（env LOGS_DIR 优先，否则 logs_all）。
- get_path_name / get_path_templates — 某根目录的展示名 / 层级模板（查 DB sources 表）。
- get_stats_roots / get_registered_roots — 参与统计/预览的所有根目录。
- dir_size / human_size — 目录大小工具。

统计聚合的根 = [活跃 env_dir] + [sources 表中其余根]，去重。
每个根下面用可变深度扫描找到含 index.jsonl 的叶子目录（见 log_scan.py），
因此同时支持:
  - 本项目格式  {root}/{hour}/index.jsonl        (root = logs_all/env-xxx)
  - new-api 格式 {root}/{day}/{hour}/index.jsonl  (root = .../logs/details)
"""

import hashlib
import os
from typing import List, Optional

_DEFAULT_BASE = "logs_all"
_DEFAULT_NAME = "default"
_DEFAULT_ID = "default"


def get_active_base() -> str:
    """进程写入的 base 名：env LOGS_DIR 优先，否则 logs_all。"""
    env_base = (os.getenv("LOGS_DIR") or "").strip()
    return env_base or _DEFAULT_BASE


def get_root_id(path: str, active_env_dir: Optional[str] = None) -> str:
    """某根目录的稳定内部唯一标识（供 log_dir / 备份 env_name 区分来源）。

    - 活跃 env_dir 固定返回 "default"（与历史行为一致，且是活跃写入目录）。
    - 其余根：normpath 路径的 md5 前 8 位。同路径必同 id，不同路径几乎不碰撞。

    用路径哈希而非独立 uuid：全系统缓存/会话 DB 本就以路径为键，
    路径哈希与之同源、确定性生成、无需持久化回填。
    """
    if not path:
        return _DEFAULT_ID
    norm = os.path.normpath(path)
    if active_env_dir and norm == os.path.normpath(active_env_dir):
        return _DEFAULT_ID
    return hashlib.md5(norm.encode("utf-8")).hexdigest()[:8]


def _source_for(path: str) -> Optional[dict]:
    """查某根目录在 sources 表里的行（含 root_id/root_path/name/templates）。无则 None。

    惰性 import：logdir_store.get_source_by_path 反向 import 本模块的 get_root_id，
    模块级 import 会成环。get_source 内部自检 DB 未就绪则返回 None。
    """
    if not path:
        return None
    try:
        import utils.logdir_store as lds
    except Exception:  # noqa: BLE001
        return None
    try:
        return lds.get_source(get_root_id(path))
    except Exception:  # noqa: BLE001
        return None


def get_path_name(path: str) -> str:
    """某根目录的展示名称；未登记（含活跃目录）回退 "default"。"""
    if not path:
        return _DEFAULT_NAME
    src = _source_for(path)
    if src:
        return (src.get("name") or "").strip() or _DEFAULT_NAME
    return _DEFAULT_NAME


def get_path_templates(path: str) -> list:
    """某根目录登记的层级模板 list；未登记返回 []（调用方回退默认模板）。"""
    src = _source_for(path)
    if src:
        return src.get("templates") or []
    return []


def _is_ancestor(anc: str, desc: str) -> bool:
    """anc 是否为 desc 的严格祖先目录（两者均须已 normpath）。

    用「加分隔符做前缀」判定，避免 `/a/details` 误判为 `/a/details2` 的祖先。
    相等不算祖先（相等由调用方的「路径已存在」分支单独处理）。
    """
    if anc == desc:
        return False
    return desc.startswith(anc + os.sep)


def _list_sources(active_env_dir: str = "", db_dir: str = "") -> List[dict]:
    """读 sources 表全量行（含 root_id/root_path/name/templates）。DB 未就绪返回 []。

    db_dir 传入时按指定目录初始化（脚本用）；否则用已初始化的默认连接。
    """
    try:
        import utils.logdir_store as lds
        if db_dir:
            lds.init_db(db_dir)
        elif not lds._ready():
            return []
        return lds.list_sources(active_env_dir)
    except Exception:  # noqa: BLE001
        return []


def get_stats_roots(active_env_dir: Optional[str] = None) -> List[str]:
    """返回参与统计/预览的所有根目录，去重。

    active_env_dir: 当前进程的活跃 env_dir（logs_all/{env-key}）；
                    传入以确保活跃目录排在第一位且不与历史重复。
    DB 为空（新机器未添加源）时仅返回活跃目录。
    """
    roots: List[str] = []
    seen = set()

    def _add(p: str):
        if not p:
            return
        key = os.path.normpath(p)
        if key in seen:
            return
        seen.add(key)
        roots.append(p)

    if active_env_dir:
        _add(active_env_dir)
    for src in _list_sources(active_env_dir or ""):
        _add(src.get("root_path") or "")
    return roots


def get_registered_roots(active_env_dir: Optional[str] = None, *, db_dir: str = "") -> List[str]:
    """返回所有已登记数据源的根目录，去重保序。

    以 DB sources 表为准（活跃行 root_id='default' 排第一）。DB 为空返回 []。
    """
    roots: List[str] = []
    seen = set()

    def _add(p: str):
        if not p:
            return
        key = os.path.normpath(p)
        if key in seen:
            return
        seen.add(key)
        roots.append(p)

    for src in _list_sources(active_env_dir or "", db_dir=db_dir):
        _add(src.get("root_path") or "")
    return roots


def get_root_by_id(root_id: str, active_env_dir: Optional[str] = None) -> Optional[str]:
    """按 root_id 反查根目录路径；找不到返回 None。

    候选集合 = 活跃 env_dir + sources 表登记路径（与 get_stats_roots 一致）。
    """
    if not root_id:
        return None
    for root in get_stats_roots(active_env_dir):
        if get_root_id(root, active_env_dir) == root_id:
            return root
    return None


def dir_size(path: str) -> int:
    """递归统计目录字节数。大目录可能较慢，调用方自行控制频率。"""
    total = 0
    try:
        for dirpath, _dirnames, filenames in os.walk(path):
            for fn in filenames:
                fp = os.path.join(dirpath, fn)
                try:
                    total += os.path.getsize(fp)
                except OSError:
                    continue
    except OSError:
        pass
    return total


def human_size(nbytes: int) -> str:
    val = float(nbytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if val < 1024 or unit == "TB":
            if unit == "B":
                return f"{int(val)}{unit}"
            return f"{val:.1f}{unit}"
        val /= 1024
    return f"{val:.1f}TB"
