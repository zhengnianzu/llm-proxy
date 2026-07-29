"""
utils/logs_config.py — 多日志路径配置

持久化文件: settings/logs_dirs.yaml

结构:
    active_base: logs_all          # 进程写入的 base，写路径 {active_base}/{env-key}/{hour}
    history:                       # 参与统计/预览的历史根目录（env-key 层或 details 层）
      - /sdc/data/newapi/logs/details
      - /mnt/path1

统计聚合的根 = [活跃 env_dir] + [history 中每个根]，去重。
每个根下面用可变深度扫描找到含 index.jsonl 的叶子目录（见 log_scan.py），
因此同时支持:
  - 本项目格式  {root}/{hour}/index.jsonl        (root = logs_all/env-xxx)
  - new-api 格式 {root}/{day}/{hour}/index.jsonl  (root = .../logs/details)
"""

import os
import threading
from pathlib import Path
from typing import List, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG_PATH = PROJECT_ROOT / "settings" / "logs_dirs.yaml"
_DEFAULT_BASE = "logs_all"

_lock = threading.Lock()


def _config_path() -> Path:
    """数据目录配置文件路径。

    优先环境变量 LOGS_DIRS_CONFIG（由 `app data config` 写入 .cli_state.yaml，
    并在 `app start` 时作为环境变量传给 app.py），相对路径按项目根解析；
    未设置时回退默认 settings/logs_dirs.yaml。
    """
    override = (os.getenv("LOGS_DIRS_CONFIG") or "").strip()
    if override:
        p = Path(override)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return p
    return _DEFAULT_CONFIG_PATH


def _default_active_base() -> str:
    """首次运行时的默认活跃 base：env LOGS_DIR 优先，否则 logs_all。"""
    env_base = (os.getenv("LOGS_DIR") or "").strip()
    return env_base or _DEFAULT_BASE


def _load_raw() -> dict:
    path = _config_path()
    if not path.is_file():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}
    except (OSError, yaml.YAMLError):
        return {}


def _save_raw(data: dict) -> None:
    path = _config_path()
    try:
        os.makedirs(str(path.parent), exist_ok=True)
        tmp = str(path) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
        os.replace(tmp, str(path))
    except OSError:
        pass


def get_config_file() -> str:
    """当前生效的数据目录配置文件绝对路径（供 CLI/UI 显示）。"""
    return str(_config_path())


def get_active_base() -> str:
    """进程写入的 base 名（相对项目根或绝对路径）。"""
    data = _load_raw()
    base = (data.get("active_base") or "").strip()
    return base or _default_active_base()


def get_history_paths() -> List[str]:
    """配置的历史根目录列表（原样返回，去空/去重）。"""
    data = _load_raw()
    raw = data.get("history") or []
    out: List[str] = []
    seen = set()
    for item in raw:
        if not item:
            continue
        p = str(item).strip()
        if not p:
            continue
        key = os.path.normpath(p)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _ensure_config() -> dict:
    """加载配置，缺省字段补齐（不写盘）。"""
    data = _load_raw()
    if "active_base" not in data or not (data.get("active_base") or "").strip():
        data["active_base"] = _default_active_base()
    if not isinstance(data.get("history"), list):
        data["history"] = []
    return data


def add_history_path(path: str) -> tuple[bool, str]:
    """新增历史路径。返回 (成功, 消息)。"""
    p = (path or "").strip()
    if not p:
        return False, "路径不能为空"
    if not os.path.isdir(p):
        return False, f"目录不存在: {p}"

    with _lock:
        data = _ensure_config()
        norm = os.path.normpath(p)
        existing = {os.path.normpath(x) for x in data["history"]}
        if norm in existing:
            return False, "路径已存在"
        # 不允许与活跃 base 的 env_dir 重复（活跃目录本就在统计中）
        data["history"].append(p)
        _save_raw(data)
    return True, "已添加"


def remove_history_path(path: str) -> tuple[bool, str]:
    """移除历史路径（只从配置移除，不删除磁盘文件）。"""
    p = (path or "").strip()
    with _lock:
        data = _ensure_config()
        norm = os.path.normpath(p)
        new_list = [x for x in data["history"] if os.path.normpath(x) != norm]
        if len(new_list) == len(data["history"]):
            return False, "路径不存在"
        data["history"] = new_list
        _save_raw(data)
    return True, "已移除（未删除磁盘文件）"


def set_active_base(base: str) -> tuple[bool, str]:
    """设置活跃 base（重启进程后生效）。"""
    b = (base or "").strip()
    if not b:
        return False, "base 不能为空"
    with _lock:
        data = _ensure_config()
        data["active_base"] = b
        _save_raw(data)
    return True, "已设置（重启后生效）"


def get_stats_roots(active_env_dir: Optional[str] = None) -> List[str]:
    """返回参与统计/预览的所有根目录，去重。

    active_env_dir: 当前进程的活跃 env_dir（logs_all/{env-key}）；
                    传入以确保活跃目录排在第一位且不与历史重复。
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
    for h in get_history_paths():
        _add(h)
    return roots


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
