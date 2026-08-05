"""
utils/logs_config.py — 多日志路径配置

持久化文件: settings/logs_dirs.yaml

结构:
    active_base: logs_all          # 进程写入的 base，写路径 {active_base}/{env-key}/{hour}
    history:                       # 参与统计/预览的历史根目录（env-key 层或 details 层）
      - path: /sdc/data/newapi/logs/details   # 带名称（新格式）
        name: new-api数据
      - /mnt/path1                            # 纯字符串（旧格式，名称回退 default）

统计聚合的根 = [活跃 env_dir] + [history 中每个根]，去重。
每个根下面用可变深度扫描找到含 index.jsonl 的叶子目录（见 log_scan.py），
因此同时支持:
  - 本项目格式  {root}/{hour}/index.jsonl        (root = logs_all/env-xxx)
  - new-api 格式 {root}/{day}/{hour}/index.jsonl  (root = .../logs/details)
"""

import hashlib
import os
from utils.atomic_write import safe_replace
import threading
from pathlib import Path
from typing import List, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG_PATH = PROJECT_ROOT / "settings" / "logs_dirs.yaml"
_DEFAULT_BASE = "logs_all"
_DEFAULT_NAME = "default"
_DEFAULT_ID = "default"

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
        safe_replace(tmp, str(path))
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


def _entry_path_name(item) -> tuple[str, str]:
    """把一条 history 记录（新格式 dict 或旧格式 str）规整为 (path, name)。"""
    if isinstance(item, dict):
        p = str(item.get("path") or "").strip()
        n = str(item.get("name") or "").strip() or _DEFAULT_NAME
        return p, n
    return str(item or "").strip(), _DEFAULT_NAME


def _entry_templates(item) -> list:
    """把一条 history 记录的 templates 字段规整为 list（多行字符串或 list）。老条目无此字段 → []。"""
    if not isinstance(item, dict):
        return []
    v = item.get("templates")
    if v is None:
        return []
    if isinstance(v, str):
        return [ln.strip() for ln in v.splitlines() if ln.strip()]
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    return []


def get_history_entries() -> List[dict]:
    """历史根目录列表，含名称与层级模板：[{"path","name","templates"}]，去空/去重（按路径）。"""
    data = _load_raw()
    raw = data.get("history") or []
    out: List[dict] = []
    seen = set()
    for item in raw:
        p, n = _entry_path_name(item)
        if not p:
            continue
        key = os.path.normpath(p)
        if key in seen:
            continue
        seen.add(key)
        out.append({"path": p, "name": n, "templates": _entry_templates(item)})
    return out


def get_history_paths() -> List[str]:
    """配置的历史根目录列表（原样返回，去空/去重）。"""
    return [e["path"] for e in get_history_entries()]


def get_path_name(path: str) -> str:
    """某根目录的展示名称；未配置或活跃目录回退 "default"。"""
    if not path:
        return _DEFAULT_NAME
    norm = os.path.normpath(path)
    for e in get_history_entries():
        if os.path.normpath(e["path"]) == norm:
            return e["name"]
    return _DEFAULT_NAME


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


def get_root_by_id(root_id: str, active_env_dir: Optional[str] = None) -> Optional[str]:
    """按 root_id 反查根目录路径；找不到返回 None。

    候选集合 = 活跃 env_dir + 配置的历史路径（与 get_stats_roots 一致）。
    """
    if not root_id:
        return None
    for root in get_stats_roots(active_env_dir):
        if get_root_id(root, active_env_dir) == root_id:
            return root
    return None


def _ensure_config() -> dict:
    """加载配置，缺省字段补齐（不写盘）。"""
    data = _load_raw()
    if "active_base" not in data or not (data.get("active_base") or "").strip():
        data["active_base"] = _default_active_base()
    if not isinstance(data.get("history"), list):
        data["history"] = []
    return data


def _is_ancestor(anc: str, desc: str) -> bool:
    """anc 是否为 desc 的严格祖先目录（两者均须已 normpath）。

    用「加分隔符做前缀」判定，避免 `/a/details` 误判为 `/a/details2` 的祖先。
    相等不算祖先（相等由调用方的「路径已存在」分支单独处理）。
    """
    if anc == desc:
        return False
    return desc.startswith(anc + os.sep)


def _normalize_templates(templates) -> list:
    """把 templates（None / 多行字符串 / list）规整为去空 list，存 YAML。"""
    if templates is None:
        return []
    if isinstance(templates, str):
        return [ln.strip() for ln in templates.splitlines() if ln.strip()]
    if isinstance(templates, list):
        return [str(x).strip() for x in templates if str(x).strip()]
    return []


def add_history_path(path: str, name: str = "", templates=None) -> tuple[bool, str]:
    """新增历史路径（可带名称与层级模板）。返回 (成功, 消息)。

    templates：占位符层级模板，多行字符串或 list（如 ["details/{日6}/{时8}"]）；
    存进 YAML history 条目，供「同步」按其枚举叶子。空则由格式默认模板兜底。
    """
    p = (path or "").strip()
    n = (name or "").strip() or _DEFAULT_NAME
    tpls = _normalize_templates(templates)
    if not p:
        return False, "路径不能为空"
    if not os.path.isdir(p):
        return False, f"目录不存在: {p}"

    with _lock:
        data = _ensure_config()
        norm = os.path.normpath(p)
        existing = {os.path.normpath(_entry_path_name(x)[0]) for x in data["history"]}
        if norm in existing:
            return False, "路径已存在"
        # 拒绝与已有历史路径「嵌套」（互为祖先/后代）：叶子扫描是递归的，
        # 外层 root 会把内层 root 的叶子也扫进来，同一物理叶子被两个源各扫各建、
        # 在 log_dir.db 里存成两条、列表里显示两遍、构建两遍（见 jumper-003 vs -latest）。
        # 让用户改成互不包含的登记，从根上避免重复。
        for x in data["history"]:
            ep, en = _entry_path_name(x)
            enorm = os.path.normpath(ep)
            if _is_ancestor(enorm, norm):
                return False, (f"新路径在已有历史「{en}」（{ep}）之下，会被其递归扫描重复收录。"
                               f"请改登记不与它嵌套的路径，或先移除该历史再登记外层。")
            if _is_ancestor(norm, enorm):
                return False, (f"新路径是已有历史「{en}」（{ep}）的上层，会把它递归收录导致重复。"
                               f"请改登记不与它嵌套的路径，或先移除该历史再登记外层。")
        # 不允许与活跃 base 的 env_dir 重复（活跃目录本就在统计中）
        data["history"].append({"path": p, "name": n, "templates": tpls})
        _save_raw(data)
    return True, "已添加"


def remove_history_path(path: str) -> tuple[bool, str]:
    """移除历史路径（只从配置移除，不删除磁盘文件）。"""
    p = (path or "").strip()
    with _lock:
        data = _ensure_config()
        norm = os.path.normpath(p)
        new_list = [x for x in data["history"]
                    if os.path.normpath(_entry_path_name(x)[0]) != norm]
        if len(new_list) == len(data["history"]):
            return False, "路径不存在"
        data["history"] = new_list
        _save_raw(data)
    return True, "已移除（未删除磁盘文件）"


def get_path_templates(path: str) -> list:
    """某历史根目录登记的层级模板 list；未配置/活跃目录返回 []（调用方回退默认模板）。"""
    if not path:
        return []
    norm = os.path.normpath(path)
    for e in get_history_entries():
        if os.path.normpath(e["path"]) == norm:
            return e.get("templates") or []
    return []


def set_history_name(path: str, name: str) -> tuple[bool, str]:
    """改某历史条目的名字（写 YAML）。路径不存在返回 (False, ...)。"""
    p = (path or "").strip()
    n = (name or "").strip() or _DEFAULT_NAME
    if not p:
        return False, "路径不能为空"
    with _lock:
        data = _ensure_config()
        norm = os.path.normpath(p)
        hit = False
        for x in data["history"]:
            if isinstance(x, dict) and os.path.normpath(str(x.get("path") or "")) == norm:
                x["name"] = n
                hit = True
            elif not isinstance(x, dict) and os.path.normpath(str(x or "")) == norm:
                # 旧格式 str 条目：就地升级为 dict
                idx = data["history"].index(x)
                data["history"][idx] = {"path": p, "name": n, "templates": []}
                hit = True
        if not hit:
            return False, "路径不存在"
        _save_raw(data)
    return True, "已改名"


def set_history_templates(path: str, templates) -> tuple[bool, str]:
    """改某历史条目的层级模板（写 YAML）。路径不存在返回 (False, ...)。"""
    p = (path or "").strip()
    tpls = _normalize_templates(templates)
    if not p:
        return False, "路径不能为空"
    with _lock:
        data = _ensure_config()
        norm = os.path.normpath(p)
        hit = False
        for x in data["history"]:
            if isinstance(x, dict) and os.path.normpath(str(x.get("path") or "")) == norm:
                x["templates"] = tpls
                hit = True
            elif not isinstance(x, dict) and os.path.normpath(str(x or "")) == norm:
                idx = data["history"].index(x)
                data["history"][idx] = {"path": p, "name": _DEFAULT_NAME, "templates": tpls}
                hit = True
        if not hit:
            return False, "路径不存在"
        _save_raw(data)
    return True, "已更新模板"


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
