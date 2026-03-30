"""
core/config.py — 配置加载

优先级（高→低）:
  1. 环境变量 MANAGE_SESSION_CONFIG（指定 config 文件路径）
  2. BASE_DIR/config.yaml
  3. 内置默认值
"""

import os
from pathlib import Path
from typing import Any

try:
    import yaml
    _YAML_OK = True
except ImportError:
    _YAML_OK = False

BASE_DIR = Path(__file__).parent.parent

_DEFAULTS: dict = {
    "dirs": {
        "tasks":      "data/tasks",
        "raw_index":  "data/raw_index",
        "manifests":  "output/manifests",
        "pair_cache": "output/pair_cache",
        "views":      "output/views",
    },
    "web": {
        "host": "127.0.0.1",
        "port": 8081,
    },
    "matching": {
        "task_query_field":  "query",
        "index_query_field": "q1",
        "match_type":        "exact",
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_yaml(path: Path) -> dict:
    if not _YAML_OK:
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load() -> dict:
    env_path = os.environ.get("MANAGE_SESSION_CONFIG")
    cfg_path = Path(env_path) if env_path else BASE_DIR / "config.yaml"
    if cfg_path.exists():
        return _deep_merge(_DEFAULTS, _load_yaml(cfg_path))
    return dict(_DEFAULTS)


_cfg: dict = _load()


def get(section: str, key: str) -> Any:
    return _cfg.get(section, {}).get(key)


def reload():
    """重新加载配置（用于测试或热重载）"""
    global _cfg
    _cfg = _load()


# ── 路径快捷方式 ──────────────────────────────────────────────────────────────

def dir_path(name: str) -> Path:
    """返回 dirs.<name> 对应的绝对路径，并确保目录存在。"""
    rel = _cfg["dirs"][name]
    p = Path(rel) if Path(rel).is_absolute() else BASE_DIR / rel
    p.mkdir(parents=True, exist_ok=True)
    return p


def tasks_dir()      -> Path: return dir_path("tasks")
def raw_index_dir()  -> Path: return dir_path("raw_index")
def manifests_dir()  -> Path: return dir_path("manifests")
def pair_cache_dir() -> Path: return dir_path("pair_cache")
def views_dir()      -> Path: return dir_path("views")
def reports_dir()    -> Path: return dir_path("reports")


def web_host()          -> str:  return _cfg["web"]["host"]
def web_port()          -> int:  return int(_cfg["web"]["port"])
def web_templates_dir() -> Path: return BASE_DIR / "web" / "templates"
def web_static_dir()    -> Path: return BASE_DIR / "web" / "static"
