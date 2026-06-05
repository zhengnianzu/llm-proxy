"""
Key 配置管理。
优先读 key_state.yaml（由 key config 写入），fallback 到 settings/keys.yaml。
"""

import os
import yaml

_state_path: str = ""
_DEFAULT_CONFIG = "settings/keys.yaml"


def init_key_config(service_log_dir: str):
    global _state_path
    _state_path = os.path.join(service_log_dir, "key_state.yaml")


def get_state_path() -> str:
    return _state_path


def load_key_state() -> dict:
    """读取配置。优先 key_state.yaml，fallback settings/keys.yaml。"""
    for path in (_state_path, _DEFAULT_CONFIG):
        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                return _normalize(cfg)
            except Exception:
                continue
    return _defaults()


def _defaults() -> dict:
    return {"user": "", "password": "", "invite_codes": [], "key_len": 24, "keys": []}


def _normalize(cfg: dict) -> dict:
    cfg.setdefault("user", "")
    cfg.setdefault("password", "")
    cfg.setdefault("key_len", 24)
    if not cfg.get("keys"):
        cfg["keys"] = []

    # invite_code 兼容：字符串 -> 列表，列表 -> 原样
    raw_codes = cfg.pop("invite_code", None) or cfg.pop("invite_codes", None) or []
    if isinstance(raw_codes, str):
        raw_codes = [raw_codes] if raw_codes.strip() else []
    cfg["invite_codes"] = [str(c).strip() for c in raw_codes if str(c).strip()]

    if cfg["user"]:
        cfg["user"] = str(cfg["user"]).strip()
    if cfg["password"]:
        cfg["password"] = str(cfg["password"]).strip()
    cfg["key_len"] = max(1, int(cfg["key_len"]))
    return cfg


def apply_config(yaml_path: str, state_path: str) -> dict:
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg = _normalize(cfg)
    os.makedirs(os.path.dirname(state_path), exist_ok=True)
    state = {k: v for k, v in cfg.items() if k != "keys"}
    with open(state_path, "w", encoding="utf-8") as f:
        yaml.dump(state, f, default_flow_style=False, allow_unicode=True)
    return cfg


def validate_invite_code(code: str) -> bool:
    """检查邀请码是否有效。"""
    codes = load_key_state().get("invite_codes", [])
    return code in codes


def validate_static_key(raw_key: str) -> str | None:
    for entry in load_key_state().get("keys") or []:
        if isinstance(entry, dict) and entry.get("value") == raw_key:
            return raw_key
    return None


def get_static_keys() -> list[dict]:
    return load_key_state().get("keys") or []
