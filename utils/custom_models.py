"""
utils/custom_models.py — 自定义模型别名 + 参数注入

从 settings/custom_models.yaml 加载配置。
客户端发送 model="THINKING:claude-opus-4-8" 时，
自动替换为实际模型名并注入预设参数（不覆盖客户端已传的同名参数）。
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

_models: Dict[str, dict] = {}
_loaded = False

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "settings" / "custom_models.yaml"


def load_custom_models(config_path: Optional[str] = None) -> Dict[str, dict]:
    global _models, _loaded
    path = Path(config_path) if config_path else _CONFIG_PATH
    if not path.is_file():
        _models = {}
        _loaded = True
        return _models
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        raw = data.get("models", {})
        _models = {}
        for alias, cfg in raw.items():
            if not isinstance(cfg, dict):
                continue
            _models[alias] = {
                "model": cfg.get("model", alias),
                "params": cfg.get("params", {}),
            }
        _loaded = True
        logging.info("custom_models: loaded %d alias(es) from %s", len(_models), path)
    except Exception as e:
        logging.warning("custom_models: failed to load %s: %s", path, e)
        _models = {}
        _loaded = True
    return _models


def resolve_custom_model(body: dict) -> Optional[str]:
    """检查 body["model"] 是否匹配自定义别名。

    匹配时：替换 model、注入 params，返回实际模型名。
    不匹配：返回 None，body 不变。
    """
    if not _loaded:
        load_custom_models()

    raw_model = body.get("model")
    if not isinstance(raw_model, str) or raw_model not in _models:
        return None

    cfg = _models[raw_model]
    actual_model = cfg["model"]
    body["model"] = actual_model

    for key, value in cfg.get("params", {}).items():
        if key not in body:
            body[key] = value

    return actual_model
