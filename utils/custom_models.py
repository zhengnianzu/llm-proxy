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
_prefixes: Dict[str, dict] = {}
_loaded = False

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "settings" / "custom_models.yaml"


def load_custom_models(config_path: Optional[str] = None) -> Dict[str, dict]:
    global _models, _prefixes, _loaded
    path = Path(config_path) if config_path else _CONFIG_PATH
    if not path.is_file():
        _models = {}
        _prefixes = {}
        _loaded = True
        return _models
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        raw = data.get("models") or {}
        _models = {}
        for alias, cfg in raw.items():
            if not isinstance(cfg, dict):
                continue
            _models[alias] = {
                "model": cfg.get("model", alias),
                "params": cfg.get("params", {}),
            }
        raw_prefixes = data.get("prefixes") or {}
        _prefixes = {}
        for prefix, cfg in raw_prefixes.items():
            if not isinstance(cfg, dict):
                continue
            # 归一化：命名空间名不含分隔符，匹配时统一按 "<name>:<model>" 处理。
            # 这样 yaml 里写 "THINKING" 或 "THINKING:" 都可以（yaml 会把尾冒号
            # 当作键值分隔符，键实际为 "THINKING"）。
            name = str(prefix).rstrip(":")
            if name:
                _prefixes[name] = {"params": cfg.get("params", {})}
        _loaded = True
        logging.info(
            "custom_models: loaded %d alias(es), %d prefix(es) from %s",
            len(_models), len(_prefixes), path,
        )
    except Exception as e:
        logging.warning("custom_models: failed to load %s: %s", path, e)
        _models = {}
        _prefixes = {}
        _loaded = True
    return _models


def resolve_custom_model(body: dict) -> Optional[str]:
    """检查 body["model"] 是否匹配自定义别名或前缀规则。

    匹配顺序：
      1. models 精确别名（alias -> 指定 model + params）
      2. prefixes 前缀规则（"<PREFIX>:<真实模型名>" -> 剥前缀得 model + params）

    匹配时：替换 model、注入 params（不覆盖客户端已显式传入的同名参数），
    返回实际模型名。不匹配：返回 None，body 不变。
    """
    if not _loaded:
        load_custom_models()

    raw_model = body.get("model")
    if not isinstance(raw_model, str):
        return None

    # 1) 精确别名优先
    if raw_model in _models:
        cfg = _models[raw_model]
        return _apply(body, cfg["model"], cfg.get("params", {}))

    # 2) 前缀规则："<NAME>:<真实模型名>"，分隔符固定为冒号。
    #    最长命名空间优先，避免多前缀歧义。
    name, sep, rest = raw_model.partition(":")
    if sep and rest and name in _prefixes:
        return _apply(body, rest, _prefixes[name].get("params", {}))

    return None


def _apply(body: dict, actual_model: str, params: dict) -> str:
    """替换 model 并注入 params（不覆盖客户端已显式传入的同名参数）。"""
    body["model"] = actual_model
    for key, value in params.items():
        body.setdefault(key, value)
    return actual_model
