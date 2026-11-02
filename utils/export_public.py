"""
utils/export_public.py — 公开导出（/export/*）的模式清单与 IP 绑定清单

两个独立能力，各对应一个端点：

1. **mode_list**：给 /export/mode_list 用，返回可公开使用的**导出模式**清单。
   直接复用 ExportModesConfig.to_public_list()（settings/export_modes.yaml 是单一
   真相源），这里只做「只保留 public=true」的过滤，不维护第二份模式定义。

2. **proxy_list**：给 /export/proxy_list 用，返回当前管理的 **IP → 来源名** 绑定。
   数据源是 log_dir.db 的 sources.ip 列（「数据管理 → 来源与路径」里登记），
   所以页面上改了 IP 这里立刻生效。**纯展示**：不参与任何过滤或鉴权。

settings/export_public.yaml 现在只剩两个用途：
  - 提供一份初始 IP 绑定（init_db 首次迁移时灌进 sources.ip，之后以表为准）；
  - 供 _seed_source_ips() 读取。运行时查询不再读它。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)

_CONFIG_PATH = (Path(__file__).resolve().parent.parent
                / "settings" / "export_public.yaml")

_config: Optional[dict] = None
_config_sig: Optional[Tuple[float, int]] = None


def _load_raw() -> dict:
    """读配置文件，带 mtime+size 缓存。缺失/损坏返回空 dict。"""
    global _config, _config_sig
    try:
        st = _CONFIG_PATH.stat()
        sig: Optional[Tuple[float, int]] = (st.st_mtime, st.st_size)
    except OSError:
        sig = None

    if _config is not None and sig is not None and sig == _config_sig:
        return _config
    if sig is None:
        if _config is not None:
            return _config
        _config, _config_sig = {}, None
        return _config

    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if not isinstance(data, dict):
            raise ValueError("top-level YAML is not a mapping")
    except Exception as exc:  # noqa: BLE001
        logger.warning("export_public: 读取 %s 失败（%s），沿用上次配置/不过滤",
                       _CONFIG_PATH, exc)
        if _config is not None:
            _config_sig = sig
            return _config
        data = {}
    _config, _config_sig = data, sig
    return _config


def get_config(force: bool = False) -> dict:
    """返回 {support_proxy: bool, proxies: {ip: [source_name, ...]}}。

    仅供 _seed_source_ips()（首次迁移把 yaml 灌进 sources.ip）读取。
    proxies 的 value 统一归一化成**列表**：yaml 里既可以写单个名字
    （`"1.2.3.4": jumper`），也可以写列表（`"1.2.3.4": [proxy-004,
    proxy-004-latest]`）。历史原因，同一台机器的数据在 sources 表里可能有
    新旧两种命名，两个都得认。

    support_proxy 字段保留以兼容旧配置；**运行时不再用它做任何过滤**
    （按 host 过滤 root 已移除：key 本身就是最强的定位条件，再过滤只会做减法）。
    """
    global _config, _config_sig
    if force:
        _config_sig = None
    raw = _load_raw()
    raw_proxies = raw.get("proxies")
    if not isinstance(raw_proxies, dict):
        raw_proxies = {}
    proxies: Dict[str, List[str]] = {}
    for ip, val in raw_proxies.items():
        if not ip:
            continue
        if isinstance(val, (list, tuple, set)):
            names = [str(x).strip() for x in val if x and str(x).strip()]
        elif val is None:
            names = []
        else:
            s = str(val).strip()
            names = [s] if s else []
        if names:
            proxies[str(ip)] = names
    return {
        "support_proxy": bool(raw.get("support_proxy", False)),
        "proxies": proxies,
    }


def support_proxy_enabled() -> bool:
    return get_config()["support_proxy"]


def _normalize_host(host: str) -> str:
    """去掉端口 / IPv6 方括号，转小写。"""
    host = (host or "").strip().lower()
    if not host:
        return ""
    # 允许 host 带端口（"115.120.113.66:8084"）或被 IPv6 括号包裹
    if host.startswith("["):
        host = host[1:].split("]", 1)[0]
    elif host.count(":") == 1:
        host = host.split(":", 1)[0]
    return host


def proxy_list() -> List[Dict[str, Any]]:
    """从 sources 表生成 IP → 来源清单（纯展示，不参与任何过滤）。

    返回 [{source, root_id, root_path, ips: [...]}, ...]，只列绑了 IP 的行。
    数据源是 logdir_store.sources.ip（逗号分隔串），因此页面上改了 IP，
    这里立刻跟着变——单一真相源。
    """
    try:
        import utils.logdir_store as lds
        rows = lds.list_sources()
    except Exception:  # noqa: BLE001
        return []
    out: List[Dict[str, Any]] = []
    for r in rows:
        ips = _split_ips(r.get("ip"))
        if not ips:
            continue
        out.append({
            "source": (r.get("name") or "").strip() or "default",
            "root_id": r.get("root_id") or "",
            "root_path": r.get("root_path") or "",
            "ips": ips,
        })
    out.sort(key=lambda x: x["source"])
    return out


def _split_ips(raw) -> List[str]:
    """把 'a,b' 或列表归一化成去重有序的 IP 列表。"""
    if not raw:
        return []
    if isinstance(raw, (list, tuple, set)):
        items = [str(x).strip() for x in raw]
    else:
        items = [s.strip() for s in str(raw).replace(";", ",").split(",")]
    return list(dict.fromkeys([s for s in items if s]))


def mode_list() -> List[Dict[str, Any]]:
    """可公开使用的导出模式清单（只保留 public=true）。

    模式定义来自 settings/export_modes.yaml（单一真相源），这里不重复维护。
    """
    from utils.export_modes import get_config as get_modes

    cfg = get_modes()
    return [m for m in cfg.to_public_list() if m.get("public")]
