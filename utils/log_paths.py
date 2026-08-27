import os
import re
from datetime import datetime
from pathlib import Path
from typing import Iterator

INDEX_FILENAME = "index.jsonl"
STARTUP_DATE_TAG = datetime.now().strftime("%y%m%d%H")


def _first_configured_upstream_key() -> str:
    raw = (os.getenv("UPSTREAM_API_KEY") or "").strip()
    if not raw:
        return ""
    for part in raw.split(","):
        part = part.strip()
        if part:
            return part
    return ""


def get_upstream_key_prefix() -> str:
    key = _first_configured_upstream_key()
    if not key:
        return "nokey"
    prefix = key[-4:]
    prefix = re.sub(r"[^A-Za-z0-9_-]", "_", prefix)
    return prefix or "nokey"


def get_log_task_tag() -> str:
    raw = (os.getenv("LOG_TASK_TAG") or "").strip()
    if not raw:
        return ""
    tag = re.sub(r"[^A-Za-z0-9_-]", "-", raw)
    tag = re.sub(r"-+", "-", tag).strip("-_")
    return tag


def _env_key_segment() -> str:
    task_tag = get_log_task_tag()
    key_prefix = get_upstream_key_prefix()
    if task_tag:
        return f"{task_tag}-{key_prefix}"
    return key_prefix


def get_service_log_dir() -> str:
    """服务实例目录（logs/port<P>/<segment>），导出/会话/备份等 DB 都落在它下面。

    可用 env LOG_DIR 直接覆盖为绝对路径 —— 这是把「路径真相」交给 .env 的逃生通道：
    .env 里定义 LOG_DIR=/root/llm-proxy-main/logs/port8084/env-99oR 后，
    无论 LOG_TASK_TAG/PROXY_PORT/UPSTREAM_API_KEY 是否在 launcher 环境里一致，
    都指向同一个目录，避免 ad-hoc 进程解析到已停用的旧实例（如 99oR）。
    """
    override = (os.getenv("LOG_DIR") or "").strip()
    if override:
        return override
    port = (os.getenv("PROXY_PORT") or "").strip() or "0"
    segment = _env_key_segment()
    return os.path.join("logs", f"port{port}", segment)


def _resolve_base(base_name: str) -> str:
    """logs_all 的 base 可由 env LOGS_DIR / DATA_DIR 覆盖（两者都是完整目录，含 segment）。
    其它 base（logs_anthropic/logs_openai 等）保持原样。
    LOGS_DIR 与 DATA_DIR 区别：LOGS_DIR 是「当前小时写入目录」，DATA_DIR 是「env 根目录」。
    get_log_dir 只追加小时段；若 base 已含 segment，调用方需自行保证不再追加。"""
    if base_name != "logs_all":
        return base_name
    try:
        from utils.logs_config import get_active_base
        return get_active_base() or base_name
    except Exception:
        return base_name


def get_log_dir(base_name: str) -> str:
    """当前小时写入目录：{base}/{segment}/{hour}。

    当 base 由 LOGS_DIR 显式给定为「小时目录」时（已含 segment + 小时），直接返回 base，
    不再追加；当 base 由 DATA_DIR 给定为「env 根目录」时（已含 segment，不含小时），
    只追加小时段。未覆盖时走原始推导。
    """
    base = _resolve_base(base_name)
    logs_dir_override = (os.getenv("LOGS_DIR") or "").strip()
    if base_name == "logs_all" and logs_dir_override:
        # LOGS_DIR 已被显式指定为最终小时目录：不追加，原样返回。
        return logs_dir_override
    data_dir_override = (os.getenv("DATA_DIR") or "").strip()
    if base_name == "logs_all" and data_dir_override:
        # DATA_DIR 是 env 根目录（含 segment），只需追加小时段。
        return os.path.join(data_dir_override, STARTUP_DATE_TAG)
    return os.path.join(base, _env_key_segment(), STARTUP_DATE_TAG)


def build_index_path(log_dir: str) -> str:
    return os.path.join(log_dir, INDEX_FILENAME)


def iter_matching_log_dirs(base_name: str, root: str = ".") -> Iterator[Path]:
    root_path = Path(root)
    current_dir = root_path / get_log_dir(base_name)
    if current_dir.is_dir():
        yield current_dir
        return

    base_dir = root_path / base_name
    if not base_dir.is_dir():
        return

    for env_dir in sorted(base_dir.iterdir()):
        if not env_dir.is_dir():
            continue
        for hour_dir in sorted(env_dir.iterdir()):
            if hour_dir.is_dir():
                yield hour_dir
