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
    port = (os.getenv("PROXY_PORT") or "").strip() or "0"
    segment = _env_key_segment()
    return os.path.join("logs", f"port{port}", segment)


def get_log_dir(base_name: str) -> str:
    return os.path.join(base_name, _env_key_segment(), STARTUP_DATE_TAG)


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
