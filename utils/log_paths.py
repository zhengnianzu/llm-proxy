import os
import re
from datetime import datetime
from pathlib import Path
from typing import Iterator

INDEX_FILENAME = "index.jsonl"
STARTUP_DATE_TAG = datetime.now().strftime("%y%m%d")


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


def get_log_dir(base_name: str) -> str:
    return f"{base_name}_{get_upstream_key_prefix()}_{STARTUP_DATE_TAG}"


def build_index_path(log_dir: str) -> str:
    return os.path.join(log_dir, INDEX_FILENAME)


def iter_matching_log_dirs(base_name: str, root: str = ".") -> Iterator[Path]:
    root_path = Path(root)
    current_dir = root_path / get_log_dir(base_name)
    if current_dir.is_dir():
        yield current_dir
        return

    prefix = f"{base_name}_"
    matches = sorted(
        path for path in root_path.iterdir()
        if path.is_dir() and path.name.startswith(prefix)
    )
    for path in matches:
        yield path
