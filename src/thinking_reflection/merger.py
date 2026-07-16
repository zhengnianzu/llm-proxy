from __future__ import annotations

import copy
from typing import Any


def _resolve(root: Any, path: str) -> dict | None:
    node = root
    cursor = path[1:]
    while cursor:
        if cursor.startswith("."):
            cursor = cursor[1:]
            key, sep, rest = cursor.partition(".")
            bracket = key.find("[")
            if bracket >= 0:
                rest = key[bracket:] + (("." + rest) if sep else "")
                key = key[:bracket]
            if not isinstance(node, dict) or key not in node:
                return None
            node = node[key]
            cursor = rest if sep or bracket >= 0 else ""
        elif cursor.startswith("["):
            end = cursor.find("]")
            if end < 0 or not isinstance(node, list): return None
            try: node = node[int(cursor[1:end])]
            except (ValueError, IndexError): return None
            cursor = cursor[end + 1:]
        else:
            return None
    return node if isinstance(node, dict) else None


def merge(raw: dict, tasks: list[dict], run_id: str) -> dict:
    out = copy.deepcopy(raw)
    for task in tasks:
        block = _resolve(out, task["block_path"])
        if block is None: continue
        block["reflect"] = {
            "status": task["status"], "text": task.get("processed_text"), "run_id": run_id,
            "model": task.get("model"), "error": task.get("last_error"),
            "retry_count": task["retry_count"], "processed_at": task.get("updated_at"),
        }
    return out
