from __future__ import annotations

from typing import Any, Iterator


def extract_signatures(value: Any) -> Iterator[dict]:
    def walk(node: Any, path: str, message_index: int | None = None):
        if isinstance(node, dict):
            signature = node.get("signature")
            if isinstance(signature, str) and signature:
                thinking = node.get("thinking") or node.get("reasoning_content")
                yield {
                    "block_path": path,
                    "message_index": message_index,
                    "signature": signature,
                    "original_thinking": thinking if isinstance(thinking, str) else None,
                }
            for key, child in node.items():
                if key.lower() in {"header", "headers", "authorization", "x-api-key", "api-key"}:
                    continue
                child_index = message_index
                if path == "$.messages" and isinstance(child, list):
                    child_index = None
                yield from walk(child, f"{path}.{key}", child_index)
        elif isinstance(node, list):
            for index, child in enumerate(node):
                current = index if path == "$.messages" else message_index
                yield from walk(child, f"{path}[{index}]", current)

    yield from walk(value, "$")
