from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class LoadedPrompt:
    name: str
    instruction: str
    unrelated_thinking: str
    tool: dict
    sha256: str
    loaded_at: str


def load_prompt(root: Path, method: str) -> LoadedPrompt:
    if method not in {"bulk", "sentence"}:
        raise ValueError("method must be bulk or sentence")
    root = root.resolve()
    path = (root / f"{method}.json").resolve()
    if path.parent != root:
        raise ValueError("invalid prompt path")
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise ValueError(f"Reflection Prompt 不可用: {path.name}") from exc
    if not raw:
        raise ValueError(f"Reflection Prompt 为空: {path.name}")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Reflection Prompt JSON 无效: {path.name}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Reflection Prompt 必须是 JSON object: {path.name}")
    instruction = value.get("instruction")
    unrelated_thinking = value.get("unrelated_thinking")
    tool = value.get("tool")
    expected_tool = (
        "reflect_on_prior_reasoning_bulk"
        if method == "bulk"
        else "reflect_on_prior_reasoning"
    )
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError(f"Reflection Prompt 缺少 instruction: {path.name}")
    if not isinstance(unrelated_thinking, str) or not unrelated_thinking.strip():
        raise ValueError(f"Reflection Prompt 缺少 unrelated_thinking: {path.name}")
    if not isinstance(tool, dict) or tool.get("name") != expected_tool:
        raise ValueError(f"Reflection Prompt tool 名称无效: {path.name}")
    if not isinstance(tool.get("input_schema"), dict):
        raise ValueError(f"Reflection Prompt 缺少 tool.input_schema: {path.name}")
    return LoadedPrompt(
        name=path.name,
        instruction=instruction.strip(),
        unrelated_thinking=unrelated_thinking.strip(),
        tool=tool,
        sha256=hashlib.sha256(raw.encode()).hexdigest(),
        loaded_at=datetime.now(timezone.utc).isoformat(),
    )
