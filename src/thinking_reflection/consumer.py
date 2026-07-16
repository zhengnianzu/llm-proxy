from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import anthropic


@dataclass(frozen=True)
class ReflectionResult:
    text: str
    sentence_count: int
    tool_input: dict[str, Any]
    model: str | None
    response_id: str | None
    stop_reason: str | None
    usage: dict[str, Any] | None
    raw_response: dict[str, Any] | None
    stream: bool
    max_tokens: int


def sentences_to_reasoning(tool_input: dict[str, Any] | str) -> tuple[str, int]:
    value = _normalize_tool_input(tool_input)
    items: list[dict[str, Any]] = []
    for item in value.get("prior_reasoning_sentences") or []:
        if isinstance(item, str):
            stripped = item.strip()
            item = json.loads(stripped) if stripped.startswith("{") else {
                "round": 1,
                "sentence_index": len(items) + 1,
                "sentence": item,
                "relevance": "irrelevant",
                "rationale": "bare-string tool item",
            }
        if not isinstance(item, dict):
            raise TypeError(f"sentence item must be dict, got {type(item).__name__}")
        items.append(item)
    items.sort(key=lambda item: (int(item.get("round") or 0), int(item.get("sentence_index") or 0)))
    parts = [str(item["sentence"]).strip() for item in items if item.get("sentence") is not None and str(item["sentence"]).strip()]
    return " ".join(parts), len(parts)


def bulk_to_reasoning(tool_input: dict[str, Any] | str) -> tuple[str, int]:
    value = _normalize_tool_input(tool_input)
    text = str(value.get("sentences") or "").strip()
    if not text:
        raise RuntimeError("tool returned empty sentences")
    return text, 1


def _normalize_tool_input(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped.startswith("{"):
            raise TypeError("tool_input string is not a JSON object")
        value = json.loads(stripped)
    if not isinstance(value, dict):
        raise TypeError(f"tool_input must be dict, got {type(value).__name__}")
    return value


def reflect(
    *, endpoint: str, api_key: str, model: str, instruction: str,
    tool: dict[str, Any], unrelated_thinking: str, signature: str,
    thinking: str | None, method: str, timeout: float = 600,
    stream: bool = False, max_tokens: int = 16384,
) -> ReflectionResult:
    if method not in {"bulk", "sentence"}:
        raise ValueError("method must be bulk or sentence")
    expected_tool = "reflect_on_prior_reasoning_bulk" if method == "bulk" else "reflect_on_prior_reasoning"
    if tool.get("name") != expected_tool:
        raise ValueError(f"tool must be {expected_tool}")
    parser = bulk_to_reasoning if method == "bulk" else sentences_to_reasoning
    thinking_text = (thinking or "").strip() or unrelated_thinking.strip()
    if not thinking_text:
        raise ValueError("unrelated_thinking must not be empty")
    messages = [
        {"role": "user", "content": "hello"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": thinking_text, "signature": signature},
                {"type": "text", "text": "Hello"},
            ],
        },
        {"role": "user", "content": instruction},
    ]
    client = anthropic.Anthropic(
        api_key=api_key,
        base_url=endpoint.rstrip("/"),
        max_retries=0,
        timeout=timeout,
    )
    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 1.0,
        "tools": [tool],
        "tool_choice": {"type": "tool", "name": tool["name"]},
        "messages": messages,
    }
    if stream:
        with client.messages.stream(**kwargs) as stream_context:
            response = stream_context.get_final_message()
    else:
        response = client.messages.create(**kwargs)
    block = next((item for item in response.content if item.type == "tool_use"), None)
    if block is None:
        raise RuntimeError(f"no tool_use in response stop={response.stop_reason}")
    tool_input = _normalize_tool_input(block.input)
    text, sentence_count = parser(tool_input)
    if method == "sentence" and sentence_count == 0:
        raise RuntimeError("tool returned zero sentences")
    usage = response.usage.model_dump() if getattr(response, "usage", None) else None
    try:
        raw_response = response.model_dump()
    except Exception:
        raw_response = None
    return ReflectionResult(
        text=text,
        sentence_count=sentence_count,
        tool_input=tool_input,
        model=response.model,
        response_id=response.id,
        stop_reason=response.stop_reason,
        usage=usage,
        raw_response=raw_response,
        stream=stream,
        max_tokens=max_tokens,
    )
