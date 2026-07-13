"""
utils/message_common.py — 消息解析共用层

app.py（在线）和 chat-log-viewer（离线）共用的核心逻辑：
- 消息提取、Q1 提取（含 session marker 跳过 + 噪声清理）
- SSE 流式响应重建
- 统一响应解析
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


_SESSION_MARKER_PREFIXES = (
    "A new session was started via /new",
    "A new session was started via /reset",
)

_SKIP_PREFIXES = (
    "[Startup context loaded by runtime]",
    "Bootstrap files like SOUL.md",
    "System (untrusted)",
    "(session bootstrap)",
    "A new session was started via /new",
    "A new session was started via /reset",
)

_SKIP_CLEANED_PREFIXES = (
    "tools:\n",
    "- name:",
    "这是可以使用的tools:",
)

_INTERNAL_REQUEST_PATTERNS = [
    re.compile(r"generate a short 1-2 word filename slug", re.IGNORECASE),
    re.compile(r"^Based on this conversation", re.IGNORECASE),
]

_NOISE_PATTERNS = [
    re.compile(r"^\s*\[[^\]]*\]\s*"),
    re.compile(
        r"^Sender\s*(?:\([^)]*\))?:\s*```json\s*\{[\s\S]*?\}\s*```\s*",
        re.IGNORECASE,
    ),
    re.compile(r"^Sender\s*(?:\([^)]*\))?\s*:\s*", re.IGNORECASE),
    re.compile(r"^System\s*\([^)]*\)\s*:\s*", re.IGNORECASE),
    re.compile(r"^下面就是新任务[^：:]*[：:]\s*"),
    re.compile(r"^任务名[：:][^，,]*[，,]\s*内容[：:]\s*"),
]

_COMPACTION_GOAL_RE = re.compile(
    r"## Goal\s*\n(.*?)(?:\n##|\n</summary>|\Z)", re.DOTALL
)
_CONVERSATION_USER_RE = re.compile(
    r"\[User\]:\s*(.*?)(?:\[Assistant\]|\Z)", re.DOTALL
)


def load_json_safe(path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def extract_messages(data: Any) -> Optional[List[dict]]:
    """从请求体中提取 messages 列表。"""
    if isinstance(data, dict) and isinstance(data.get("messages"), list):
        return data["messages"]
    if isinstance(data, list) and data and isinstance(data[0], dict) and "role" in data[0]:
        return data
    return None


def count_user_messages(messages: List[dict]) -> int:
    # OpenAI 格式中工具结果的 role 是 "tool"（而非 Anthropic 的 "user"），也需计入以确保聚合正确
    return sum(1 for m in messages if m.get("role") in ("user", "tool"))


def count_real_user_turns(messages: List[dict]) -> int:
    """计算真实用户轮次数（排除噪声消息）。

    跳过 session marker、bootstrap context、tool definitions、
    internal requests 等非真实用户输入的 user 消息。
    用于判断请求属于第几轮对话。
    """
    if not messages:
        return 0

    first_text = get_text_from_content(messages[0].get("content", ""))
    is_new_session = any(first_text.startswith(p) for p in _SESSION_MARKER_PREFIXES)
    start = 1 if is_new_session else 0

    count = 0
    for msg in messages[start:]:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            if not any(
                isinstance(b, dict) and b.get("type") == "text"
                for b in content
            ):
                continue
        raw_text = get_text_from_content(content)

        if any(raw_text.startswith(p) for p in _SKIP_PREFIXES):
            continue
        if any(pat.search(raw_text) for pat in _INTERNAL_REQUEST_PATTERNS):
            continue

        cleaned = raw_text.strip()
        if cleaned and any(cleaned.startswith(p) for p in _SKIP_CLEANED_PREFIXES):
            continue

        count += 1
    return count


def get_text_from_content(content) -> str:
    """从 message content 提取纯文本，拼接所有 text block。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif "text" in block:
                    parts.append(block["text"])
        if parts:
            return "\n".join(p for p in parts if p)
        if content:
            return str(content[0])[:500]
        return ""
    return str(content)[:500]


def _strip_noise(text: str) -> str:
    """循环去除已知噪声前缀，直到稳定。"""
    while True:
        prev = text
        for pat in _NOISE_PATTERNS:
            text = pat.sub("", text)
        if text == prev:
            break
    return text.strip()


def get_first_user_text(messages: List[dict], return_index: bool = False) -> Union[str, Tuple[str, int]]:
    """统一 Q1 提取：跳过 session marker + 循环噪声清理。

    处理 OpenClaw 特有的噪声模式：
    - session marker (/new, /reset)
    - [Startup context...] / Bootstrap files...
    - Sender (untrusted metadata): ```json...```
    - [timestamp] 前缀
    - compaction summary → 提取 ## Goal
    - <conversation> wrapper → 提取 [User]: 内容
    - filename slug 内部请求 → 标记为空

    return_index=True 时返回 (text, msg_index) 元组。
    """
    if not messages:
        return ("", 0) if return_index else ""

    first_text = get_text_from_content(messages[0].get("content", ""))
    is_new_session = any(first_text.startswith(p) for p in _SESSION_MARKER_PREFIXES)

    start = 1 if is_new_session else 0

    for i, msg in enumerate(messages[start:], start):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            if not any(
                isinstance(b, dict) and b.get("type") == "text"
                for b in content
            ):
                continue
        raw_text = get_text_from_content(content)

        if any(raw_text.startswith(p) for p in _SKIP_PREFIXES):
            continue

        if any(pat.search(raw_text) for pat in _INTERNAL_REQUEST_PATTERNS):
            continue

        if "conversation history before this point was compacted" in raw_text:
            m = _COMPACTION_GOAL_RE.search(raw_text)
            if m:
                goal = m.group(1).strip()
                if goal:
                    return (goal[:500], i) if return_index else goal[:500]
            continue

        if raw_text.startswith("<conversation>"):
            m = _CONVERSATION_USER_RE.search(raw_text)
            if m:
                inner = m.group(1).strip()
                inner_cleaned = _strip_noise(inner)
                if inner_cleaned and not any(inner_cleaned.startswith(p) for p in _SKIP_PREFIXES):
                    return (inner_cleaned, i) if return_index else inner_cleaned
            continue

        cleaned = _strip_noise(raw_text)
        if cleaned:
            return (cleaned, i) if return_index else cleaned

    return ("", 0) if return_index else ""


def build_chain_key(messages: List[dict]) -> str:
    """构建聚合用的 chain_key：直接使用 Q1 文本。"""
    return get_first_user_text(messages)[:500]


def parse_streaming_response(chunks: List[dict]) -> dict:
    """从 anthropic_passthrough_sse_capture 的 chunks 中重建完整 message。"""
    message: dict = {}
    blocks: dict = {}
    block_json_buf: dict = {}

    for chunk in chunks:
        t = chunk.get("type")

        if t == "message_start":
            message = dict(chunk.get("message", {}))
            message["content"] = []

        elif t == "content_block_start":
            idx = chunk.get("index", 0)
            cb = dict(chunk.get("content_block", {}))
            blocks[idx] = cb
            if cb.get("type") == "tool_use":
                block_json_buf[idx] = ""

        elif t == "content_block_delta":
            idx = chunk.get("index", 0)
            delta = chunk.get("delta", {})
            dtype = delta.get("type")
            if dtype == "text_delta":
                blocks.setdefault(idx, {"type": "text", "text": ""})
                blocks[idx]["text"] = blocks[idx].get("text", "") + delta.get("text", "")
            elif dtype == "thinking_delta":
                blocks.setdefault(idx, {"type": "thinking", "thinking": ""})
                blocks[idx]["thinking"] = blocks[idx].get("thinking", "") + delta.get("thinking", "")
            elif dtype == "signature_delta":
                blocks.setdefault(idx, {"type": "thinking", "thinking": "", "signature": ""})
                blocks[idx]["signature"] = blocks[idx].get("signature", "") + delta.get("signature", "")
            elif dtype == "input_json_delta":
                block_json_buf[idx] = block_json_buf.get(idx, "") + delta.get("partial_json", "")

        elif t == "content_block_stop":
            idx = chunk.get("index", 0)
            if idx in block_json_buf:
                try:
                    blocks[idx]["input"] = json.loads(block_json_buf[idx])
                except json.JSONDecodeError:
                    blocks[idx]["input"] = block_json_buf[idx]

        elif t == "message_delta":
            delta = chunk.get("delta", {})
            if "stop_reason" in delta:
                message["stop_reason"] = delta["stop_reason"]
            if "stop_sequence" in delta:
                message["stop_sequence"] = delta["stop_sequence"]
            if "usage" in chunk:
                base_usage = dict(message.get("usage") or {})
                base_usage.update(chunk["usage"])
                message["usage"] = base_usage

    if blocks:
        message["content"] = [blocks[i] for i in sorted(blocks)]

    return message


def parse_streaming_response_content(chunks: List[dict]) -> Optional[list]:
    """SSE 重建后只返回 content 列表（兼容 app.py 原有接口）。"""
    msg = parse_streaming_response(chunks)
    content = msg.get("content")
    return content if content else None


def extract_res_usage(res_data: dict) -> Optional[dict]:
    """从 res.json 提取 usage 信息，支持 Anthropic/OpenAI 的流式和非流式格式。"""
    if not res_data:
        return None

    rtype = res_data.get("type", "")

    if rtype == "anthropic_passthrough_sse_capture":
        chunks = res_data.get("chunks", [])
        msg = parse_streaming_response(
            [c for c in chunks if c.get("type") != "anthropic_passthrough_sse_meta"]
        )
        return msg.get("usage") or None

    if rtype == "openai_passthrough_sse_capture":
        chunks = res_data.get("chunks", [])
        msg = parse_openai_streaming_response(chunks)
        return msg.get("usage") or None

    body = res_data.get("json") or {}
    return body.get("usage") or None


def parse_openai_streaming_response(chunks: List[dict]) -> dict:
    """从 openai_passthrough_sse_capture 的 chunks 中重建完整 message。

    OpenAI SSE delta 结构: choices[0].delta.{role, content, tool_calls}
    """
    message: dict = {"role": "assistant", "content": ""}
    tool_calls_buf: dict = {}
    usage: dict = {}

    for chunk in chunks:
        if chunk.get("type") in ("openai_passthrough_sse_meta",):
            continue

        choices = chunk.get("choices", [])
        if not choices:
            if chunk.get("usage"):
                usage.update(chunk["usage"])
            continue

        choice = choices[0]
        delta = choice.get("delta", {})

        if delta.get("role"):
            message["role"] = delta["role"]

        if delta.get("content"):
            message["content"] = (message.get("content") or "") + delta["content"]

        if delta.get("reasoning_content"):
            message["reasoning_content"] = (message.get("reasoning_content") or "") + delta["reasoning_content"]

        if delta.get("tool_calls"):
            for tc in delta["tool_calls"]:
                idx = tc.get("index", 0)
                if idx not in tool_calls_buf:
                    tool_calls_buf[idx] = {
                        "id": tc.get("id", ""),
                        "type": tc.get("type", "function"),
                        "function": {"name": "", "arguments": ""},
                    }
                buf = tool_calls_buf[idx]
                fn = tc.get("function", {})
                if fn.get("name"):
                    buf["function"]["name"] = fn["name"]
                if fn.get("arguments"):
                    buf["function"]["arguments"] += fn["arguments"]
                if tc.get("id"):
                    buf["id"] = tc["id"]

        finish_reason = choice.get("finish_reason")
        if finish_reason:
            message["finish_reason"] = finish_reason

        if chunk.get("usage"):
            usage.update(chunk["usage"])

        if chunk.get("model"):
            message["model"] = chunk["model"]

    if tool_calls_buf:
        message["tool_calls"] = [tool_calls_buf[i] for i in sorted(tool_calls_buf)]

    if usage:
        message["usage"] = usage

    return message


def _detect_provider(res_data: dict) -> str:
    """从 res.json 的内容判断 provider 类型。"""
    rtype = res_data.get("type", "")
    if rtype == "anthropic_passthrough_sse_capture":
        return "anthropic"
    if rtype == "openai_passthrough_sse_capture":
        return "openai"
    if rtype == "responses_passthrough_sse_capture":
        return "responses"

    body = res_data.get("json") or {}
    if body.get("type") == "message":
        return "anthropic"
    if body.get("object") == "chat.completion":
        return "openai"
    if body.get("object") == "response":
        return "responses"
    if "choices" in body:
        return "openai"
    if isinstance(body.get("content"), list):
        return "anthropic"
    return "unknown"


def _normalize_openai_response(body: dict) -> dict:
    """将 OpenAI 响应结构规范化为统一格式（content 为列表）。"""
    msg = {}
    choices = body.get("choices", [])
    if choices:
        msg = dict(choices[0].get("message") or choices[0].get("delta") or {})

    content_blocks = []
    if msg.get("reasoning_content"):
        content_blocks.append({"type": "thinking", "thinking": msg["reasoning_content"]})
    text = msg.get("content")
    if text:
        content_blocks.append({"type": "text", "text": text})

    if msg.get("tool_calls"):
        for tc in msg["tool_calls"]:
            fn = tc.get("function", {})
            try:
                inp = json.loads(fn.get("arguments", "{}"))
            except (json.JSONDecodeError, TypeError):
                inp = fn.get("arguments", "")
            content_blocks.append({
                "type": "tool_use",
                "id": tc.get("id", ""),
                "name": fn.get("name", ""),
                "input": inp,
            })

    result = {
        "role": msg.get("role", "assistant"),
        "content": content_blocks,
        "model": body.get("model", ""),
    }
    if msg.get("finish_reason") or (choices and choices[0].get("finish_reason")):
        result["stop_reason"] = msg.get("finish_reason") or choices[0].get("finish_reason")
    if body.get("usage"):
        u = body["usage"]
        result["usage"] = {
            "input_tokens": u.get("prompt_tokens", 0),
            "output_tokens": u.get("completion_tokens", 0),
        }
    return result


def parse_response(res_data: dict) -> dict:
    """统一解析 res.json，返回规范化的 response 对象。

    支持三种格式：Anthropic / OpenAI / Responses API
    判断依据：res.json 的 type 标签（流式）或 json.object / json.type（非流式）
    """
    provider = _detect_provider(res_data)

    if provider == "anthropic":
        rtype = res_data.get("type")
        if rtype == "anthropic_passthrough_sse_capture":
            chunks = res_data.get("chunks", [])
            status_code = 200
            for c in chunks:
                if c.get("type") == "anthropic_passthrough_sse_meta":
                    status_code = c.get("status_code", 200)
                    break
            msg = parse_streaming_response(
                [c for c in chunks if c.get("type") != "anthropic_passthrough_sse_meta"]
            )
            return {"status_code": status_code, **msg}
        else:
            body = dict(res_data.get("json") or {})
            body["status_code"] = res_data.get("status_code", 200)
            return body

    if provider == "openai":
        rtype = res_data.get("type")
        if rtype == "openai_passthrough_sse_capture":
            chunks = res_data.get("chunks", [])
            status_code = 200
            for c in chunks:
                if c.get("type") == "openai_passthrough_sse_meta":
                    status_code = c.get("status_code", 200)
                    break
            msg = parse_openai_streaming_response(
                [c for c in chunks if c.get("type") != "openai_passthrough_sse_meta"]
            )
            normalized = _normalize_openai_response({"choices": [{"message": msg}], "usage": msg.pop("usage", None), "model": msg.pop("model", "")})
            normalized["status_code"] = status_code
            return normalized
        else:
            body = res_data.get("json") or {}
            normalized = _normalize_openai_response(body)
            normalized["status_code"] = res_data.get("status_code", 200)
            return normalized

    # responses / unknown — 直接返回 json body
    body = dict(res_data.get("json") or {})
    body["status_code"] = res_data.get("status_code", 200)
    return body
