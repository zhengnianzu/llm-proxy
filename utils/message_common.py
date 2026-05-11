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
)

_SKIP_CLEANED_PREFIXES = (
    "tools:\n",
    "- name:",
    "这是可以使用的tools:",
)

_INTERNAL_REQUEST_PATTERNS = [
    re.compile(r"generate a short 1-2 word filename slug", re.IGNORECASE),
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
    return sum(1 for m in messages if m.get("role") == "user")


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


def parse_response(res_data: dict) -> dict:
    """统一解析 res.json，返回规范化的 response 对象。"""
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
