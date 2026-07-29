from __future__ import annotations

import copy
import re
from typing import Any

# JSONPath 片段：.key 或 [index]，用于把 "$.messages[1].content[0]" 拆成 token 序列
_PATH_TOKEN = re.compile(r"\.([^.\[\]]+)|\[(\d+)\]")


def _resolve(root: Any, path: str) -> dict | None:
    """按 block_path（如 $.messages[1].content[0] 或 $.response.content[0]）定位节点。

    注意：response 侧的 signature 任务 block_path 形如 $.response.content[0]
    （连续点号、中间无括号），必须与 messages 侧同样支持。
    """
    if not path.startswith("$"):
        return None
    node: Any = root
    for m in _PATH_TOKEN.finditer(path[1:]):
        key, idx = m.group(1), m.group(2)
        if key is not None:
            if not isinstance(node, dict) or key not in node:
                return None
            node = node[key]
        else:
            if not isinstance(node, list):
                return None
            i = int(idx)
            if i < 0 or i >= len(node):
                return None
            node = node[i]
    return node if isinstance(node, dict) else None


def _assistant_msg_from_response(resp: Any) -> dict | None:
    """把轨迹文件内嵌的 response 转成一条 assistant 消息（供前端渲染）。

    轨迹文件请求侧只存 messages（不含本次调用的返回），返回存在同级 response 里。
    支持两种格式（参见 [[newapi-resp-dual-format]]）：
      - Anthropic 原生：{"role":"assistant","content":[blocks...], ...}
      - OpenAI：       {"choices":[{"message":{"role":"assistant","content":...}}]}
    仅返回有内容的消息；纯 {status_code}（如报错）或无法识别时返回 None。
    """
    if not isinstance(resp, dict):
        return None
    # OpenAI 格式优先判断（choices 存在即视为 openai）
    choices = resp.get("choices")
    if isinstance(choices, list) and choices:
        msg = choices[0].get("message") if isinstance(choices[0], dict) else None
        if isinstance(msg, dict) and msg.get("content") is not None:
            return {"role": "assistant", "content": msg.get("content"),
                    "tool_calls": msg.get("tool_calls"), "_from_res": True}
        return None
    # Anthropic 原生格式
    content = resp.get("content")
    # 空内容（[] 或 ""）通常是最后一次调用无返回/截断，不追加空气泡
    if content is not None and content != [] and content != "":
        return {"role": "assistant", "content": content, "_from_res": True}
    return None


def append_response_message(merged: dict, raw: dict) -> dict:
    """若 raw 内嵌 response 有内容，则在 merged.messages 末尾追加一条 assistant 回复。

    就地修改并返回 merged。merged 必须是 raw 的副本（merge 已 deepcopy），
    直接从 raw 读 response 避免依赖 merge 是否保留该字段。
    """
    msg = _assistant_msg_from_response(raw.get("response") if isinstance(raw, dict) else None)
    if msg is None:
        return merged
    msgs = merged.get("messages")
    if not isinstance(msgs, list):
        return merged
    # 幂等：避免重复追加（例如缓存/二次合并）
    if msgs and isinstance(msgs[-1], dict) and msgs[-1].get("_from_res"):
        return merged
    msgs.append(msg)
    return merged


def merge(raw: dict, tasks: list[dict], run_id: str) -> dict:
    out = copy.deepcopy(raw)
    for task in tasks:
        block = _resolve(out, task["block_path"])
        if block is None: continue
        block["reflect"] = {
            "status": task.get("latest_status") or task.get("status"),
            "text": task.get("processed_text") or task.get("latest_processed_text"),
            "run_id": task.get("latest_run_id") or run_id,
            "model": task.get("latest_model") or task.get("model"),
            "error": task.get("last_error"),
            "retry_count": task["retry_count"],
            "processed_at": task.get("updated_at"),
        }
    return out
