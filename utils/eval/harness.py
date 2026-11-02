"""
utils/eval/harness.py — 跨 harness 的请求适配层

不同 harness 打上来的请求形态差异很大，但「关键帧导出」只关心三件事：

    1. 这条请求的 messages 是什么（用来判上下文连续性）
    2. 某条消息是否每次调用都变（system prompt / 动态注入 → 不能参与比较）
    3. 这条请求是不是压缩、或者是压缩助手自己在跑

其余（响应解析、工具转换、OpenAI/Anthropic 互转）对关键帧导出都不需要——
产物是原始文件透传，不做格式改写。

实测差异（同一时间窗的两类流量）：

    claude_code (lp5i)              openclaw (ODeM)
    ---------------------------------------------------------------
    system: 顶层 block 列表          system: messages[0] 单字符串
      [billing-header, 角色, 长提示]    "You are a personal assistant…"
    tool_result 嵌在 user block 里    role="tool" 独立消息
    assistant.thinking block         assistant.reasoning_content
    tools: 26 个大写名词式            tools: 24-34 个小写动词式（动态子集）
    压缩: msg0 第二 block             压缩: msg0 整段
      "This session is being continued  "The conversation history before
       from a previous conversation      this point was compacted into
       that ran out of context."         the following summary:"

两类流量的共同点：**`role == "system"` 的消息每次都可能变**（lp5i 每轮插
`<total_tokens>` 提示，ODeM 的 system 从 33311 字节涨到 34148）。所以连续性比较
必须跳过所有 system 消息，否则公共前缀直接归零——实测 13→14 那对：含 system 时
前缀 0，剥掉后 44。

注意 ``traces.msg_count`` 比 ``len(req.messages)`` 恒多 1（同步阶段 +1 计入响应），
所有计数一律以 messages 为准。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

__all__ = [
    "HarnessAdapter",
    "ADAPTERS",
    "detect_adapter",
    "get_adapter",
    "message_text",
    "COMPACTION_MARKERS",
]

# 压缩摘要的标志性文本。命中即判为压缩边界（强信号，优先于前缀比对）。
COMPACTION_MARKERS = (
    # Claude Code
    "This session is being continued from a previous conversation that ran out of context",
    "The conversation history before this point was compacted into the following summary",
    # jiuwenswarm 系
    "<memory_block_current>",
    "<memory_block_round>",
    "<memory_block_dialogue>",
    "<recovered_context>",
    "<state_snapshot>",
)

# 压缩助手自己发起的请求（harness 在执行压缩，不是用户对话）。
COMPACTION_ASSISTANT_MARKERS = (
    "You are a context summarization assistant",
    "You are an Execution State Compression Assistant",
)

# harness 每轮动态注入的元数据信封。同一段对话的同一个用户提问，
# 可能这轮带信封、下轮不带，或信封内容不同——比较前必须剥掉。
_DYNAMIC_ENVELOPE_PATTERNS = (
    # OpenClaw: "Sender (untrusted metadata): ```json {...}``` "
    re.compile(r"Sender\s*\([^)]*\)\s*:\s*```json\s*\{[\s\S]*?\}\s*```\s*", re.I),
    re.compile(r"Sender\s*\([^)]*\)\s*:\s*", re.I),
    # new-api: "Conversation info (untrusted metadata): ```json {...}``` "
    re.compile(r"Conversation info\s*\([^)]*\)\s*:\s*```json\s*\{[\s\S]*?\}\s*```\s*", re.I),
    re.compile(r"Conversation info\s*\([^)]*\)\s*:\s*", re.I),
)


def message_text(msg: dict) -> str:
    """把一条消息的 content 压成纯文本，只用于判定。"""
    value = msg.get("content")
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for block in value:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                for key in ("text", "thinking", "content"):
                    val = block.get(key)
                    if isinstance(val, str):
                        parts.append(val)
                        break
        return "\n".join(parts)
    if value is None:
        return ""
    return str(value)


@dataclass(frozen=True)
class HarnessAdapter:
    """一个 harness 的适配规则。所有字段都是纯函数，可安全跨进程传递。"""

    name: str
    # obj -> 请求体；找不到 messages 返回 None
    extract_request: Callable[[dict], Optional[dict]]
    # 消息 -> 用于连续性比较的规范化字符串（应剥掉可变字段如 cache_control）
    canonical: Callable[[dict], str]
    # 该消息是否「每次调用都可能变」，不参与前缀比较
    is_variable: Callable[[dict], bool]
    # 消息 -> 角色（统一成 user/assistant/system/tool）
    role_of: Callable[[dict], str]

    def messages(self, obj: dict) -> Optional[List[dict]]:
        req = self.extract_request(obj)
        if req is None:
            return None
        msgs = req.get("messages")
        return msgs if isinstance(msgs, list) else None

    def tools_key(self, obj: dict) -> str:
        """工具定义的比较键。两请求工具集不同 → 视为边界。"""
        req = self.extract_request(obj) or {}
        raw = req.get("tools")
        if not isinstance(raw, list):
            return ""
        names = []
        for item in raw:
            if isinstance(item, dict):
                fn = item.get("function") if isinstance(item.get("function"), dict) else item
                name = fn.get("name")
                if isinstance(name, str):
                    names.append(name)
        return "|".join(sorted(names))

    def compaction_marker(self, obj: dict) -> Optional[str]:
        """命中的压缩标记（返回标记本身便于审计），没有返回 None。

        注意：压缩摘要会**一直留在历史里**，所以后续所有请求都能命中。
        只看「有没有」会把整段会话判成每步都压缩——判定「是否新发生一次压缩」
        必须由调用方比对前后两条请求的标记是否相同（见 keyframes.classify_boundary）。
        """
        msgs = self.messages(obj)
        if not msgs:
            return None
        # 只看前两条：摘要永远在最前面
        for msg in msgs[:2]:
            text = message_text(msg)
            for mark in COMPACTION_MARKERS:
                if mark in text:
                    return mark
        return None

    def summary_digest(self, obj: dict) -> str:
        """前两条消息的文本摘要指纹。

        用于判断「压缩是否是新发生的」：相邻两条请求若摘要文本一致，说明这段
        摘要早就在历史里，不是本次新增的压缩边界。
        """
        msgs = self.messages(obj)
        if not msgs:
            return ""
        head = "\n".join(message_text(m) for m in msgs[:2])
        return head[:2000]

    def is_compaction_assistant(self, obj: dict) -> bool:
        msgs = self.messages(obj)
        if not msgs:
            return False
        for msg in msgs[:2]:
            text = message_text(msg)
            if any(mark in text for mark in COMPACTION_ASSISTANT_MARKERS):
                return True
        return False


# ── 公共规范化 ────────────────────────────────────────────────────────────


def _strip_dynamic_envelopes(text: str) -> str:
    """剥掉 harness 每轮动态注入的元数据信封。

    OpenClaw 会在同一条 user 消息前插入 ``Sender (untrusted metadata): ```json {...}``` ``，
    内容随轮次/渠道变化。实测同一段对话的相邻两条请求：不剥时公共前缀 0，剥后 41。
    new-api 侧还有对称的 ``Conversation info`` 信封，一并处理。
    """
    for pattern in _DYNAMIC_ENVELOPE_PATTERNS:
        text = pattern.sub("", text)
    return text


def _canonical_message(msg: dict) -> str:
    """剥 reasoning / cache_control / thinking / 动态信封后序列化。

    cache_control 会随轮次搬移到最新 block（实测 Claude Code 每对请求都会搬），
    不剥会把正常的续写全判成断裂。
    """
    body = {k: v for k, v in msg.items() if k not in ("reasoning_content", "cache_control")}
    content = body.get("content")
    if isinstance(content, str):
        body["content"] = _strip_dynamic_envelopes(content)
    elif isinstance(content, list):
        cleaned = []
        for block in content:
            if not isinstance(block, dict):
                cleaned.append(block)
                continue
            if block.get("type") == "thinking":
                continue
            item = {k: v for k, v in block.items() if k != "cache_control"}
            if isinstance(item.get("text"), str):
                item["text"] = _strip_dynamic_envelopes(item["text"])
            cleaned.append(item)
        body["content"] = cleaned
    return json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _role_of(msg: dict) -> str:
    role = msg.get("role")
    return role if isinstance(role, str) else "unknown"


def _is_system(msg: dict) -> bool:
    return _role_of(msg) == "system"


# ── 各 harness 的取请求方式 ────────────────────────────────────────────────


def _request_any(obj: dict) -> Optional[dict]:
    """宽松探测：new-api 合并文件里请求体可能叫 request/req/up_req/body。"""
    for key in ("req", "request", "up_req", "body"):
        val = obj.get(key)
        if isinstance(val, dict) and isinstance(val.get("messages"), list):
            return val
        if isinstance(val, str):
            try:
                parsed = json.loads(val)
            except (ValueError, TypeError):
                continue
            if isinstance(parsed, dict) and isinstance(parsed.get("messages"), list):
                return parsed
    if isinstance(obj.get("messages"), list):
        return obj
    return None


def _request_claude_code(obj: dict) -> Optional[dict]:
    # Claude Code 只认 req：up_req 少一个 system block（billing-header），混用会漏。
    req = _request_any(obj)
    return req


def _request_openclaw(obj: dict) -> Optional[dict]:
    return _request_any(obj)


ADAPTERS: Dict[str, HarnessAdapter] = {
    "claude_code": HarnessAdapter(
        name="claude_code",
        extract_request=_request_claude_code,
        canonical=_canonical_message,
        is_variable=_is_system,
        role_of=_role_of,
    ),
    "openclaw": HarnessAdapter(
        name="openclaw",
        extract_request=_request_openclaw,
        canonical=_canonical_message,
        is_variable=_is_system,
        role_of=_role_of,
    ),
    "generic": HarnessAdapter(
        name="generic",
        extract_request=_request_any,
        canonical=_canonical_message,
        is_variable=_is_system,
        role_of=_role_of,
    ),
}

DEFAULT_ADAPTER = "generic"


def detect_adapter(obj: dict) -> HarnessAdapter:
    """按请求字段指纹自动选 adapter。

    实测指纹：
      claude_code  顶层 system（block 列表）+ thinking + context_management
      openclaw     prompt_cache_key + tool_choice，system 在 messages[0]
    识别不出就退回 generic（宽松探测，行为与旧代码一致）。
    """
    req = _request_any(obj)
    if req is None:
        return ADAPTERS[DEFAULT_ADAPTER]

    has_top_system = "system" in req
    has_thinking = "thinking" in req
    has_ctx_mgmt = "context_management" in req
    has_cache_key = "prompt_cache_key" in req

    if has_top_system and (has_thinking or has_ctx_mgmt):
        return ADAPTERS["claude_code"]
    if has_cache_key and not has_top_system:
        return ADAPTERS["openclaw"]
    if has_top_system:
        return ADAPTERS["claude_code"]
    return ADAPTERS[DEFAULT_ADAPTER]


def get_adapter(name: str) -> HarnessAdapter:
    """按名字取 adapter；"auto" 表示逐条请求自动识别。"""
    if name == "auto":
        return ADAPTERS[DEFAULT_ADAPTER]
    adapter = ADAPTERS.get(name)
    if adapter is None:
        raise ValueError(
            f"unknown harness {name!r}; expected 'auto' or one of {sorted(ADAPTERS)}"
        )
    return adapter
