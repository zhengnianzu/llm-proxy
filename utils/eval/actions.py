"""
utils/eval/actions.py — 动作抽取（纯函数，无 IO）

把一条 trace 的 ``messages`` 压成有序的**动作链**，供「动作决策图」渲染。

## 为什么要动作级

会话级的 trace 列表只能回答「哪次 API 调用失败了」。而 ``success`` 的口径是
``token_out > 0``（见 ``utils/newapi_format.parse_combined_file``）——它只说
API 通不通，不说 agent 的动作成不成。

实测（8226910d / 26110718 / 2026-11-07T18:00:46.296）：19 条 trace 严格线性，
msg_count 100→102→…→134 每步 +2，只有最后一条 success=0。真正的试错全在最后
那条 trace 内部：135 条消息、65 次 tool_call，其中 3 次失败
（TypeError canvasToWorld / SyntaxError Unexpected token ',' / VL API 401）。

所以「a1✗ a2✗ a3✓」这种决策过程只有下沉到 action 级才看得见。

## 成败判定

**不能直接复用 ``eval.eval._has_error_keywords``。** 那套规则是为「判断一条工具结果
是否失败」调的，含 ``\\berror\\b`` / ``\\btimed?\\s*out\\b`` / ``\\bexception\\b`` 这类
宽泛匹配，遇到「把源码/日志打印出来」的结果会假阳性——实测 ``sed -n '1,60p' game.py``
的输出里有 ``timeout=30`` 和 ``except Exception:``，被判成了失败（真实的失败只有 3 个，
它给出了 4 个）。

这里改用一组**只认诊断形态**的强信号，按优先级：
  1. ``is_error`` 标记（Anthropic ``tool_result.is_error``）—— 最硬的信号
  2. ``"success": false`` —— 工具自己在结构化结果里报的失败
  3. 异常类型 + 冒号（``TypeError:`` / ``SyntaxError:`` …），类型表沿用
     ``eval.eval._ERROR_PATTERNS`` 里的那份（**不含裸 Exception/Error**，
     所以 ``except Exception:`` 不会误触发）
  4. 行首的诊断词（``error:`` / ``failed:`` / ``fatal:``）
  5. 进程/文件系统级失败短语（command not found / no such file or directory …）

注意：判定输入是**结果全文**，不是预览截断——错误常在很靠后的位置。

## 两种消息形态

  OpenAI 形态     assistant.tool_calls[] + role="tool" 的 tool_call_id
  Anthropic 形态  assistant.content[] 的 tool_use + user.content[] 的 tool_result

同一份请求里两种可以共存（见 doc/README_traj.md §1），故两边都要扫。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from utils.eval.eval import _collect_text

__all__ = ["Action", "extract_actions", "merge_action_chains", "summarize",
           "ACTION_PREVIEW_CHARS", "FAIL_PATTERNS"]

# 预览截断长度。图上一个节点只放得下这么多，完整内容由前端按需再取。
ACTION_PREVIEW_CHARS = 200
RESULT_PREVIEW_CHARS = 240
# 失败动作后找 assistant 自述的回看窗口（消息条数）。见 _attach_repair_hints。
REPAIR_LOOKAHEAD = 16

# 工具自己吐的失败标记：`{"success": false}` / `"success":false`。
# 只匹配布尔 false，避免 `"success": true` 与 `"successful": false` 之类误伤。
_JSON_FAIL_RE = re.compile(r'"success"\s*:\s*false\b')

# 失败强信号。刻意不含裸 `error` / `exception` / `timeout`——见模块 docstring。
FAIL_PATTERNS = (
    # Python 诊断形态：类型名 + 冒号。类型表与 eval._ERROR_PATTERNS 的异常分支一致，
    # 但**不含**裸 Exception/Error，所以源码里的 `except Exception:` 不会误触发。
    re.compile(
        r"\b(SyntaxError|NameError|TypeError|ValueError|AttributeError|"
        r"ImportError|ModuleNotFoundError|RuntimeError|KeyError|IndexError|"
        r"FileNotFoundError|PermissionError|OSError|IOError|ZeroDivisionError|"
        r"RecursionError|MemoryError|ReferenceError|UnhandledPromiseRejection)"
        r"\s*:"
    ),
    re.compile(r"Traceback \(most recent call last\)", re.I),
    # 行首的诊断词：`error: ...` / `failed: ...` / `fatal: ...` / `FAILED (1.23s)`
    re.compile(r"(?m)^\s*(?:error|failed|fatal|failure)\s*[:：(]", re.I),
    # pytest 风格
    re.compile(r"(?m)^\s*(?:\d+\s+failed|FAILED\s+\S)", re.I),
    # 进程 / 文件系统级失败
    re.compile(r"permission denied", re.I),
    re.compile(r"operation not permitted", re.I),
    re.compile(r"command not found", re.I),
    re.compile(r"no such file or directory", re.I),
    re.compile(r"segmentation fault", re.I),
    re.compile(r"(?m)^\s*killed\b", re.I),
)

# assistant 正文里「我发现了问题、正在修」的自述。用于把「✗ → 下一个 ✓」这对
# 标成「已定位并修复」，纯展示标注，不参与成败判定。
_REPAIR_HINT_RE = re.compile(
    r"(发现一个真实的?\s*bug|真实\s*bug|这是个实际问题|我(?:先|来)?修|修正|修复)"
    r"|(\bbug\b[^。\n]{0,20}(fix|修))|(\bfix(?:ing|ed)?\b)",
    re.I,
)


@dataclass(slots=True)
class Action:
    """一次工具调用的判定视图。"""

    index: int = 0                 # 本 trace 内的序号，0 起
    call_id: str = ""
    name: str = ""                 # exec / write / edit / browser / image_reading / ...
    args_preview: str = ""         # 参数摘要
    ok: bool = True
    fail_kind: str = ""            # "" | "is_error" | "error_keyword" | "json_success_false"
    fail_text: str = ""            # 失败信息首行
    result_preview: str = ""       # 结果摘要
    repair_hint: str = ""          # 后续 assistant 的自述（「发现 bug / 先修」）
    has_result: bool = True        # 有没有配上结果（截断的 trace 可能缺尾部结果）

    def as_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "call_id": self.call_id,
            "name": self.name,
            "args_preview": self.args_preview,
            "ok": self.ok,
            "fail_kind": self.fail_kind,
            "fail_text": self.fail_text,
            "result_preview": self.result_preview,
            "repair_hint": self.repair_hint,
            "has_result": self.has_result,
        }


def _clip(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


def _first_line(text: str, limit: int = 300) -> str:
    for line in (text or "").splitlines():
        line = line.strip()
        if line:
            return _clip(line, limit)
    return ""


def _squeeze(text: str) -> str:
    """把多行压成一行（预览用）。"""
    return re.sub(r"\s+", " ", (text or "").strip())


# 只有进程退出码、没有诊断信息的行。`is_error` 的结果常常以它开头，
# 真正的错误在后面（实测 Claude Code 的 Bash 工具：`Exit code 1` + 完整 traceback）。
_EXIT_ONLY_RE = re.compile(r"^\s*(?:Process )?[Ee]xit(?:ed)? (?:code|status)[:\s]*\d+\s*$")


def _diagnostic_line(text: str) -> str:
    """从工具结果里挑一行最能说明问题的文本。

    优先命中失败强信号的那一行；其次第一条不是「Exit code N」的非空行；
    都没有就返回第一条非空行。
    """
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return ""
    for pattern in FAIL_PATTERNS:
        for line in lines:
            if pattern.search(line):
                return line
    for line in lines:
        if not _EXIT_ONLY_RE.match(line):
            return line
    return lines[0]


def _blocks(content: Any):
    """遍历 content 里的 block（字符串内容当作单个 text block）。"""
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                yield block
    elif isinstance(content, str) and content:
        yield {"type": "text", "text": content}


def _args_text(raw: Any) -> str:
    """工具参数压成一行可读文本。"""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        try:
            return json.dumps(raw, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError):
            return str(raw)
    return "" if raw is None else str(raw)


def _classify(text: str, is_error_flag: bool) -> tuple[bool, str, str]:
    """判一次工具结果成败。返回 (ok, fail_kind, 命中的那一行)。

    返回**命中行**而不是全文首行：工具结果的第一行常常是正常输出（实测
    ``process`` 工具的第一行是 ``prompt: 与解说员 · 林交谈``，真正的错误在
    最后几行的 stack trace 里）。取命中行才能让节点标签有信息量。
    """
    if is_error_flag:
        return False, "is_error", _clip(_diagnostic_line(text), 300)
    m = _JSON_FAIL_RE.search(text)
    if m:
        # 只取 `"success": false` 那一行没有信息量（结构化结果通常是格式化 JSON，
        # 失败说明在相邻字段里）。取命中位置前后一段窗口，把 caption / message
        # 之类的说明带出来。
        lo = max(0, m.start() - 220)
        hi = min(len(text), m.end() + 60)
        return False, "json_success_false", _clip(_squeeze(text[lo:hi]), 300)
    for pattern in FAIL_PATTERNS:
        m = pattern.search(text)
        if not m:
            continue
        # 命中位置所在行的完整内容
        start = text.rfind("\n", 0, m.start()) + 1
        end = text.find("\n", m.end())
        line = text[start:end if end != -1 else len(text)]
        return False, "error_keyword", _clip(line.strip(), 300)
    return True, "", ""


def _collect_calls(messages: List[dict]) -> List[Action]:
    """按出现顺序收集所有工具调用（只有调用、还没有结果）。"""
    actions: List[Action] = []
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        # OpenAI 形态
        for tc in (msg.get("tool_calls") or []):
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
            actions.append(Action(
                index=len(actions),
                call_id=str(tc.get("id") or ""),
                name=str(fn.get("name") or "unknown"),
                args_preview=_clip(_args_text(fn.get("arguments")), ACTION_PREVIEW_CHARS),
            ))
        # Anthropic 形态
        for block in _blocks(msg.get("content")):
            if block.get("type") != "tool_use":
                continue
            actions.append(Action(
                index=len(actions),
                call_id=str(block.get("id") or ""),
                name=str(block.get("name") or "unknown"),
                args_preview=_clip(_args_text(block.get("input")), ACTION_PREVIEW_CHARS),
            ))
    return actions


def _apply_results(messages: List[dict], actions: List[Action]) -> tuple[Dict[str, Action], List[Action]]:
    """把工具结果配到调用上。返回 (call_id → Action, 没有 id 的调用列表)。

    有 ``tool_call_id`` 走精确配对；没有 id 的（个别 harness 省略）按出现顺序
    顺次消费——两种情况在实测数据里都存在。
    """
    by_id = {a.call_id: a for a in actions if a.call_id}
    no_id = [a for a in actions if not a.call_id]
    cursor = 0

    def _fill(action: Action, text: str, is_error_flag: bool) -> None:
        ok, kind, line = _classify(text, is_error_flag)
        action.ok = ok
        action.fail_kind = kind
        action.result_preview = _clip(text, RESULT_PREVIEW_CHARS)
        action.has_result = True
        if not ok:
            action.fail_text = line

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")

        # OpenAI 形态：独立的 tool 消息
        if role == "tool":
            text = _collect_text(msg.get("content", ""))
            target: Optional[Action] = None
            cid = str(msg.get("tool_call_id") or "")
            if cid:
                target = by_id.get(cid)
            elif cursor < len(no_id):
                target = no_id[cursor]
                cursor += 1
            if target is not None:
                _fill(target, text, bool(msg.get("is_error")))
            continue

        # Anthropic 形态：user 消息里嵌 tool_result block
        if role == "user":
            for block in _blocks(msg.get("content")):
                if block.get("type") != "tool_result":
                    continue
                text = _collect_text(block.get("content", ""))
                target = by_id.get(str(block.get("tool_use_id") or ""))
                if target is None and cursor < len(no_id):
                    target = no_id[cursor]
                    cursor += 1
                if target is not None:
                    _fill(target, text, bool(block.get("is_error")))

    return by_id, no_id


def _attach_repair_hints(messages: List[dict], actions: List[Action]) -> None:
    """失败动作之后最近的 assistant 自述，标成 repair_hint。

    只认失败动作（成功的动作不需要「修」），往后找第一条**有可见正文**的
    assistant 消息。窗口要放宽到十几条：实测 agent 在失败后往往先连打几个
    诊断性的 exec（结果里没有可见正文），自述出现在好几步之后——
    例：8226910d 会话里失败在两个 exec 之后，自述在 10 条消息之后的 msg 103。

    纯展示标注，不参与成败判定，也不改变 ``ok``。
    """
    # 先建「消息序号 → 该消息里的动作」索引
    pos_actions: Dict[int, List[Action]] = {}
    ai = 0
    for mi, msg in enumerate(messages):
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        n = 0
        n += len(msg.get("tool_calls") or [])
        n += sum(1 for b in _blocks(msg.get("content")) if b.get("type") == "tool_use")
        if n:
            pos_actions[mi] = actions[ai:ai + n]
            ai += n

    order = sorted(pos_actions)
    for mi in order:
        for action in pos_actions[mi]:
            if action.ok or action.repair_hint:
                continue
            # 往后最多看 REPAIR_LOOKAHEAD 条消息，找第一条 assistant 的可见正文
            for nxt in messages[mi + 1: mi + 1 + REPAIR_LOOKAHEAD]:
                if not isinstance(nxt, dict) or nxt.get("role") != "assistant":
                    continue
                text = _collect_text(nxt.get("content", ""))
                if not text.strip():
                    continue
                if _REPAIR_HINT_RE.search(text):
                    action.repair_hint = _clip(_first_line(text), 160)
                break


def extract_actions(messages: List[dict]) -> List[Action]:
    """把一条 trace 的 messages 抽成有序动作链。

    只有调用没有结果的（trace 被截断、或响应里还没落工具结果）保留为
    ``has_result=False`` 且 ``ok=True``——「未知」不该被当成失败。
    """
    if not isinstance(messages, list) or not messages:
        return []
    actions = _collect_calls(messages)
    if not actions:
        return []
    _apply_results(messages, actions)
    _attach_repair_hints(messages, actions)
    return actions


def merge_action_chains(per_trace: List[List[Action]]) -> List[Action]:
    """把逐 trace 的动作链合成**一条**全局链。

    必须合并：同一个会话里每条 trace 都携带完整历史，后面的 trace 是前面 trace 的
    超集。实测 8226910d 会话：19 条 trace，动作数 47→48→…→65，直接拼接会得到
    1064 个「动作」，而真实动作只有 65 个——图上会变成 19 倍的重复幻觉。

    合并依据是稳定的 ``call_id``（实测 trace1 的 id 序列是 trace19 的严格前缀）。
    没有 id 的动作无法跨 trace 对齐，按 (name, args_preview) 的口径保留一次。

    保序：以最后一条 trace 的顺序为准（它是超集，顺序也最完整），把更早 trace 里
    出现过、但最后一条没覆盖的（trace 被截断的情形）补在最前面。
    """
    chains = [c for c in per_trace if c]
    if not chains:
        return []
    last = chains[-1]

    seen_ids = set()
    seen_keys = set()
    merged: List[Action] = []

    def _key(a: Action) -> tuple:
        return (a.name, a.args_preview)

    # 先补最后一条之前、但没被最后一条覆盖的动作（保序）
    tail_ids = {a.call_id for a in last if a.call_id}
    tail_keys = {_key(a) for a in last}
    prefix_extra: List[Action] = []
    for chain in chains[:-1]:
        for a in chain:
            if a.call_id:
                if a.call_id in tail_ids:
                    continue
            elif _key(a) in tail_keys:
                continue
            # 同一条更早 trace 内部也要去重（理论上不该重复，防御性处理）
            k = ("id", a.call_id) if a.call_id else ("k", _key(a))
            if k in seen_keys:
                continue
            seen_keys.add(k)
            cp = Action(**a.as_dict())
            prefix_extra.append(cp)

    merged.extend(prefix_extra)

    for a in last:
        if a.call_id:
            if a.call_id in seen_ids:
                continue
            seen_ids.add(a.call_id)
        merged.append(a)

    # 重编号：全局序号，前端按下标定位
    for i, a in enumerate(merged):
        a.index = i
    return merged


def summarize(actions: List[Action]) -> Dict[str, int]:
    """一次动作链的统计口径（与前端指标行对齐）。"""
    total = len(actions)
    fail = sum(1 for a in actions if not a.ok)
    return {
        "total": total,
        "ok": total - fail,
        "fail": fail,
        "unknown": sum(1 for a in actions if not a.has_result),
        "repaired": sum(1 for a in actions if a.repair_hint),
    }
