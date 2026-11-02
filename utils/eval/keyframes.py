"""
utils/eval/keyframes.py — 关键帧切分（纯函数，无 IO）

把一次会话的 trace_list 切成若干「关键帧」。每帧只保留**一个代表文件**——
段内消息最多的那条。

判定的是上下文连续性：相邻两条请求，后者是否还是前者的增量？

    是  → 同一段，跳过
    否  → 上下文断裂，前一帧收口，从这条起新帧

与 ``utils/eval/segments.grows`` 的区别：这里的比较**跳过所有 system 消息**。
实测 ODeM 的 system prompt 在同一会话内从 33311 字节涨到 34148，不跳过的话
公共前缀直接归零——13→14 那对含 system 时前缀 0、剥掉后 44，会把正常续写
全判成断裂。

边界类型（按判定优先级，强信号优先）：

    compaction    命中压缩摘要标记（各 harness 文本不同，见 harness.COMPACTION_MARKERS）
    auxiliary     压缩助手自己发起的请求，隔离，不参与分段
    tools_changed 工具定义变了（会话中途换工具集）
    tail_truncate 同段对话，只是尾部被裁/重跑，**不切分**
    restart       无标记但历史对不上（重开 / 换 persona / 上下文丢失）
    continuation  正常增量，不切分
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from utils.eval.harness import HarnessAdapter, detect_adapter, get_adapter

__all__ = ["KeyFrame", "TraceNode", "split_keyframes"]

# 同段判定的重叠阈值。低于这个比例视为上下文真的换了（重开/换 persona）。
# 实测 ODeM 的分布呈明显双峰：重开集中在 0.0-0.2，同段重写集中在 0.67-0.97，
# 中间几乎没有样本，0.5 是安全的分界。
SAME_SEGMENT_RATIO = 0.5


@dataclass(slots=True)
class TraceNode:
    """一条请求（一次 API 调用）的判定视图。"""

    index: int
    filename: str
    messages: int
    canonical: List[str]          # 已剥 system 的规范化消息
    roles: List[str]
    tools_key: str
    compaction: Optional[str]     # 命中的压缩标记
    compaction_head: str          # 摘要文本指纹（判「是否新压缩」用）
    auxiliary: bool               # 是否是压缩助手请求
    system_count: int = 0
    turns: int = 0                # 真实用户轮次（排除 marker/bootstrap/内部请求）

    @property
    def stable_prefix(self) -> List[str]:
        return self.canonical


@dataclass(slots=True)
class KeyFrame:
    """一个关键帧 = 一段对话的代表。"""

    index: int                     # 帧序号，1 起
    filename: str                  # 代表文件（段内 messages 最多）
    first_file: str                # 段内第一条
    last_file: str                 # 段内最后一条
    messages: int                  # 代表文件的消息数
    traces: int                    # 段内 trace 数
    boundary: str                  # 本帧的起始原因
    turns: int = 0                 # 代表文件的真实用户轮次
    evidence: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_mutation(self) -> bool:
        """本帧起始是否构成「突变」——上下文真的换了，而非正常续写或尾部重跑。"""
        return self.boundary not in ("start", "continuation", "tail_truncate")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "filename": self.filename,
            "first_file": self.first_file,
            "last_file": self.last_file,
            "messages": self.messages,
            "traces": self.traces,
            "boundary": self.boundary,
            "turns": self.turns,
            "mutation": self.is_mutation,
            "evidence": self.evidence,
        }


def build_node(obj: dict, index: int, filename: str, adapter: HarnessAdapter) -> Optional[TraceNode]:
    """把一份原始请求建成判定视图。无法解析返回 None。"""
    msgs = adapter.messages(obj)
    if msgs is None:
        return None

    canonical: List[str] = []
    roles: List[str] = []
    system_count = 0
    for msg in msgs:
        role = adapter.role_of(msg)
        roles.append(role)
        if adapter.is_variable(msg):
            system_count += 1
            continue
        canonical.append(adapter.canonical(msg))

    return TraceNode(
        index=index,
        filename=filename,
        messages=len(msgs),
        canonical=canonical,
        roles=roles,
        tools_key=adapter.tools_key(obj),
        compaction=adapter.compaction_marker(obj),
        compaction_head=adapter.summary_digest(obj),
        auxiliary=adapter.is_compaction_assistant(obj),
        system_count=system_count,
        turns=_count_turns(msgs),
    )


def _count_turns(msgs: List[dict]) -> int:
    """真实用户轮次。复用 message_common 的口径（排除 session marker /
    bootstrap context / 内部请求），保证与其它模块的 turn 统计一致。"""
    try:
        from utils.message_common import count_real_user_turns
        return int(count_real_user_turns(msgs) or 0)
    except Exception:  # noqa: BLE001  统计失败不该让分析整体失败
        return 0


def _common_prefix(a: Sequence[str], b: Sequence[str]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x == y:
            n += 1
        else:
            break
    return n


def classify_boundary(prev: TraceNode, cur: TraceNode) -> Tuple[str, Dict[str, Any]]:
    """判定 cur 相对 prev 的边界类型。返回 (类型, 证据)。"""
    prefix = _common_prefix(prev.stable_prefix, cur.stable_prefix)
    evidence: Dict[str, Any] = {
        "prev_msgs": prev.messages,
        "next_msgs": cur.messages,
        "common_prefix": prefix,
        "prev_stable": len(prev.stable_prefix),
    }

    # 1. 压缩标记：只在**摘要文本发生变化**时才算新压缩。
    #    摘要会一直留在历史里，后续每条请求都命中同一段文本；若只看「有没有」，
    #    整段会话会被判成每步都压缩（实测 lp5i 10 条 trace 全中）。
    if cur.compaction and cur.compaction_head != prev.compaction_head:
        evidence["marker"] = cur.compaction
        return "compaction", evidence

    # 2. 压缩助手请求：harness 在执行压缩，不是用户对话
    if cur.auxiliary:
        return "auxiliary", evidence

    # 3. 工具集变化：会话中途换了工具配置
    if cur.tools_key != prev.tools_key:
        evidence["tools_changed"] = True
        return "tools_changed", evidence

    # 4. 正常增量：更长且完整包含前者
    if (
        len(cur.stable_prefix) > len(prev.stable_prefix)
        and prefix == len(prev.stable_prefix)
    ):
        return "continuation", evidence

    # 5. 同段的重写：前缀重叠足够高，只是尾部被裁/重跑/改写了某几轮。
    #    实测两类：一方是另一方前缀（152→128，prefix 85/101），
    #    或中途某条工具结果被重跑（84→86，前 57 条相同、第 57 条不同）。
    #
    #    分母必须用 max 而不是 min：用 min 时「短的新请求」会把比值抬高。
    #    实测 ODeM 的 50→16（丢掉 39 条）用 min 算出 10/15=0.67 被误判成同段，
    #    实际它只保留了 10/49=0.20 的历史——这正是报告里说的「重开/上下文丢失」，
    #    必须切开，否则那 39 条消息的丢失在导出里完全看不见。
    prev_len = len(prev.stable_prefix)
    cur_len = len(cur.stable_prefix)
    if prev_len and cur_len:
        overlap = prefix / max(prev_len, cur_len)
        evidence["overlap"] = round(overlap, 3)
        if overlap >= SAME_SEGMENT_RATIO:
            return "tail_truncate", evidence

    # 6. 其余：重开 / 换 persona / 上下文丢失
    return "restart", evidence


def split_keyframes(
    nodes: Sequence[TraceNode],
    *,
    drop_auxiliary: bool = True,
) -> List[KeyFrame]:
    """把有序的 TraceNode 列表切成关键帧。

    drop_auxiliary: 是否把压缩助手请求整段丢弃（默认丢——它不是用户对话）。
        设为 False 时它们各自成帧，便于排查 harness 的压缩行为。

    段内代表文件取 messages 最多者（并列取靠后的，即更新的那次）。
    帧的 boundary 记录**该段是如何开始的**。
    """
    frames: List[KeyFrame] = []
    segment: List[TraceNode] = []
    boundary = "start"
    evidence: Dict[str, Any] = {}

    def close() -> None:
        nonlocal segment, boundary, evidence
        if not segment:
            return
        # 代表文件：消息最多；并列取靠后
        rep = segment[0]
        for node in segment[1:]:
            if node.messages >= rep.messages:
                rep = node
        frames.append(KeyFrame(
            index=len(frames) + 1,
            filename=rep.filename,
            first_file=segment[0].filename,
            last_file=segment[-1].filename,
            messages=rep.messages,
            traces=len(segment),
            boundary=boundary,
            turns=rep.turns,
            evidence=evidence,
        ))
        segment = []
        boundary = "continuation"
        evidence = {}

    prev: Optional[TraceNode] = None
    for node in nodes:
        if prev is None:
            # 首条就是压缩助手请求：整场会话都不是用户对话，直接丢弃。
            if node.auxiliary and drop_auxiliary:
                prev = node
                continue
            boundary = "start"
            evidence = {"msgs": node.messages}
        else:
            kind, ev = classify_boundary(prev, node)
            if kind in ("continuation", "tail_truncate"):
                pass                      # 同段，继续累积
            elif kind == "auxiliary" and drop_auxiliary:
                # 压缩助手请求不入段，也不打断当前段（它是旁路请求）。
                prev = node
                continue
            else:
                close()
                boundary = kind
                evidence = ev
        segment.append(node)
        prev = node

    close()
    return frames
