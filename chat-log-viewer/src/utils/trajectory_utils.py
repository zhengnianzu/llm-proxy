import difflib
import html
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.utils.message_utils import extract_messages, load_json


COMPACTION_MARKER = "[compacted: tool output removed to free context]"


def _append_response_message(data: Dict[str, Any], messages: List[dict]) -> List[dict]:
    response = data.get("response")
    if isinstance(response, dict) and isinstance(response.get("content"), list) and response["content"]:
        return list(messages) + [{"role": "assistant", "content": response["content"]}]
    return list(messages)


def load_snapshot_messages(path: Path) -> List[dict]:
    data = load_json(path)
    messages = extract_messages(data) or []
    return _append_response_message(data, messages)


def _canonicalize(value: Any) -> Any:
    if isinstance(value, dict):
        out = {}
        for key, val in value.items():
            if key == "cache_control":
                continue
            out[key] = _canonicalize(val)
        return out
    if isinstance(value, list):
        return [_canonicalize(item) for item in value]
    return value


def _content_to_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    try:
        return json.dumps(value, ensure_ascii=False, indent=2).strip()
    except TypeError:
        return str(value).strip()


def _message_preview(message: Dict[str, Any], limit: int = 220) -> str:
    content = message.get("content")
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if not isinstance(block, dict):
                parts.append(str(block))
                continue
            btype = block.get("type")
            if btype in {"text", "thinking"}:
                parts.append(block.get("text") or block.get("thinking") or "")
            elif btype == "tool_use":
                parts.append(f"tool_use:{block.get('name') or ''}")
            elif btype == "tool_result":
                parts.append(_content_to_text(block.get("content") or ""))
            else:
                parts.append(_content_to_text(block))
        text = "\n".join(p for p in parts if p)
    else:
        text = _content_to_text(content)
    if len(text) > limit:
        return text[:limit] + "..."
    return text


def _contains_compaction(value: Any) -> bool:
    if isinstance(value, str):
        return COMPACTION_MARKER in value
    if isinstance(value, list):
        return any(_contains_compaction(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_compaction(v) for v in value.values())
    return False


def _message_summary(message: Dict[str, Any], index: int) -> Dict[str, Any]:
    return {
        "index": index,
        "role": message.get("role") or "unknown",
        "preview": _message_preview(message),
        "canonical": _canonicalize(message),
        "raw": message,
        "compacted": _contains_compaction(message),
    }


def _classify_against_base(base: Dict[str, Any], other: Dict[str, Any]) -> str:
    if base["raw"] == other["raw"]:
        return "same"
    if base["canonical"] == other["canonical"]:
        return "metadata_only"
    if other["compacted"]:
        return "compaction"
    return "rewrite"


def build_session_trajectory(session_dir: Path) -> Dict[str, Any]:
    files = sorted(session_dir.glob("*.json"))
    snapshots: List[Dict[str, Any]] = []
    loaded: List[List[dict]] = []

    for path in files:
        messages = load_snapshot_messages(path)
        loaded.append(messages)
        snapshots.append({
            "file": path.name,
            "message_count": len(messages),
        })

    if not loaded:
        return {
            "session_id": session_dir.name,
            "snapshot_count": 0,
            "snapshots": [],
            "base_snapshot": None,
            "blocks": [],
            "rounds": [],
        }

    base_idx = len(loaded) - 1
    base_messages = loaded[base_idx]
    base_snapshot = snapshots[base_idx]
    blocks: List[Dict[str, Any]] = []

    for msg_idx, message in enumerate(base_messages):
        base_summary = _message_summary(message, msg_idx)
        per_snapshot: List[Dict[str, Any]] = []
        changed_rounds: List[int] = []
        first_diff_round = None
        stats = {"same": 0, "metadata_only": 0, "compaction": 0, "rewrite": 0, "missing": 0}

        for snap_idx, snap_msgs in enumerate(loaded):
            if msg_idx >= len(snap_msgs):
                state = {
                    "snapshot_index": snap_idx,
                    "file": snapshots[snap_idx]["file"],
                    "status": "missing",
                    "preview": "",
                    "message_count": len(snap_msgs),
                }
                stats["missing"] += 1
            else:
                snap_summary = _message_summary(snap_msgs[msg_idx], msg_idx)
                status = _classify_against_base(base_summary, snap_summary)
                state = {
                    "snapshot_index": snap_idx,
                    "file": snapshots[snap_idx]["file"],
                    "status": status,
                    "preview": snap_summary["preview"],
                    "message_count": len(snap_msgs),
                }
                stats[status] += 1
                if status != "same":
                    changed_rounds.append(snap_idx)
                    if first_diff_round is None:
                        first_diff_round = snap_idx
            per_snapshot.append(state)

        if changed_rounds:
            blocks.append({
                "base_index": msg_idx,
                "role": base_summary["role"],
                "base_preview": base_summary["preview"],
                "base_file": base_snapshot["file"],
                "first_diff_round": first_diff_round,
                "changed_rounds": changed_rounds,
                "stats": stats,
                "states": per_snapshot,
            })

    rounds: List[Dict[str, Any]] = []
    for snap_idx, snap in enumerate(snapshots):
        changed_blocks = []
        summary = {"same": 0, "metadata_only": 0, "compaction": 0, "rewrite": 0, "missing": 0}
        for block in blocks:
            state = block["states"][snap_idx]
            summary[state["status"]] += 1
            if state["status"] != "same":
                changed_blocks.append({
                    "base_index": block["base_index"],
                    "role": block["role"],
                    "status": state["status"],
                    "base_preview": block["base_preview"],
                    "snapshot_preview": state["preview"],
                })
        rounds.append({
            "snapshot_index": snap_idx,
            "file": snap["file"],
            "message_count": snap["message_count"],
            "is_base": snap_idx == base_idx,
            "changed_block_count": len(changed_blocks),
            "summary": summary,
            "blocks": changed_blocks,
        })

    return {
        "session_id": session_dir.name,
        "snapshot_count": len(files),
        "snapshots": snapshots,
        "base_snapshot": {
            "snapshot_index": base_idx,
            "file": base_snapshot["file"],
            "message_count": base_snapshot["message_count"],
        },
        "blocks": blocks,
        "rounds": rounds,
    }


# ---------------------------------------------------------------------------
# Trajectory Diff — 为前端提供预计算的 diff 结果
# ---------------------------------------------------------------------------

def _msg_content_to_text(message: Dict[str, Any]) -> str:
    """将消息 content 转为纯文本用于 diff 比较。"""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if not isinstance(block, dict):
                parts.append(str(block))
                continue
            btype = block.get("type", "")
            if btype in ("text", "thinking"):
                parts.append(block.get("text") or block.get("thinking") or "")
            elif btype == "tool_use":
                name = block.get("name", "")
                inp = block.get("input", {})
                parts.append(f"[tool_use: {name}]\n{json.dumps(inp, ensure_ascii=False, indent=2)}")
            elif btype == "tool_result":
                c = block.get("content", "")
                parts.append(f"[tool_result]\n{c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)}")
            else:
                parts.append(json.dumps(block, ensure_ascii=False, indent=2))
        return "\n".join(parts)
    return json.dumps(content, ensure_ascii=False) if content else ""


def _normalize_whitespace(text: str) -> str:
    """规范化空白用于相似度比较。"""
    return re.sub(r"\s+", " ", text).strip()


def _word_diff_html(original: str, final: str) -> Tuple[str, str]:
    """生成词级 diff HTML（左右两栏）。"""
    orig_words = original.split()
    final_words = final.split()

    # 超长文本直接标记整块差异
    if len(orig_words) > 2000 or len(final_words) > 2000:
        left = f'<span class="diff-del">{html.escape(original)}</span>'
        right = f'<span class="diff-ins">{html.escape(final)}</span>'
        return left, right

    sm = difflib.SequenceMatcher(None, orig_words, final_words)
    left_parts: List[str] = []
    right_parts: List[str] = []

    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            text = html.escape(" ".join(orig_words[i1:i2]))
            left_parts.append(text)
            right_parts.append(text)
        elif op == "replace":
            left_parts.append(f'<span class="diff-del">{html.escape(" ".join(orig_words[i1:i2]))}</span>')
            right_parts.append(f'<span class="diff-ins">{html.escape(" ".join(final_words[j1:j2]))}</span>')
        elif op == "delete":
            left_parts.append(f'<span class="diff-del">{html.escape(" ".join(orig_words[i1:i2]))}</span>')
        elif op == "insert":
            right_parts.append(f'<span class="diff-ins">{html.escape(" ".join(final_words[j1:j2]))}</span>')

    return " ".join(left_parts), " ".join(right_parts)


def _get_content_blocks(message: Dict[str, Any]) -> List[Dict[str, Any]]:
    """提取消息的 content blocks 列表。"""
    content = message.get("content")
    if isinstance(content, list):
        return content
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return [{"type": "text", "text": json.dumps(content, ensure_ascii=False) if content else ""}]


def _block_to_text(block: Dict[str, Any]) -> str:
    """单个 content block 转文本。"""
    btype = block.get("type", "")
    if btype in ("text", "thinking"):
        return block.get("text") or block.get("thinking") or ""
    if btype == "tool_use":
        name = block.get("name", "")
        inp = block.get("input", {})
        return f"[tool_use: {name}]\n{json.dumps(inp, ensure_ascii=False, indent=2)}"
    if btype == "tool_result":
        c = block.get("content", "")
        return c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)
    return json.dumps(block, ensure_ascii=False, indent=2)


def _compute_block_diffs(orig_msg: Dict[str, Any], final_msg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """对比两条消息的 content blocks，返回每个 block 的 diff HTML。"""
    orig_blocks = _get_content_blocks(orig_msg)
    final_blocks = _get_content_blocks(final_msg)
    max_len = max(len(orig_blocks), len(final_blocks))
    block_diffs: List[Dict[str, Any]] = []

    for i in range(max_len):
        ob = orig_blocks[i] if i < len(orig_blocks) else None
        fb = final_blocks[i] if i < len(final_blocks) else None
        orig_text = _block_to_text(ob) if ob else ""
        final_text = _block_to_text(fb) if fb else ""

        # 规范化比较
        if _normalize_whitespace(orig_text) == _normalize_whitespace(final_text):
            continue

        left_html, right_html = _word_diff_html(orig_text, final_text)
        block_diffs.append({
            "block_index": i,
            "block_type": (ob or fb or {}).get("type", "text"),
            "left_html": left_html,
            "right_html": right_html,
        })

    return block_diffs


def build_trajectory_diff(session_dir: Path) -> Dict[str, Any]:
    """构建 trajectory diff 结果，包含预计算的词级 diff HTML。"""
    files = sorted(session_dir.glob("*.json"))
    loaded: List[List[dict]] = []
    snapshots: List[Dict[str, Any]] = []

    for path in files:
        messages = load_snapshot_messages(path)
        loaded.append(messages)
        snapshots.append({
            "file": path.name,
            "rel_path": f"{session_dir.name}/{path.name}",
            "msg_count": len(messages),
        })

    if len(loaded) < 2:
        return {
            "session_id": session_dir.name,
            "snapshot_count": len(loaded),
            "rounds": [{
                "round": i + 1,
                **s,
                "diff_count": 0,
                "summary": {"same": s["msg_count"], "rewrite": 0, "compaction": 0, "missing": 0},
            } for i, s in enumerate(snapshots)],
            "diffs": {},
        }

    base_idx = len(loaded) - 1
    base_messages = loaded[base_idx]

    rounds_out: List[Dict[str, Any]] = []
    diffs_out: Dict[str, List[Dict[str, Any]]] = {}

    for snap_idx, snap_msgs in enumerate(loaded):
        snap = snapshots[snap_idx]
        if snap_idx == base_idx:
            rounds_out.append({
                "round": snap_idx + 1,
                **snap,
                "diff_count": 0,
                "is_base": True,
                "summary": {"same": snap["msg_count"], "rewrite": 0, "compaction": 0, "missing": 0},
            })
            continue

        round_diffs: List[Dict[str, Any]] = []
        summary = {"same": 0, "rewrite": 0, "compaction": 0, "missing": 0}

        for msg_idx in range(len(snap_msgs)):
            snap_msg = snap_msgs[msg_idx]
            if msg_idx >= len(base_messages):
                # 该消息在 base 中不存在（不太常见）
                summary["same"] += 1
                continue

            base_msg = base_messages[msg_idx]
            base_canonical = _canonicalize(base_msg)
            snap_canonical = _canonicalize(snap_msg)

            if base_msg == snap_msg or base_canonical == snap_canonical:
                summary["same"] += 1
                continue

            # 有差异 — 判断类型
            if _contains_compaction(snap_msg):
                status = "compaction"
            else:
                status = "rewrite"
            summary[status] += 1

            # 计算 block-level diff HTML
            block_diffs = _compute_block_diffs(snap_msg, base_msg)

            round_diffs.append({
                "msg_index": msg_idx,
                "role": snap_msg.get("role", "unknown"),
                "status": status,
                "original_blocks": _get_content_blocks(snap_msg),
                "final_blocks": _get_content_blocks(base_msg),
                "block_diffs": block_diffs,
            })

        # base 中多出的消息（snap 中 missing）
        if len(snap_msgs) < len(base_messages):
            summary["missing"] = len(base_messages) - len(snap_msgs)

        diffs_out[str(snap_idx)] = round_diffs
        rounds_out.append({
            "round": snap_idx + 1,
            **snap,
            "diff_count": len(round_diffs),
            "is_base": False,
            "summary": summary,
        })

    return {
        "session_id": session_dir.name,
        "snapshot_count": len(files),
        "rounds": rounds_out,
        "diffs": diffs_out,
    }
