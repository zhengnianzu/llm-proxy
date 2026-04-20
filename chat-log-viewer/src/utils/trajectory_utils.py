import json
from pathlib import Path
from typing import Any, Dict, List

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
