"""请求 index.jsonl 写入和计数。"""

import json
import os

from utils.log_paths import build_index_path
from utils.message_common import build_chain_key, get_first_user_text


_first_count: int = 0
_total_count: int = 0
_valid_count: int = 0


def get_index_counts() -> tuple:
    return _first_count, _total_count, _valid_count


def index_path_for_req_file(req_file: str, logs_dir: str) -> str:
    req_path = os.path.normpath(req_file)
    main_root = os.path.normpath(logs_dir)
    if req_path == main_root or req_path.startswith(main_root + os.sep):
        return build_index_path(logs_dir)
    return build_index_path(os.path.dirname(req_file) or ".")


def load_index(logs_dir: str):
    global _first_count, _total_count, _valid_count
    root_index = build_index_path(logs_dir)
    if not os.path.exists(root_index):
        return
    with open(root_index, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                _first_count += 1
                _total_count += entry.get("total_attempts", 1)
                if entry.get("valid") or entry.get("success"):
                    _valid_count += 1
            except json.JSONDecodeError:
                pass


def _extract_q1_preview(messages, kind="anthropic"):
    return get_first_user_text(messages or [])[:100]


def _extract_chain_key_responses(input_data) -> str:
    if isinstance(input_data, str):
        return input_data[:500]
    if isinstance(input_data, list):
        for item in input_data:
            if not isinstance(item, dict):
                continue
            if item.get("role") == "user":
                content = item.get("content", "")
                if isinstance(content, str):
                    return content[:500]
                if isinstance(content, list):
                    texts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "input_text":
                            texts.append(part.get("text", ""))
                    return "|".join(texts)[:500]
    return ""


def _extract_q1_preview_responses(input_data) -> str:
    if isinstance(input_data, str):
        return input_data[:100]
    if isinstance(input_data, list):
        for item in input_data:
            if not isinstance(item, dict):
                continue
            if item.get("role") == "user":
                content = item.get("content", "")
                if isinstance(content, str):
                    return content[:100]
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "input_text":
                            return part.get("text", "")[:100]
    return ""


def append_index(ts: str, req_file: str, provider: str, logs_dir: str, model: str = "",
                 tok_in: int = 0, tok_out: int = 0, cache_in: int = 0,
                 success: bool = True,
                 api_key: str = "", chain_key: str = "", q1_preview: str = "",
                 total_attempts: int = 1, start_turn: int = 0,
                 channel_key: str = "", usage: dict = None,
                 debug_file: str = "",
                 msg_count: int = 0, user_turns: int = 0):
    global _first_count, _total_count, _valid_count
    entry = {
        "ts": ts,
        "req_file": req_file,
        "provider": provider,
        "model": model,
        "tok_in": tok_in,
        "tok_out": tok_out,
        "cache_in": cache_in,
        "success": success,
        "api_key": api_key,
        "chain_key": chain_key,
        "q1_preview": q1_preview,
        "total_attempts": total_attempts,
        "retried": total_attempts > 1,
        "start_turn": start_turn,
        "channel_key": channel_key,
        "usage": usage or {},
        "msg_count": msg_count,
        "user_turns": user_turns,
    }
    if debug_file:
        entry["debug_file"] = debug_file
    index_file = index_path_for_req_file(req_file, logs_dir)
    os.makedirs(os.path.dirname(index_file) or ".", exist_ok=True)
    with open(index_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    _first_count += 1
    _total_count += total_attempts
    if success:
        _valid_count += 1


def append_index_anthropic(ts, req_path, total_attempts, valid, logs_dir, model="", tok_in=0, tok_out=0, cache_in=0, api_key="", messages=None, channel_key="", usage=None, debug_file=""):
    msgs = messages or []
    from utils.message_common import count_real_user_turns
    append_index(
        ts, req_path, provider="anthropic", logs_dir=logs_dir, model=model,
        tok_in=tok_in, tok_out=tok_out, cache_in=cache_in, success=valid,
        api_key=api_key,
        chain_key=build_chain_key(msgs),
        q1_preview=_extract_q1_preview(msgs),
        total_attempts=total_attempts,
        start_turn=get_first_user_text(msgs, return_index=True)[1],
        channel_key=channel_key,
        usage=usage,
        debug_file=debug_file,
        msg_count=len(msgs),
        user_turns=count_real_user_turns(msgs),
    )


def append_index_openai(ts, req_path, logs_dir, model="", tok_in=0, tok_out=0, success=True, api_key="", messages=None, channel_key="", usage=None, debug_file=""):
    msgs = messages or []
    from utils.message_common import count_real_user_turns
    append_index(
        ts, req_path, provider="openai", logs_dir=logs_dir, model=model,
        tok_in=tok_in, tok_out=tok_out, success=success,
        api_key=api_key,
        chain_key=build_chain_key(msgs),
        q1_preview=_extract_q1_preview(msgs),
        channel_key=channel_key,
        usage=usage,
        debug_file=debug_file,
        msg_count=len(msgs),
        user_turns=count_real_user_turns(msgs),
    )


def append_index_responses(ts, req_path, logs_dir, model="", tok_in=0, tok_out=0, success=True, api_key="", input_data=None, channel_key="", usage=None, debug_file=""):
    append_index(
        ts, req_path, provider="responses", logs_dir=logs_dir, model=model,
        tok_in=tok_in, tok_out=tok_out, success=success,
        api_key=api_key,
        chain_key=_extract_chain_key_responses(input_data),
        q1_preview=_extract_q1_preview_responses(input_data),
        channel_key=channel_key,
        usage=usage,
        debug_file=debug_file,
    )
