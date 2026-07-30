"""
utils/newapi_format.py — new-api 合并日志文件解析

new-api 每个请求写一个合并 JSON 文件：
  {ts, rid, uid, model, api_key, usage:{token_in,token_out},
   req:  "<请求体 JSON 字符串>",
   resp: "<原始响应文本：Anthropic SSE 或 OpenAI（流式 chunk / 非流式整块）>",
   up_req, up_resp}

与本项目「三元组 + 精简 index」不同，合并文件本身含全部信息，缺的只是
会话聚合派生字段（q1_hash/msg_count/user_turns/chain_key）。本模块负责：
  - 把合并文件拆成 messages + assistant content（供预览）
  - 计算会话聚合字段（供 session_cache.db 归并）
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.message_common import (
    parse_streaming_response_content,
    parse_openai_response_content,
    compute_q1_hash,
    get_first_user_text,
    count_real_user_turns,
    build_chain_key,
)


def split_sse_text(text: str) -> List[dict]:
    """把原始 SSE 文本（event:/data: 行）解析为 chunk dict 列表。"""
    chunks: List[dict] = []
    if not text:
        return chunks
    for line in text.split("\n"):
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            chunks.append(json.loads(payload))
        except json.JSONDecodeError:
            continue
    return chunks


def _is_openai_resp(resp: str) -> bool:
    """判断 resp 是否 OpenAI 格式（流式 chat.completion.chunk 或非流式整块）。

    Anthropic 口径用 content_block_* / message_start；OpenAI 用 choices[].delta/message。
    取前若干字符探测即可（响应可能很大）。
    """
    head = resp[:4000]
    if "content_block" in head or "message_start" in head:
        return False
    return '"choices"' in head and ('"delta"' in head or '"message"' in head or '"reasoning_content"' in head)


def _assistant_content_from_resp(resp: str):
    """从 resp（原始 SSE 文本，Anthropic SSE 或 OpenAI 流式/非流式）重建 assistant content 列表。"""
    if not isinstance(resp, str) or not resp:
        return None
    # OpenAI 口径（流式 choices.delta / 非流式 choices.message）单独解析
    if _is_openai_resp(resp):
        return parse_openai_response_content(resp)
    chunks = split_sse_text(resp)
    if not chunks:
        return None
    return parse_streaming_response_content(
        [c for c in chunks if c.get("type") != "anthropic_passthrough_sse_meta"]
    )


def load_combined(path: Path) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _req_to_dict(req: Any) -> Optional[dict]:
    """把 req 字段（可能是 JSON 字符串或已是 dict）解析为完整请求体 dict。"""
    if isinstance(req, str):
        try:
            req = json.loads(req)
        except json.JSONDecodeError:
            return None
    return req if isinstance(req, dict) else None


def _messages_from_req(req: Any) -> Optional[List[dict]]:
    req_obj = _req_to_dict(req)
    if isinstance(req_obj, dict) and isinstance(req_obj.get("messages"), list):
        return req_obj["messages"]
    return None


def parse_combined_file(path: Path) -> Optional[Dict[str, Any]]:
    """解析合并文件，返回统一结构。用于预览与会话聚合。

    返回 None 表示文件损坏/无法解析。
    """
    data = load_combined(path)
    if data is None:
        return None

    req_body = _req_to_dict(data.get("req"))
    messages = (req_body.get("messages") if isinstance(req_body, dict) else None) or []
    resp_raw = data.get("resp", "")
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    tok_in = usage.get("token_in") or 0
    tok_out = usage.get("token_out") or 0

    return {
        "ts": data.get("ts", "") or "",
        "model": data.get("model", "") or "",
        "api_key": data.get("api_key", "") or "",
        "messages": messages,
        # 完整请求体（含 tools/system/tool_choice 等所有字段），不裁剪
        "req_body": req_body if isinstance(req_body, dict) else {},
        # 响应原文（Anthropic SSE / OpenAI 流式或非流式整块），不裁剪
        "resp_raw": resp_raw if isinstance(resp_raw, str) else "",
        "assistant_content": _assistant_content_from_resp(resp_raw),
        "tok_in": tok_in,
        "tok_out": tok_out,
        "success": tok_out > 0,
    }


def compute_session_fields(messages: List[dict]) -> Dict[str, Any]:
    """从 messages 计算会话聚合派生字段。"""
    if not messages:
        return {
            "q1_hash": "",
            "q1_preview": "",
            "chain_key": "",
            "msg_count": 0,
            "user_turns": 0,
        }
    q1_text = get_first_user_text(messages)
    return {
        "q1_hash": compute_q1_hash(messages),
        "q1_preview": (q1_text or "")[:200],
        "chain_key": build_chain_key(messages),
        "msg_count": len(messages),
        "user_turns": count_real_user_turns(messages),
    }


def build_merged_for_eval(path: Path) -> Optional[Dict[str, Any]]:
    """供质检 reformat 使用：把合并文件拆成 analyze_best_data 期望的 merged 结构。

    analyze_best_data 读 merged["messages"] 与 merged["response"]["content"]。
    与三元组路径 (_load_json(req) + merged["response"]=parse_response(res)) 对齐。
    """
    parsed = parse_combined_file(path)
    if parsed is None:
        return None
    content = parsed["assistant_content"]
    # 以完整请求体为基底，保留 tools/system/tool_choice 等所有字段，
    # 再覆盖 model/messages 与派生的 response，避免遗漏原始请求信息。
    merged: Dict[str, Any] = dict(parsed["req_body"])
    merged["model"] = parsed["model"] or merged.get("model", "")
    merged["messages"] = parsed["messages"]
    merged["header"] = {}
    merged["response"] = {
        "content": content if content is not None else [],
        "status_code": 200 if parsed["success"] else 400,
    }
    return merged


def build_preview_payload(path: Path) -> Optional[Dict[str, Any]]:
    """供 /logs/file 使用：返回 messages（末尾附 assistant）+ usage，
    结构与原生 anthropic 预览对齐。"""
    parsed = parse_combined_file(path)
    if parsed is None:
        return None
    messages = list(parsed["messages"])
    if parsed["assistant_content"] is not None:
        messages.append({
            "role": "assistant",
            "content": parsed["assistant_content"],
            "_from_res": True,
        })
    # 以完整请求体为基底，保留 tools/system/tool_choice 等所有字段，
    # 再覆盖 model/messages（messages 末尾已附 assistant）。
    payload: Dict[str, Any] = dict(parsed["req_body"])
    payload["model"] = parsed["model"] or payload.get("model", "")
    payload["messages"] = messages
    if parsed["tok_in"] or parsed["tok_out"]:
        payload["_usage"] = {
            "input_tokens": parsed["tok_in"],
            "output_tokens": parsed["tok_out"],
        }
    return payload
