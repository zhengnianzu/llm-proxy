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
from typing import Any, Dict, List, Optional, Tuple

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


def _split_ndjson(text: str) -> List[dict]:
    """把 NDJSON 文本（每行一个 JSON 对象，无 data: 前缀）解析为 chunk 列表。

    up_resp 在 resp 被省略后以此格式写入：Anthropic SSE 和 OpenAI Responses API
    都以 { 开头的裸行拼接，不带 "data: " 前缀。
    """
    chunks: List[dict] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            chunks.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return chunks


def _parse_responses_api_content(chunks: List[dict]) -> Optional[list]:
    """从 OpenAI Responses API 的 NDJSON 流里提取 assistant content block 列表。

    优先取 response.completed 里的完整 output；没有则从各类 delta 事件累加：
      - response.output_text.delta / response.output_text.done  → text
      - response.reasoning_text.delta                           → thinking (reasoning)
      - response.output_item.done (type=function_call)          → tool_use
    """
    for c in reversed(chunks):
        if c.get("type") == "response.completed":
            resp_obj = c.get("response") or {}
            blocks: list = []
            for item in resp_obj.get("output") or []:
                if not isinstance(item, dict):
                    continue
                for b in item.get("content") or []:
                    if not isinstance(b, dict):
                        continue
                    bt = b.get("type", "")
                    if bt == "output_text":
                        t = b.get("text", "")
                        if t:
                            blocks.append({"type": "text", "text": t})
                    elif bt == "tool_use":
                        blocks.append({"type": "tool_use",
                                       "name": b.get("name", ""),
                                       "input": b.get("input") or {}})
                    else:
                        blocks.append(b)
            if blocks:
                return blocks

    # 没有 response.completed：从 delta 事件流重建。
    # 每个 output_item 对应一个 slot，用 output_index 分槽累加。
    # 槽里可能是 reasoning text、output text 或 tool_call。
    slots: dict = {}  # output_index → {"type": str, "text": str, "tool": dict}
    for c in chunks:
        t = c.get("type", "")
        idx = c.get("output_index", 0)
        slot = slots.setdefault(idx, {"kind": "text", "reasoning": "", "text": "", "tool": None})
        if t == "response.reasoning_text.delta":
            slot["reasoning"] += c.get("delta") or ""
        elif t == "response.output_text.delta":
            slot["text"] += c.get("delta") or ""
        elif t == "response.output_text.done":
            slot["text"] = c.get("text") or slot["text"]
        elif t == "response.output_item.added":
            item = c.get("item") or {}
            if item.get("type") == "function_call":
                slot["kind"] = "tool"
                slot["tool"] = {"name": item.get("name", ""), "args": ""}
        elif t == "response.function_call_arguments.delta":
            if slot.get("kind") == "tool" and slot["tool"]:
                slot["tool"]["args"] += c.get("delta") or ""
        elif t == "response.output_item.done":
            item = c.get("item") or {}
            if item.get("type") == "function_call":
                slot["kind"] = "tool"
                try:
                    import json as _json
                    args = _json.loads(item.get("arguments") or "{}")
                except Exception:
                    args = {}
                slot["tool"] = {"name": item.get("name", ""), "parsed_args": args}

    blocks = []
    for idx in sorted(slots):
        s = slots[idx]
        if s.get("reasoning"):
            blocks.append({"type": "thinking", "thinking": s["reasoning"]})
        if s.get("kind") == "tool" and s.get("tool"):
            tl = s["tool"]
            inp = tl.get("parsed_args")
            if inp is None:
                try:
                    import json as _json
                    inp = _json.loads(tl.get("args") or "{}")
                except Exception:
                    inp = {}
            blocks.append({"type": "tool_use", "name": tl.get("name", ""), "input": inp})
        elif s.get("text"):
            blocks.append({"type": "text", "text": s["text"]})
    return blocks if blocks else None


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
    """从 resp（原始 SSE 文本，Anthropic SSE 或 OpenAI 流式/非流式）重建 assistant content 列表。

    up_resp 写入的格式是 NDJSON（每行一个 JSON 对象，无 data: 前缀），需要额外处理：
      - OpenAI chat.completion.chunk NDJSON（有 choices）→ 转成标准 SSE 再解析
      - Anthropic SSE 变体（{"type":"message_start",...}\n{...}...）
      - OpenAI Responses API（{"type":"response.created",...}\n{...}...）
    """
    if not isinstance(resp, str) or not resp:
        return None

    # NDJSON 检测：首行是 { 但整体 json.loads 失败，说明是多行 JSON 拼接。
    # 必须在 _is_openai_resp 之前处理，因为后者只看前 4000 字节，对
    # NDJSON 形态的 OpenAI 流式响应会误判（它找 choices 关键词发现有，
    # 但 parse_openai_response_content 要求 "data: " 前缀而拿不到内容）。
    stripped = resp.lstrip()
    if stripped.startswith("{"):
        try:
            obj = json.loads(resp)
            # 单体 JSON：Anthropic 非流式（{"role":"assistant","content":[...]})
            if isinstance(obj, dict) and obj.get("role") == "assistant":
                content = obj.get("content")
                if isinstance(content, list) and content:
                    return content
        except json.JSONDecodeError:
            # 多行 NDJSON 进这里
            ndjson_chunks = _split_ndjson(resp)
            if ndjson_chunks:
                first = ndjson_chunks[0]
                # OpenAI chat.completion.chunk（有 choices）→ 转标准 SSE 再解析
                if "choices" in first:
                    sse_text = "\n".join("data: " + json.dumps(c, ensure_ascii=False)
                                         for c in ndjson_chunks)
                    return parse_openai_response_content(sse_text)
                # Responses API（有 response.* 事件）
                if any(c.get("type", "").startswith("response.") for c in ndjson_chunks):
                    return _parse_responses_api_content(ndjson_chunks)
                # Anthropic NDJSON（message_start / content_block_*）
                if any(c.get("type") in ("message_start", "content_block_start", "content_block_delta")
                       for c in ndjson_chunks):
                    from utils.message_common import parse_streaming_response_content
                    return parse_streaming_response_content(
                        [c for c in ndjson_chunks if c.get("type") != "anthropic_passthrough_sse_meta"]
                    )
            return None

    # OpenAI 口径（流式 choices.delta / 非流式 choices.message，以 "data: " 开头）
    if _is_openai_resp(resp):
        return parse_openai_response_content(resp)

    # Anthropic SSE（data: 行）
    chunks = split_sse_text(resp)
    if not chunks:
        return None
    from utils.message_common import parse_streaming_response_content
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


def _normalize_body(body: Any) -> Tuple[List[dict], Any]:
    """把任意形态的请求体归一成 (messages, system)。

    真实采集里出现过四种形态（实测 2026-11-24 proxy-004 抽样）：
      * {"messages": [...]}                      OpenAI chat
      * {"messages": [...], "system": [...]}     Anthropic messages
      * {"contents": [{"role","parts":[{"text"}]}]}   Gemini generateContent
      * {"input": [{"type":"message","role","content":[{"type":"input_text"}]}]}
                                                 OpenAI Responses
    req 字段为空时上游只留下 up_req，而这些 up_req 恰是上面这些形态；不归一就会
    解析出 0 条消息，进而 Q1 全空、会话列表看似没有新轨迹。
    """
    obj = body if isinstance(body, dict) else {}
    system = obj.get("system")
    msgs: List[dict] = []

    raw_msgs = obj.get("messages")
    if isinstance(raw_msgs, list) and raw_msgs:
        return raw_msgs, system

    contents = obj.get("contents")
    if isinstance(contents, list) and contents:
        role_map = {"model": "assistant", "user": "user"}
        for c in contents:
            if not isinstance(c, dict):
                continue
            role = role_map.get(c.get("role") or "", c.get("role") or "user")
            texts = [
                p.get("text") for p in (c.get("parts") or [])
                if isinstance(p, dict) and p.get("text")
            ]
            if texts:
                msgs.append({"role": role, "content": "\n".join(texts)})
        return msgs, system

    items = obj.get("input")
    if isinstance(items, list) and items:
        for it in items:
            if not isinstance(it, dict) or it.get("type") != "message":
                continue
            role = it.get("role") or "user"
            if role in ("developer", "system"):
                # Responses 的 developer/system 归到 system，不占用户轮次
                continue
            content = it.get("content")
            if isinstance(content, list):
                texts = [
                    b.get("text") for b in content
                    if isinstance(b, dict) and b.get("text")
                ]
                text = "\n".join(texts)
            else:
                text = str(content or "")
            if text:
                msgs.append({"role": role, "content": text})
        # Responses 把系统提示放在 instructions，不当用户消息
        if isinstance(obj.get("instructions"), str) and not system:
            system = obj["instructions"]
        return msgs, system

    return msgs, system


def parse_combined_file(path: Path) -> Optional[Dict[str, Any]]:
    """解析合并文件，返回统一结构。用于预览与会话聚合。

    返回 None 表示文件损坏/无法解析。
    """
    data = load_combined(path)
    if data is None:
        return None

    req_body = _req_to_dict(data.get("req"))
    messages, system = _normalize_body(req_body)
    # 采集侧从 2026-11-19 起 req 写空（up_req 正常）。没有 messages 就回退到 up_req，
    # 否则整批新数据解析成空，Q1/会话列表全部失效。
    if not messages:
        up_body = _req_to_dict(data.get("up_req"))
        up_msgs, up_system = _normalize_body(up_body)
        if up_msgs:
            messages, req_body = up_msgs, up_body
            system = system or up_system
    resp_raw = data.get("resp", "")
    # 采集侧从 2026-11-19 起 resp 也写空（up_resp 正常）。没有响应就回退到 up_resp，
    # 否则 assistant_content 解析为 None，浏览页看不到助手回复内容。
    if not (isinstance(resp_raw, str) and resp_raw.strip()):
        up_resp = data.get("up_resp", "")
        if isinstance(up_resp, str) and up_resp.strip():
            resp_raw = up_resp
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    tok_in = usage.get("token_in") or 0
    tok_out = usage.get("token_out") or 0

    return {
        "ts": data.get("ts", "") or "",
        "model": data.get("model", "") or "",
        "api_key": data.get("api_key", "") or "",
        "messages": messages,
        # 系统提示（Anthropic 放在 body.system；Responses 放在 instructions）。
        # 单独带出，浏览页据此在消息前渲染系统块。
        "system": system,
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
