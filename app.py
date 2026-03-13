import copy
import re
import os
import json
import time
import glob
import httpx
import asyncio
import logging

from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from typing import Any, Dict, List, Optional, AsyncIterator
from fastapi.responses import JSONResponse, StreamingResponse, Response

from print_stats_summary import statistic_tokens

load_dotenv(override=True)

# 全局默认：是否屏蔽 Task 工具里的 "- Explore:" 行
BAN_EXPLORE = os.getenv("BAN_EXPLORE", "false").lower() == "true"
BAN_STREAM = os.getenv("BAN_STREAM", "false").lower() == "true"
EXPOSE_THINKING = os.getenv("EXPOSE_THINKING", "true").lower() == "true"
TRUST_ENV = os.getenv("TRUST_ENV", "true").lower() == "true"

# 全局默认：重试次数（不从环境变量读取）
MAX_RETRIES = 20

LOGS_SESSION_ANTHROPIC = "logs_session_anthropic"
LOGS_ANTHROPIC = "logs_anthropic"
LOGS_SESSION_OPENAI = "logs_session_openai"
LOGS_OPENAI = "logs_openai"

app = FastAPI(title="Anthropic+OpenAI Proxy (FastAPI)")

app.mount("/static", StaticFiles(directory="static"), name="static")


# templates = Jinja2Templates(directory="api/templates")

async def get_x_auth_token(request) -> str:
    keys = ['authorization', 'x-api-key']
    for key in keys:
        ack = request.headers.get(key)
        if isinstance(ack, str) and ack.startswith('Bearer '):
            ack = ack.split('Bearer ')[1].strip()
            return ack
    return ''


def _resolve_model_name(raw_model: Any) -> str:
    """
    统一处理 model 名：
    - "byenv" 或空值 -> 从环境变量 MODEL_ID 读取
    - 其他值 -> 原样返回
    """
    if raw_model == "byenv" or not raw_model:
        return os.environ.get("MODEL_ID") or "unknown"
    return raw_model


def _strip_task_explore_line(
        tools: Optional[List[Dict[str, Any]]],
        ban_explore: Optional[bool] = None,
) -> Optional[List[Dict[str, Any]]]:
    """
    根据 ban_explore 决定是否从 Task 工具描述中移除 "- Explore:" 行。
    - ban_explore 为 None 时，采用全局 BAN_EXPLORE 开关。
    """

    def _remove_explore_from_desc(desc: Any) -> Optional[str]:
        if not isinstance(desc, str):
            return None
        lines = desc.splitlines()
        filtered_lines = []
        changed = False
        for line in lines:
            if line.lstrip().startswith("- Explore:") or line.lstrip().startswith("- **Explore**:"):
                changed = True
                continue
            filtered_lines.append(line)
        return "\n".join(filtered_lines) if changed else None

    if ban_explore is None:
        ban_explore = BAN_EXPLORE

    if not ban_explore or not tools:
        return tools

    cleaned: List[Any] = []
    for t in tools:
        if not isinstance(t, dict):
            cleaned.append(t)
            continue
        # Anthropic tools: {name, description, input_schema}
        if t.get("name") == "Task":
            new_desc = _remove_explore_from_desc(t.get("description"))
            if new_desc is not None:
                cleaned.append({**t, "description": new_desc})
            else:
                cleaned.append(t)
            continue

        # OpenAI tools: {type:"function", function:{name, description, parameters}}
        if t.get("type") == "function":
            func = t.get("function")
            if isinstance(func, dict) and func.get("name") == "Task":
                new_desc = _remove_explore_from_desc(func.get("description"))
                if new_desc is not None:
                    cleaned.append({**t, "function": {**func, "description": new_desc}})
                else:
                    cleaned.append(t)
                continue

        cleaned.append(t)

    return cleaned


def build_upstream_headers(x_auth_token: str, model_id: str) -> Dict[str, str]:
    """
    你网关/后端需要的 headers 全放这。
    如果你还需要额外的（比如 X-ID / X-APPKEY），也在这里加。
    """
    ack = os.getenv('UPSTREAM_API_KEY') or x_auth_token
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {ack}",
        "x-api-key": ack,
        "Model-Id": os.environ.get("MODEL_ID") or model_id,
    }


def _ssl_verify() -> bool:
    return os.getenv("SSL_VERIFY", "true").lower() != "false"


# -----------------------------
# Rate limit helpers
# -----------------------------
# 所有需要视为「限流且可重试」的上游状态码，统一维护在这里，便于后续扩展（如再加入 503 等）
RATE_LIMIT_STATUS_CODES = {406, 429}


def is_rate_limit_status(status_code: int) -> bool:
    """
    判断上游响应码是否属于「限流/可重试」错误。
    所有调用处统一依赖本函数，而不是直接写死 (406, 429)。
    """
    return status_code in RATE_LIMIT_STATUS_CODES


# -----------------------------
# Endpoints
# -----------------------------
@app.get("/hi")
async def health():
    return {"LLM_PROXY": "hello !!!"}


# ---------- Anthropic Messages ----------
def _dump_json(path: str, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def _resp_to_obj(r):  # httpx.Response -> dict
    base = {"status_code": r.status_code, "headers": dict(r.headers)}
    try:
        base["json"] = r.json()
    except Exception:
        base["text"] = r.text
    return base


def _extract_text_from_blocks(blocks: Any) -> str:
    if blocks is None:
        return ""
    if isinstance(blocks, str):
        return blocks
    if isinstance(blocks, list):
        parts = []
        for b in blocks:
            if isinstance(b, dict) and b.get("type") == "text":
                parts.append(b.get("text", ""))
        return "".join(parts)
    return str(blocks)


def _extract_first_user_text(body: Dict[str, Any]) -> str:
    """
    提取首条 user 消息的 text 内容，用于 warmup 识别。
    """
    msgs = body.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return ""
    first = msgs[0] or {}
    content = first.get("content")
    return _extract_text_from_blocks(content).strip().lower()


def _system_texts(body: Dict[str, Any]) -> List[str]:
    """
    拉平 system 字段的所有 text 段，便于关键词匹配。
    """
    systems = body.get("system")
    if systems is None:
        return []
    if isinstance(systems, list):
        texts = []
        for s in systems:
            if isinstance(s, dict):
                texts.append(_extract_text_from_blocks(s.get("text")))
            else:
                texts.append(_extract_text_from_blocks(s))
        return texts
    return [_extract_text_from_blocks(systems)]


def _should_skip_session_logging(body: Dict[str, Any]) -> bool:
    """
    按规则过滤不需要写入 session 目录的请求：
    - warmup：首个 user content 为 'warmup'（不区分大小写）
    - topic：system 含 “Analyze if this message indicates a new conversation topic”
    - summary：system 含 “Summarize this coding conversation”
    """
    first_text = _extract_first_user_text(body)
    # _extract_first_user_text 已经 lower，直接匹配小写
    if first_text == "warmup":
        return True

    sys_texts = " ".join(t.lower() for t in _system_texts(body))
    if "analyze if this message indicates a new conversation topic" in sys_texts:
        return True
    if "summarize this coding conversation" in sys_texts:
        return True
    return False


@app.post("/v1/messages")
async def anthropic_messages(req: Request):
    """anthropic透传"""
    body = await req.json()
    stream = bool(body.get("stream", False))
    body_model = body.get("model")
    ban_explore = BAN_EXPLORE

    model_from_body: Optional[str] = body_model if isinstance(body_model, str) else None
    suffix = "--ban_explore"
    if model_from_body and model_from_body.endswith(suffix):
        # 任何带 "--ban_explore" 后缀的模型名，都强制开启屏蔽 Explore
        ban_explore = True
        base_model = model_from_body[: -len(suffix)]
        model = _resolve_model_name(base_model or "byenv")
    else:
        model = _resolve_model_name(body_model)
    # session相关
    session_id = None
    session_metadata = body.get("metadata")
    if isinstance(session_metadata, dict):
        user_id = session_metadata.get("user_id") or ""
        m = re.search(r"session_([A-Za-z0-9-]+)", str(user_id))
        if m:
            session_id = m.group(1)
    else:
        # 适配codeagent获取session id
        session_id = req.headers.get("X-Session-Id")
    skip_session_logging = False
    if session_id:
        skip_session_logging = _should_skip_session_logging(body)

    # 保存请求/响应日志（anthropic 直通）
    if session_id and not skip_session_logging:
        os.makedirs(LOGS_SESSION_ANTHROPIC, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")[:-3]  # 带毫秒，避免并发重名
        # 若该 session_id 已有目录则复用，否则按当前时间戳新建
        existing_dirs = sorted(glob.glob(os.path.join(LOGS_SESSION_ANTHROPIC, f"*_{session_id}")))
        session_dir = existing_dirs[0] if existing_dirs else os.path.join(LOGS_SESSION_ANTHROPIC,
                                                                          f"{ts}_{session_id}")
        os.makedirs(session_dir, exist_ok=True)
        req_path = os.path.join(session_dir, f"{ts}-req.json")
        res_path = os.path.join(session_dir, f"{ts}-res.json")
        head_path = os.path.join(session_dir, f"{ts}-headers.json")
    else:
        os.makedirs(LOGS_ANTHROPIC, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")[:-3]  # 带毫秒，避免并发重名
        req_path = os.path.join(LOGS_ANTHROPIC, f"{ts}-req.json")
        res_path = os.path.join(LOGS_ANTHROPIC, f"{ts}-res.json")
        head_path = os.path.join(LOGS_ANTHROPIC, f"{ts}-headers.json")

    upstream_url = f"{os.environ['UPSTREAM_URL'].rstrip('/')}/messages"
    verify = _ssl_verify()

    x_auth_token = await get_x_auth_token(req)
    upstream_headers = build_upstream_headers(x_auth_token, model)
    body["model"] = upstream_headers['Model-Id']

    # 根据当前请求是否开启 ban_explore 来处理 Task 工具描述
    tools = _strip_task_explore_line(body.get("tools"), ban_explore=ban_explore)
    if tools is not None:
        body["tools"] = tools
    elif "tools" in body:
        body.pop("tools", None)

    # headers = dict(req.headers)
    headers = dict()
    headers.update(upstream_headers)
    _dump_json(head_path, headers)
    _dump_json(req_path, body)
    # ---- non-stream ----
    if not stream:
        async with httpx.AsyncClient(
                verify=verify,
                timeout=httpx.Timeout(500.0),
                trust_env=TRUST_ENV,
        ) as client:
            # 限流状态码重试逻辑：最多重试5次（与 /v1/messages 保持一致，详见 RATE_LIMIT_STATUS_CODES）
            max_retries = MAX_RETRIES
            r = None
            last_retry_response = None

            for attempt in range(max_retries):
                r = await client.post(upstream_url, headers=upstream_headers, json=body)

                if not is_rate_limit_status(r.status_code):
                    # 不是限流错误，直接使用这个响应
                    break

                # 是限流错误，保存响应用于最后返回
                last_retry_response = r
                logging.warning(f"{attempt} retryable response (chat/completions non-stream): {r.status_code} {r.text}")
                # 如果不是最后一次重试，等待后继续
                if attempt < max_retries - 1:
                    # 指数退避：1s, 2s, 4s, 8s
                    await asyncio.sleep(1 * (2 ** attempt))
                    # 重新获取 token（可能已更新）
                    x_auth_token = await get_x_auth_token(req)
                    upstream_headers = build_upstream_headers(x_auth_token, model)

            # 如果所有重试都是限流错误，使用最后一次的响应
            if is_rate_limit_status(r.status_code) and last_retry_response is not None:
                r = last_retry_response

        # 记录上下游响应（非流式）
        _dump_json(res_path, _resp_to_obj(r))

        # ✅ 上游错误透传（状态码+body 原样返回）
        return Response(
            content=r.content,
            status_code=r.status_code,
            media_type=r.headers.get("content-type", "application/json"),
        )

    # ---- stream SSE (OpenAI SSE pass-through) ----
    async def anthropic_sse_passthrough() -> AsyncIterator[bytes]:
        up_chunks: List[Any] = []
        stop_msg = f'event: message_stop\ndata: {json.dumps({"type": "message_stop"}, ensure_ascii=False)}\n\n'.encode(
            "utf-8")
        try:
            async with httpx.AsyncClient(
                    verify=verify,
                    timeout=httpx.Timeout(500.0),
                    trust_env=TRUST_ENV,
            ) as client:
                # Rate limit retry logic: max 5 retries (same as /v1/messages streaming)
                max_retries = MAX_RETRIES
                last_retry_err_text = None
                last_retry_status = None
                retry_headers = upstream_headers
                retry_token = x_auth_token
                connection_established = False

                # For tracking streaming response data
                has_valid_content = False
                content_buffer = ""

                for attempt in range(max_retries):
                    async with client.stream("POST", upstream_url, headers=retry_headers, json=body) as r:
                        meta = {
                            "type": "anthropic_passthrough_sse_meta",
                            "status_code": r.status_code,
                            "headers": dict(r.headers),
                        }
                        up_chunks.append(meta)

                        if is_rate_limit_status(r.status_code):
                            # Rate limit error, save error and close connection
                            err = await r.aread()
                            last_retry_err_text = err.decode("utf-8", errors="replace")
                            last_retry_status = r.status_code
                            up_chunks.append({"type": "error_body", "body": last_retry_err_text})
                            logging.warning(
                                f"{attempt} retryable response (anthropic stream): {r.status_code} {last_retry_err_text}")
                            # Close connection and prepare for retry
                            if attempt < max_retries - 1:
                                # Exponential backoff
                                await asyncio.sleep(1 * (2 ** attempt))
                                # Refresh token (might be updated)
                                retry_token = await get_x_auth_token(req)
                                retry_headers = build_upstream_headers(retry_token, model)
                            continue

                        # Not a rate limit error, continue processing
                        connection_established = True

                        # Handle other errors (non-406)
                        if r.status_code >= 400:
                            err = await r.aread()
                            err_text = err.decode("utf-8", errors="replace")
                            up_chunks.append({"type": "error_body", "body": err_text})
                            # Return Anthropic-formatted error response
                            error_data = {
                                "type": "error",
                                "error": {
                                    "type": "api_error",
                                    "message": err_text,
                                    "status_code": r.status_code
                                }
                            }

                            yield f"event: error\ndata: {json.dumps(error_data, ensure_ascii=False)}\n\n".encode(
                                "utf-8")
                            yield stop_msg
                            return

                        # Normal case, read stream data
                        res = r.aiter_lines()
                        async for line in res:
                            if not line:
                                continue

                            # Add to chunks log
                            # up_chunks.append(line)

                            # Check if it's an SSE event line
                            if line.startswith("event:"):
                                event_type = line[6:].strip()  # Remove "event:" prefix

                                # Wait for the corresponding data line
                                data_line = await res.__anext__()

                                if not data_line.startswith("data:"):
                                    continue

                                data_part = data_line[5:].strip()  # Remove "data:" prefix
                                up_chunks.append(json.loads(data_part))
                                # Handle different event types
                                if event_type == "ping":
                                    # Just forward pings
                                    yield f"event: ping\ndata: {data_part}\n\n".encode("utf-8")
                                    continue

                                if event_type == "message_stop":
                                    # Forward message_stop event
                                    yield stop_msg
                                    return

                                try:
                                    chunk_data = json.loads(data_part)

                                    # Handle content block deltas
                                    if event_type == "content_block_delta":
                                        if "delta" in chunk_data and "text" in chunk_data["delta"]:
                                            text = chunk_data["delta"]["text"]
                                            if text:
                                                has_valid_content = True
                                                content_buffer += text

                                        # Forward the event
                                        yield f"event: content_block_delta\ndata: {json.dumps(chunk_data, ensure_ascii=False)}\n\n".encode(
                                            "utf-8")

                                    # Handle message delta events
                                    elif event_type == "message_delta":
                                        if "delta" in chunk_data and "stop_reason" in chunk_data["delta"]:
                                            has_valid_content = True

                                        # Forward the event
                                        yield f"event: message_delta\ndata: {json.dumps(chunk_data, ensure_ascii=False)}\n\n".encode(
                                            "utf-8")

                                    # Handle content block start events
                                    elif event_type == "content_block_start":
                                        if "content_block" in chunk_data and chunk_data["content_block"].get("text"):
                                            has_valid_content = True

                                        # Forward the event
                                        yield f"event: content_block_start\ndata: {json.dumps(chunk_data, ensure_ascii=False)}\n\n".encode(
                                            "utf-8")

                                    # Handle other events
                                    else:
                                        yield f"event: {event_type}\ndata: {json.dumps(chunk_data, ensure_ascii=False)}\n\n".encode(
                                            "utf-8")

                                except json.JSONDecodeError:
                                    # If not valid JSON, forward as-is
                                    yield f"event: {event_type}\ndata: {data_part}\n\n".encode("utf-8")
                            else:
                                # For non-event lines, just forward
                                yield line.encode("utf-8") + b"\n"

                    # If connection was established and streaming completed, don't retry
                    if connection_established:
                        # If we never got valid content, send an empty message to prevent client hanging
                        if not has_valid_content:
                            empty_chunk = {
                                "type": "message",
                                "content": [],
                                "stop_reason": None,
                                "model": body.get("model", "unknown")
                            }
                            yield f"event: message\ndata: {json.dumps(empty_chunk, ensure_ascii=False)}\n\n".encode(
                                "utf-8")
                            yield stop_msg
                        break

                    # If not last retry and connection failed (406/429), wait and continue
                    if not connection_established and attempt < max_retries - 1:
                        # Exponential backoff
                        await asyncio.sleep(1 * (2 ** attempt))
                        # Refresh token
                        retry_token = await get_x_auth_token(req)
                        retry_headers = build_upstream_headers(retry_token, model)

                # If all retries were rate limit errors, return last error
                if not connection_established and last_retry_status is not None and is_rate_limit_status(
                        last_retry_status):
                    if last_retry_err_text is not None:
                        yield last_retry_err_text.encode("utf-8", errors="replace")
                    yield stop_msg
                    return
        finally:
            # Always log the chunks
            _dump_json(res_path, {"type": "anthropic_passthrough_sse_capture", "chunks": up_chunks})

    return StreamingResponse(anthropic_sse_passthrough(), media_type="text/event-stream")


# ---------- OpenAI Chat Completions ----------
@app.post("/chat/completions")
async def openai_chat_completions(req: Request):
    """
    OpenAI-compatible endpoint:
      - non-stream: upstream JSON pass-through
      - stream: upstream OpenAI SSE pass-through
    """
    body = await req.json()
    stream = bool(body.get("stream", False))
    body_model = body.get("model")
    ban_explore = BAN_EXPLORE

    model_from_body: Optional[str] = body_model if isinstance(body_model, str) else None
    suffix = "--ban_explore"
    if model_from_body and model_from_body.endswith(suffix):
        # 任何带 "--ban_explore" 后缀的模型名，都强制开启屏蔽 Explore
        ban_explore = True
        base_model = model_from_body[: -len(suffix)]
        model = _resolve_model_name(base_model or "byenv")
    else:
        model = _resolve_model_name(body_model)

    # session相关
    session_id = None
    session_metadata = body.get("metadata")
    if isinstance(session_metadata, dict):
        user_id = session_metadata.get("user_id") or ""
        m = re.search(r"session_([A-Za-z0-9-]+)", str(user_id))
        if m:
            session_id = m.group(1)
    else:
        # 适配codeagent获取session id
        session_id = req.headers.get("X-Session-Id")
    skip_session_logging = False
    if session_id:
        skip_session_logging = _should_skip_session_logging(body)

    # 保存请求/响应日志（OpenAI 直通）
    if session_id and not skip_session_logging:
        os.makedirs(LOGS_SESSION_OPENAI, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")[:-3]  # 带毫秒，避免并发重名
        # 若该 session_id 已有目录则复用，否则按当前时间戳新建
        existing_dirs = sorted(glob.glob(os.path.join(LOGS_SESSION_OPENAI, f"*_{session_id}")))
        session_dir = existing_dirs[0] if existing_dirs else os.path.join(LOGS_SESSION_OPENAI, f"{ts}_{session_id}")
        os.makedirs(session_dir, exist_ok=True)
        req_path = os.path.join(session_dir, f"{ts}-req.json")
        res_path = os.path.join(session_dir, f"{ts}-res.json")
        head_path = os.path.join(session_dir, f"{ts}-headers.json")
    else:
        os.makedirs(LOGS_OPENAI, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")[:-3]  # 带毫秒，避免并发重名
        req_path = os.path.join(LOGS_OPENAI, f"{ts}-req.json")
        res_path = os.path.join(LOGS_OPENAI, f"{ts}-res.json")
        head_path = os.path.join(LOGS_OPENAI, f"{ts}-headers.json")

    upstream_url = f"{os.environ['UPSTREAM_URL'].rstrip('/')}/chat/completions"
    verify = _ssl_verify()

    x_auth_token = await get_x_auth_token(req)
    upstream_headers = build_upstream_headers(x_auth_token, model)
    body["model"] = upstream_headers['Model-Id']
    # 根据当前请求是否开启 ban_explore 来处理 Task 工具描述
    tools = _strip_task_explore_line(body.get("tools"), ban_explore=ban_explore)
    if tools is not None:
        body["tools"] = tools
    elif "tools" in body:
        body.pop("tools", None)

    headers = dict(req.headers)
    headers.update(upstream_headers)
    _dump_json(head_path, headers)
    _dump_json(head_path, dict(req.headers))
    _dump_json(req_path, body)

    # ---- non-stream ----
    if not stream:
        async with httpx.AsyncClient(
                verify=verify,
                timeout=httpx.Timeout(500.0),
                trust_env=TRUST_ENV,
        ) as client:
            # 限流状态码重试逻辑：最多重试5次（与 /v1/messages 保持一致，详见 RATE_LIMIT_STATUS_CODES）
            max_retries = MAX_RETRIES
            r = None
            last_retry_response = None

            for attempt in range(max_retries):
                r = await client.post(upstream_url, headers=upstream_headers, json=body)

                if not is_rate_limit_status(r.status_code):
                    # 不是限流错误，直接使用这个响应
                    break

                # 是限流错误，保存响应用于最后返回
                last_retry_response = r
                logging.warning(f"{attempt} retryable response (chat/completions non-stream): {r.status_code} {r.text}")
                # 如果不是最后一次重试，等待后继续
                if attempt < max_retries - 1:
                    # 指数退避：1s, 2s, 4s, 8s
                    await asyncio.sleep(1 * (2 ** attempt))
                    # 重新获取 token（可能已更新）
                    x_auth_token = await get_x_auth_token(req)
                    upstream_headers = build_upstream_headers(x_auth_token, model)

            # 如果所有重试都是限流错误，使用最后一次的响应
            if is_rate_limit_status(r.status_code) and last_retry_response is not None:
                r = last_retry_response

        # 记录上下游响应（非流式）
        _dump_json(res_path, _resp_to_obj(r))

        # ✅ 上游错误透传（状态码+body 原样返回）
        return Response(
            content=r.content,
            status_code=r.status_code,
            media_type=r.headers.get("content-type", "application/json"),
        )

    # ---- stream SSE (OpenAI SSE pass-through) ----
    async def sse_passthrough() -> AsyncIterator[bytes]:
        up_chunks: List[Any] = []
        try:
            async with httpx.AsyncClient(
                    verify=verify,
                    timeout=httpx.Timeout(500.0),
                    trust_env=TRUST_ENV,
            ) as client:
                # 限流状态码重试逻辑：最多重试5次（与 /v1/messages 流式保持一致，详见 RATE_LIMIT_STATUS_CODES）
                max_retries = MAX_RETRIES
                last_retry_err_text = None
                last_retry_status = None
                retry_headers = upstream_headers
                retry_token = x_auth_token
                connection_established = False

                # 用于跟踪流式响应的数据
                has_valid_content = False
                content_buffer = ""

                for attempt in range(max_retries):
                    async with client.stream("POST", upstream_url, headers=retry_headers, json=body) as r:
                        meta = {
                            "type": "openai_passthrough_sse_meta",
                            "status_code": r.status_code,
                            "headers": dict(r.headers),
                        }
                        up_chunks.append(meta)

                        if is_rate_limit_status(r.status_code):
                            # 是限流错误，保存错误信息并关闭连接
                            err = await r.aread()
                            last_retry_err_text = err.decode("utf-8", errors="replace")
                            last_retry_status = r.status_code
                            up_chunks.append({"type": "error_body", "body": last_retry_err_text})
                            logging.warning(
                                f"{attempt} retryable response (chat/completions stream): {r.status_code} {last_retry_err_text}")
                            # 关闭连接，准备重试（进行下一次for循环）
                            if attempt < max_retries - 1:
                                # 指数退避
                                await asyncio.sleep(1 * (2 ** attempt))
                                # 重新获取 token（可能已更新）
                                retry_token = await get_x_auth_token(req)
                                retry_headers = build_upstream_headers(retry_token, model)
                            continue

                        # 不是限流错误，继续在这个连接上处理
                        connection_established = True

                        # 处理其他错误（非406）
                        if r.status_code >= 400:
                            err = await r.aread()
                            err_text = err.decode("utf-8", errors="replace")
                            up_chunks.append({"type": "error_body", "body": err_text})
                            # 返回符合OpenAI格式的错误响应
                            error_data = {
                                "id": "chatcmpl-error",
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": body.get("model", "unknown"),
                                "choices": [{
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": "error"
                                }]
                            }

                            # 如果是JSON格式的错误响应，尝试解析并包含详细信息
                            try:
                                error_json = json.loads(err_text)
                                if isinstance(error_json, dict):
                                    error_data["error"] = error_json
                            except:
                                # 如果不是有效的JSON，直接作为消息返回
                                error_data["error"] = {
                                    "message": err_text,
                                    "type": "upstream_error",
                                    "code": r.status_code
                                }

                            yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n".encode("utf-8")
                            yield b"data: [DONE]\n\n"
                            return

                        # 正常情况，读取流数据
                        async for line in r.aiter_lines():
                            if not line:
                                continue

                            # 添加到 chunks 日志中
                            # up_chunks.append(line)

                            # 检查是否是 SSE 数据行
                            if line.startswith("data:"):
                                data_part = line[5:].strip()  # 移除 "data:" 前缀
                                # 检查是否是结束标记
                                if data_part == "[DONE]":
                                    # 只有当我们收到了有效内容时才发送 DONE 标记
                                    if has_valid_content:
                                        yield b"data: [DONE]\n\n"
                                    break

                                # 尝试解析 JSON 数据
                                try:
                                    chunk_data = json.loads(data_part)
                                    up_chunks.append(copy.deepcopy(chunk_data))
                                    # 检查是否有有效的内容
                                    choices = chunk_data.get("choices", [])
                                    if choices and len(choices) > 0:
                                        choice = choices[0]
                                        delta = choice.get("delta", {})
                                        content = delta.get("content")
                                        reasoning_content = delta.get("reasoning_content")
                                        reasoning = delta.get("reasoning")
                                        tool_calls = delta.get("tool_calls")
                                        finish_reason = choice.get("finish_reason")

                                        # 检查是否有任何有效内容（content 或 reasoning_content 或 tool_calls）
                                        # 分别处理每个字段，避免使用 elif 导致某些字段被忽略
                                        if content is not None and content != "":
                                            has_valid_content = True
                                            content_buffer += content
                                        elif content is not None and content == "" and len(content_buffer) > 0:
                                            # 空字符串但前面有内容，也认为是有效的
                                            has_valid_content = True
                                        if reasoning_content is not None and reasoning_content != "":
                                            has_valid_content = True
                                            content_buffer += reasoning_content
                                        if reasoning is not None and reasoning != "":
                                            has_valid_content = True
                                            content_buffer += reasoning
                                        if tool_calls is not None and len(tool_calls) > 0:
                                            has_valid_content = True

                                        # 如果有 finish_reason，也标记为有效
                                        if finish_reason is not None:
                                            has_valid_content = True

                                    # 适配CodeaAgent代码：只在有finish_reason时保留usage
                                    # 移除中间chunk的usage信息，避免被CodeAgent代码误判
                                    if "usage" in chunk_data:
                                        choices = chunk_data.get("choices", [])
                                        has_finish_reason = False
                                        if choices and len(choices) > 0:
                                            finish_reason = choices[0].get("finish_reason")
                                            if finish_reason is not None:
                                                has_finish_reason = True

                                        # 如果没有finish_reason，移除usage字段
                                        if not has_finish_reason:
                                            chunk_data.pop("usage", None)

                                    # 检查是否有工具调用完成
                                    should_emit_tool_calls = False
                                    if finish_reason == "tool_calls" and choices:
                                        # 检查是否有任何tool_calls（哪怕空数组）
                                        tool_calls_flat = []
                                        for choice in choices:
                                            if isinstance(choice, dict):
                                                delta = choice.get("delta", {})
                                                tcs = delta.get("tool_calls", [])
                                                if isinstance(tcs, list):
                                                    tool_calls_flat.extend(tcs)

                                        # 如果有tool_calls（哪怕是空数组）且之前已经有tool_calls记录，则触发返回
                                        if tool_calls_flat or has_valid_content:
                                            should_emit_tool_calls = True

                                    # 传递处理后的数据行
                                    yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n".encode("utf-8")

                                    # 如果检测到tool_calls完成，立即发送[i/]D[/i]标记并结束流
                                    if should_emit_tool_calls:
                                        # 确保所有tool calls都已经输出
                                        if has_valid_content:
                                            # 发送最终的有效内容块以触发tool calls处理
                                            final_chunk = {
                                                "id": "chatcmpl-final",
                                                "object": "chat.completion.chunk",
                                                "created": chunk_data.get("created", int(time.time())),
                                                "model": chunk_data.get("model", body.get("model", "unknown")),
                                                "choices": [{
                                                    "index": 0,
                                                    "delta": {"content": ""},
                                                    "finish_reason": "tool_calls"
                                                }]
                                            }
                                            yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n".encode(
                                                "utf-8")

                                        # 立即发送DONE标记以结束流
                                        yield b"data: [DONE]\n\n"
                                        return
                                except json.JSONDecodeError:
                                    # 如果不是有效的 JSON，直接传递
                                    yield line.encode("utf-8") + b"\n\n"
                            else:
                                # 对于非数据行，直接传递
                                yield line.encode("utf-8") + b"\n"

                    # 如果已经成功建立连接且完成流式传输，则不再重试
                    if connection_established:
                        # 如果我们从未收到有效内容，添加一个最终的空内容块以防止客户端挂起
                        if not has_valid_content:
                            empty_chunk = {
                                "id": "chatcmpl-empty",
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": body.get("model", "unknown"),
                                "choices": [{
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": "stop"
                                }]
                            }
                            yield f"data: {json.dumps(empty_chunk, ensure_ascii=False)}\n\n".encode("utf-8")

                        # 确保发送 DONE 标记
                        yield b"data: [DONE]\n\n"
                        break

                    # 如果不是最后一次重试且连接失败（406/429），等待后继续
                    if not connection_established and attempt < max_retries - 1:
                        # 指数退避
                        await asyncio.sleep(1 * (2 ** attempt))
                        # 重新获取 token（可能已更新）
                        retry_token = await get_x_auth_token(req)
                        retry_headers = build_upstream_headers(retry_token, model)

                # 如果所有重试都是限流错误，返回最后一次错误
                if not connection_established and last_retry_status is not None and is_rate_limit_status(
                        last_retry_status):
                    # 直接把错误原样吐回（客户端一般也能看到）
                    if last_retry_err_text is not None:
                        yield last_retry_err_text.encode("utf-8", errors="replace")
                    # 确保发送 DONE 标记
                    yield b"data: [DONE]\n\n"
                    return
        finally:
            # 无论正常/异常/客户端断开，尽最大努力落盘
            _dump_json(res_path, {"type": "openai_passthrough_sse_capture", "chunks": up_chunks})

    return StreamingResponse(sse_passthrough(), media_type="text/event-stream")


# ===============================================
# 以下为新增的统计功能

@app.get("/")
async def index_statistic():
    return FileResponse(path="templates/statistic.html")


@app.get("/chat_viewer")
async def chat_viewer():
    return FileResponse(path="templates/chat-viewer.html")


@app.get("/statistic")
def statistic_tokens_web(model: str = '', date_start: str = '', date_end: str = '', status: str = '全部'):
    res = statistic_tokens(model=model, date_start=date_start, date_end=date_end, status=status)
    return JSONResponse(res)


@app.get("/logs/anthropic/list")
def logs_anthropic_list(min_messages: int = 10):
    """列出 logs_anthropic 目录下满足条件的 req.json 文件（含 messages 字段且消息数 > min_messages）"""
    result = []
    pattern = os.path.join(LOGS_ANTHROPIC, "*-req.json")
    for path in sorted(glob.glob(pattern), reverse=True):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            messages = data.get("messages")
            if not isinstance(messages, list) or len(messages) <= min_messages:
                continue
            filename = os.path.basename(path)
            result.append({
                "filename": filename,
                "message_count": len(messages),
                "model": data.get("model", ""),
            })
        except Exception:
            continue
    return JSONResponse(result)


@app.get("/logs/anthropic/file")
def logs_anthropic_file(filename: str):
    """返回 logs_anthropic 目录下指定文件的内容（仅允许 -req.json 文件）"""
    if not filename.endswith("-req.json") or "/" in filename or "\\" in filename or ".." in filename:
        return JSONResponse({"error": "invalid filename"}, status_code=400)
    path = os.path.join(LOGS_ANTHROPIC, filename)
    if not os.path.isfile(path):
        return JSONResponse({"error": "file not found"}, status_code=404)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return JSONResponse(data)


@app.get("/logs/openai/list")
def logs_openai_list(min_messages: int = 10):
    """列出 logs_openai 目录下满足条件的 req.json 文件（含 messages 字段且消息数 > min_messages）"""
    result = []
    pattern = os.path.join(LOGS_OPENAI, "*-req.json")
    for path in sorted(glob.glob(pattern), reverse=True):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            messages = data.get("messages")
            if not isinstance(messages, list) or len(messages) <= min_messages:
                continue
            filename = os.path.basename(path)
            result.append({
                "filename": filename,
                "message_count": len(messages),
                "model": data.get("model", ""),
            })
        except Exception:
            continue
    return JSONResponse(result)


@app.get("/logs/anthropic/aggregate")
def logs_anthropic_aggregate(min_messages: int = 1):
    """
    扫描 logs_anthropic 目录下所有 req+res 文件对，
    按时间戳排序后聚合成独立的对话链（每条链以第一条 user 消息为 key）。
    返回格式: [{ chain_id, first_time, last_time, file_count, messages: [...] }, ...]
    每条 message 附加 _source_file 和 _is_assistant_response 字段。
    """
    import re as _re

    def extract_res_content(res_path: str):
        """从 res.json 提取 assistant 回复的 content 列表"""
        if not os.path.isfile(res_path):
            return None
        try:
            with open(res_path, "r", encoding="utf-8") as f:
                d = json.load(f)
            # 非流式: {"json": {"content": [...], "role": "assistant", ...}}
            if isinstance(d, dict) and "json" in d:
                j = d["json"]
                if isinstance(j, dict) and j.get("role") == "assistant":
                    return j.get("content")
            # 流式: 列表 chunks，找最后一个有 message.content 的
            if isinstance(d, list):
                for chunk in reversed(d):
                    msg = chunk.get("message", {})
                    if isinstance(msg, dict) and msg.get("role") == "assistant" and msg.get("content") is not None:
                        return msg["content"]
        except Exception:
            pass
        return None

    def msg_fingerprint(msg):
        """消息指纹：role + content 前200字符，用于去重比较"""
        role = msg.get("role", "")
        c = msg.get("content", "")
        if isinstance(c, list):
            parts = []
            for b in c:
                if isinstance(b, dict):
                    parts.append(b.get("text") or b.get("id") or str(b)[:80])
            c = "|".join(parts)
        return f"{role}::{str(c)[:200]}"

    # 收集所有 req 文件
    req_files = sorted(glob.glob(os.path.join(LOGS_ANTHROPIC, "*-req.json")))

    # 按时间戳提取文件名前缀
    ts_pat = _re.compile(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_\d+)-req\.json$")

    entries = []
    for path in req_files:
        m = ts_pat.search(os.path.basename(path))
        if not m:
            continue
        ts_str = m.group(1)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        msgs = data.get("messages")
        if not isinstance(msgs, list) or len(msgs) < min_messages:
            continue
        res_path = path.replace("-req.json", "-res.json")
        entries.append({
            "ts": ts_str,
            "path": path,
            "res_path": res_path,
            "messages": msgs,
            "model": data.get("model", ""),
        })

    if not entries:
        return JSONResponse([])

    # --- 聚合逻辑 ---
    # Q1 完整内容作为 key（Claude Code 每次会话的第一条 user 消息含时间戳+任务，天然唯一）。
    # 同一 Q1 key 的所有文件属于同一条链，取消息数最多的文件作为完整轨迹。

    def get_q1_full(msgs):
        """提取第一条 user 消息的完整文本作为会话 key"""
        if not msgs:
            return ""
        m = msgs[0]
        c = m.get("content", "")
        if isinstance(c, list):
            parts = []
            for b in c:
                if isinstance(b, dict):
                    parts.append(b.get("text") or b.get("id") or str(b)[:200])
            c = "|".join(parts)
        return str(c)  # 完整内容，不截断

    from collections import OrderedDict
    chains_map = OrderedDict()  # q1_key -> { chain_id, entries, messages, model }

    for entry in entries:
        msgs = entry["messages"]
        if not msgs:
            continue
        q1_key = get_q1_full(msgs)
        if q1_key not in chains_map:
            chains_map[q1_key] = {
                "chain_id": len(chains_map),
                "q1_key": q1_key,
                "messages": msgs,
                "entries": [entry],
                "model": entry["model"],
            }
        else:
            chain = chains_map[q1_key]
            if len(msgs) > len(chain["messages"]):
                chain["messages"] = msgs
            chain["entries"].append(entry)

    chains = list(chains_map.values())

    # 构建返回结构
    result = []
    for chain in chains:
        first_ts = chain["entries"][0]["ts"]
        last_ts = chain["entries"][-1]["ts"]
        # 取最长消息列表所在 entry 的 res.json
        best_entry = max(chain["entries"], key=lambda e: len(e["messages"]))
        res_content = extract_res_content(best_entry["res_path"])

        # 构建完整 messages（history + assistant 回复）
        full_messages = list(chain["messages"])
        if res_content is not None:
            full_messages.append({
                "role": "assistant",
                "content": res_content,
                "_from_res": True,
                "_source_file": os.path.basename(best_entry["res_path"]),
            })

        result.append({
            "chain_id": chain["chain_id"],
            "first_time": first_ts.replace("_", " ", 1).replace("_", ".").replace("-", ":"),
            "last_time": last_ts.replace("_", " ", 1).replace("_", ".").replace("-", ":"),
            "file_count": len(chain["entries"]),
            "message_count": len(full_messages),
            "model": chain["model"],
            "messages": full_messages,
        })

    # 按 first_time 倒序
    result.sort(key=lambda x: x["first_time"], reverse=True)
    return JSONResponse(result)


@app.get("/logs/openai/aggregate")
def logs_openai_aggregate(min_messages: int = 1):
    """
    扫描 logs_openai 目录下所有 req 文件，按 Origin_query（system 消息中）或首条 user 消息聚合。
    返回格式与 /logs/anthropic/aggregate 相同。
    """
    import re as _re
    from collections import OrderedDict

    ts_pat = _re.compile(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_\d+)-req\.json$")

    def get_oai_chain_key(msgs):
        """用 system 消息里的 Origin_query 作为链 key，没有则用第一条 user 消息完整内容"""
        for m in msgs:
            if m.get("role") == "system":
                c = str(m.get("content", ""))
                match = _re.search(r"# Origin_query\s*\n+(.*?)(\n---|\Z)", c, _re.DOTALL)
                if match:
                    return "oq:" + match.group(1).strip()
        for m in msgs:
            if m.get("role") == "user":
                c = m.get("content", "")
                if isinstance(c, list):
                    c = "|".join(b.get("text", "") for b in c if isinstance(b, dict))
                return "u:" + str(c)
        return str(msgs[0].get("content", ""))

    def extract_oai_res_content(res_path):
        """从 OpenAI res.json 提取 assistant 消息（含 tool_calls）"""
        if not os.path.isfile(res_path):
            return None
        try:
            with open(res_path, "r", encoding="utf-8") as f:
                d = json.load(f)
            msg = d.get("json", {}).get("choices", [{}])[0].get("message")
            if msg and msg.get("role") == "assistant":
                return msg
        except Exception:
            pass
        return None

    req_files = sorted(glob.glob(os.path.join(LOGS_OPENAI, "*-req.json")))
    entries = []
    for path in req_files:
        m = ts_pat.search(os.path.basename(path))
        if not m:
            continue
        ts_str = m.group(1)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        msgs = data.get("messages")
        if not isinstance(msgs, list) or len(msgs) < min_messages:
            continue
        res_path = path.replace("-req.json", "-res.json")
        entries.append({
            "ts": ts_str,
            "path": path,
            "res_path": res_path,
            "messages": msgs,
            "model": data.get("model", ""),
        })

    if not entries:
        return JSONResponse([])

    chains_map = OrderedDict()
    for entry in entries:
        msgs = entry["messages"]
        key = get_oai_chain_key(msgs)
        if key not in chains_map:
            chains_map[key] = {
                "chain_id": len(chains_map),
                "chain_key": key,
                "messages": msgs,
                "entries": [entry],
                "model": entry["model"],
            }
        else:
            chain = chains_map[key]
            if len(msgs) > len(chain["messages"]):
                chain["messages"] = msgs
            chain["entries"].append(entry)

    result = []
    for chain in chains_map.values():
        first_ts = chain["entries"][0]["ts"]
        last_ts = chain["entries"][-1]["ts"]
        # 取消息最多的 entry 对应的 res.json
        best_entry = max(chain["entries"], key=lambda e: len(e["messages"]))
        res_msg = extract_oai_res_content(best_entry["res_path"])

        full_messages = list(chain["messages"])
        if res_msg is not None:
            full_messages.append({
                **res_msg,
                "_from_res": True,
                "_source_file": os.path.basename(best_entry["res_path"]),
            })

        result.append({
            "chain_id": chain["chain_id"],
            "first_time": first_ts.replace("_", " ", 1).replace("_", ".").replace("-", ":"),
            "last_time": last_ts.replace("_", " ", 1).replace("_", ".").replace("-", ":"),
            "file_count": len(chain["entries"]),
            "message_count": len(full_messages),
            "model": chain["model"],
            "messages": full_messages,
            "chain_key": chain["chain_key"],
        })

    result.sort(key=lambda x: x["first_time"], reverse=True)
    return JSONResponse(result)


@app.get("/logs/openai/file")
def logs_openai_file(filename: str):
    """返回 logs_openai 目录下指定文件的内容（仅允许 -req.json 文件）"""
    if not filename.endswith("-req.json") or "/" in filename or "\\" in filename or ".." in filename:
        return JSONResponse({"error": "invalid filename"}, status_code=400)
    path = os.path.join(LOGS_OPENAI, filename)
    if not os.path.isfile(path):
        return JSONResponse({"error": "file not found"}, status_code=404)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return JSONResponse(data)


if __name__ == "__main__":
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser(description="Start the LLM proxy")
    parser.add_argument(
        "--ban_explore",
        action="store_true",
        help="Remove '- Explore:' line from Task tool descriptions in /v1/messages",
    )
    parser.add_argument(
        "--ban_stream",
        action="store_true",
        help="Disable stream requests for anthropic api /v1/messages",
    )
    args = parser.parse_args()

    if args.ban_explore:
        os.environ["BAN_EXPLORE"] = "true"

    if args.ban_stream:
        os.environ["BAN_STREAM"] = "true"

    host = os.getenv("PROXY_HOST", "127.0.0.1")
    port = int(os.getenv("PROXY_PORT", "4000"))
    uvicorn.run("app:app", host=host, port=port, log_level="info")
