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
from auth import validate_api_key
from utils.metrics import record_request, get_metrics_snapshot
from utils.log_routes import register_log_routes

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
    await validate_api_key(req)
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
        r = None
        last_exception = None
        success = False
        try:
            async with httpx.AsyncClient(
                    verify=verify,
                    timeout=httpx.Timeout(500.0),
                    trust_env=TRUST_ENV,
            ) as client:
                for attempt in range(MAX_RETRIES):
                    try:
                        r = await client.post(upstream_url, headers=upstream_headers, json=body)
                        last_exception = None
                        if not is_rate_limit_status(r.status_code):
                            success = True
                            break
                        logging.warning(f"Attempt {attempt} rate limit (anthropic non-stream): {r.status_code} {r.text}")
                    except Exception as e:
                        last_exception = e
                        logging.warning(f"Attempt {attempt} upstream error (anthropic non-stream): {e}")

                    if attempt < MAX_RETRIES - 1:
                        await asyncio.sleep(0.5 )
                        x_auth_token = await get_x_auth_token(req)
                        upstream_headers = build_upstream_headers(x_auth_token, model)
        except Exception as e:
            last_exception = e
            logging.error(f"Failed to create httpx client (anthropic non-stream): {e}")

        if not success:
            error_msg = str(last_exception) if last_exception else (f"HTTP {r.status_code}" if r else "unknown")
            logging.error(f"All retries exhausted (anthropic non-stream): {error_msg}")
            _dump_json(res_path, {"error": "max_retries_exceeded", "detail": error_msg})
            return JSONResponse(
                status_code=502,
                content={"type": "error", "error": {"type": "max_retries_exceeded", "message": f"上游多次失败({MAX_RETRIES}次): {error_msg}"}},
            )

        _dump_json(res_path, _resp_to_obj(r))
        try:
            resp_json = r.json()
            usage = resp_json.get("usage", {})
            tok_in = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
            tok_out = usage.get("output_tokens") or usage.get("completion_tokens") or 0
            record_request(tok_in, tok_out, success=r.status_code < 400)
        except Exception:
            record_request(success=r.status_code < 400)
        return Response(
            content=r.content,
            status_code=r.status_code,
            media_type=r.headers.get("content-type", "application/json"),
        )

    # ---- stream SSE (pure pass-through) ----
    async def anthropic_sse_passthrough() -> AsyncIterator[bytes]:
        up_chunks: List[Any] = []
        connection_established = False
        last_exception = None
        last_retry_status = None
        retry_headers = upstream_headers
        retry_token = x_auth_token

        try:
            async with httpx.AsyncClient(
                    verify=verify,
                    timeout=httpx.Timeout(500.0),
                    trust_env=TRUST_ENV,
            ) as client:
                # Retry loop: only retries BEFORE any bytes are yielded to the client
                for attempt in range(MAX_RETRIES):
                    try:
                        async with client.stream("POST", upstream_url, headers=retry_headers, json=body) as r:
                            up_chunks.append({
                                "type": "anthropic_passthrough_sse_meta",
                                "status_code": r.status_code,
                                "headers": dict(r.headers),
                            })

                            if is_rate_limit_status(r.status_code):
                                err = await r.aread()
                                last_retry_err_text = err.decode("utf-8", errors="replace")
                                last_retry_status = r.status_code
                                up_chunks.append({"type": "error_body", "body": last_retry_err_text})
                                logging.warning(f"Attempt {attempt} rate limit (anthropic stream): {r.status_code}")
                                if attempt < MAX_RETRIES - 1:
                                    await asyncio.sleep(0.5)
                                    retry_token = await get_x_auth_token(req)
                                    retry_headers = build_upstream_headers(retry_token, model)
                                continue

                            # Connection established — from here we yield directly, no more retries
                            connection_established = True

                            if r.status_code >= 400:
                                err = await r.aread()
                                err_text = err.decode("utf-8", errors="replace")
                                up_chunks.append({"type": "error_body", "body": err_text})
                                error_data = {"type": "error", "error": {"type": "api_error", "message": err_text}}
                                yield f"event: error\ndata: {json.dumps(error_data, ensure_ascii=False)}\n\n".encode("utf-8")
                                return

                            # Pure pass-through: tee raw bytes to client and capture for logging
                            raw_buf = bytearray()
                            async for raw in r.aiter_bytes():
                                raw_buf.extend(raw)
                                yield raw

                            # Parse captured SSE for logging (best-effort)
                            for line in raw_buf.decode("utf-8", errors="replace").splitlines():
                                if line.startswith("data:"):
                                    data_part = line[5:].strip()
                                    if data_part and data_part != "[DONE]":
                                        try:
                                            up_chunks.append(json.loads(data_part))
                                        except json.JSONDecodeError:
                                            pass
                            return

                    except Exception as e:
                        if connection_established:
                            # Already streaming to client, can't retry
                            logging.warning(f"Stream interrupted (anthropic stream): {e}")
                            return
                        last_exception = e
                        logging.warning(f"Attempt {attempt} upstream error (anthropic stream): {e}")
                        if attempt < MAX_RETRIES - 1:
                            await asyncio.sleep(0.5)
                            retry_token = await get_x_auth_token(req)
                            retry_headers = build_upstream_headers(retry_token, model)

                # All retries exhausted without connecting
                error_msg = str(last_exception) if last_exception else (f"HTTP {last_retry_status}" if last_retry_status else "unknown")
                logging.error(f"All retries exhausted (anthropic stream): {error_msg}")
                err_event = {"type": "error", "error": {"type": "max_retries_exceeded", "message": f"上游多次失败({MAX_RETRIES}次): {error_msg}"}}
                yield f"event: error\ndata: {json.dumps(err_event, ensure_ascii=False)}\n\n".encode("utf-8")

        except Exception as e:
            logging.error(f"Failed to create httpx client (anthropic stream): {e}")
            err_event = {"type": "error", "error": {"type": "connection_error", "message": str(e)}}
            yield f"event: error\ndata: {json.dumps(err_event, ensure_ascii=False)}\n\n".encode("utf-8")
        finally:
            _dump_json(res_path, {"type": "anthropic_passthrough_sse_capture", "chunks": up_chunks})
            _tok_in, _tok_out = 0, 0
            for _c in up_chunks:
                if isinstance(_c, dict):
                    _u = _c.get("message", {}).get("usage") or _c.get("usage") or {}
                    _tok_in = _tok_in or (_u.get("input_tokens") or 0)
                    _tok_out = _tok_out or (_u.get("output_tokens") or 0)
            record_request(_tok_in, _tok_out, success=connection_established)


    return StreamingResponse(anthropic_sse_passthrough(), media_type="text/event-stream")


# ---------- OpenAI Chat Completions ----------
@app.post("/chat/completions")
async def openai_chat_completions(req: Request):
    """
    OpenAI-compatible endpoint:
      - non-stream: upstream JSON pass-through
      - stream: upstream OpenAI SSE pass-through
    """
    await validate_api_key(req)
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
        r = None
        last_exception = None
        success = False
        try:
            async with httpx.AsyncClient(
                    verify=verify,
                    timeout=httpx.Timeout(500.0),
                    trust_env=TRUST_ENV,
            ) as client:
                for attempt in range(MAX_RETRIES):
                    try:
                        r = await client.post(upstream_url, headers=upstream_headers, json=body)
                        last_exception = None
                        if not is_rate_limit_status(r.status_code):
                            success = True
                            break
                        logging.warning(f"Attempt {attempt} rate limit (openai non-stream): {r.status_code} {r.text}")
                    except Exception as e:
                        last_exception = e
                        logging.warning(f"Attempt {attempt} upstream error (openai non-stream): {e}")

                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(0.5 )
                    x_auth_token = await get_x_auth_token(req)
                    upstream_headers = build_upstream_headers(x_auth_token, model)
        except Exception as e:
            last_exception = e
            logging.error(f"Failed to create httpx client (openai non-stream): {e}")

        if not success:
            error_msg = str(last_exception) if last_exception else (f"HTTP {r.status_code}" if r else "unknown")
            logging.error(f"All retries exhausted (openai non-stream): {error_msg}")
            _dump_json(res_path, {"error": "max_retries_exceeded", "detail": error_msg})
            return JSONResponse(
                status_code=502,
                content={"error": {"message": f"上游多次失败({MAX_RETRIES}次): {error_msg}", "type": "max_retries_exceeded"}},
            )

        _dump_json(res_path, _resp_to_obj(r))
        try:
            resp_json = r.json()
            usage = resp_json.get("usage", {})
            tok_in = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
            tok_out = usage.get("output_tokens") or usage.get("completion_tokens") or 0
            record_request(tok_in, tok_out, success=r.status_code < 400)
        except Exception:
            record_request(success=r.status_code < 400)
        return Response(
            content=r.content,
            status_code=r.status_code,
            media_type=r.headers.get("content-type", "application/json"),
        )

    # ---- stream SSE (OpenAI SSE pass-through) ----
    async def sse_passthrough() -> AsyncIterator[bytes]:
        up_chunks: List[Any] = []
        connection_established = False
        last_exception = None
        last_retry_status = None
        retry_headers = upstream_headers
        retry_token = x_auth_token

        try:
            async with httpx.AsyncClient(
                    verify=verify,
                    timeout=httpx.Timeout(500.0),
                    trust_env=TRUST_ENV,
            ) as client:
                for attempt in range(MAX_RETRIES):
                    try:
                        async with client.stream("POST", upstream_url, headers=retry_headers, json=body) as r:
                            up_chunks.append({
                                "type": "openai_passthrough_sse_meta",
                                "status_code": r.status_code,
                                "headers": dict(r.headers),
                            })

                            if is_rate_limit_status(r.status_code):
                                err = await r.aread()
                                last_retry_err_text = err.decode("utf-8", errors="replace")
                                last_retry_status = r.status_code
                                up_chunks.append({"type": "error_body", "body": last_retry_err_text})
                                logging.warning(f"Attempt {attempt} rate limit (openai stream): {r.status_code}")
                                if attempt < MAX_RETRIES - 1:
                                    await asyncio.sleep(0.5)
                                    retry_token = await get_x_auth_token(req)
                                    retry_headers = build_upstream_headers(retry_token, model)
                                continue

                            connection_established = True

                            if r.status_code >= 400:
                                err = await r.aread()
                                err_text = err.decode("utf-8", errors="replace")
                                up_chunks.append({"type": "error_body", "body": err_text})
                                error_data = {"error": {"message": err_text, "type": "api_error", "code": r.status_code}}
                                yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n".encode("utf-8")
                                yield b"data: [DONE]\n\n"
                                return

                            # Pure pass-through: tee raw bytes to client and capture for logging
                            raw_buf = bytearray()
                            async for raw in r.aiter_bytes():
                                raw_buf.extend(raw)
                                yield raw

                            # Parse captured SSE for logging (best-effort)
                            for line in raw_buf.decode("utf-8", errors="replace").splitlines():
                                if line.startswith("data:"):
                                    data_part = line[5:].strip()
                                    if data_part and data_part != "[DONE]":
                                        try:
                                            up_chunks.append(json.loads(data_part))
                                        except json.JSONDecodeError:
                                            pass
                            return

                    except Exception as e:
                        if connection_established:
                            logging.warning(f"Stream interrupted (openai stream): {e}")
                            return
                        last_exception = e
                        logging.warning(f"Attempt {attempt} upstream error (openai stream): {e}")
                        if attempt < MAX_RETRIES - 1:
                            await asyncio.sleep(0.5)
                            retry_token = await get_x_auth_token(req)
                            retry_headers = build_upstream_headers(retry_token, model)

                # All retries exhausted
                error_msg = str(last_exception) if last_exception else (f"HTTP {last_retry_status}" if last_retry_status else "unknown")
                logging.error(f"All retries exhausted (openai stream): {error_msg}")
                error_data = {"error": {"message": f"上游多次失败({MAX_RETRIES}次): {error_msg}", "type": "max_retries_exceeded"}}
                yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n".encode("utf-8")
                yield b"data: [DONE]\n\n"

        except Exception as e:
            logging.error(f"Failed to create httpx client (openai stream): {e}")
            error_data = {"error": {"message": str(e), "type": "connection_error"}}
            yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n".encode("utf-8")
            yield b"data: [DONE]\n\n"
        finally:
            # 无论正常/异常/客户端断开，尽最大努力落盘
            _dump_json(res_path, {"type": "openai_passthrough_sse_capture", "chunks": up_chunks})
            # 统计 token（从 usage chunk 提取）
            _tok_in, _tok_out = 0, 0
            for _c in up_chunks:
                if isinstance(_c, dict):
                    _u = _c.get("usage") or {}
                    _tok_in = _tok_in or (_u.get("prompt_tokens") or 0)
                    _tok_out = _tok_out or (_u.get("completion_tokens") or 0)
            record_request(_tok_in, _tok_out, success=connection_established)

    return StreamingResponse(sse_passthrough(), media_type="text/event-stream")


# ===============================================
# 以下为新增的统计功能

@app.get("/")
async def index_statistic():
    return FileResponse(path="templates/statistic.html")


@app.get("/history")
async def chat_viewer():
    return FileResponse(path="templates/chat-viewer.html")

@app.get("/statistic")
def statistic_tokens_web(model: str = '', date_start: str = '', date_end: str = '', status: str = '全部'):
    res = statistic_tokens(model=model, date_start=date_start, date_end=date_end, status=status)
    return JSONResponse(res)


@app.get("/metrics/realtime")
def metrics_realtime():
    """返回最近 120 分钟的 RPM/TPM 数据"""
    return JSONResponse(get_metrics_snapshot())


register_log_routes(app)


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
