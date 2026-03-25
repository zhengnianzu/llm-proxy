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
from utils.metrics import record_request, record_validity, get_metrics_snapshot, get_rate_history
from utils.log_routes import register_log_routes

load_dotenv(os.environ.get("ENV_FILE", ".env"), override=True)

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
LOGS_DEBUG = os.path.join("logs", "debug")
INDEX_FILE_ANTHROPIC = os.path.join(LOGS_ANTHROPIC, "index.jsonl")
INDEX_FILE_OPENAI    = os.path.join(LOGS_OPENAI,    "index.jsonl")

# 请求计数（启动时从 index.jsonl 加载，运行时在内存中累计）
_first_count: int = 0   # 首次请求数（每次 endpoint 调用 = 1）
_total_count: int = 0   # 总体上游请求数（含重试）
_valid_count: int = 0   # 有效响应数（获得有效 Anthropic 内容的首次请求）

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


_VALID_CONTENT_TYPES = {"text", "thinking", "tool_use"}


def _has_valid_content(resp_json: dict) -> bool:
    """非流式：content 数组中至少有一个有效内容块（text/thinking/tool_use 非空）。"""
    for block in resp_json.get("content", []):
        btype = block.get("type")
        if btype == "text" and block.get("text"):
            return True
        if btype == "thinking" and block.get("thinking"):
            return True
        if btype == "tool_use" and block.get("input") is not None:
            return True
    return False


def _has_valid_sse_content(chunks: list) -> bool:
    """流式：SSE chunks 中至少有一个有效 delta（text/thinking/tool_use input）。"""
    for chunk in chunks:
        if chunk.get("type") == "content_block_delta":
            delta = chunk.get("delta", {})
            dtype = delta.get("type")
            if dtype == "text_delta" and delta.get("text"):
                return True
            if dtype == "thinking_delta" and delta.get("thinking"):
                return True
            if dtype == "input_json_delta" and delta.get("partial_json") is not None:
                return True
    return False


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


def _write_debug(ts: str, attempt: int, model: str, reason: str, body: str):
    """将失败尝试的原始响应写入 logs/debug/ 目录，便于排查问题。返回文件名。"""
    try:
        os.makedirs(LOGS_DEBUG, exist_ok=True)
        safe_model = model.replace("/", "_").replace(":", "_")
        filename = f"{ts}_attempt{attempt}_{safe_model}_{reason}.txt"
        path = os.path.join(LOGS_DEBUG, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(body)
        return filename
    except Exception as ex:
        logging.warning(f"Failed to write debug log: {ex}")
        return None


def _resp_to_obj(r):  # httpx.Response -> dict
    base = {"status_code": r.status_code, "headers": dict(r.headers)}
    try:
        base["json"] = r.json()
    except Exception:
        base["text"] = r.text
    return base


def _load_index_anthropic():
    """启动时从 index.jsonl 恢复历史计数，避免重启后归零。"""
    global _first_count, _total_count, _valid_count
    if not os.path.exists(INDEX_FILE_ANTHROPIC):
        return
    with open(INDEX_FILE_ANTHROPIC, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                _first_count += 1
                _total_count += entry.get("total_attempts", 1)
                if entry.get("valid"):
                    _valid_count += 1
            except json.JSONDecodeError:
                pass


def _append_index_anthropic(ts: str, req_file: str, total_attempts: int, valid: bool, model: str = "", tok_in: int = 0, tok_out: int = 0):
    """追加一条请求记录到 index.jsonl，并更新内存计数。"""
    global _first_count, _total_count, _valid_count
    entry = {
        "ts": ts,
        "req_file": req_file,
        "model": model,
        "total_attempts": total_attempts,
        "retried": total_attempts > 1,
        "valid": valid,
        "tok_in": tok_in,
        "tok_out": tok_out,
    }
    os.makedirs(LOGS_ANTHROPIC, exist_ok=True)
    with open(INDEX_FILE_ANTHROPIC, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    _first_count += 1
    _total_count += total_attempts
    if valid:
        _valid_count += 1


def _append_index_openai(ts: str, req_file: str, model: str = "", tok_in: int = 0, tok_out: int = 0, success: bool = True):
    """追加 OpenAI 请求记录到 index.jsonl。"""
    entry = {
        "ts": ts,
        "req_file": req_file,
        "model": model,
        "tok_in": tok_in,
        "tok_out": tok_out,
        "success": success,
    }
    os.makedirs(LOGS_OPENAI, exist_ok=True)
    with open(INDEX_FILE_OPENAI, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# 启动时加载历史 index
_load_index_anthropic()


def _sanitize_messages(messages: Any) -> Any:
    """
    清洗 messages 列表，过滤掉空 text content block。
    Anthropic API 要求 text content blocks 必须非空，否则返回 400。
    - content 为 list 时：过滤 type==text 且 text 为空/空白的块；
      若过滤后 list 为空，用单个空格占位块替代，避免整条消息丢失。
    - content 为字符串时：原样保留（空字符串同理由上游决定，非我们的问题范围）。
    """
    if not isinstance(messages, list):
        return messages
    cleaned = []
    for msg in messages:
        if not isinstance(msg, dict):
            cleaned.append(msg)
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            cleaned.append(msg)
            continue
        # thinking/redacted_thinking blocks in assistant messages must remain exactly
        # as they were in the original response; skip sanitization for such messages.
        if msg.get("role") == "assistant" and any(
            isinstance(b, dict) and b.get("type") in ("thinking", "redacted_thinking")
            for b in content
        ):
            cleaned.append(msg)
            continue
        def _is_empty_text_block(b: Any) -> bool:
            return isinstance(b, dict) and b.get("type") == "text" and not (b.get("text") or "").strip()

        def _sanitize_block(b: Any) -> Any:
            """对 tool_result 块递归清理其嵌套 content 中的空 text 块。"""
            if not isinstance(b, dict) or b.get("type") != "tool_result":
                return b
            nested = b.get("content")
            if not isinstance(nested, list):
                return b
            filtered_nested = [nb for nb in nested if not _is_empty_text_block(nb)]
            if len(filtered_nested) == len(nested):
                return b
            return {**b, "content": filtered_nested if filtered_nested else [{"type": "text", "text": " "}]}

        new_blocks = [_sanitize_block(b) for b in content if not _is_empty_text_block(b)]
        if len(new_blocks) == len(content) and all(new_blocks[i] is content[i] for i in range(len(content))):
            cleaned.append(msg)
        elif new_blocks:
            cleaned.append({**msg, "content": new_blocks})
        else:
            # 所有块都被过滤掉了，用占位符保留消息结构
            cleaned.append({**msg, "content": [{"type": "text", "text": " "}]})
    return cleaned


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

    # 清洗空 text content block，避免上游 400 "text content blocks must be non-empty"
    if "messages" in body:
        body["messages"] = _sanitize_messages(body["messages"])

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
        final_valid = False
        upstream_attempts = 0
        try:
            async with httpx.AsyncClient(
                    verify=verify,
                    timeout=httpx.Timeout(500.0),
                    trust_env=TRUST_ENV,
            ) as client:
                for attempt in range(MAX_RETRIES):
                    upstream_attempts += 1
                    try:
                        r = await client.post(upstream_url, headers=upstream_headers, json=body)
                        last_exception = None
                        if r.status_code == 200:
                            try:
                                resp_json = r.json()
                                if _has_valid_content(resp_json):
                                    success = True
                                    final_valid = True
                                    break
                                logging.warning(f"Attempt {attempt} empty content (anthropic non-stream), retrying: {r.text[:200]}")
                                _dbg = _write_debug(ts, attempt, model, "empty_content", r.text[:2000])
                                if _dbg: logging.warning(f"  -> debug: {_dbg}")
                            except Exception:
                                # JSON 解析失败：透传原始响应
                                success = True
                                break
                        else:
                            # 非 200 一律重试，最大次数后透传
                            _dbg = _write_debug(ts, attempt, model, f"http_{r.status_code}", r.text[:2000])
                            logging.warning(f"Attempt {attempt} non-200 (anthropic non-stream): {r.status_code} {r.text[:200]}" + (f" -> debug: {_dbg}" if _dbg else ""))
                    except Exception as e:
                        last_exception = e
                        logging.warning(f"Attempt {attempt} upstream error (anthropic non-stream): {e}")

                    if attempt < MAX_RETRIES - 1:
                        await asyncio.sleep(0.5)
                        x_auth_token = await get_x_auth_token(req)
                        upstream_headers = build_upstream_headers(x_auth_token, model)
        except Exception as e:
            last_exception = e
            logging.error(f"Failed to create httpx client (anthropic non-stream): {e}")

        if not success:
            if r is not None:
                # 透传上游最后一次错误响应
                error_msg = f"HTTP {r.status_code}"
                logging.error(f"All retries exhausted (anthropic non-stream), passing through: {error_msg}")
                _dump_json(res_path, _resp_to_obj(r))
                _append_index_anthropic(ts, req_path, upstream_attempts, False, model)
                record_validity(False, model)
                record_request(0, 0, success=False, model=model)
                return Response(
                    content=r.content,
                    status_code=r.status_code,
                    media_type=r.headers.get("content-type", "application/json"),
                )
            else:
                error_msg = str(last_exception) if last_exception else "unknown"
                logging.error(f"All retries exhausted (anthropic non-stream): {error_msg}")
                _dump_json(res_path, {"error": "max_retries_exceeded", "detail": error_msg})
                _append_index_anthropic(ts, req_path, upstream_attempts, False, model)
                record_validity(False, model)
                return JSONResponse(
                    status_code=502,
                    content={"type": "error", "error": {"type": "max_retries_exceeded", "message": f"上游多次失败({MAX_RETRIES}次): {error_msg}"}},
                )

        _dump_json(res_path, _resp_to_obj(r))
        tok_in, tok_out = 0, 0
        try:
            resp_json = r.json()
            usage = resp_json.get("usage", {})
            tok_in = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
            tok_out = usage.get("output_tokens") or usage.get("completion_tokens") or 0
            record_request(tok_in, tok_out, success=r.status_code < 400, model=model)
        except Exception:
            record_request(success=r.status_code < 400, model=model)
        _append_index_anthropic(ts, req_path, upstream_attempts, final_valid, model, tok_in, tok_out)
        record_validity(final_valid, model)
        return Response(
            content=r.content,
            status_code=r.status_code,
            media_type=r.headers.get("content-type", "application/json"),
        )

    # ---- stream SSE (pure pass-through) ----
    async def anthropic_sse_passthrough() -> AsyncIterator[bytes]:
        up_chunks: List[Any] = []
        connection_established = False
        upstream_attempts = 0
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
                    upstream_attempts += 1
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
                                _dbg = _write_debug(ts, attempt, model, "rate_limit", last_retry_err_text[:2000])
                                logging.warning(f"Attempt {attempt} rate limit (anthropic stream): {r.status_code}" + (f" -> debug: {_dbg}" if _dbg else ""))
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

                            # Stream with early commit on message_start:
                            # Buffer until we see message_start, then flush + pass-through directly.
                            # If stream ends without message_start → no valid response → retry.
                            log_buf = bytearray()   # full capture for logging
                            raw_buf = bytearray()   # pre-commit buffer only
                            committed = False
                            line_buf = ""

                            async for raw in r.aiter_bytes():
                                log_buf.extend(raw)
                                if committed:
                                    yield raw
                                else:
                                    raw_buf.extend(raw)
                                    line_buf += raw.decode("utf-8", errors="replace")
                                    while "\n" in line_buf:
                                        line, line_buf = line_buf.split("\n", 1)
                                        if line.startswith("data:"):
                                            data_part = line[5:].strip()
                                            if data_part and data_part != "[DONE]":
                                                try:
                                                    if json.loads(data_part).get("type") == "message_start":
                                                        committed = True
                                                        connection_established = True
                                                        yield bytes(raw_buf)
                                                        raw_buf = bytearray()
                                                        break
                                                except json.JSONDecodeError:
                                                    pass

                            if not committed:
                                raw_text = raw_buf.decode("utf-8", errors="replace")
                                if attempt < MAX_RETRIES - 1:
                                    _dbg = _write_debug(ts, attempt, model, "no_message_start", raw_text[:4000])
                                    logging.warning(f"Attempt {attempt} no message_start in SSE (anthropic stream), retrying" + (f" -> debug: {_dbg}" if _dbg else ""))
                                    connection_established = False
                                    up_chunks.clear()
                                    await asyncio.sleep(0.5)
                                    retry_token = await get_x_auth_token(req)
                                    retry_headers = build_upstream_headers(retry_token, model)
                                    continue
                                else:
                                    connection_established = True
                                    yield bytes(raw_buf)

                            # Parse full log_buf for logging
                            for line in log_buf.decode("utf-8", errors="replace").splitlines():
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
            record_request(_tok_in, _tok_out, success=connection_established, model=model)
            _append_index_anthropic(ts, req_path, upstream_attempts, connection_established, model)
            record_validity(connection_established, model)


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
        upstream_attempts = 0
        try:
            async with httpx.AsyncClient(
                    verify=verify,
                    timeout=httpx.Timeout(500.0),
                    trust_env=TRUST_ENV,
            ) as client:
                for attempt in range(MAX_RETRIES):
                    upstream_attempts += 1
                    try:
                        r = await client.post(upstream_url, headers=upstream_headers, json=body)
                        last_exception = None
                        if r.status_code == 200:
                            success = True
                            break
                        # 非 200 一律重试，最大次数后透传
                        _dbg = _write_debug(ts, attempt, model, f"http_{r.status_code}", r.text[:2000])
                        logging.warning(f"Attempt {attempt} non-200 (openai non-stream): {r.status_code} {r.text[:200]}" + (f" -> debug: {_dbg}" if _dbg else ""))
                    except Exception as e:
                        last_exception = e
                        logging.warning(f"Attempt {attempt} upstream error (openai non-stream): {e}")

                    if attempt < MAX_RETRIES - 1:
                        await asyncio.sleep(0.5)
                        x_auth_token = await get_x_auth_token(req)
                        upstream_headers = build_upstream_headers(x_auth_token, model)
        except Exception as e:
            last_exception = e
            logging.error(f"Failed to create httpx client (openai non-stream): {e}")

        if not success:
            if r is not None:
                # 透传上游最后一次错误响应
                error_msg = f"HTTP {r.status_code}"
                logging.error(f"All retries exhausted (openai non-stream), passing through: {error_msg}")
                _dump_json(res_path, _resp_to_obj(r))
                _append_index_openai(ts, req_path, model=model, success=False)
                record_request(0, 0, success=False, model=model)
                return Response(
                    content=r.content,
                    status_code=r.status_code,
                    media_type=r.headers.get("content-type", "application/json"),
                )
            else:
                error_msg = str(last_exception) if last_exception else "unknown"
                logging.error(f"All retries exhausted (openai non-stream): {error_msg}")
                _dump_json(res_path, {"error": "max_retries_exceeded", "detail": error_msg})
                _append_index_openai(ts, req_path, model=model, success=False)
                return JSONResponse(
                    status_code=502,
                    content={"error": {"message": f"上游多次失败({MAX_RETRIES}次): {error_msg}", "type": "max_retries_exceeded"}},
                )

        _dump_json(res_path, _resp_to_obj(r))
        tok_in, tok_out = 0, 0
        try:
            resp_json = r.json()
            usage = resp_json.get("usage", {})
            tok_in = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
            tok_out = usage.get("output_tokens") or usage.get("completion_tokens") or 0
            record_request(tok_in, tok_out, success=r.status_code < 400, model=model)
        except Exception:
            record_request(success=r.status_code < 400, model=model)
        _append_index_openai(ts, req_path, model=model, tok_in=tok_in, tok_out=tok_out, success=r.status_code < 400)
        return Response(
            content=r.content,
            status_code=r.status_code,
            media_type=r.headers.get("content-type", "application/json"),
        )

    # ---- stream SSE (OpenAI SSE pass-through) ----
    async def sse_passthrough() -> AsyncIterator[bytes]:
        up_chunks: List[Any] = []
        connection_established = False
        upstream_attempts = 0
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
                    upstream_attempts += 1
                    try:
                        async with client.stream("POST", upstream_url, headers=retry_headers, json=body) as r:
                            up_chunks.append({
                                "type": "openai_passthrough_sse_meta",
                                "status_code": r.status_code,
                                "headers": dict(r.headers),
                            })

                            if r.status_code != 200:
                                # 非 200 一律重试，最大次数后透传
                                err = await r.aread()
                                last_retry_err_text = err.decode("utf-8", errors="replace")
                                last_retry_status = r.status_code
                                up_chunks.append({"type": "error_body", "body": last_retry_err_text})
                                _dbg = _write_debug(ts, attempt, model, f"http_{r.status_code}", last_retry_err_text[:2000])
                                logging.warning(f"Attempt {attempt} non-200 (openai stream): {r.status_code}" + (f" -> debug: {_dbg}" if _dbg else ""))
                                if attempt < MAX_RETRIES - 1:
                                    await asyncio.sleep(0.5)
                                    retry_token = await get_x_auth_token(req)
                                    retry_headers = build_upstream_headers(retry_token, model)
                                continue

                            connection_established = True

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
            record_request(_tok_in, _tok_out, success=connection_established, model=model)
            _append_index_openai(ts, req_path, model=model, tok_in=_tok_in, tok_out=_tok_out, success=connection_established)

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


@app.get("/metrics/index-stats")
def index_stats():
    """返回 Anthropic 请求的首次/总体/有效次数及成功率。"""
    rate = (_valid_count / _first_count) if _first_count > 0 else 0.0
    return JSONResponse({
        "first_count": _first_count,
        "total_count": _total_count,
        "valid_count": _valid_count,
        "success_rate": round(rate, 4),
        "index_file": INDEX_FILE_ANTHROPIC,
    })


@app.get("/metrics/rate-history")
def rate_history():
    """返回最近 60 分钟的有效率时序数据。"""
    return JSONResponse(get_rate_history())


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
