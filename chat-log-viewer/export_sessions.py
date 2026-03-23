"""
export_sessions.py — 将 logs_anthropic 中的三元组文件导出为 session 格式

用法:
    python export_sessions.py --src /path/to/logs_anthropic --out /path/to/logs_session_anthropic

分组逻辑:
  按时间戳顺序处理所有三元组：
  - messages 里只有 1 条 user 消息（纯 Q1）→ 开启新 session，文件夹以该时间戳命名
  - messages 里有多条 user 消息（[Q1, Q2, ...]）→ 归入最近一次同 Q1 的 session
  - 相同 Q1 文本在不同时间出现的纯 Q1 请求，各自开启独立 session

每个 API call 生成一个 json 文件：
  req 内容作为主体
  + "header" 字段 ← headers.json 内容
  + "response" 字段 ← res.json 解析后的完整 message 对象
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Q1 提取（与 server.py 保持一致的去噪逻辑）
# ---------------------------------------------------------------------------

def get_first_user_text(messages: List[dict]) -> str:
    for msg in messages:
        if msg.get("role") != "user":
            continue
        c = msg.get("content", "")
        if isinstance(c, str):
            text = c
        elif isinstance(c, list):
            parts = []
            for b in c:
                if isinstance(b, dict):
                    parts.append(b.get("text") or b.get("id") or "")
            text = "\n".join(p for p in parts if p)
        else:
            text = str(c)

        # 反复去除头部噪声直到稳定
        while True:
            prev = text
            text = re.sub(r"^\s*\[[^\]]*\]\s*", "", text)
            text = re.sub(
                r"^Sender\s*(?:\([^)]*\))?:\s*```json\s*\{[\s\S]*?\}\s*```\s*",
                "", text, flags=re.IGNORECASE,
            )
            text = re.sub(r"^Sender\s*(?:\([^)]*\))?:[^\n]*\n?", "", text, flags=re.IGNORECASE)
            if text == prev:
                break

        return text.strip()
    return ""


def extract_messages(data: Any) -> Optional[List[dict]]:
    if isinstance(data, dict) and isinstance(data.get("messages"), list):
        return data["messages"]
    if isinstance(data, list) and data and isinstance(data[0], dict) and "role" in data[0]:
        return data
    return None


# ---------------------------------------------------------------------------
# 流式 res 解析：将 SSE chunks 重组为完整 message 对象
# ---------------------------------------------------------------------------

def parse_streaming_response(chunks: List[dict]) -> dict:
    """从 anthropic_passthrough_sse_capture 的 chunks 中重建完整 message。"""
    message: dict = {}
    blocks: dict = {}          # index -> block dict
    block_json_buf: dict = {}  # index -> partial_json string (for tool_use)

    for chunk in chunks:
        t = chunk.get("type")

        if t == "message_start":
            message = dict(chunk.get("message", {}))
            message["content"] = []

        elif t == "content_block_start":
            idx = chunk.get("index", 0)
            cb = dict(chunk.get("content_block", {}))
            blocks[idx] = cb
            if cb.get("type") == "tool_use":
                block_json_buf[idx] = ""

        elif t == "content_block_delta":
            idx = chunk.get("index", 0)
            delta = chunk.get("delta", {})
            dtype = delta.get("type")
            if dtype == "text_delta":
                blocks.setdefault(idx, {"type": "text", "text": ""})
                blocks[idx]["text"] = blocks[idx].get("text", "") + delta.get("text", "")
            elif dtype == "input_json_delta":
                block_json_buf[idx] = block_json_buf.get(idx, "") + delta.get("partial_json", "")

        elif t == "content_block_stop":
            idx = chunk.get("index", 0)
            if idx in block_json_buf:
                try:
                    blocks[idx]["input"] = json.loads(block_json_buf[idx])
                except json.JSONDecodeError:
                    blocks[idx]["input"] = block_json_buf[idx]

        elif t == "message_delta":
            delta = chunk.get("delta", {})
            if "stop_reason" in delta:
                message["stop_reason"] = delta["stop_reason"]
            if "stop_sequence" in delta:
                message["stop_sequence"] = delta["stop_sequence"]
            # 合并 usage
            if "usage" in chunk:
                base_usage = dict(message.get("usage") or {})
                base_usage.update(chunk["usage"])
                message["usage"] = base_usage

    # 按 index 顺序组装 content
    if blocks:
        message["content"] = [blocks[i] for i in sorted(blocks)]

    return message


def parse_response(res_data: dict) -> dict:
    """统一解析 res.json，返回规范化的 response 对象。"""
    rtype = res_data.get("type")

    if rtype == "anthropic_passthrough_sse_capture":
        chunks = res_data.get("chunks", [])
        # 找出第一个 meta chunk 的 status_code
        status_code = 200
        for c in chunks:
            if c.get("type") == "anthropic_passthrough_sse_meta":
                status_code = c.get("status_code", 200)
                break
        msg = parse_streaming_response(
            [c for c in chunks if c.get("type") != "anthropic_passthrough_sse_meta"]
        )
        return {"status_code": status_code, **msg}

    else:
        # 非流式：res.json = {"status_code":..., "headers":..., "json":{...}}
        body = dict(res_data.get("json") or {})
        body["status_code"] = res_data.get("status_code", 200)
        return body


# ---------------------------------------------------------------------------
# 主逻辑
# ---------------------------------------------------------------------------

TIMESTAMP_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_\d{3})-(req|headers|res)\.json$")


def collect_triplets(src: Path) -> Dict[str, Dict[str, Path]]:
    """返回 {timestamp_prefix: {"req": Path, "headers": Path, "res": Path}}"""
    triplets: Dict[str, Dict[str, Path]] = {}
    for f in src.rglob("*.json"):
        m = TIMESTAMP_RE.match(f.name)
        if m:
            prefix, kind = m.group(1), m.group(2)
            if prefix not in triplets:
                triplets[prefix] = {}
            triplets[prefix][kind] = f
    return triplets


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def count_user_messages(messages: List[dict]) -> int:
    return sum(1 for m in messages if m.get("role") == "user")


def main():
    parser = argparse.ArgumentParser(description="Export logs_anthropic → logs_session_anthropic")
    parser.add_argument("--src", "-s", required=True, help="logs_anthropic 目录")
    parser.add_argument("--out", "-o", required=True, help="输出目录 logs_session_anthropic")
    args = parser.parse_args()

    src = Path(args.src).resolve()
    out = Path(args.out).resolve()
    if not src.is_dir():
        sys.exit(f"[error] 源目录不存在: {src}")
    out.mkdir(parents=True, exist_ok=True)

    triplets = collect_triplets(src)
    if not triplets:
        sys.exit("[error] 未找到任何 req/headers/res 文件")

    # 按时间戳顺序处理，构建 session 列表
    # sessions: list of {"folder_prefix": str, "q1": str, "items": [(prefix, tri), ...]}
    sessions: List[dict] = []
    # latest_session_by_q1: q1_text -> 最近一个以该 Q1 开头的 session dict
    latest_session_by_q1: Dict[str, dict] = {}
    skipped = 0

    for prefix in sorted(triplets):
        tri = triplets[prefix]
        if "req" not in tri:
            skipped += 1
            continue
        try:
            req_data = load_json(tri["req"])
        except Exception as e:
            print(f"[warn] 读取 req 失败 {tri['req']}: {e}")
            skipped += 1
            continue

        messages = extract_messages(req_data)
        if not messages:
            skipped += 1
            continue

        q1 = get_first_user_text(messages)
        user_count = count_user_messages(messages)

        if user_count <= 1:
            # 纯 Q1 请求 → 开启新 session
            session = {"folder_prefix": prefix, "q1": q1, "items": [(prefix, tri)]}
            sessions.append(session)
            latest_session_by_q1[q1] = session
        else:
            # [Q1, Q2, ...] 请求 → 归入最近同 Q1 的 session
            session = latest_session_by_q1.get(q1)
            if session is None:
                # 没有找到前置 Q1 session，单独建一个
                session = {"folder_prefix": prefix, "q1": q1, "items": []}
                sessions.append(session)
                latest_session_by_q1[q1] = session
            session["items"].append((prefix, tri))

    print(f"[info] 共 {len(triplets)} 个三元组，分为 {len(sessions)} 个 session，跳过 {skipped} 个")

    # 导出
    exported_files = 0
    index_entries: List[dict] = []

    for session in sessions:
        session_dir = out / session["folder_prefix"]
        session_dir.mkdir(parents=True, exist_ok=True)

        best_file: Optional[str] = None   # msg_count 最多的文件名
        best_msg_count = -1
        best_model = ""

        for prefix, tri in sorted(session["items"], key=lambda x: x[0]):
            try:
                req_data = load_json(tri["req"])
            except Exception as e:
                print(f"[warn] 读取 req 失败 {prefix}: {e}")
                continue

            merged = dict(req_data)

            if "headers" in tri:
                try:
                    merged["header"] = load_json(tri["headers"])
                except Exception as e:
                    print(f"[warn] 读取 headers 失败 {prefix}: {e}")
                    merged["header"] = {}
            else:
                merged["header"] = {}

            if "res" in tri:
                try:
                    res_data = load_json(tri["res"])
                    merged["response"] = parse_response(res_data)
                except Exception as e:
                    print(f"[warn] 解析 res 失败 {prefix}: {e}")
                    merged["response"] = {}
            else:
                merged["response"] = {}

            out_file = session_dir / f"{prefix}.json"
            with open(out_file, "w", encoding="utf-8") as fh:
                json.dump(merged, fh, ensure_ascii=False, indent=2)
            exported_files += 1

            # 追踪 msg_count 最多的文件作为 latest_file
            messages = extract_messages(req_data) or []
            has_response = bool((merged.get("response") or {}).get("content"))
            msg_count = len(messages) + (1 if has_response else 0)
            if msg_count > best_msg_count:
                best_msg_count = msg_count
                best_file = f"{prefix}.json"
                best_model = req_data.get("model", "")

        if best_file:
            index_entries.append({
                "folder": session["folder_prefix"],
                "q1": session["q1"],
                "latest_file": best_file,
                "msg_count": best_msg_count,
                "model": best_model,
            })

    # 写入 index.json
    index_path = out / "index.json"
    with open(index_path, "w", encoding="utf-8") as fh:
        json.dump(index_entries, fh, ensure_ascii=False, indent=2)

    print(f"[done] 导出 {exported_files} 个文件，{len(index_entries)} 个 session → {out}")
    print(f"[done] index.json 已生成: {index_path}")


if __name__ == "__main__":
    main()
