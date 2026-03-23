"""
export_sessions.py — 将 logs_anthropic 中的三元组文件导出为 session 格式

用法:
    # 全量导出
    python export_sessions.py --src /path/to/logs_anthropic --out /path/to/output

    # 增量导出（基于上次输出继续）
    python export_sessions.py --src /path/to/logs_anthropic --out /path/to/new_output \\
                              --base-output /path/to/prev_output

增量逻辑:
  读取 --base-output/index.json，找出最新已处理时间戳作为 cutoff。
  只处理 cutoff 之后的新三元组，分两种情况：
  - 已有 session 的续接（[Q1,Q2,...]）：将 base-output 中的文件夹复制到新 out，追加新文件
  - 全新 session（纯 Q1）：直接在 out 中创建新文件夹
  base-output 中无变化的 session 文件夹也原样复制到 out，保证 out 是完整输出。

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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kwargs):
        desc = kwargs.get("desc", "")
        total = kwargs.get("total", None)
        if desc:
            print(f"{desc}...")
        return it


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
    parser.add_argument("--out", "-o", required=True, help="输出目录")
    parser.add_argument("--base-output", "-b", default=None,
                        help="上次的输出目录（增量模式），含 index.json")
    args = parser.parse_args()

    src = Path(args.src).resolve()
    out = Path(args.out).resolve()
    if not src.is_dir():
        sys.exit(f"[error] 源目录不存在: {src}")
    out.mkdir(parents=True, exist_ok=True)

    # ── 加载 base-output（增量模式）──────────────────────────────
    base_out: Optional[Path] = Path(args.base_output).resolve() if args.base_output else None
    base_index: List[dict] = []
    cutoff: Optional[str] = None   # 最新已处理时间戳前缀，如 "2026-03-17_17-21-00_871"

    if base_out:
        if not base_out.is_dir():
            sys.exit(f"[error] base-output 目录不存在: {base_out}")
        idx_path = base_out / "index.json"
        if not idx_path.exists():
            sys.exit(f"[error] base-output 中未找到 index.json: {idx_path}")
        with open(idx_path, "r", encoding="utf-8") as f:
            base_index = json.load(f)
        # cutoff = 所有 latest_file / folder 中最大的时间戳前缀
        ts_candidates = []
        for entry in base_index:
            if entry.get("folder"):
                ts_candidates.append(entry["folder"])
            lf = entry.get("latest_file", "")
            if lf.endswith(".json"):
                ts_candidates.append(lf[:-5])
        cutoff = max(ts_candidates) if ts_candidates else None
        print(f"[info] 增量模式：base-output={base_out}，cutoff={cutoff}")

    # ── 收集三元组并过滤 ─────────────────────────────────────────
    triplets = collect_triplets(src)
    if not triplets:
        sys.exit("[error] 未找到任何 req/headers/res 文件")

    all_prefixes = sorted(triplets)
    if cutoff:
        new_prefixes = [p for p in all_prefixes if p > cutoff]
        print(f"[info] 总三元组 {len(all_prefixes)}，cutoff 后新增 {len(new_prefixes)} 个")
    else:
        new_prefixes = all_prefixes

    # ── 构建 session 列表 ────────────────────────────────────────
    # sessions: list of dict，每个 dict 含：
    #   folder_prefix, q1, items, from_base(bool)
    sessions: List[dict] = []
    latest_session_by_q1: Dict[str, dict] = {}

    # 增量模式：将 base_index 的 session 预填入 latest_session_by_q1
    # 新的纯 Q1 请求会覆盖同 Q1 的旧 session；[Q1,Q2...] 会续接最近的
    if base_out:
        for entry in base_index:
            q1 = entry.get("q1", "")
            if not q1:
                continue
            session = {
                "folder_prefix": entry["folder"],
                "q1": q1,
                "items": [],
                "from_base": True,
            }
            sessions.append(session)
            latest_session_by_q1[q1] = session

    skipped = 0
    for prefix in new_prefixes:
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
            # 纯 Q1 → 开启新 session（覆盖 latest_session_by_q1 中同 Q1 的旧记录）
            session = {"folder_prefix": prefix, "q1": q1, "items": [(prefix, tri)], "from_base": False}
            sessions.append(session)
            latest_session_by_q1[q1] = session
        else:
            # [Q1, Q2, ...] → 续接最近同 Q1 的 session
            session = latest_session_by_q1.get(q1)
            if session is None:
                session = {"folder_prefix": prefix, "q1": q1, "items": [], "from_base": False}
                sessions.append(session)
                latest_session_by_q1[q1] = session
            session["items"].append((prefix, tri))

    print(f"[info] 共 {len(sessions)} 个 session，跳过 {skipped} 个三元组")

    # ── 导出 ─────────────────────────────────────────────────────
    active_sessions = [s for s in sessions if s.get("items")]
    index_entries: List[dict] = []
    all_items: List[Tuple] = [
        (s, prefix, tri)
        for s in active_sessions
        for prefix, tri in sorted(s["items"], key=lambda x: x[0])
    ]

    # 预先创建所有 session 目录（避免多线程竞争 mkdir）
    for s in active_sessions:
        (out / s["folder_prefix"]).mkdir(parents=True, exist_ok=True)

    def _export_one(task: Tuple) -> Tuple[str, str, Optional[dict]]:
        """返回 (folder_prefix, out_filename, merged_data | None)"""
        session, prefix, tri = task
        try:
            req_data = load_json(tri["req"])
        except Exception as e:
            print(f"[warn] 读取 req 失败 {prefix}: {e}")
            return session["folder_prefix"], f"{prefix}.json", None

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
                merged["response"] = parse_response(load_json(tri["res"]))
            except Exception as e:
                print(f"[warn] 解析 res 失败 {prefix}: {e}")
                merged["response"] = {}
        else:
            merged["response"] = {}

        out_file = out / session["folder_prefix"] / f"{prefix}.json"
        with open(out_file, "w", encoding="utf-8") as fh:
            json.dump(merged, fh, ensure_ascii=False, indent=2)
        return session["folder_prefix"], f"{prefix}.json", merged

    # 按 folder_prefix 收集写出结果，用于后续计算 index entry
    results: Dict[str, List[Tuple[str, dict]]] = {s["folder_prefix"]: [] for s in active_sessions}
    exported_files = 0

    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(_export_one, task): task for task in all_items}
        with tqdm(total=len(all_items), desc="导出文件", unit="file") as bar:
            for future in as_completed(futures):
                folder_prefix, filename, merged = future.result()
                bar.update(1)
                if merged is not None:
                    results[folder_prefix].append((filename, merged))
                    exported_files += 1

    # 为每个 active session 构建 index entry（取 msg_count 最多的文件）
    for session in active_sessions:
        best_file: Optional[str] = None
        best_msg_count = -1
        best_model = ""

        # 续接旧 session：先以 base_index 中的记录为基准
        if session.get("from_base"):
            for e in base_index:
                if e.get("folder") == session["folder_prefix"]:
                    best_file = e.get("latest_file")
                    best_msg_count = e.get("msg_count", -1)
                    best_model = e.get("model", "")
                    break

        for filename, merged in results[session["folder_prefix"]]:
            messages = extract_messages(merged) or []
            has_response = bool((merged.get("response") or {}).get("content"))
            msg_count = len(messages) + (1 if has_response else 0)
            if msg_count > best_msg_count:
                best_msg_count = msg_count
                best_file = filename
                best_model = merged.get("model", "")

        if best_file:
            index_entries.append({
                "folder": session["folder_prefix"],
                "q1": session["q1"],
                "latest_file": best_file,
                "msg_count": best_msg_count,
                "model": best_model,
            })

    # ── 写 index.json ─────────────────────────────────────────────
    index_path = out / "index.json"
    with open(index_path, "w", encoding="utf-8") as fh:
        json.dump(index_entries, fh, ensure_ascii=False, indent=2)

    print(f"[done] 导出 {exported_files} 个新文件，{len(index_entries)} 个 session → {out}")
    print(f"[done] index.json 已生成: {index_path}")


if __name__ == "__main__":
    main()
