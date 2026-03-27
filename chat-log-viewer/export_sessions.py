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

from utils.message_utils import (
    count_user_messages,
    extract_messages,
    get_first_user_text,
    load_json,
    parse_response,
)
from utils.triplet_collector import collect_new_triplets


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
    cutoff: Optional[str] = None

    if base_out:
        if not base_out.is_dir():
            sys.exit(f"[error] base-output 目录不存在: {base_out}")
        idx_path = base_out / "index.json"
        if not idx_path.exists():
            sys.exit(f"[error] base-output 中未找到 index.json: {idx_path}")
        with open(idx_path, "r", encoding="utf-8") as f:
            base_index = json.load(f)
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
    new_triplets, new_prefixes, _ = collect_new_triplets(src, cutoff)

    if not new_triplets and not base_index:
        sys.exit("[error] 未找到任何 req/headers/res 文件")

    total_msg = f"总三元组 cutoff 后新增 {len(new_prefixes)} 个" if cutoff else f"全量 {len(new_prefixes)} 个三元组"
    print(f"[info] {total_msg}")

    # ── 构建 session 列表 ────────────────────────────────────────
    sessions: List[dict] = []
    latest_session_by_q1: Dict[str, dict] = {}

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
        tri = new_triplets[prefix]
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
            session = {"folder_prefix": prefix, "q1": q1, "items": [(prefix, tri)], "from_base": False}
            sessions.append(session)
            latest_session_by_q1[q1] = session
        else:
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

    for s in active_sessions:
        (out / s["folder_prefix"]).mkdir(parents=True, exist_ok=True)

    def _export_one(task: Tuple) -> Tuple[str, str, Optional[dict]]:
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

    for session in active_sessions:
        best_file: Optional[str] = None
        best_msg_count = -1
        best_model = ""

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
