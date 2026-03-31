"""
utils/triplet_collector.py — logs_anthropic 三元组收集 & session index.jsonl 读取

优先从 index.jsonl 读取（增量友好），降级到 rglob 全量扫描。

logs_anthropic/index.jsonl 格式（每行一条）:
    {"ts": "2026-03-25_15-42-10_366",
     "req_file": "logs_anthropic/2026-03-25_15-42-10_366-req.json",
     ...}
    req_file 仅用于提供 req 文件名，实际定位时使用 src / basename(req_file)。

session_dir/index.jsonl 格式（每行一条，与 index.json 的 entry 结构一致）:
    {"folder": "2026-03-25_15-42-10_366", "q1": "...",
     "latest_file": "2026-03-25_15-42-10_366.json", "msg_count": 5, "model": "..."}
    每次 session 新增或更新时追加一行。

三元组结构:
    {timestamp_prefix: {"req": Path, "headers": Path, "res": Path}}
    headers/res 可能缺失，由调用方决定如何处理。
"""

import json
import re
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

TIMESTAMP_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_\d{3})-(req|headers|res)\.json$"
)

# ---------------------------------------------------------------------------
# index.jsonl 读取
# ---------------------------------------------------------------------------

def _iter_index_entries(
    src: Path,
    start_line: int = 0,
) -> Tuple[List[dict], int]:
    """
    读取 src/index.jsonl，返回 (entries, total_lines)。
    entries: start_line 之后的所有有效行解析结果列表。
    仅返回 valid=True 且含 ts/req_file 的条目。
    """
    index_path = src / "index.jsonl"
    entries: List[dict] = []
    total = 0
    with open(index_path, "r", encoding="utf-8") as f:
        for i, raw in enumerate(f):
            total += 1
            if i < start_line:
                continue
            line = raw.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not entry.get("ts") or not entry.get("req_file"):
                continue
            entries.append(entry)
    return entries, total


def collect_triplets_from_index(
    src: Path,
    start_line: int = 0,
) -> Tuple[Dict[str, Dict[str, Path]], int]:
    """
    从 index.jsonl 收集三元组。根据每条 entry 的 ts 推导 headers/res 路径。

    Args:
        src: logs_anthropic 目录
        start_line: 从第几行开始读（增量模式用）

    Returns:
        (triplets, total_lines)
        triplets: {ts_prefix: {"req": Path, "headers": Path?, "res": Path?}}
    """
    entries, total = _iter_index_entries(src, start_line)

    triplets: Dict[str, Dict[str, Path]] = {}
    for entry in entries:
        ts = entry["ts"]
        req_path = src / Path(entry["req_file"]).name
        if not req_path.is_file():
            continue
        tri: Dict[str, Path] = {"req": req_path}
        # headers/res 与 req 在同一目录，名称由 ts 推导
        parent = req_path.parent
        headers_path = parent / f"{ts}-headers.json"
        res_path = parent / f"{ts}-res.json"
        if headers_path.is_file():
            tri["headers"] = headers_path
        if res_path.is_file():
            tri["res"] = res_path
        triplets[ts] = tri
    return triplets, total


# ---------------------------------------------------------------------------
# rglob 全量扫描（降级路径）
# ---------------------------------------------------------------------------

def collect_triplets_by_scan(src: Path) -> Dict[str, Dict[str, Path]]:
    """
    遍历 src 目录，按文件名正则匹配收集三元组。
    在没有 index.jsonl 时使用。
    """
    triplets: Dict[str, Dict[str, Path]] = {}
    for f in src.rglob("*.json"):
        m = TIMESTAMP_RE.match(f.name)
        if m:
            prefix, kind = m.group(1), m.group(2)
            if prefix not in triplets:
                triplets[prefix] = {}
            triplets[prefix][kind] = f
    return triplets


# ---------------------------------------------------------------------------
# 统一入口
# ---------------------------------------------------------------------------

def collect_triplets(
    src: Path,
    start_line: int = 0,
) -> Tuple[Dict[str, Dict[str, Path]], Optional[int]]:
    """
    优先使用 index.jsonl，降级到 rglob。

    Args:
        src: logs_anthropic 目录
        start_line: 仅在 index.jsonl 模式下生效，用于增量读取

    Returns:
        (triplets, total_index_lines)
        - triplets: {ts_prefix: {"req": Path, ...}}
        - total_index_lines: index.jsonl 总行数（rglob 模式下为 None）
    """
    index_path = src / "index.jsonl"
    if index_path.exists():
        triplets, total = collect_triplets_from_index(src, start_line)
        return triplets, total
    else:
        return collect_triplets_by_scan(src), None


def collect_new_triplets(
    src: Path,
    cutoff_ts: Optional[str],
    index_line_offset: int = 0,
) -> Tuple[Dict[str, Dict[str, Path]], List[str], Optional[int]]:
    """
    收集 cutoff_ts 之后的新三元组（按时间戳前缀字典序比较）。

    Args:
        src: logs_anthropic 目录
        cutoff_ts: 上次处理的最大时间戳前缀，None 表示全量
        index_line_offset: index.jsonl 已读行数（增量模式下可跳过已处理行）

    Returns:
        (new_triplets, sorted_new_prefixes, total_index_lines)
        - new_triplets: 仅包含 prefix > cutoff_ts 的条目
        - sorted_new_prefixes: 按时间戳升序排列的前缀列表
        - total_index_lines: index.jsonl 总行数（rglob 模式下为 None）
    """
    triplets, total = collect_triplets(src, start_line=index_line_offset)
    all_prefixes = sorted(triplets)
    if cutoff_ts:
        new_prefixes = [p for p in all_prefixes if p > cutoff_ts]
        new_triplets = {p: triplets[p] for p in new_prefixes}
    else:
        new_prefixes = all_prefixes
        new_triplets = triplets
    return new_triplets, new_prefixes, total


# ---------------------------------------------------------------------------
# session_dir/index.jsonl 读取（upload-only 模式）
# ---------------------------------------------------------------------------

def read_session_index_jsonl(
    session_dir: Path,
    start_line: int = 0,
) -> Tuple[List[dict], int]:
    """
    读取 session_dir/index.jsonl，返回 start_line 之后的新增 session 条目。

    每行格式（与 index.json entry 一致）:
        {"folder": "...", "q1": "...", "latest_file": "...", "msg_count": N, "model": "..."}

    Returns:
        (new_entries, total_lines)
        - new_entries: start_line 之后的有效条目（含 folder 字段）
        - total_lines: 文件总行数（用于下次调用的 start_line）
    """
    index_path = session_dir / "index.jsonl"
    entries: List[dict] = []
    total = 0
    with open(index_path, "r", encoding="utf-8") as f:
        for i, raw in enumerate(f):
            total += 1
            if i < start_line:
                continue
            line = raw.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("folder"):
                entries.append(entry)
    return entries, total
