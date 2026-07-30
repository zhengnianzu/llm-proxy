"""
utils/log_scan.py — 可变深度日志目录发现 + 索引字段归一化

支持两种目录布局，统一以「含 index.jsonl 的叶子目录」为扫描单元：
  - 本项目:  {root}/{hour}/index.jsonl              (root = logs_all/env-xxx)
  - new-api: {root}/{day}/{hour}/index.jsonl        (root = .../logs/details)

叶子目录的稳定标识 dir_key = 相对 root 的 posix 路径（如 "26072520" 或 "260727/26072717"）。

索引条目字段在两种格式间有差异，normalize_entry 统一为:
  ts, model, api_key, tok_in, tok_out, cache_in, success, channel_key,
  q1_hash, chain_key, req_file
"""

import json
from pathlib import Path
from typing import Iterator, Optional

_INDEX_NAME = "index.jsonl"
_MAX_DEPTH = 3  # root 自身算 depth 0；new-api 需要 depth 2


def iter_index_dirs(root: Path, max_depth: int = _MAX_DEPTH) -> Iterator[Path]:
    """递归查找 root 下所有直接包含 index.jsonl 的目录（含 root 自身）。

    找到含 index.jsonl 的目录后不再深入其子目录。
    """
    if not root.is_dir():
        return

    def _walk(d: Path, depth: int) -> Iterator[Path]:
        if (d / _INDEX_NAME).is_file():
            yield d
            return
        if depth >= max_depth:
            return
        try:
            children = sorted(d.iterdir())
        except OSError:
            return
        for sub in children:
            if sub.is_dir() and not sub.name.startswith("."):
                yield from _walk(sub, depth + 1)

    yield from _walk(root, 0)


def dir_key_for(root: Path, leaf: Path) -> str:
    """叶子目录相对 root 的标识；用于索引缓存的键。

    折叠"末段与父段同名"的冗余层：某些 new-api 源把真叶子多套一层同名子目录
    （如 260727/26072717/26072717/index.jsonl），此处归一化为标准 天/小时
    （260727/26072717），与其它源口径一致。与 resolve_leaf 成对，可逆。
    """
    try:
        rel = leaf.relative_to(root).as_posix()
    except ValueError:
        return leaf.name
    parts = rel.split("/")
    if len(parts) >= 2 and parts[-1] == parts[-2]:
        rel = "/".join(parts[:-1])
    return rel


def resolve_leaf(root, dir_key: str) -> Path:
    """dir_key（可能已被 dir_key_for 折叠）反解为真实叶子绝对路径。

    先试 root/dir_key（标准结构直接命中）；若该目录不含 index.jsonl 但存在
    同名子目录 root/dir_key/<末段> 且含 index.jsonl，则补回被折叠的层。
    与 dir_key_for 成对。
    """
    root_path = Path(root)
    cand = root_path / dir_key
    if (cand / _INDEX_NAME).is_file():
        return cand
    if dir_key:
        last = dir_key.split("/")[-1]
        nested = cand / last
        if (nested / _INDEX_NAME).is_file():
            return nested
    return cand


def detect_format(root: str) -> str:
    """采样一个叶子的 index.jsonl 首条，判断 'native' / 'newapi' / 'unknown' / 'empty'。"""
    root_path = Path(root)
    if not root_path.is_dir():
        return "missing"
    for leaf in iter_index_dirs(root_path):
        entry = _first_entry(leaf / _INDEX_NAME)
        if entry is None:
            continue
        if "tok_in" in entry or "channel_key" in entry or "q1_hash" in entry:
            return "native"
        usage = entry.get("usage")
        if isinstance(usage, dict) and ("token_in" in usage or "token_out" in usage):
            return "newapi"
        return "unknown"
    return "empty"


def _first_entry(index_file: Path) -> Optional[dict]:
    try:
        with open(index_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("_meta"):
                    continue
                return obj
    except OSError:
        return None
    return None


def normalize_entry(entry: dict) -> dict:
    """把两种格式的索引条目归一化为统一字段。

    - 本项目已有 tok_in/tok_out/success/valid/channel_key/q1_hash/chain_key。
    - new-api 只有 usage.token_in/token_out、req_file、model、api_key、ts；
      无 success/channel_key/会话字段。成功判定退化为 tok_out > 0。
    """
    usage = entry.get("usage") if isinstance(entry.get("usage"), dict) else {}

    tok_in = entry.get("tok_in")
    if tok_in is None:
        tok_in = usage.get("token_in") or usage.get("prompt_tokens") or 0
    tok_out = entry.get("tok_out")
    if tok_out is None:
        tok_out = usage.get("token_out") or usage.get("completion_tokens") or 0
    tok_in = tok_in or 0
    tok_out = tok_out or 0

    cache_in = entry.get("cache_in") or 0

    if "valid" in entry:
        success = bool(entry["valid"]) and tok_out > 0
    elif "success" in entry:
        success = bool(entry["success"]) and tok_out > 0
    else:
        # new-api：无显式成功标记，用 tok_out 判定
        success = tok_out > 0

    ts = entry.get("ts", "") or ""

    return {
        "ts": ts,
        "date": ts[:10] if len(ts) >= 10 else "unknown",
        "model": entry.get("model", "") or "",
        "api_key": entry.get("api_key", "") or "",
        "tok_in": tok_in,
        "tok_out": tok_out,
        "cache_in": cache_in,
        "success": success,
        "channel_key": entry.get("channel_key", "") or "",
        "q1_hash": entry.get("q1_hash", "") or "",
        "chain_key": entry.get("chain_key", "") or "",
        "req_file": entry.get("req_file", "") or "",
    }


def date_from_ts(ts: str) -> str:
    """ts 可能是 '2026-07-25_00-01-41_930' 或 '2026-07-27T17:43:38.096+08:00'。
    统一取前 10 位 YYYY-MM-DD。"""
    return ts[:10] if len(ts) >= 10 else "unknown"
