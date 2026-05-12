#!/usr/bin/env python3
"""紧急修复 index.jsonl 中不正确的 chain_key 和 q1_preview。

扫描指定目录（或所有 logs 目录）的 index.jsonl，对每条记录检查
chain_key/q1_preview 是否为噪声，如果是则从对应的 req 文件重新提取。

用法:
  python fix_index_q1.py logs_all_env-ann_xnGZ_260511
  python fix_index_q1.py --all
  python fix_index_q1.py logs_all_env-ann_xnGZ_260511 --dry-run
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.message_common import (
    get_first_user_text,
    build_chain_key,
    extract_messages,
    load_json_safe,
    _SKIP_PREFIXES,
    _INTERNAL_REQUEST_PATTERNS,
    _SKIP_CLEANED_PREFIXES,
)

ROOT = Path(__file__).parent.parent


def is_noise(text: str) -> bool:
    if not text:
        return False
    if any(text.startswith(p) for p in _SKIP_PREFIXES):
        return True
    if any(pat.search(text) for pat in _INTERNAL_REQUEST_PATTERNS):
        return True
    cleaned = text.strip()
    if cleaned and any(cleaned.startswith(p) for p in _SKIP_CLEANED_PREFIXES):
        return True
    return False


def fix_index(index_path: Path, dry_run: bool):
    log_dir = index_path.parent
    lines = index_path.read_text(encoding="utf-8").splitlines()
    fixed_count = 0
    new_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            new_lines.append(line)
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            new_lines.append(line)
            continue

        chain_key = entry.get("chain_key", "")
        q1_preview = entry.get("q1_preview", "")
        needs_fix = is_noise(chain_key) or is_noise(q1_preview)

        if not needs_fix:
            new_lines.append(line)
            continue

        req_file = entry.get("req_file", "")
        req_path = Path(req_file) if req_file.startswith("/") else log_dir / Path(req_file).name
        if not req_path.is_file():
            req_path = log_dir / Path(req_file).name
        if not req_path.is_file():
            new_lines.append(line)
            continue

        data = load_json_safe(req_path)
        if not data:
            new_lines.append(line)
            continue

        messages = extract_messages(data)
        if not messages:
            new_lines.append(line)
            continue

        new_chain_key = build_chain_key(messages)
        new_q1 = get_first_user_text(messages)[:100]

        changed = False
        if is_noise(chain_key) and new_chain_key != chain_key:
            entry["chain_key"] = new_chain_key
            changed = True
        if is_noise(q1_preview) and new_q1 != q1_preview:
            entry["q1_preview"] = new_q1
            changed = True

        if changed:
            fixed_count += 1
            if dry_run:
                print(f"  [FIX] {req_file}")
                print(f"         chain_key: {chain_key!r} -> {entry['chain_key']!r}")
                print(f"         q1_preview: {q1_preview!r} -> {entry['q1_preview']!r}")
            new_lines.append(json.dumps(entry, ensure_ascii=False))
        else:
            new_lines.append(line)

    if fixed_count > 0:
        print(f"  {index_path}: {fixed_count} 条已修复")
        if not dry_run:
            index_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    else:
        print(f"  {index_path}: 无需修复")


def main():
    parser = argparse.ArgumentParser(description="修复 index.jsonl 中的噪声 q1/chain_key")
    parser.add_argument("dir", nargs="?", help="指定要修复的日志目录")
    parser.add_argument("--all", action="store_true", help="扫描项目下所有 index.jsonl")
    parser.add_argument("--dry-run", action="store_true", help="仅预览，不实际写入")
    args = parser.parse_args()

    if not args.dir and not args.all:
        parser.error("请指定目录或使用 --all 扫描全部")

    if args.dry_run:
        print("=== DRY RUN 模式 ===\n")

    if args.all:
        index_files = sorted(ROOT.rglob("index.jsonl"))
    else:
        target = Path(args.dir)
        if not target.is_absolute():
            target = ROOT / target
        idx = target / "index.jsonl"
        if not idx.is_file():
            print(f"错误: {idx} 不存在")
            sys.exit(1)
        index_files = [idx]

    print(f"找到 {len(index_files)} 个 index.jsonl\n")

    for idx_path in index_files:
        fix_index(idx_path, args.dry_run)

    print("\n完成。" + (" (dry-run, 未实际写入)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
