#!/usr/bin/env python3
"""回填 newapi 叶子 index.db 中被 <system-reminder> 污染的 Q1 派生字段。

背景
----
Claude Code 把工作区上下文（gitStatus / attribution 规则）作为首条 user 消息注入，
整体包在 <system-reminder> 里。旧版 utils/message_common.get_first_user_text() 不认识
这个模式，于是把整段注入内容当成了 Q1 存进 requests 表，导致 /export/view 的 Q1 列
显示 system prompt 而非真实提问。

提取逻辑已修复，但 q1_hash / q1_preview / chain_key / msg_count / user_turns 是
**持久化派生字段**，改逻辑不会重算存量行。本脚本按 _worker_enrich 的同一代码路径
（_resolve_combined_path + parse_combined_file + compute_session_fields）就地重算
并写回。

注意：不能只对存量 q1_preview 做字符串剥离 —— 它被截断到 200 字符，真实提问往往
整个落在截断之外，剥完只剩空串。必须回原始合并文件重解析。

用法
----
  # 预览（默认，不写入）
  python backfill_reminder_q1.py '<leaf>/index.db'
  python backfill_reminder_q1.py '<details>/261118/*/index.db'

  # 实际写入
  python backfill_reminder_q1.py '<details>/261118/*/index.db' --write

  # 只回填某个 key（单叶子可达 29 万行、混着很多 key，建议先按 key 限定验证）
  python backfill_reminder_q1.py '<details>/261112/*/index.db' --key sk-xxxx --write
"""
import argparse
import glob
import os
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.newapi_consumer import _resolve_combined_path
from utils.newapi_format import compute_session_fields, parse_combined_file

# 只挑选 q1_preview 明显是注入块的行：要么以标签起头，要么以那句话起头（无包裹变体）。
# 已 enrich（q1_hash IS NOT NULL）才有重算的意义；q1_hash='' 是解析失败的死行，
# msg_count=-1，重算只会再失败一次，所以一并排除。
_SELECT_BAD = """
SELECT req_file, q1_preview
  FROM requests
 WHERE q1_hash IS NOT NULL
   AND q1_hash != ''
   AND ( q1_preview LIKE '<system-reminder>%'
      OR q1_preview LIKE 'As you answer the user''s questions%' )
 ORDER BY rowid
"""

# --key 只回填指定 api_key 的行。同一叶子里混着很多 key（单叶子可达 29 万行），
# 限定后能把一次回填压到可控范围、也便于先验证某个 key 的页面效果。
_SELECT_BAD_KEY = """
SELECT req_file, q1_preview
  FROM requests
 WHERE api_key = ?
   AND q1_hash IS NOT NULL
   AND q1_hash != ''
   AND ( q1_preview LIKE '<system-reminder>%'
      OR q1_preview LIKE 'As you answer the user''s questions%' )
 ORDER BY rowid
"""

_UPDATE = """
UPDATE requests
   SET q1_hash = ?, q1_preview = ?, chain_key = ?, msg_count = ?, user_turns = ?
 WHERE req_file = ?
"""


def backfill_leaf(db: str, write: bool, verbose: bool, key: str = "") -> tuple:
    """返回 (bad, fixed, failed)。key 非空时只处理该 api_key 的行。"""
    leaf = Path(os.path.dirname(db))
    mode = "" if write else "?mode=ro"
    conn = sqlite3.connect(f"file:{db}{mode}", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        if key:
            rows = conn.execute(_SELECT_BAD_KEY, (key,)).fetchall()
        else:
            rows = conn.execute(_SELECT_BAD).fetchall()
        fixed = failed = 0
        updates = []
        for r in rows:
            path = _resolve_combined_path(leaf, r["req_file"])
            parsed = parse_combined_file(path) if path is not None else None
            if parsed is None:
                failed += 1
                continue
            f = compute_session_fields(parsed["messages"])
            new_q1 = f["q1_preview"] or ""
            # 重算后仍带标签说明清洗没生效（畸形输入），保守跳过而不是写入更差的值。
            if "<system-reminder>" in new_q1:
                failed += 1
                continue
            updates.append((
                f["q1_hash"] or "", new_q1, f["chain_key"],
                f["msg_count"], f["user_turns"], r["req_file"],
            ))
            fixed += 1
            if verbose and fixed <= 3:
                print(f"    旧: {r['q1_preview'][:70]!r}")
                print(f"    新: {new_q1[:70]!r}")
        if write and updates:
            with conn:
                conn.executemany(_UPDATE, updates)
        return len(rows), fixed, failed
    finally:
        conn.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="回填被 system-reminder 污染的 Q1")
    ap.add_argument("pattern", help="index.db 路径或 glob")
    ap.add_argument("--write", action="store_true", help="实际写入（默认仅预览）")
    ap.add_argument("--key", default="", help="只回填该 api_key 的行（默认全部 key）")
    ap.add_argument("-v", "--verbose", action="store_true", help="打印新旧 Q1 样例")
    args = ap.parse_args()

    dbs = sorted(glob.glob(args.pattern))
    if not dbs:
        print(f"没有匹配到 index.db: {args.pattern}")
        return 1

    print("=== WRITE 模式 ===" if args.write else "=== DRY RUN（不写入）===")
    print(f"叶子数: {len(dbs)}")
    if args.key:
        print(f"限定 api_key: {args.key}")
    print()
    t_bad = t_fixed = t_failed = 0
    for db in dbs:
        name = os.path.basename(os.path.dirname(db))
        try:
            bad, fixed, failed = backfill_leaf(db, args.write, args.verbose, args.key)
        except Exception as e:
            print(f"  {name}: ERROR {type(e).__name__}: {e}")
            continue
        t_bad += bad
        t_fixed += fixed
        t_failed += failed
        if bad:
            verb = "已修复" if args.write else "可修复"
            print(f"  {name}: 受影响 {bad}  {verb} {fixed}  跳过 {failed}")
        else:
            print(f"  {name}: 无需修复")

    print(f"\n合计: 受影响 {t_bad}  {'已修复' if args.write else '可修复'} {t_fixed}  跳过 {t_failed}")
    if not args.write:
        print("（dry-run，未写入；加 --write 实际执行）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
