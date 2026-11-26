#!/usr/bin/env python3
"""批量重置「因 req 为空而算错 Q1」的请求行，让 Stage 2 enrich 用新解析逻辑重算。

背景
----
采集侧从 2026-11-19 中午起把合并文件里的 `req` 字段写成空串，只保留 `up_req`。
旧版 utils/newapi_format.parse_combined_file() 只读 `req`，于是 messages 解析成
0 条 → Q1 为空 → 会话折叠后 q1 全空，浏览页看起来「没有新轨迹」。

解析层已修（parse_combined_file 在 req 空时回退 up_req，并归一
messages / contents / input 四种形态）。但 q1_hash / q1_preview / chain_key 是
**已落库的派生字段**，改代码不会自动重算；且 rebuild_sessions 是从 requests 表读
q1（不是重读文件），所以只置 sessions_dirty 只会用同样空的 q1 再折叠一遍，无效。

因此本脚本做两件事：
  1. 把受影响行的 q1_hash / q1_preview / chain_key / msg_count / user_turns 置 NULL
     —— enrich 只处理 `q1_hash IS NULL` 的行，置空即触发重算；
  2. 置 meta.sessions_dirty='1'，让折叠阶段在新 meta 就绪后重建 sessions。

受影响行的精确特征
------------------
  q1_hash = ''        （q1_hash_from_text("") 返回空串）
  AND msg_count >= 0  （排除 msg_count=-1 的「解析失败死行」，那是另一种状态）

实测 2026-11-24：新叶子 100% 命中该特征，2026-11-19 之前的老叶子仅 0.14%（真实空 Q1，
重算后仍是空串，无害但不必要），所以默认从 --start 指定的日期开始扫。

用法
----
  # 预览（默认，只读）
  python reset_q1_empty.py --start 26111912
  python reset_q1_empty.py --start 26111912 --root proxy-004

  # 实际写入（按日期分批跑，避免一次触发过重的 enrich）
  python reset_q1_empty.py --start 26111912 --end 26111923 --write
"""
import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

HARNESS = Path("/mnt/sfs_turbo/s3-asset-b-hd-cce-aifm-nlp-exp/raw/harness")
ROOTS = {
    "proxy-004": HARNESS / "proxy-004/data/newapi/logs/details",
    "jumper-003": HARNESS / "jumper-003/data/newapi/logs/details",
    "jumper-001": HARNESS / "jumper-001/data/newapi/logs/details",
    "jumper": HARNESS / "jumper/data/newapi/logs/details",
}

_SELECT_BROKEN = (
    "SELECT COUNT(*) FROM requests WHERE q1_hash = '' AND COALESCE(msg_count, 1) >= 0"
)
_RESET_ROWS = """
UPDATE requests
   SET q1_hash = NULL, q1_preview = NULL, chain_key = NULL,
       msg_count = NULL, user_turns = NULL, attempts = 0
 WHERE q1_hash = '' AND COALESCE(msg_count, 1) >= 0
"""


def set_sessions_dirty(conn) -> bool:
    """置 meta.sessions_dirty='1'。返回是否真的改了。"""
    try:
        conn.execute(
            "INSERT INTO meta(k, v) VALUES('sessions_dirty','1') "
            "ON CONFLICT(k) DO UPDATE SET v='1'"
        )
        return True
    except sqlite3.OperationalError:
        # 老库 meta 可能是 key/value 列名不同，退回逐条尝试
        try:
            cur = conn.execute("UPDATE meta SET v='1' WHERE k='sessions_dirty'")
            if cur.rowcount == 0:
                conn.execute("INSERT INTO meta(k, v) VALUES('sessions_dirty','1')")
            return True
        except Exception:
            return False


def process_leaf(db: Path, write: bool, verbose: bool) -> tuple:
    """返回 (broken, reset, ok)。"""
    try:
        conn = sqlite3.connect(str(db), timeout=30)
    except sqlite3.Error:
        return (-1, 0, False)
    try:
        conn.execute("PRAGMA busy_timeout=30000")
        broken = conn.execute(_SELECT_BROKEN).fetchone()[0]
        if not broken:
            return (0, 0, True)
        if not write:
            return (broken, broken, True)
        with conn:
            cur = conn.execute(_RESET_ROWS)
            reset = cur.rowcount or 0
            dirty = set_sessions_dirty(conn)
        if verbose:
            print(f"    {db.parent.name}: 置空 {reset} 行, sessions_dirty={dirty}")
        return (broken, reset, dirty)
    except sqlite3.Error as e:
        print(f"    {db.parent.name}: ERROR {e}")
        return (-1, 0, False)
    finally:
        conn.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="重置 req 空导致的错误 Q1，触发重算")
    ap.add_argument("--start", default="26111912", help="起始叶子（含），如 26111912")
    ap.add_argument("--end", default="", help="结束叶子（含），默认到最新")
    ap.add_argument("--root", default="", help="只处理某个源（proxy-004/jumper-003/...），默认全部")
    ap.add_argument("--write", action="store_true", help="实际写入（默认仅预览）")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    roots = {args.root: ROOTS[args.root]} if args.root in ROOTS else ROOTS
    if args.root and args.root not in ROOTS:
        print(f"未知 root: {args.root}，可选 {list(ROOTS)}", file=sys.stderr)
        return 2

    print("=== WRITE 模式 ===" if args.write else "=== DRY RUN（只读）===")
    print(f"范围: {args.start} .. {args.end or '最新'}  roots={list(roots)}\n")

    grand_broken = grand_reset = 0
    grand_leaves = 0
    failed = []
    for name, root in roots.items():
        if not root.is_dir():
            print(f"{name}: 路径不存在，跳过")
            continue
        b = r = n = 0
        for d in sorted(os.listdir(root)):
            if not d.isdigit():
                continue
            dd = root / d
            if not dd.is_dir():
                continue
            for leaf in sorted(os.listdir(dd)):
                if not leaf.isdigit():
                    continue
                if leaf < args.start or (args.end and leaf > args.end):
                    continue
                db = dd / leaf / "index.db"
                if not db.exists():
                    continue
                n += 1
                broken, reset, ok = process_leaf(db, args.write, args.verbose)
                if broken > 0:
                    b += broken
                    r += reset
                if not ok:
                    failed.append(f"{name}/{leaf}")
        print(f"{name:12} 叶子={n:>4}  受影响行={b:>8}  {'已重置' if args.write else '可重置'}={r:>8}")
        grand_broken += b
        grand_reset += r
        grand_leaves += n

    print(f"\n合计: 叶子 {grand_leaves}  受影响行 {grand_broken:,}  "
          f"{'已重置' if args.write else '可重置'} {grand_reset:,}")
    if failed:
        print(f"失败叶子 {len(failed)} 个: {failed[:5]}{' ...' if len(failed) > 5 else ''}")
    if not args.write:
        print("（dry-run，未写入；加 --write 实际执行）")
    else:
        print("已写入。enrich 会自动接手重算，sessions 折叠随后重建。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
