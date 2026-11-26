#!/usr/bin/env python3
"""清理 leaf_status 里指向「已无本地数据源」的叶子，止住 pending 假积压。

背景
----
同步链路（src.watch.worker + src.backfill.worker）本身是好的，但 leaf_status 会对
每个登记过的小时叶子留一行。源目录被备份/轮转/清理删掉后，这些行不会自动消失：

  * 发现线程对 os.stat 失败的叶子直接 return，不会去动 leaf_status；
  * 对账线程跑 needs_build() 也补不了（目录都没了）；
  * 于是它们的 pending 计数永远挂着，状态面板显示上千万行「待同步」，看起来像同步停了。

实测（2026-11-24）：236 个 pending 叶子共 34.9M 行，其中
  - 233 个目录已不存在（幽灵），占 34,177,875 行
  - 3 个目录存在但 index.jsonl 与全部 .json 请求文件都没了，只剩 index.db，
    其 q1_hash IS NULL 占 97%，enrich 因缺原始文件永远补不完

这两类都该打 archived 标记（沿用 utils/logdir_store.mark_archived 的语义：归档 =
无本地可读数据源）。注意 archived 不影响 /sessions 统计（那里刻意不过滤 archived），
所以历史数据仍可查，只是不再计入待同步、session worker 也不再来回重试。

判定
----
  ghost : leaf_path 目录不存在
  dead  : 目录存在，但既无 index.jsonl、也无任何 *.json 请求文件（只剩 index.db）
  ok    : 其余（有 index.jsonl 或至少还有请求文件可解析）

用法
----
  # 预览（默认，只读）
  python cleanup_leaf_status.py
  python cleanup_leaf_status.py --db logs/port8084/env-99oR/log_dir.db -v

  # 实际写入
  python cleanup_leaf_status.py --write

  # 撤销归档（目录恢复后想重新纳入同步）
  python cleanup_leaf_status.py --unarchive --write
"""
import argparse
import os
import sqlite3
import sys
from collections import defaultdict

DEFAULT_DB = "/root/llm-proxy-main/logs/port8084/env-99oR/log_dir.db"


def _now() -> str:
    import datetime
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def classify(leaf_path: str) -> str:
    """返回 ghost / dead / ok。"""
    if not leaf_path:
        return "ghost"          # 早期登记的行没有 leaf_path，无从同步
    if not os.path.isdir(leaf_path):
        return "ghost"
    # 目录在：看还有没有可读数据源
    if os.path.exists(os.path.join(leaf_path, "index.jsonl")):
        return "ok"
    try:
        entries = os.listdir(leaf_path)
    except OSError:
        return "dead"
    if any(e.endswith(".json") for e in entries):
        return "ok"
    return "dead"


def main() -> int:
    ap = argparse.ArgumentParser(description="归档无本地数据源的叶子，清理 pending 假积压")
    ap.add_argument("--db", default=DEFAULT_DB, help="log_dir.db 路径")
    ap.add_argument("--write", action="store_true", help="实际写入（默认仅预览）")
    ap.add_argument("--unarchive", action="store_true", help="撤销归档而不是归档")
    ap.add_argument("-v", "--verbose", action="store_true", help="打印每条明细")
    ap.add_argument("--only-pending", action="store_true",
                    help="只处理 pending>0 的叶子（默认检查全部）")
    args = ap.parse_args()

    if not os.path.exists(args.db):
        print(f"找不到 db: {args.db}", file=sys.stderr)
        return 1

    mode = "" if args.write else "?mode=ro"
    conn = sqlite3.connect(f"file:{args.db}{mode}", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        sql = ("SELECT root_id, leaf_path, pending, archived, state FROM leaf_status"
               + (" WHERE COALESCE(pending,0) > 0" if args.only_pending else ""))
        rows = conn.execute(sql).fetchall()

        print("=== WRITE 模式 ===" if args.write else "=== DRY RUN（只读）===")
        print(f"db: {args.db}")
        print(f"检查叶子: {len(rows)}\n")

        by_root = defaultdict(lambda: {"ghost": [0, 0], "dead": [0, 0], "ok": [0, 0]})
        to_archive = []

        for r in rows:
            lp = r["leaf_path"] or ""
            kind = classify(lp)
            root = (r["root_id"] or "?")
            by_root[root][kind][0] += 1
            by_root[root][kind][1] += r["pending"] or 0
            if kind in ("ghost", "dead") and not r["archived"]:
                to_archive.append(r)
                if args.verbose:
                    print(f"  [{kind:5}] pending={r['pending'] or 0:>8} "
                          f"{(lp or '<无 leaf_path>').split('/details/')[-1]}")

        print(f"{'root_id':12} {'ghost':>16} {'dead':>16} {'ok':>8}")
        for root, c in sorted(by_root.items()):
            print(f"{root:12} {c['ghost'][0]:>5}叶/{c['ghost'][1]:>9}行 "
                  f"{c['dead'][0]:>5}叶/{c['dead'][1]:>9}行 {c['ok'][0]:>8}")

        # 已归档却仍挂 state='pending' / pending>0 的行。count_summary() 用
        # SUM(state='pending') 且不过滤 archived，只要这些行不改，列表页就会把
        # 几百个「目录早已删除」的叶子显示成待同步。归档语义上就是不待同步，故归一化。
        stale_archived = conn.execute(
            "SELECT COUNT(*) FROM leaf_status WHERE archived = 1 "
            "AND (state = 'pending' OR COALESCE(pending,0) > 0)"
        ).fetchone()[0]

        total_pending = sum((r["pending"] or 0) for r in to_archive)
        print(f"\n待归档: {len(to_archive)} 叶，涉及 pending {total_pending:,} 行")
        print(f"待归一化的已归档行（state=pending 或 pending>0）: {stale_archived} 叶")
        for root, c in sorted(by_root.items()):
            if c["ghost"][0] or c["dead"][0]:
                meaningful = c["ghost"][0] + c["dead"][0]
                print(f"  {root}: {meaningful} 叶")

        if args.write:
            stamp = _now()
            with conn:
                for r in to_archive:
                    conn.execute(
                        "UPDATE leaf_status SET archived = 1, archived_at = ?, pending = 0, "
                        "state = CASE WHEN state = 'pending' THEN 'done' ELSE state END "
                        "WHERE root_id = ? AND COALESCE(leaf_path,'') = ?",
                        (stamp, r["root_id"], r["leaf_path"] or ""),
                    )
                norm = conn.execute(
                    "UPDATE leaf_status SET pending = 0, "
                    "state = CASE WHEN state = 'pending' THEN 'done' ELSE state END "
                    "WHERE archived = 1 AND (state = 'pending' OR COALESCE(pending,0) > 0)"
                ).rowcount
            print(f"\n已归档 {len(to_archive)} 叶，归一化已归档行 {norm} 叶")
        elif not args.write:
            print("\n（dry-run，未写入；加 --write 实际执行）")
        else:
            print("\n没有需要归档的叶子")
    finally:
        conn.close()
    return 0


def _unarchive(db: str, write: bool) -> int:
    """把已归档但目录已恢复的叶子取消归档。"""
    mode = "" if write else "?mode=ro"
    conn = sqlite3.connect(f"file:{db}{mode}", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT root_id, leaf_path FROM leaf_status WHERE archived = 1"
        ).fetchall()
        back = [r for r in rows if r["leaf_path"] and os.path.isdir(r["leaf_path"])]
        print(f"已归档 {len(rows)} 叶，其中目录已恢复 {len(back)} 个")
        if write and back:
            with conn:
                for r in back:
                    conn.execute(
                        "UPDATE leaf_status SET archived = 0, archived_at = '' "
                        "WHERE root_id = ? AND leaf_path = ?",
                        (r["root_id"], r["leaf_path"]),
                    )
            print(f"已取消归档 {len(back)} 叶")
        elif not write:
            print("（dry-run，未写入；加 --write 实际执行）")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    if "--unarchive" in sys.argv:
        a = argparse.ArgumentParser()
        a.add_argument("--db", default=DEFAULT_DB)
        a.add_argument("--write", action="store_true")
        a.add_argument("--unarchive", action="store_true")
        ns = a.parse_args()
        sys.exit(_unarchive(ns.db, ns.write))
    sys.exit(main())
