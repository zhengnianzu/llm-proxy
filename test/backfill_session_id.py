#!/usr/bin/env python3
"""离线回填最近两天的 session 身份（metadata.session_id / user_id），并重建 sessions。

背景
----
上游 new-api 从 2026-11 起在 `up_req.metadata` 里写了 `session_id` / `user_id`，
聚合口径已改成 session_id > user_id > q1_hash（见 doc/watcher.md「聚合身份」）。

但这两个字段**只能在 enrich 解析 ~1MB 合并文件时取到**（`index.jsonl` 里没有），
而 enrich 的待办判据是 `WHERE q1_hash IS NULL` —— 已 enrich 的行 q1_hash 非空，
永远不会被重扫，新列就一直是 NULL，历史叶子继续按 q1_hash 聚合。

本脚本就是那个「按需回填」的口子：把指定范围内的行 meta 置 NULL → enrich 重算
→ rebuild_sessions 用新口径重新折叠。

每个叶子依次做：
  1. 置空 meta（q1_hash/msg_count/... + session_id/session_src），attempts 归零
  2. enrich(leaf, workers)   —— 重新解析合并文件，写入 q1 + 会话身份
  3. rebuild_sessions(leaf)  —— 全量重折叠（不是增量，口径变了必须全量）

⚠ 本脚本会写共享盘上的线上 index.db（与 audit_session_agg.py 不同）
--------------------------------------------------------------------
所以做了三层保护：
  - **跨进程锁**：每叶持 `<leaf>/.indexer.lock`（utils.leaf_lock），与
    src/watch/worker.py、src/backfill/worker.py 互斥。拿不到就跳过，不等不抢。
  - **版本检查**：若 watch/backfill worker 启动早于代码修改时间，它们在跑旧代码，
    会把叶子重算回旧口径（新列写不进去），白跑。默认拒绝启动，--force 可跳过。
  - **--dry-run**：只统计影响面，不写一个字节。

幂等 / 断点续跑
---------------
重跑安全。判据是「requests 里存在 q1_hash 非空但 session_src IS NULL 的行」——
回填过的叶子 session_src 已非 NULL（没 metadata 的行是空串 ''，不是 NULL），
不会被重复处理。可随时 Ctrl-C。

用法
----
  # 先看影响面（只读，强烈建议先跑）
  python test/backfill_session_id.py --dry-run

  # 正式回填最近两天的 proxy-004
  python test/backfill_session_id.py

  # 全部 newapi 来源 / 指定范围
  python test/backfill_session_id.py --root all --days 2
  python test/backfill_session_id.py --start 26112500 --end 26112523

  # 已经确认 worker 重启过了，跳过版本检查
  python test/backfill_session_id.py --force
"""
import argparse
import os
import sqlite3
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

HARNESS = Path("/mnt/sfs_turbo/s3-asset-b-hd-cce-aifm-nlp-exp/raw/harness")
ROOTS = {
    "proxy-004": HARNESS / "proxy-004/data/newapi/logs/details",
    "jumper-003": HARNESS / "jumper-003/data/newapi/logs/details",
    "jumper-001": HARNESS / "jumper-001/data/newapi/logs/details",
    "jumper": HARNESS / "jumper/data/newapi/logs/details",
    "proxy45": Path("/mnt/sfs_turbo/s3-asset-b-hd-cce-aifm-nlp-exp/data/newapi/"
                    "115.120.127.45/logs/details"),
    "proxy8": Path("/mnt/sfs_turbo/s3-asset-b-hd-cce-aifm-nlp-exp/data/newapi/"
                   "115.120.50.8/logs/details"),
    "proxy118": HARNESS / "coding-plan/data/newapi/logs/details",
}

# 待回填行：已 enrich（q1_hash 非 NULL）但身份列还没算过（session_src IS NULL）。
# 注意区分 NULL 与 ''：回填后没有 metadata 的行是 ''（已算过、上游没给），
# 只有从没算过的行才是 NULL。这个区别就是幂等的依据。
_COUNT_PENDING = """
SELECT COUNT(*) FROM requests
 WHERE q1_hash IS NOT NULL AND session_src IS NULL AND COALESCE(msg_count, 0) >= 0
"""
# 置空 meta 让 enrich 重扫。与 refill_q1_worker 同款手法。
_RESET_ROWS = """
UPDATE requests
   SET q1_hash = NULL, q1_preview = NULL, chain_key = NULL,
       msg_count = NULL, user_turns = NULL, attempts = 0
 WHERE q1_hash IS NOT NULL AND session_src IS NULL AND COALESCE(msg_count, 0) >= 0
"""
_COUNT_NULL = "SELECT COUNT(*) FROM requests WHERE q1_hash IS NULL"


def _conn(db: Path, write: bool = False) -> sqlite3.Connection:
    c = sqlite3.connect(str(db), timeout=60)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA busy_timeout=60000")
    if write:
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA synchronous=NORMAL")
    return c


def _has_cols(c: sqlite3.Connection) -> bool:
    cols = {r[1] for r in c.execute("PRAGMA table_info(requests)")}
    return "session_id" in cols and "session_src" in cols


# ─────────────────────────── 子进程：处理单叶 ───────────────────────────

def process_leaf(job: tuple) -> dict:
    """job = (leaf_str, workers, dry)。在子进程里完成一个叶子。"""
    leaf_str, workers, dry = job
    leaf = Path(leaf_str)
    db = leaf / "index.db"
    res = {"leaf": leaf_str, "dir_key": f"{leaf.parent.name}/{leaf.name}",
           "pending": 0, "enriched": 0, "sessions": 0, "by_src": {},
           "error": "", "took": 0.0, "skipped": "", "locked": False}
    t0 = time.time()
    if not db.exists():
        res["skipped"] = "no index.db"
        return res

    try:
        from utils import newapi_index_db as nidb
        from utils.leaf_lock import leaf_lock, LeafLockBusy

        # ── dry-run：只读统计，不加锁（只读不会与 worker 冲突）──
        if dry:
            c = _conn(db)
            try:
                if not _has_cols(c):
                    # 老库还没 ALTER：所有已 enrich 的行都算待回填
                    res["pending"] = c.execute(
                        "SELECT COUNT(*) FROM requests WHERE q1_hash IS NOT NULL "
                        "AND COALESCE(msg_count,0) >= 0").fetchone()[0]
                    res["skipped"] = "需先补列(由 _migrate 自动)"
                else:
                    res["pending"] = c.execute(_COUNT_PENDING).fetchone()[0]
            finally:
                c.close()
            res["took"] = time.time() - t0
            return res

        # ── 正式回填：全程持叶子跨进程锁 ──
        try:
            with leaf_lock(leaf_str, timeout=10):
                # 1) 置空。_connect(write=True) 会跑 _migrate 补列，所以先用它开一次。
                wc = nidb._connect(leaf_str, write=True)
                wc.close()
                c = _conn(db, write=True)
                try:
                    with c:
                        n = c.execute(_RESET_ROWS).rowcount or 0
                        if n:
                            nidb._meta_set(c, "sessions_dirty", "1")
                    res["pending"] = n
                    nulls = c.execute(_COUNT_NULL).fetchone()[0]
                finally:
                    c.close()

                if not n and not nulls:
                    res["skipped"] = "无待回填行"
                    res["took"] = time.time() - t0
                    return res

                # 2) enrich：重新解析合并文件，写入 q1 + session_id/session_src
                out = nidb.enrich(leaf_str, workers=workers)
                res["enriched"] = int((out or {}).get("enriched") or 0)

                # 3) 全量重建 sessions。**不能用增量 sync_sessions**：它只折叠
                #    rowid 水位之后的行，而这里是「已折叠过的行换了口径」，
                #    必须整叶重折。rebuild_sessions 自己会重置水位。
                res["sessions"] = int(nidb.rebuild_sessions(leaf_str) or 0)

                # 回填效果统计
                c = _conn(db)
                try:
                    res["by_src"] = {
                        (r["session_src"] or ""): r["n"] for r in c.execute(
                            "SELECT session_src, COUNT(*) n FROM requests "
                            "WHERE q1_hash IS NOT NULL GROUP BY 1")}
                finally:
                    c.close()
        except LeafLockBusy:
            # 另一个进程（watcher / backfill）正在写它，交给下一轮，不抢
            res["locked"] = True
            res["skipped"] = "leaf locked"
    except Exception as e:  # noqa: BLE001 - 单叶失败不中断全局
        res["error"] = f"{type(e).__name__}: {e}"
    res["took"] = time.time() - t0
    return res


# ─────────────────────────── 叶子发现 ───────────────────────────

def discover(args) -> list:
    roots = ROOTS if args.root == "all" else {args.root: ROOTS.get(args.root)}
    if args.root != "all" and roots.get(args.root) is None:
        raise SystemExit(f"未知 root: {args.root!r}，可选 {', '.join(ROOTS)} 或 all")

    leaves = []
    for name, root in roots.items():
        if root is None or not root.is_dir():
            continue
        days = sorted((d for d in os.listdir(root)
                       if d.isdigit() and len(d) == 6 and (root / d).is_dir()),
                      reverse=True)
        if args.days > 0 and not (args.start or args.end):
            days = days[:args.days]
        for d in days:
            dd = root / d
            for h in sorted(os.listdir(dd), reverse=True):
                if not h.isdigit():
                    continue
                if args.start and h < args.start:
                    continue
                if args.end and h > args.end:
                    continue
                lp = dd / h
                if (lp / "index.db").is_file():
                    leaves.append(str(lp))
    # 最新在前：叶子名 YYMMDDHH 字典序 = 时间序，当前小时的数据最先可查
    leaves.sort(reverse=True)
    if args.limit:
        leaves = leaves[:args.limit]
    return leaves


def check_worker_safety(force: bool) -> bool:
    """确认 watch/backfill worker 已加载新代码。

    它们若在跑旧代码，_SCHEMA/_migrate 里没有新列、enrich 也不写身份字段，
    会把本脚本刚回填好的叶子重算回旧口径 —— 白跑，而且结果看起来像"没生效"。
    """
    root = Path(__file__).resolve().parent.parent
    newest = 0.0
    for rel in ("utils/newapi_index_db.py", "utils/newapi_format.py"):
        try:
            newest = max(newest, (root / rel).stat().st_mtime)
        except OSError:
            pass
    if not newest:
        return True

    stale = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            cmd = (p / "cmdline").read_bytes().decode("utf-8", "replace")
            if "src.watch.worker" not in cmd and "src.backfill.worker" not in cmd:
                continue
            if p.stat().st_mtime < newest:
                tag = "watch" if "src.watch.worker" in cmd else "backfill"
                stale.append(f"{tag}(PID {p.name})")
        except OSError:
            continue
    if not stale:
        return True

    print(f"\n⚠ 以下常驻进程启动早于代码修改时间，仍在跑旧代码：")
    for s in stale:
        print(f"      {s}")
    print("  旧代码的 index.db schema 没有 session_id/session_src 列，enrich 也不写这两个")
    print("  字段，会把本脚本回填好的叶子重算回旧口径 —— 本次回填会白跑。")
    print("  先重启（没有 restart 子命令）：")
    print("      ./app watch stop && ./app watch start")
    print("      # backfill worker 按你的部署方式重启\n")
    if not force:
        print("已中止。确认要强行继续请加 --force。")
        return False
    print("--force 已指定，继续。\n")
    return True


def _heartbeat(stop_evt, total: int, counter: dict, t0: float):
    while not stop_evt.wait(15):
        print(f"  … 进度 {counter['done']}/{total}  已用时 {time.time()-t0:.0f}s",
              flush=True)


SRC_LABEL = {
    "session_id": "metadata.session_id",
    "user_id_json": "user_id 内嵌 session_id",
    "user_id": "metadata.user_id",
    "": "(回退 q1_hash)",
}
SRC_ORDER = ["session_id", "user_id_json", "user_id", ""]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="离线回填最近两天的 session 身份并重建 sessions（会写线上 index.db）")
    ap.add_argument("--dry-run", action="store_true", help="只统计影响面，不写入")
    ap.add_argument("--root", default="proxy-004",
                    help="来源名或 all（默认 proxy-004）。可选：" + ", ".join(ROOTS))
    ap.add_argument("--days", type=int, default=2, help="最近几天（默认 2）")
    ap.add_argument("--start", default="", help="起始叶子名（含），如 26112500")
    ap.add_argument("--end", default="", help="结束叶子名（含）")
    ap.add_argument("--limit", type=int, default=0, help="最多处理多少个叶子")
    ap.add_argument("--leaves", type=int, default=6, help="并行叶子数（默认 6）")
    ap.add_argument("--workers", type=int, default=4,
                    help="每叶 enrich 子进程数（默认 4）")
    ap.add_argument("--force", action="store_true", help="跳过 worker 版本检查")
    args = ap.parse_args()

    if not args.dry_run and not check_worker_safety(args.force):
        return 3

    leaves = discover(args)
    if not leaves:
        print("没有匹配到叶子（检查 --root/--days/--start/--end）")
        return 1

    cpu = os.cpu_count() or 8
    par = args.leaves * args.workers
    print(f"叶子 {len(leaves)} 个（最新在前），root={args.root}，"
          f"并发 {args.leaves} × {args.workers} = {par} 进程（CPU {cpu}）")
    if par > cpu * 2:
        print(f"  提示：{par} 进程超过 CPU×2，建议调小 --leaves/--workers")
    print("=== DRY RUN（只读）===" if args.dry_run else "=== 开始回填 ===", flush=True)

    t0 = time.time()
    counter = {"done": 0}
    stop_evt = threading.Event()
    threading.Thread(target=_heartbeat,
                     args=(stop_evt, len(leaves), counter, t0), daemon=True).start()

    jobs = [(lf, args.workers, args.dry_run) for lf in leaves]
    results = []
    try:
        with ProcessPoolExecutor(max_workers=args.leaves) as pool:
            futs = {pool.submit(process_leaf, j): j for j in jobs}
            for fut in as_completed(futs):
                r = fut.result()
                results.append(r)
                counter["done"] += 1
                if args.dry_run:
                    if r["pending"] or r["error"]:
                        note = r["error"] or r["skipped"]
                        print(f"  {r['dir_key']:20} 待回填 {r['pending']:6} 行"
                              f"{'  ' + note if note else ''}", flush=True)
                else:
                    mark = "✗" if r["error"] else ("·" if r["skipped"] else "✓")
                    detail = (r["error"] or r["skipped"] or
                              f"{r['enriched']:5} 行 → {r['sessions']:5} 会话")
                    print(f"  [{counter['done']}/{len(jobs)}] {mark} "
                          f"{r['dir_key']:20} {r['took']:6.1f}s  {detail}", flush=True)
    except KeyboardInterrupt:
        print("\n已中断（重跑会跳过已完成叶子）")
    finally:
        stop_evt.set()

    # ── 汇总 ──
    pend = sum(r["pending"] for r in results)
    enr = sum(r["enriched"] for r in results)
    sess = sum(r["sessions"] for r in results)
    bad = [r for r in results if r["error"]]
    lock = [r for r in results if r["locked"]]
    print(f"\n{'=' * 70}")
    if args.dry_run:
        print(f"DRY RUN：{len(results)} 个叶子，待回填 {pend} 行")
        need_col = [r for r in results if "补列" in r["skipped"]]
        if need_col:
            print(f"  其中 {len(need_col)} 个叶子还没有 session_id/session_src 列，"
                  f"回填时由 _migrate 自动补上")
        print(f"\n正式执行：去掉 --dry-run")
    else:
        print(f"完成：{len(results)} 个叶子，回填 {enr} 行，重建 {sess} 个会话，"
              f"耗时 {time.time()-t0:.0f}s")
        tot = {}
        for r in results:
            for k, v in r["by_src"].items():
                tot[k] = tot.get(k, 0) + v
        if tot:
            print(f"\n── 回填后的请求级 metadata 覆盖率 ──")
            s = sum(tot.values())
            for k in [x for x in SRC_ORDER if x in tot] + \
                     [x for x in tot if x not in SRC_ORDER]:
                print(f"    {SRC_LABEL.get(k, k):26} {tot[k]:7} 条  "
                      f"{100.0*tot[k]/max(s,1):5.1f}%")
            cov = sum(v for k, v in tot.items() if k)
            print(f"    {'→ 有上游会话标识':26} {cov:7} 条  "
                  f"{100.0*cov/max(s,1):5.1f}%")
    if lock:
        print(f"\n{len(lock)} 个叶子被其他进程占用（watcher/backfill 正在写），"
              f"稍后重跑即可：")
        for r in lock[:10]:
            print(f"    {r['dir_key']}")
    if bad:
        print(f"\n{len(bad)} 个叶子失败：")
        for r in bad[:15]:
            print(f"    ✗ {r['dir_key']}: {r['error']}")
    print()
    if not args.dry_run:
        print("建议跑一次核验：python test/audit_session_agg.py --days 2")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
