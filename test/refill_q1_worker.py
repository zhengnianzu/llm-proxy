#!/usr/bin/env python3
"""从最新叶子往前，批量重算「req 为空导致 Q1 错」的行，并重建 sessions。

背景
----
采集侧从 2026-11-19 中午起把合并文件的 `req` 写成空串，只保留等价的 `up_req`。
旧解析只读 `req` → messages 空 → Q1 空 → 会话折叠后 q1 全空，浏览页看不到新轨迹。

解析层已修（parse_combined_file 在 req 空时回退 up_req）。但已落库的
q1_hash/q1_preview/chain_key 不会自动重算，且 rebuild_sessions 是从 requests 表读
q1，所以必须先把受影响行置 NULL，再跑 enrich，最后重建 sessions。

每个叶子依次做：
  1. `q1_hash='' AND msg_count>=0` 的行 → meta 字段置 NULL（enrich 只处理 NULL 行）
  2. enrich(leaf, workers)     —— 重新解析合并文件，写入正确 q1
  3. rebuild_sessions(leaf)    —— 用新 q1 重建 sessions/traces

性能注意
--------
`q1_hash=''` 没有索引（只有 `q1_hash IS NULL` 的部分索引），判定受影响行必须全表
扫描。所以**不做**串行预扫描：直接把所有叶子丢进进程池，各叶子在自己的进程里并行
扫描，谁都不会拖住别人。判定也合并成一次扫描 —— 直接跑 UPDATE 拿 rowcount，
不再先 COUNT 一遍。

顺序：叶子名倒序（最新在前）。可随时 Ctrl-C 中断，重跑会跳过已完成叶子
（已修复的行不再是 `q1_hash=''`，也不会再触发 enrich）。

⚠ 运行前请重启 src.watch.worker
    它若仍在跑旧代码，会在后台把同一个叶子重算回空 Q1，与本脚本互相覆盖。
    本脚本会检查该进程启动时间 vs utils/newapi_format.py 修改时间，不一致时拒绝启动
    （确需继续可加 --force）。
    重启方式（没有 restart 子命令）：
        ./app watch stop && ./app watch start

用法
----
  # 预览：并行统计每个叶子待处理行数（只读）
  python test/refill_q1_worker.py --dry-run

  # 正式跑：默认 8 并发叶子 × 每叶 8 进程（64 核留余量）
  python test/refill_q1_worker.py

  # 开更大并发（注意 叶子×每叶 不要明显超过核数，否则进程调度反而变慢）
  python test/refill_q1_worker.py --leaves 16 --workers 4

  # 限定范围
  python test/refill_q1_worker.py --start 261124 --limit 20
  python test/refill_q1_worker.py --root proxy-004
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
}

# 受影响行的精确特征：空 Q1（q1_hash_from_text("") 返回空串）且解析成功。
# msg_count=-1 是「文件缺失/损坏」的死行状态，不属于本问题，排除。
_RESET_ROWS = """
UPDATE requests
   SET q1_hash = NULL, q1_preview = NULL, chain_key = NULL,
       msg_count = NULL, user_turns = NULL, attempts = 0
 WHERE q1_hash = '' AND COALESCE(msg_count, 1) >= 0
"""
_COUNT_BROKEN = (
    "SELECT COUNT(*) FROM requests WHERE q1_hash = '' AND COALESCE(msg_count, 1) >= 0"
)
# 走 requests_pending 部分索引，接近 O(1)
_COUNT_NULL = "SELECT COUNT(*) FROM requests WHERE q1_hash IS NULL"


def _conn(db: Path, write: bool = False) -> sqlite3.Connection:
    c = sqlite3.connect(str(db), timeout=60)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA busy_timeout=60000")
    if write:
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA synchronous=NORMAL")
    return c


# ─────────────────────────── 子进程：处理单个叶子 ───────────────────────────

def process_leaf(job: tuple) -> dict:
    """job = (leaf_str, workers, dry)。在子进程里完成一个叶子。"""
    leaf_str, workers, dry = job
    leaf = Path(leaf_str)
    db = leaf / "index.db"
    res = {"leaf": leaf_str, "broken": 0, "nulls": 0, "enriched": 0,
           "sessions": 0, "error": "", "took": 0.0, "skipped": False}
    if not db.exists():
        res["error"] = "no index.db"
        return res
    t0 = time.time()

    if dry:
        try:
            c = _conn(db)
            res["broken"] = c.execute(_COUNT_BROKEN).fetchone()[0]
            res["nulls"] = c.execute(_COUNT_NULL).fetchone()[0]
            c.close()
        except sqlite3.Error as e:
            res["error"] = str(e)
        res["took"] = time.time() - t0
        return res

    from utils import newapi_index_db as nidb

    # 1) 直接 UPDATE 拿 rowcount —— 一次全表扫描同时完成「判定 + 置空」，
    #    不再先 COUNT 再 UPDATE（那样同一个叶子要扫两遍）。
    try:
        c = _conn(db, write=True)
        try:
            with c:
                broken = c.execute(_RESET_ROWS).rowcount or 0
                if broken:
                    nidb._meta_set(c, "sessions_dirty", "1")
            res["broken"] = broken
            res["nulls"] = c.execute(_COUNT_NULL).fetchone()[0]
        finally:
            c.close()
    except sqlite3.Error as e:
        res["error"] = f"reset: {e}"
        return res

    # 无事可做：既没有刚置空的行，也没有历史遗留的 NULL 行。
    # 但还有一种残留：requests 已被修好、sessions 却仍是旧的空 Q1 —— 因为 enrich
    # 改的是 requests 表，而 sessions 在修复前就折叠完了，且增量 fold 按
    # sessions_synced_rowid 前进，不会回头重算已同步过的行。这里做一次廉价探测
    # （两个 EXISTS，命中即返回），只重建不重算。
    stale_sessions = False
    if not res["broken"] and not res["nulls"]:
        try:
            c2 = _conn(db)
            try:
                has_empty = c2.execute(
                    "SELECT EXISTS(SELECT 1 FROM sessions WHERE q1 = '' OR q1 IS NULL)"
                ).fetchone()[0]
                has_full = c2.execute(
                    "SELECT EXISTS(SELECT 1 FROM requests "
                    "WHERE q1_hash IS NOT NULL AND q1_hash != '')"
                ).fetchone()[0]
            finally:
                c2.close()
            stale_sessions = bool(has_empty and has_full)
        except sqlite3.Error:
            stale_sessions = False

    if not res["broken"] and not res["nulls"] and not stale_sessions:
        res["skipped"] = True
        res["took"] = time.time() - t0
        return res

    if not res["broken"] and not res["nulls"]:
        # 只差重建 sessions，跳过代价高昂的 enrich
        try:
            res["sessions"] = int(nidb.rebuild_sessions(leaf_str) or 0)
        except Exception as e:  # noqa: BLE001
            res["error"] = f"sessions: {type(e).__name__}: {e}"
        res["took"] = time.time() - t0
        return res

    # 2) enrich（重新解析合并文件，回填正确 q1）
    try:
        out = nidb.enrich(leaf_str, workers=workers)
        res["enriched"] = int((out or {}).get("enriched") or 0)
    except Exception as e:  # noqa: BLE001
        res["error"] = f"enrich: {type(e).__name__}: {e}"
        res["took"] = time.time() - t0
        return res

    # 3) 重建 sessions（从 requests 表按新 q1 折叠）
    try:
        res["sessions"] = int(nidb.rebuild_sessions(leaf_str) or 0)
    except Exception as e:  # noqa: BLE001
        res["error"] = f"sessions: {type(e).__name__}: {e}"

    res["took"] = time.time() - t0
    return res


# ─────────────────────────── 叶子发现 ───────────────────────────

def discover(args) -> list:
    roots = {args.root: ROOTS[args.root]} if args.root in ROOTS else ROOTS
    if args.root and args.root not in ROOTS:
        raise SystemExit(f"未知 root: {args.root}，可选 {list(ROOTS)}")
    leaves = []
    for name, root in roots.items():
        if not root.is_dir():
            continue
        try:
            dates = os.listdir(root)
        except OSError:
            continue
        for d in dates:
            dd = root / d
            if not d.isdigit() or not dd.is_dir():
                continue
            for leaf in os.listdir(dd):
                if not leaf.isdigit():
                    continue
                if args.start and leaf < args.start:
                    continue
                if args.end and leaf > args.end:
                    continue
                lp = dd / leaf
                if (lp / "index.db").exists():
                    leaves.append(str(lp))
    # 最新在前（叶子名即 YYMMDDHH，字典序等于时间序）
    leaves.sort(reverse=True)
    if args.limit:
        leaves = leaves[:args.limit]
    return leaves


def check_watcher_safety(force: bool) -> bool:
    """确认运行中的 watch worker 已加载新解析代码。"""
    src = Path(__file__).resolve().parent.parent / "utils/newapi_format.py"
    try:
        src_mtime = src.stat().st_mtime
    except OSError:
        return True
    old_pids = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            cmd = (p / "cmdline").read_bytes().decode("utf-8", "replace")
            if "src.watch.worker" not in cmd:
                continue
            # /proc/<pid> 目录的 mtime 即进程创建时间
            if p.stat().st_mtime < src_mtime:
                old_pids.append(p.name)
        except OSError:
            continue
    if not old_pids:
        return True
    print(f"\n⚠ src.watch.worker 仍在运行旧代码（PID {', '.join(old_pids)}），"
          f"启动早于 utils/newapi_format.py 的修改时间。")
    print("  它会在后台把同一批叶子重算回空 Q1，导致本脚本白跑。先重启：")
    print("      ./app watch stop && ./app watch start\n")
    if not force:
        print("已中止。确认要强行继续请加 --force。")
        return False
    print("--force 已指定，继续。")
    return True


def _heartbeat(stop_evt: threading.Event, total: int, counter: dict, t0: float):
    while not stop_evt.wait(15):
        done = counter["done"]
        print(f"  … 进度 {done}/{total}  已用时 {time.time()-t0:.0f}s", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="从最新叶子往前重算 Q1 并重建 sessions")
    ap.add_argument("--dry-run", action="store_true", help="只统计，不写入")
    ap.add_argument("--start", default="", help="起始叶子（含），如 26111912")
    ap.add_argument("--end", default="", help="结束叶子（含）")
    ap.add_argument("--root", default="", help="只处理某个源")
    ap.add_argument("--limit", type=int, default=0, help="最多处理多少个叶子")
    ap.add_argument("--leaves", type=int, default=8, help="并行叶子数（默认 8）")
    ap.add_argument("--workers", type=int, default=8, help="每叶 enrich 进程数（默认 8）")
    ap.add_argument("--force", action="store_true", help="跳过 watch worker 版本检查")
    args = ap.parse_args()

    if not args.dry_run and not check_watcher_safety(args.force):
        return 3

    leaves = discover(args)
    if not leaves:
        print("没有匹配到叶子（检查 --start/--end/--root）")
        return 1

    cpu = os.cpu_count() or 8
    par = args.leaves * args.workers
    print(f"叶子 {len(leaves)} 个，最新在前。并发 叶子={args.leaves} × 每叶={args.workers}"
          f" = {par} 进程（CPU {cpu}）")
    if par > cpu * 2:
        print(f"  提示：{par} 进程超过 CPU×2，调度开销可能反而拖慢，建议调小 --leaves/--workers")
    print("=== DRY RUN（只读）===" if args.dry_run else "=== 开始处理 ===", flush=True)

    t0 = time.time()
    counter = {"done": 0}
    stop_evt = threading.Event()
    hb = threading.Thread(target=_heartbeat, args=(stop_evt, len(leaves), counter, t0),
                          daemon=True)
    hb.start()

    agg_b = agg_n = agg_e = agg_s = 0
    skipped = ok = 0
    errors = []
    jobs = [(lp, args.workers, args.dry_run) for lp in leaves]
    try:
        with ProcessPoolExecutor(max_workers=args.leaves) as ex:
            futs = [ex.submit(process_leaf, j) for j in jobs]
            for f in as_completed(futs):
                r = f.result()
                counter["done"] += 1
                tag = r["leaf"].split("/details/")[-1]
                if r["error"]:
                    errors.append((r["leaf"], r["error"]))
                    print(f"  ✗ {tag}  {r['error']}", flush=True)
                elif r["skipped"]:
                    skipped += 1
                else:
                    ok += 1
                    agg_b += r["broken"]
                    agg_n += r["nulls"]
                    agg_e += r["enriched"]
                    agg_s += r["sessions"]
                    print(f"  ✓ {tag}  重置={r['broken']} enrich={r['enriched']} "
                          f"sessions={r['sessions']} {r['took']:.1f}s", flush=True)
    except KeyboardInterrupt:
        print("\n中断：已完成的叶子已落盘，重跑会跳过它们。")
    finally:
        stop_evt.set()

    print(f"\n完成 处理={ok} 跳过={skipped} 失败={len(errors)}  "
          f"耗时 {time.time()-t0:.0f}s")
    if args.dry_run:
        print(f"合计待重置 {agg_b:,} 行，已 NULL {agg_n:,} 行")
    else:
        print(f"重置 {agg_b:,} 行，enrich {agg_e:,} 行，重建 sessions {agg_s:,} 条")
    if errors:
        print("失败明细（前 10）：")
        for lp, e in errors[:10]:
            print(f"  {lp}: {e}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
