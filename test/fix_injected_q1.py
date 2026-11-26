#!/usr/bin/env python3
"""修正被注入块污染的 Q1（存量重算），并重建 sessions。

背景
----
Q1 提取层陆续遇到几类「注入块被当成用户诉求」的问题：

  * `<system-reminder>`（Claude Code 的工作区上下文）
  * `Current runtime context.` … `</current_session_workspace>`（小艺/openclaw-task
    的运行时快照，实测影响 proxy-004 上多个 key、跨 kimi-k3 / deepseek / openPangu）

utils/message_common._strip_noise 已能剥离这些块。但 q1_hash / q1_preview /
chain_key 是**已落库的派生字段**，改代码不会重算存量行。

与 test/refill_q1_worker.py 的区别
----------------------------------
那个脚本只挑 `q1_hash = ''`（req 为空导致 Q1 算成空串）的行。而本问题的行
**q1_hash 是有值的，只是值错了** —— 算的是注入块的 hash，所以那个脚本捞不到。
这里按 q1_preview 的内容特征筛选。

每个叶子依次做：
  1. 按 q1_preview 前缀特征找出受污染行 → meta 字段置 NULL（enrich 只处理 NULL 行）
  2. enrich(leaf, workers)   —— 用新的剥离规则重算 Q1
  3. rebuild_sessions(leaf)  —— 用新 Q1 重建 sessions/traces

默认只处理最近两天的叶子（--days 2）。可随时 Ctrl-C，重跑会跳过已修好的叶子。

⚠ 运行前请重启 src.watch.worker
    它若仍在跑旧代码，会在后台把同一批叶子重算回错误 Q1。
    重启（没有 restart 子命令）：./app watch stop && ./app watch start

用法
----
  # 预览（只读）：列出每个叶子的受污染行数与修复前后对照
  python test/fix_injected_q1.py --dry-run
  python test/fix_injected_q1.py --dry-run -v        # 带新旧 Q1 对照

  # 正式跑
  python test/fix_injected_q1.py
  python test/fix_injected_q1.py --days 3 --leaves 8 --workers 8
  python test/fix_injected_q1.py --root proxy-004
"""
import argparse
import datetime
import os
import sqlite3
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

HARNESS = Path("/mnt/sfs_turbo/s3-asset-b-hd-cce-aifm-nlp-exp/raw/harness")
NEWAPI = Path("/mnt/sfs_turbo/s3-asset-b-hd-cce-aifm-nlp-exp/data/newapi")
ROOTS = {
    "proxy-004": HARNESS / "proxy-004/data/newapi/logs/details",
    "jumper-003": HARNESS / "jumper-003/data/newapi/logs/details",
    "jumper-001": HARNESS / "jumper-001/data/newapi/logs/details",
    "jumper": HARNESS / "jumper/data/newapi/logs/details",
    "coding-plan": HARNESS / "coding-plan/data/newapi/logs/details",
    "proxy45": NEWAPI / "115.120.127.45/logs/details",
    "proxy8": NEWAPI / "115.120.50.8/logs/details",
}

# 受污染行的特征：q1_preview 以注入块抬头开始。
# 这些行的 q1_hash 有值（算的是注入块的 hash），所以不能用 q1_hash='' 来筛。
# msg_count=-1 是「文件缺失/损坏」的死行，排除（重算只会再失败一次）。
_BAD_PREFIXES = (
    "Current runtime context.",
    "<system-reminder>",
    "As you answer the user's questions",
)
_WHERE_BAD = (
    "(" + " OR ".join(f"q1_preview LIKE '{p}%'" for p in
                      (p.replace("'", "''") for p in _BAD_PREFIXES)) + ")"
    + " AND COALESCE(msg_count, 0) >= 0"
)
_COUNT_BAD = f"SELECT COUNT(*) FROM requests WHERE {_WHERE_BAD}"
_RESET_BAD = f"""
UPDATE requests
   SET q1_hash = NULL, q1_preview = NULL, chain_key = NULL,
       msg_count = NULL, user_turns = NULL, attempts = 0
 WHERE {_WHERE_BAD}
"""
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
    """job = (leaf_str, workers, dry, verbose)。"""
    leaf_str, workers, dry, verbose = job
    leaf = Path(leaf_str)
    db = leaf / "index.db"
    res = {"leaf": leaf_str, "bad": 0, "nulls": 0, "enriched": 0,
           "sessions": 0, "error": "", "took": 0.0, "skipped": False,
           "samples": []}
    if not db.exists():
        res["error"] = "no index.db"
        return res
    t0 = time.time()

    if dry:
        try:
            c = _conn(db)
            try:
                res["bad"] = c.execute(_COUNT_BAD).fetchone()[0]
                res["nulls"] = c.execute(_COUNT_NULL).fetchone()[0]
                if verbose and res["bad"]:
                    # 用新规则现场算一遍，给出修复前后对照（只读，不写库）
                    from utils.newapi_consumer import _resolve_combined_path
                    from utils.newapi_format import (parse_combined_file,
                                                     compute_session_fields)
                    rows = c.execute(
                        f"SELECT req_file, q1_preview FROM requests "
                        f"WHERE {_WHERE_BAD} LIMIT 2"
                    ).fetchall()
                    for r in rows:
                        p = _resolve_combined_path(leaf, r["req_file"])
                        parsed = parse_combined_file(p) if p else None
                        if not parsed:
                            continue
                        f = compute_session_fields(parsed["messages"])
                        res["samples"].append(
                            (r["q1_preview"][:50], f["q1_preview"][:50]))
            finally:
                c.close()
        except sqlite3.Error as e:
            res["error"] = str(e)
        res["took"] = time.time() - t0
        return res

    from utils import newapi_index_db as nidb

    # 1) 一次扫描同时完成「判定 + 置空」：直接 UPDATE 取 rowcount
    try:
        c = _conn(db, write=True)
        try:
            with c:
                bad = c.execute(_RESET_BAD).rowcount or 0
                if bad:
                    nidb._meta_set(c, "sessions_dirty", "1")
            res["bad"] = bad
            res["nulls"] = c.execute(_COUNT_NULL).fetchone()[0]
        finally:
            c.close()
    except sqlite3.Error as e:
        res["error"] = f"reset: {e}"
        return res

    if not res["bad"] and not res["nulls"]:
        res["skipped"] = True
        res["took"] = time.time() - t0
        return res

    # 2) enrich：用新的剥离规则重算 Q1
    try:
        out = nidb.enrich(leaf_str, workers=workers)
        res["enriched"] = int((out or {}).get("enriched") or 0)
    except Exception as e:  # noqa: BLE001
        res["error"] = f"enrich: {type(e).__name__}: {e}"
        res["took"] = time.time() - t0
        return res

    # 3) 重建 sessions（从 requests 表按新 Q1 折叠）
    try:
        res["sessions"] = int(nidb.rebuild_sessions(leaf_str) or 0)
    except Exception as e:  # noqa: BLE001
        res["error"] = f"sessions: {type(e).__name__}: {e}"

    res["took"] = time.time() - t0
    return res


# ─────────────────────────── 叶子发现 ───────────────────────────

def discover(args) -> list:
    """按 --days 取最近 N 天的叶子；--start/--end 可覆盖。"""
    if args.root:
        if args.root not in ROOTS:
            raise SystemExit(f"未知 root: {args.root}，可选 {list(ROOTS)}")
        roots = {args.root: ROOTS[args.root]}
    else:
        roots = ROOTS

    if args.start:
        start_date = args.start[:6]
    else:
        # 叶子目录名是 YYMMDD，按本地日期回推 N-1 天（含今天）
        d = datetime.date.today() - datetime.timedelta(days=max(0, args.days - 1))
        start_date = d.strftime("%y%m%d")
    end_date = args.end[:6] if args.end else ""

    leaves = []
    for name, root in roots.items():
        if not root.is_dir():
            continue
        try:
            dates = os.listdir(root)
        except OSError:
            continue
        for d in dates:
            if not d.isdigit() or len(d) != 6:
                continue
            if d < start_date or (end_date and d > end_date):
                continue
            dd = root / d
            if not dd.is_dir():
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
    leaves.sort(reverse=True)          # 最新在前
    if args.limit:
        leaves = leaves[:args.limit]
    return leaves


def check_watcher_safety(force: bool) -> bool:
    """确认运行中的 watch worker 已加载新的剥离规则。"""
    src = Path(__file__).resolve().parent.parent / "utils/message_common.py"
    try:
        src_mtime = src.stat().st_mtime
    except OSError:
        return True
    old = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            cmd = (p / "cmdline").read_bytes().decode("utf-8", "replace")
            if "src.watch.worker" not in cmd:
                continue
            if p.stat().st_mtime < src_mtime:
                old.append(p.name)
        except OSError:
            continue
    if not old:
        return True
    print(f"\n⚠ src.watch.worker 仍在运行旧代码（PID {', '.join(old)}），"
          f"启动早于 utils/message_common.py 的修改时间。")
    print("  它会在后台把同一批叶子重算回错误 Q1，导致本脚本白跑。先重启：")
    print("      ./app watch stop && ./app watch start\n")
    if not force:
        print("已中止。确认要强行继续请加 --force。")
        return False
    print("--force 已指定，继续。")
    return True


def _heartbeat(stop_evt, total, counter, t0):
    while not stop_evt.wait(15):
        print(f"  … 进度 {counter['done']}/{total}  已用时 {time.time()-t0:.0f}s",
              flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="修正被注入块污染的 Q1 并重建 sessions")
    ap.add_argument("--dry-run", action="store_true", help="只统计，不写入")
    ap.add_argument("--days", type=int, default=2, help="处理最近 N 天的叶子（默认 2）")
    ap.add_argument("--start", default="", help="起始叶子，如 26112500（覆盖 --days）")
    ap.add_argument("--end", default="", help="结束叶子（含）")
    ap.add_argument("--root", default="", help=f"只处理某个源，可选 {list(ROOTS)}")
    ap.add_argument("--limit", type=int, default=0, help="最多处理多少个叶子")
    ap.add_argument("--leaves", type=int, default=8, help="并行叶子数（默认 8）")
    ap.add_argument("--workers", type=int, default=8, help="每叶 enrich 进程数（默认 8）")
    ap.add_argument("--force", action="store_true", help="跳过 watch worker 版本检查")
    ap.add_argument("-v", "--verbose", action="store_true", help="dry-run 时显示新旧 Q1 对照")
    args = ap.parse_args()

    if not args.dry_run and not check_watcher_safety(args.force):
        return 3

    leaves = discover(args)
    if not leaves:
        print("没有匹配到叶子（检查 --days/--start/--end/--root）")
        return 1

    cpu = os.cpu_count() or 8
    par = args.leaves * args.workers
    rng = args.start or f"最近 {args.days} 天"
    print(f"叶子 {len(leaves)} 个（{rng}），最新在前。"
          f"并发 叶子={args.leaves} × 每叶={args.workers} = {par} 进程（CPU {cpu}）")
    if par > cpu * 2:
        print(f"  提示：{par} 进程超过 CPU×2，调度开销可能反而拖慢")
    print("=== DRY RUN（只读）===" if args.dry_run else "=== 开始处理 ===", flush=True)

    t0 = time.time()
    counter = {"done": 0}
    stop_evt = threading.Event()
    threading.Thread(target=_heartbeat,
                     args=(stop_evt, len(leaves), counter, t0),
                     daemon=True).start()

    agg_bad = agg_e = agg_s = 0
    ok = skipped = 0
    errors = []
    jobs = [(lp, args.workers, args.dry_run, args.verbose) for lp in leaves]
    try:
        with ProcessPoolExecutor(max_workers=args.leaves) as ex:
            futs = [ex.submit(process_leaf, j) for j in jobs]
            for f in as_completed(futs):
                r = f.result()
                counter["done"] += 1
                tag = r["leaf"].split("/details/")[-1]
                if r["error"]:
                    errors.append((tag, r["error"]))
                    print(f"  ✗ {tag}  {r['error']}", flush=True)
                    continue
                if r["skipped"] or (args.dry_run and not r["bad"]):
                    skipped += 1
                    continue
                ok += 1
                agg_bad += r["bad"]
                agg_e += r["enriched"]
                agg_s += r["sessions"]
                if args.dry_run:
                    print(f"  [FIX] 污染={r['bad']:>6} 待enrich={r['nulls']:>6}  {tag}",
                          flush=True)
                    for old, new in r["samples"]:
                        print(f"          旧: {old!r}")
                        print(f"          新: {new!r}")
                else:
                    print(f"  ✓ {tag}  重置={r['bad']} enrich={r['enriched']} "
                          f"sessions={r['sessions']} {r['took']:.1f}s", flush=True)
    except KeyboardInterrupt:
        print("\n中断：已完成的叶子已落盘，重跑会跳过它们。")
    finally:
        stop_evt.set()

    print(f"\n完成 处理={ok} 跳过={skipped} 失败={len(errors)}  耗时 {time.time()-t0:.0f}s")
    if args.dry_run:
        print(f"合计待修正 {agg_bad:,} 行")
        print("（dry-run，未写入；去掉 --dry-run 实际执行）")
    else:
        print(f"重置 {agg_bad:,} 行，enrich {agg_e:,} 行，重建 sessions {agg_s:,} 条")
    if errors:
        print("失败明细（前 10）：")
        for tag, e in errors[:10]:
            print(f"  {tag}: {e}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
