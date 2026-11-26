#!/usr/bin/env python3
"""离线核验最近两天的 session 聚合口径（session_id > user_id > q1_hash）。

做什么
------
对每个叶子，在**本地临时目录**里从 `index.jsonl` 起全新建一份 index.db
（ingest → enrich → sync_sessions），然后拿同一批 requests 行跑两种口径的
`rebuild_sessions` 做对照：

  新口径：session_id > user_id > q1_hash   （当前代码）
  旧口径：纯 api_key||q1_hash              （把 session_id/session_src 置空模拟）

报告每档 session_src 的会话数、trace 分布、以及「旧口径多切出来多少段」。

⚠ 只读共享盘，绝不碰线上 index.db
------------------------------------
共享盘上的 `<leaf>/index.db` 由 src/watch/worker.py 和回填 worker 持有
（`.indexer.lock`），本脚本若就地重算会与它们互相覆盖。所以：
  - 共享盘只读 `index.jsonl` 和合并文件；
  - index.db 一律建在 --scratch（默认 /tmp/sess_audit）下的镜像目录里，
    合并文件用软链引过去（不复制，省 IO 和空间）。
跑完可直接删 scratch，线上状态零影响。

enrich 要解析每个 ~1MB 合并文件，这是全流程最贵的一步（NFS 读 + JSON 解析），
所以默认只抽样每叶 --sample 条；要全量核验传 --sample 0。

用法
----
  # 默认：最近 2 天、proxy-004、每叶抽样 300 条
  python test/audit_session_agg.py

  # 全量核验某一天（慢，单叶数万条）
  python test/audit_session_agg.py --days 1 --sample 0

  # 看所有 newapi 来源
  python test/audit_session_agg.py --root all

  # 指定来源与并发
  python test/audit_session_agg.py --root jumper-003 --leaves 4 --workers 8

  # 导出逐会话明细，便于人工抽查
  python test/audit_session_agg.py --dump-sessions /tmp/sessions.csv
"""
import argparse
import csv
import json
import os
import shutil
import sqlite3
import sys
import time
from collections import Counter, defaultdict
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

SRC_LABEL = {
    "session_id": "metadata.session_id",
    "user_id_json": "user_id 内嵌 session_id",
    "user_id": "metadata.user_id",
    "": "(回退 q1_hash)",
}
# 报告里的固定列顺序，保证多次运行输出可比对
SRC_ORDER = ["session_id", "user_id_json", "user_id", ""]


# ─────────────────────────── 叶子发现 ───────────────────────────

def find_leaves(root: Path, days: int) -> list:
    """取最近 days 个「日」目录下的全部小时叶子（含 index.jsonl 的那层）。

    目录形如 <root>/<日6>/<时8>/index.jsonl。只认纯数字的日目录，避开
    proxy-004 下混进来的 'details' 这类非日期目录。
    """
    if not root.is_dir():
        return []
    day_dirs = sorted((d for d in root.iterdir()
                       if d.is_dir() and d.name.isdigit() and len(d.name) == 6),
                      key=lambda p: p.name, reverse=True)[:days]
    leaves = []
    for d in day_dirs:
        for h in sorted(d.iterdir(), key=lambda p: p.name, reverse=True):
            if h.is_dir() and (h / "index.jsonl").is_file():
                leaves.append(h)
    return leaves


# ─────────────────────────── 子进程：核验单叶 ───────────────────────────

def _mirror_leaf(leaf: Path, scratch: Path, sample: int) -> Path:
    """在 scratch 下做叶子镜像：拷 index.jsonl（可截断抽样）+ 软链合并文件。

    返回镜像目录。软链而非拷贝：合并文件单个 ~1MB，整叶可达数十 GB。
    """
    mirror = scratch / leaf.parent.name / leaf.name
    if mirror.exists():
        shutil.rmtree(mirror, ignore_errors=True)
    mirror.mkdir(parents=True, exist_ok=True)

    src_idx = leaf / "index.jsonl"
    dst_idx = mirror / "index.jsonl"
    if sample > 0:
        # 取**最后** sample 行：最新的请求更可能带 metadata
        with open(src_idx, "rb") as f:
            lines = f.readlines()
        keep = lines[-sample:]
        with open(dst_idx, "wb") as f:
            f.writelines(keep)
    else:
        shutil.copy2(src_idx, dst_idx)

    # 只给 index.jsonl 里实际引用到的 req_file 建软链（抽样时省下大量 symlink）
    wanted = set()
    with open(dst_idx, "rb") as f:
        for raw in f:
            try:
                e = json.loads(raw.decode("utf-8", "ignore"))
            except json.JSONDecodeError:
                continue
            rf = (e.get("req_file") or "").strip()
            if rf:
                wanted.add(os.path.basename(rf))
    for name in wanted:
        srcf = leaf / name
        if srcf.exists():
            try:
                os.symlink(srcf, mirror / name)
            except FileExistsError:
                pass
    return mirror


def _agg_counts(db: Path) -> dict:
    """从一份建好的 index.db 里统计聚合结果。"""
    c = sqlite3.connect(str(db))
    c.row_factory = sqlite3.Row
    try:
        traces = {r["session_key"]: r["c"] for r in c.execute(
            "SELECT session_key, COUNT(*) c FROM traces GROUP BY 1")}
        by_src = defaultdict(lambda: {"sessions": 0, "traces": 0, "max": 0})
        rows = []
        for r in c.execute("SELECT session_key, session_src, session_id, api_key, q1, "
                           "first_ts, last_ts FROM sessions"):
            src = r["session_src"] or ""
            n = traces.get(r["session_key"], 0)
            b = by_src[src]
            b["sessions"] += 1
            b["traces"] += n
            b["max"] = max(b["max"], n)
            rows.append({
                "session_src": src,
                "session_id": r["session_id"] or "",
                "traces": n,
                "api_key": (r["api_key"] or "")[:12],
                "first_ts": r["first_ts"] or "",
                "last_ts": r["last_ts"] or "",
                "q1": (r["q1"] or "")[:60].replace("\n", " "),
            })
        req_src = Counter(
            (r[0] or "") for r in c.execute("SELECT session_src FROM requests "
                                            "WHERE q1_hash IS NOT NULL AND msg_count >= 0"))
        # 口径交叉诊断：同一个 q1_hash 下混有「有 metadata」与「无 metadata」的行。
        # 这种组在新口径下必然分裂（有标识的按 sid 走、没标识的留在 q1_hash），
        # 是**正确行为**，也是会话总数可能反而变多的唯一来源。
        groups = defaultdict(set)
        for r in c.execute("SELECT q1_hash, CASE WHEN COALESCE(session_src,'')='' "
                           "THEN 0 ELSE 1 END h FROM requests "
                           "WHERE q1_hash IS NOT NULL AND msg_count >= 0 AND q1_hash <> ''"):
            groups[r["q1_hash"]].add(r["h"])
        mixed = sum(1 for v in groups.values() if len(v) > 1)
        return {"by_src": {k: dict(v) for k, v in by_src.items()},
                "req_src": dict(req_src),
                "sessions": len(rows),
                "mixed_q1": mixed,
                "rows": rows}
    finally:
        c.close()


def audit_leaf(job: tuple) -> dict:
    """job = (leaf_str, scratch_str, sample, workers, keep)。在子进程里核验一个叶子。"""
    leaf_str, scratch_str, sample, workers, keep = job
    leaf = Path(leaf_str)
    res = {"leaf": leaf_str, "dir_key": f"{leaf.parent.name}/{leaf.name}",
           "error": "", "took": 0.0, "requests": 0,
           "new": {}, "old": {}, "consistent": None}
    t0 = time.time()
    try:
        from utils import newapi_index_db as nidb

        mirror = _mirror_leaf(leaf, Path(scratch_str), sample)

        # 1) 新口径：正常三段式建库
        built = nidb.build_leaf(str(mirror), workers=workers)
        res["requests"] = built.get("ingested", 0)
        res["pending"] = built.get("pending", 0)
        db = mirror / "index.db"
        res["new"] = _agg_counts(db)

        # 2) 一致性：增量 sync_sessions 与全量 rebuild_sessions 必须给出同样结果
        #    （doc/watcher.md「数据一致性」要求的那条回归）
        res["consistent"] = _check_consistency(mirror, nidb)

        # 3) 旧口径：同一批 requests 行，把身份列置空 → 退化成纯 q1_hash 聚合
        old_dir = mirror.parent / (mirror.name + "__old")
        shutil.rmtree(old_dir, ignore_errors=True)
        old_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(db, old_dir / "index.db")
        oc = sqlite3.connect(str(old_dir / "index.db"))
        oc.execute("UPDATE requests SET session_id=NULL, session_src=NULL")
        oc.commit()
        oc.close()
        nidb.rebuild_sessions(str(old_dir))
        res["old"] = _agg_counts(old_dir / "index.db")

        if not keep:
            shutil.rmtree(mirror, ignore_errors=True)
            shutil.rmtree(old_dir, ignore_errors=True)
    except Exception as e:                       # noqa: BLE001 - 单叶失败不该中断全局
        res["error"] = f"{type(e).__name__}: {e}"
    res["took"] = time.time() - t0
    return res


def _check_consistency(mirror: Path, nidb) -> bool:
    """增量折叠 vs 全量重建，三表逐列比对。True=一致。"""
    def dump(p: Path):
        c = sqlite3.connect(str(p / "index.db"))
        c.row_factory = sqlite3.Row
        try:
            s = {r["session_key"]: tuple(sorted(dict(r).items()))
                 for r in c.execute("SELECT * FROM sessions")}
            t = sorted(tuple(sorted(dict(r).items())) for r in c.execute(
                "SELECT session_key, filename, model, msg_count, ts, success FROM traces"))
            ci = {r["lookup_key"]: r["session_key"]
                  for r in c.execute("SELECT * FROM chain_index")}
            return s, t, ci
        finally:
            c.close()

    incr = mirror.parent / (mirror.name + "__incr")
    full = mirror.parent / (mirror.name + "__full")
    try:
        for p in (incr, full):
            shutil.rmtree(p, ignore_errors=True)
            p.mkdir(parents=True, exist_ok=True)
            shutil.copy2(mirror / "index.db", p / "index.db")
        # 增量：清空聚合结果、水位归零，小批次逐行折叠
        c = sqlite3.connect(str(incr / "index.db"))
        c.executescript("DELETE FROM sessions; DELETE FROM traces; DELETE FROM chain_index;")
        c.execute("INSERT INTO meta(k,v) VALUES('sessions_synced_rowid','0') "
                  "ON CONFLICT(k) DO UPDATE SET v='0'")
        c.commit()
        c.close()
        while nidb.sync_sessions(str(incr), batch=37)["synced"]:
            pass
        nidb.rebuild_sessions(str(full))
        return dump(incr) == dump(full)
    finally:
        shutil.rmtree(incr, ignore_errors=True)
        shutil.rmtree(full, ignore_errors=True)


# ─────────────────────────── 报告 ───────────────────────────

def _fmt_src_table(by_src: dict, total_label: str = "") -> list:
    lines = []
    srcs = [s for s in SRC_ORDER if s in by_src] + \
           [s for s in by_src if s not in SRC_ORDER]
    for s in srcs:
        d = by_src[s]
        avg = d["traces"] / d["sessions"] if d["sessions"] else 0
        lines.append(f"    {SRC_LABEL.get(s, s):26} "
                     f"会话 {d['sessions']:6}  请求 {d['traces']:7}  "
                     f"均 {avg:6.1f}  最大 {d['max']:5}")
    return lines


def report(results: list, args) -> int:
    ok = [r for r in results if not r["error"]]
    bad = [r for r in results if r["error"]]

    print()
    print("=" * 78)
    print(f"session 聚合核验报告（最近 {args.days} 天，root={args.root}，"
          f"抽样 {'全量' if args.sample == 0 else str(args.sample) + '/叶'}）")
    print("=" * 78)

    if not ok:
        print("\n没有成功核验的叶子。")
        for r in bad:
            print(f"  ✗ {r['dir_key']}: {r['error']}")
        return 1

    # ── 汇总 ──
    tot_new = defaultdict(lambda: {"sessions": 0, "traces": 0, "max": 0})
    tot_old = defaultdict(lambda: {"sessions": 0, "traces": 0, "max": 0})
    tot_req = Counter()
    n_new = n_old = n_req = 0
    for r in ok:
        for src, d in r["new"]["by_src"].items():
            t = tot_new[src]
            t["sessions"] += d["sessions"]; t["traces"] += d["traces"]
            t["max"] = max(t["max"], d["max"])
        for src, d in r["old"]["by_src"].items():
            t = tot_old[src]
            t["sessions"] += d["sessions"]; t["traces"] += d["traces"]
            t["max"] = max(t["max"], d["max"])
        tot_req.update(r["new"]["req_src"])
        n_new += r["new"]["sessions"]
        n_old += r["old"]["sessions"]
        n_req += r["requests"]

    print(f"\n叶子 {len(ok)} 个成功 / {len(bad)} 个失败，请求 {n_req} 条\n")

    print("── 请求级：metadata 覆盖率 ──")
    for s in [x for x in SRC_ORDER if x in tot_req] + \
             [x for x in tot_req if x not in SRC_ORDER]:
        n = tot_req[s]
        pct = 100.0 * n / max(sum(tot_req.values()), 1)
        print(f"    {SRC_LABEL.get(s, s):26} {n:7} 条  {pct:5.1f}%")
    covered = sum(v for k, v in tot_req.items() if k)
    print(f"    {'→ 有上游会话标识':26} {covered:7} 条  "
          f"{100.0 * covered / max(sum(tot_req.values()), 1):5.1f}%")

    print("\n── 会话级：新口径（session_id > user_id > q1_hash）──")
    for ln in _fmt_src_table(tot_new):
        print(ln)
    print(f"    {'合计':26} 会话 {n_new:6}")

    print("\n── 会话级：旧口径（纯 api_key||q1_hash）──")
    for ln in _fmt_src_table(tot_old):
        print(ln)
    print(f"    {'合计':26} 会话 {n_old:6}")

    delta = n_old - n_new
    mixed = sum(r["new"].get("mixed_q1", 0) for r in ok)
    print(f"\n── 口径差异 ──")
    print(f"    旧口径 {n_old} 个会话 → 新口径 {n_new} 个会话（{delta:+d}）")
    print(f"    其中「同一 q1_hash 下混有 有/无 metadata 两种行」的组：{mixed} 个")
    print()
    print("    两种相反的效应同时存在，净差取决于这批数据的 metadata 覆盖率：")
    print(f"      合并(-)：被 q1_hash 错切开的同一会话，靠上游 session_id 合回去")
    print(f"      分裂(+)：同一 q1_hash 里有标识的行按 sid 走、无标识的留在 q1_hash，")
    print(f"              必然各成一条 —— 这是**正确行为**，上限就是上面那 {mixed} 个混合组")
    if delta > 0:
        print(f"\n    → 净减少 {delta} 个：合并效应占优（metadata 覆盖较完整）")
    elif delta < 0:
        if -delta <= mixed:
            print(f"\n    → 净增加 {-delta} 个，且 ≤ 混合组数 {mixed}，分裂效应占优，符合预期。")
            print(f"      覆盖率低的时段会这样：大量 q1_hash 组里只有少数行带 metadata。")
        else:
            print(f"\n    ⚠ 净增加 {-delta} 个，**超过**混合组数 {mixed}，无法只用口径交叉解释，需排查。")
    else:
        print("\n    → 两种效应恰好抵消，或这批数据的 metadata 没有改变分组。")

    # ── 一致性回归 ──
    cons = [r for r in ok if r["consistent"] is not None]
    bad_cons = [r for r in cons if not r["consistent"]]
    print(f"\n── 增量 vs 全量一致性（doc/watcher.md 要求）──")
    if bad_cons:
        print(f"    ✗ {len(bad_cons)}/{len(cons)} 个叶子不一致：")
        for r in bad_cons[:10]:
            print(f"        {r['dir_key']}")
    else:
        print(f"    ✓ {len(cons)}/{len(cons)} 个叶子：sessions/traces/chain_index 三表逐列一致")

    # ── 过度合并体检 ──
    print(f"\n── 过度合并体检（user_id 是「人」不是「会话」）──")
    mx_sid = tot_new.get("session_id", {}).get("max", 0)
    mx_uid = tot_new.get("user_id", {}).get("max", 0)
    mx_q1 = tot_new.get("", {}).get("max", 0)
    print(f"    单会话最大请求数：session_id={mx_sid}  user_id={mx_uid}  q1_hash={mx_q1}")
    if mx_uid and mx_sid and mx_uid > 3 * max(mx_sid, mx_q1, 1):
        print("    ⚠ user_id 档明显偏大，疑似把同一用户的多段对话并成一条。")
        print("      回退：_lookup_identity() 里 user_id 档改成 f'uid||{uid}||{q1_hash}'")
    else:
        print("    ✓ user_id 档与其他档同量级，未见异常合并")

    # ── 最大的几个会话，便于人工抽查 ──
    allrows = [row for r in ok for row in r["new"]["rows"]]
    allrows.sort(key=lambda x: -x["traces"])
    print(f"\n── 最大的 10 个会话（人工抽查用）──")
    print(f"    {'来源':14} {'请求':>5}  {'标识/Q1':52}")
    for row in allrows[:10]:
        ident = row["session_id"] or row["q1"]
        print(f"    {row['session_src'] or '(q1)':14} {row['traces']:5}  {ident[:52]}")

    if bad:
        print(f"\n── 失败的叶子 ──")
        for r in bad[:15]:
            print(f"    ✗ {r['dir_key']}: {r['error']}")

    if args.dump_sessions:
        with open(args.dump_sessions, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["session_src", "session_id", "traces",
                                              "api_key", "first_ts", "last_ts", "q1"])
            w.writeheader()
            w.writerows(allrows)
        print(f"\n逐会话明细已写入 {args.dump_sessions}（{len(allrows)} 行）")

    print()
    # 退出码只认真正的异常：一致性回归失败，或会话增量无法用口径交叉解释。
    unexplained = delta < 0 and -delta > mixed
    return 1 if (bad_cons or unexplained) else 0


# ─────────────────────────── main ───────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description="离线核验最近两天的 session 聚合口径（只读共享盘）")
    ap.add_argument("--root", default="proxy-004",
                    help="来源名或 all（默认 proxy-004）。可选：" + ", ".join(ROOTS))
    ap.add_argument("--days", type=int, default=2, help="最近几天（默认 2）")
    ap.add_argument("--sample", type=int, default=300,
                    help="每叶抽样最后 N 条请求，0=全量（默认 300）")
    ap.add_argument("--leaves", type=int, default=6, help="并发叶子数（默认 6）")
    ap.add_argument("--workers", type=int, default=4,
                    help="单叶 enrich 并发子进程数（默认 4）")
    ap.add_argument("--scratch", default="/tmp/sess_audit",
                    help="本地临时目录（默认 /tmp/sess_audit）")
    ap.add_argument("--keep", action="store_true", help="保留临时 index.db 供排查")
    ap.add_argument("--dump-sessions", default="", help="把逐会话明细写入 CSV")
    ap.add_argument("--list", action="store_true", help="只列出将要核验的叶子")
    args = ap.parse_args()

    roots = ROOTS if args.root == "all" else {args.root: ROOTS.get(args.root)}
    if args.root != "all" and roots[args.root] is None:
        print(f"未知来源 {args.root!r}，可选：{', '.join(ROOTS)} 或 all", file=sys.stderr)
        return 2

    leaves = []
    for name, root in roots.items():
        got = find_leaves(root, args.days)
        print(f"{name}: {len(got)} 个叶子  ({root})")
        leaves.extend(got)
    if not leaves:
        print("没找到叶子。", file=sys.stderr)
        return 2

    if args.list:
        for lf in leaves:
            n = sum(1 for _ in open(lf / "index.jsonl", "rb"))
            print(f"  {lf.parent.name}/{lf.name}  {n} 行")
        return 0

    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    print(f"\n临时目录 {scratch}（只读共享盘，线上 index.db 不受影响）")
    print(f"核验 {len(leaves)} 个叶子，{args.leaves} 并发 × 每叶 {args.workers} 进程 …\n")

    jobs = [(str(lf), str(scratch), args.sample, args.workers, args.keep)
            for lf in leaves]
    results = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.leaves) as pool:
        futs = {pool.submit(audit_leaf, j): j for j in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            results.append(r)
            mark = "✗" if r["error"] else "✓"
            extra = ""
            if not r["error"]:
                extra = (f"{r['requests']:5} 条 → 新 {r['new']['sessions']:4} / "
                         f"旧 {r['old']['sessions']:4} 会话"
                         f"{'' if r['consistent'] else '  ⚠一致性不符'}")
            else:
                extra = r["error"][:60]
            print(f"  [{i}/{len(jobs)}] {mark} {r['dir_key']:20} "
                  f"{r['took']:6.1f}s  {extra}")

    print(f"\n总耗时 {time.time() - t0:.1f}s")
    rc = report(results, args)
    if not args.keep:
        shutil.rmtree(scratch, ignore_errors=True)
    else:
        print(f"临时产物保留在 {scratch}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
