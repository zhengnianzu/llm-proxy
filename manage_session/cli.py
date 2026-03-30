#!/usr/bin/env python3
"""
cli.py — manage_session CLI

Commands:
    match       [--full]    自动扫描 tasks/ 和 raw_index/ 并匹配（增量 / 全量）
    status      [--task ID] 查看批次状态
    summary                 查看全局汇总
    list-caches             列出 pair_cache
    clear-cache             清空所有缓存（谨慎）
"""

import argparse
import json
import sys
from pathlib import Path

# 保证 core 包可 import
sys.path.insert(0, str(Path(__file__).parent))

from core import manifest as manifest_mod
from core import matcher  as matcher_mod
from core import cache    as cache_mod
from core import views    as views_mod


# ── helpers ───────────────────────────────────────────────────────────────────

def _print_json(data):
    print(json.dumps(data, ensure_ascii=False, indent=2))


def _do_pair_match(task_id: str, index_id: str, force: bool = False) -> dict:
    """匹配一对 (task, index)，命中缓存则跳过。返回 result dict。"""
    t = manifest_mod.get_task(task_id)
    i = manifest_mod.get_index(index_id)

    cached = cache_mod.get_cache(t["hash"], i["hash"])
    if cached and not force:
        print(f"  [cache hit] {task_id} × {index_id}  "
              f"(matched={cached['matched_count']})")
        return cached

    print(f"  [matching] {task_id} × {index_id} …")
    task_queries = manifest_mod.load_task_queries(task_id)
    index_q1s    = manifest_mod.load_index_q1s(index_id)

    result = matcher_mod.match(
        task_queries=task_queries,
        index_q1s=index_q1s,
        task_id=task_id,
        index_id=index_id,
        task_hash=t["hash"],
        index_hash=i["hash"],
    )
    result_dict = result.to_dict()
    cache_mod.save_cache(result_dict)
    print(f"    matched={result.matched_count}  "
          f"unmatched_tasks={result.unmatched_task_count}  "
          f"unmatched_idx={result.unmatched_index_count}")
    return result_dict


# ── commands ──────────────────────────────────────────────────────────────────

def cmd_match(args):
    all_tasks   = manifest_mod.get_all_tasks()
    all_indexes = manifest_mod.get_all_indexes()

    if not all_tasks:
        print("[warn] No task files found in tasks/ directory.")
        return
    if not all_indexes:
        print("[warn] No index files found in raw_index/ directory.")
        return

    if args.full:
        print("Full re-match (clearing cache)…")
        cache_mod.clear_all()

    print(f"Matching {len(all_tasks)} task(s) × {len(all_indexes)} index(es)…\n")

    for task_id in all_tasks:
        for index_id in all_indexes:
            _do_pair_match(task_id, index_id, force=args.full)
        views_mod.update_batch_status(task_id)

    summary = views_mod.update_global_summary()
    print(f"\n✓ Done. Global: tasks={summary['total_tasks']}  "
          f"trajectories={summary['total_trajectories']}  "
          f"matched={summary['total_matched']}  "
          f"unmatched_tasks={summary['total_unmatched_tasks']}")


def cmd_status(args):
    if args.task:
        task_id = args.task
        view = views_mod.update_batch_status(task_id)
        print(f"\nBatch: {task_id}")
        print(f"  Total tasks      : {view['total_tasks']}")
        print(f"  Matched          : {view['matched_count']}")
        print(f"  Unmatched        : {view['unmatched_count']}")
        print(f"  Completion rate  : {view['completion_rate']:.1%}")
        if view['matched_indexes']:
            print("  Matched from indexes:")
            for idx_id, cnt in view['matched_indexes'].items():
                print(f"    {cnt:6d}  {idx_id}")
        if view['topics']:
            print("  Topics:")
            for t, c in list(view['topics'].items())[:5]:
                print(f"    {c:6d}  {t}")
    else:
        # 列出所有批次
        all_tasks = manifest_mod.get_all_tasks()
        print(f"{'TASK':<40} {'TOTAL':>7} {'MATCHED':>8} {'RATE':>7}")
        print("-" * 68)
        for task_id, meta in sorted(all_tasks.items()):
            safe = task_id.replace("/", "__")
            bv_path = views_mod._batch_status_dir() / f"{safe}.json"
            if bv_path.exists():
                import json as _j
                bv = _j.loads(bv_path.read_text())
                mc = bv.get("matched_count", 0)
                rate = bv.get("completion_rate", 0)
            else:
                mc, rate = 0, 0.0
            print(f"{task_id:<40} {meta['count']:>7} {mc:>8} {rate:>6.1%}")


def cmd_summary(args):
    summary = views_mod.update_global_summary()
    print("\n" + "=" * 60)
    print(f"  Global Summary  —  {summary['updated_at']}")
    print("=" * 60)
    print(f"  Task batches         : {summary['total_task_batches']}")
    print(f"  Total tasks          : {summary['total_tasks']}")
    print(f"  Index files          : {summary['total_index_files']}")
    print(f"  Total trajectories   : {summary['total_trajectories']}")
    print(f"  Matched              : {summary['total_matched']}")
    print(f"  Unmatched tasks      : {summary['total_unmatched_tasks']}")
    print(f"  Unmatched indexes    : {summary['total_unmatched_indexes']}")
    print()
    print(f"  {'BATCH':<40} {'TASKS':>7} {'MATCHED':>8} {'RATE':>7}")
    print("  " + "-" * 66)
    for b in summary['batches']:
        print(f"  {b['task_id']:<40} {b['total_tasks']:>7} "
              f"{b['matched_count']:>8} {b['completion_rate']:>6.1%}")
    print()
    print(f"  {'INDEX':<50} {'COUNT':>7}")
    print("  " + "-" * 58)
    for i in summary['indexes']:
        print(f"  {i['index_id']:<50} {i['count']:>7}")
    print("=" * 60)


def cmd_list_caches(args):
    caches = cache_mod.list_caches()
    if not caches:
        print("No caches.")
        return
    for c in caches:
        print(f"  {c['pair_key']}  task={c['task_id']}  "
              f"index={c['index_id']}  matched={c['matched_count']}")


def cmd_clear_cache(args):
    confirm = input("Clear ALL pair caches? [y/N] ").strip().lower()
    if confirm == "y":
        cache_mod.clear_all()
        print("Cache cleared.")
    else:
        print("Aborted.")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="python cli.py",
        description="manage_session — task/index matching system",
    )
    sub = parser.add_subparsers(dest="cmd")

    p = sub.add_parser("match", help="Auto-scan and run matching (incremental by default)")
    p.add_argument("--full", action="store_true", help="Force full re-match, clear cache")

    p = sub.add_parser("status", help="Show batch status")
    p.add_argument("--task", default=None, help="task_id to show details")

    sub.add_parser("summary", help="Show global summary")
    sub.add_parser("list-caches", help="List pair cache files")
    sub.add_parser("clear-cache", help="Clear all pair caches")

    args = parser.parse_args()

    dispatch = {
        "match":       cmd_match,
        "status":      cmd_status,
        "summary":     cmd_summary,
        "list-caches": cmd_list_caches,
        "clear-cache": cmd_clear_cache,
    }

    if args.cmd not in dispatch:
        parser.print_help()
        sys.exit(1)

    dispatch[args.cmd](args)


if __name__ == "__main__":
    main()
