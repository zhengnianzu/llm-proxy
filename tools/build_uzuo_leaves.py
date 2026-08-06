#!/usr/bin/env python3
"""build_uzuo_leaves.py — 为 UZuO 那次导出涉及的 17 个叶子强制构建索引。

这些叶子（240fa79b / proxy-004-latest 源下 260731 15个 + 260730 2个）只有
index.jsonl、没有 index.db，故 needs_build=True、导出被跳过 → "无 session 数据"。
删坏库解决了整根 malformed 崩溃，但这 17 个较旧叶子始终没被 backfill 轮到，
索引仍未生成。这里直接调 ensure_fresh 把它们逐个建出来。

用法:
    python3 tools/build_uzuo_leaves.py            # 只对 needs_build 为真的叶子构建
    python3 tools/build_uzuo_leaves.py --dry      # 只打印各叶当前状态，不构建

建完后回导出页对 key-UZuO 重新发起导出即可。
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import newapi_index_db as nidb  # noqa: E402

ROOT = "/mnt/sfs_turbo/s3-asset-b-hd-cce-aifm-nlp-exp/raw/harness/proxy-004/data/newapi/logs/details"

LEAVES = [
    "details/260731/26073114", "details/260731/26073113", "details/260731/26073112",
    "details/260731/26073111", "details/260731/26073110", "details/260731/26073109",
    "details/260731/26073108", "details/260731/26073107", "details/260731/26073106",
    "details/260731/26073105", "details/260731/26073104", "details/260731/26073103",
    "details/260731/26073102", "details/260731/26073101", "details/260731/26073100",
    "details/260730/26073023", "details/260730/26073022",
]


def main() -> int:
    dry = "--dry" in sys.argv[1:]
    print(f"root = {ROOT}")
    print(f"mode = {'DRY (只看状态)' if dry else 'BUILD (真正构建)'}")
    print("-" * 64, flush=True)

    built = skipped = failed = 0
    t_all = time.time()
    for lk in LEAVES:
        d = os.path.join(ROOT, lk)
        has_jsonl = os.path.isfile(os.path.join(d, "index.jsonl"))
        if not has_jsonl:
            print(f"[SKIP] {lk}  (无 index.jsonl，不是有效叶子)")
            skipped += 1
            continue
        try:
            nb = nidb.needs_build(d)
        except Exception as e:  # noqa: BLE001
            print(f"[ERR ] {lk}  needs_build 异常: {type(e).__name__}: {e}")
            failed += 1
            continue

        if not nb:
            print(f"[OK  ] {lk}  已构建，无需重建")
            skipped += 1
            continue

        if dry:
            print(f"[TODO] {lk}  needs_build=True，将构建")
            continue

        t0 = time.time()
        try:
            st = nidb.ensure_fresh(d, workers=8)
            still = nidb.needs_build(d)
            tag = "BUILT" if not still else "PARTIAL"
            print(f"[{tag}] {lk}  took={time.time()-t0:.1f}s  status={st}", flush=True)
            if still:
                failed += 1
            else:
                built += 1
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] {lk}  {type(e).__name__}: {e}", flush=True)
            failed += 1

    print("-" * 64)
    print(f"完成：built={built} skipped={skipped} failed={failed} 耗时={time.time()-t_all:.0f}s")
    if not dry and failed == 0 and built >= 0:
        print("现在回导出页对 key-UZuO 重新发起导出即可。")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
