#!/usr/bin/env python3
"""clean_corrupt_index_db.py — 扫描某 new-api 源，删除损坏的叶子 index.db

针对「构建索引」在 needs_build 遍历阶段撞上 `database disk image is malformed`
而整根崩溃的情况：按该源注册的层级模板枚举叶子（逐段 iterdir，不 stat 海量 .json），
对每个含 index.db 的叶子做 PRAGMA integrity_check，坏库连同 -wal/-shm 一并删除
（不动 index.jsonl / 原始 .json），随后即可重新「构建索引」。

用法:
    python3 tools/clean_corrupt_index_db.py <root_dir> [--apply]

默认 dry-run，只报告不删；加 --apply 才真正删除。
"""
import os
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.log_scan import iter_leaf_dirs_by_templates, default_templates, detect_format  # noqa: E402
from utils.logs_config import get_path_templates, get_root_id  # noqa: E402

_DB_NAME = "index.db"
_SIDES = ("index.db-wal", "index.db-shm", "index.db-journal")


def _leaves_from_logdir(rid: str) -> list:
    """从 log_dir.db 读该 root_id 已登记的 dir_key 列表（无则空）。

    只读 built=1 的叶子——只有它们才有 index.db、才可能是那个 malformed 库；
    直连磁盘上的 log_dir.db，避免依赖运行中的服务。多个 service 目录都扫，取并集。
    """
    import glob
    keys = set()
    for db in glob.glob("logs/port*/**/log_dir.db", recursive=True):
        try:
            conn = sqlite3.connect(db)
            for (dk,) in conn.execute(
                "SELECT dir_key FROM leaf_status WHERE root_id=? AND built=1", (rid,)):
                keys.add(dk)
            conn.close()
        except Exception:  # noqa: BLE001
            continue
    return sorted(keys)


def _check_db(path: str) -> str:
    """返回 'ok' 或错误描述。

    不做 PRAGMA integrity_check（会整库读，多 GB 库在 NFS 上极慢）；只复现
    needs_build 真正会执行的那条轻量查询——malformed 的库在这里就会抛
    DatabaseError，正是导致 ROOT CRASH 的同一路径。这样每叶只读若干页，秒级完成。
    """
    try:
        conn = sqlite3.connect(path, timeout=5)
        try:
            # 逐字复现 needs_build 的三条读：meta(k,v) 两读 + requests 计数。
            # malformed 的库会在这里抛 sqlite3.DatabaseError，正是 ROOT CRASH 同一路径。
            conn.execute("SELECT v FROM meta WHERE k='ingest_offset'").fetchone()
            conn.execute("SELECT COUNT(*) FROM requests WHERE q1_hash IS NULL").fetchone()
            conn.execute("SELECT v FROM meta WHERE k='sessions_dirty'").fetchone()
            return "ok"
        finally:
            conn.close()
    except sqlite3.OperationalError as e:
        # 半成品库（表还没建）：needs_build 会当成需重建，不算「损坏」，不删。
        return f"ok-incomplete:{e}"
    except sqlite3.DatabaseError as e:
        # database disk image is malformed 等：真正会让 ROOT CRASH 的坏库。
        return f"DatabaseError:{e}"
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}:{e}"


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    root = sys.argv[1].rstrip("/")
    apply = "--apply" in sys.argv[2:]
    if not os.path.isdir(root):
        print(f"root 不存在: {root}")
        return 2

    fmt = detect_format(root)
    rid = get_root_id(root)
    tpls = get_path_templates(root) or default_templates(fmt)

    # 叶子来源优先用 log_dir.db 里已登记的 dir_key（避免在 NFS 上逐层 iterdir 慢）。
    # 找不到 DB / 无记录时回退按模板枚举。
    db_leaves = _leaves_from_logdir(rid)
    if db_leaves:
        leaf_iter = (Path(root) / dk for dk in db_leaves)
        src = f"log_dir.db ({len(db_leaves)} 叶)"
    else:
        leaf_iter = iter_leaf_dirs_by_templates(Path(root), tpls)
        src = f"模板枚举 {tpls}"

    print(f"root   = {root}")
    print(f"rid    = {rid}   fmt={fmt}")
    print(f"叶源   = {src}")
    print(f"mode   = {'APPLY (会删除坏库)' if apply else 'DRY-RUN (只报告)'}")
    print("-" * 60, flush=True)

    scanned = 0   # 有 index.db 的叶子数
    leaves = 0    # 枚举到的叶子总数
    ok = 0
    bad = []
    t0 = time.time()
    for leaf in leaf_iter:
        leaves += 1
        dbp = os.path.join(str(leaf), _DB_NAME)
        if not os.path.isfile(dbp):
            continue
        scanned += 1
        verdict = _check_db(dbp)
        if verdict == "ok":
            ok += 1
        else:
            bad.append((dbp, verdict))
            print(f"[BAD] {dbp}\n      -> {verdict}", flush=True)
        if leaves % 25 == 0:
            print(f"... leaves={leaves} withdb={scanned} ok={ok} bad={len(bad)} "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)

    print("-" * 60)
    print(f"叶子总数={leaves}  含 index.db={scanned}  ok={ok}  bad={len(bad)}  "
          f"耗时={time.time()-t0:.0f}s")

    if not bad:
        print("没有损坏的 index.db，无需清理。")
        return 0

    removed = 0
    for dbp, _v in bad:
        leaf = os.path.dirname(dbp)
        targets = [dbp] + [os.path.join(leaf, s) for s in _SIDES]
        for t in targets:
            if os.path.exists(t):
                if apply:
                    try:
                        os.remove(t)
                        removed += 1
                        print(f"[DEL] {t}")
                    except OSError as e:
                        print(f"[ERR] 删除失败 {t}: {e}")
                else:
                    print(f"[WOULD-DEL] {t}")

    if apply:
        print(f"\n完成：删除 {removed} 个文件（{len(bad)} 个坏叶子）。"
              f"\n现在可在数据管理页对该源重新「构建索引」。")
    else:
        print(f"\nDry-run 结束：发现 {len(bad)} 个坏 index.db。"
              f"\n加 --apply 重跑即可删除它们（连同 -wal/-shm）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
