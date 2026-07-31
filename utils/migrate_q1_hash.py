"""
utils/migrate_q1_hash.py — 为历史数据回填 q1_hash 聚合键。

背景：会话聚合键从「Q1 文本（截断 500）」改为「完整 Q1 的 md5 → q1_hash」。
本脚本把 q1_hash 回填进历史 index.jsonl，并清空对应 root_dir 的 DB 聚合表，
让运行时（utils.log_routes）用回填后的 index.jsonl 从头重建 session，
保证 DB 里的聚合口径与 index.jsonl 完全一致。

用法：
    # 默认 dry-run，只报告将影响多少行 / 多少 DB session：
    python -m utils.migrate_q1_hash --logs-dir logs_all

    # 实际回填 index.jsonl（写 .bak 备份后原子替换）：
    python -m utils.migrate_q1_hash --logs-dir logs_all --yes

    # 回填 + 清空 DB 聚合表（sessions/traces/chain_index/index_progress）：
    # DB 与 index.jsonl 解耦：每个服务实例的 session_cache.db 位于
    # logs/port<N>/<env-key>/ 下，而其 root_dir 指向 logs_all/... 目录。
    # --reset-db 会扫描 --db-scan-root（默认 logs/）下所有 session_cache.db，
    # 清除 root_dir 落在本次回填目录集合内的行。
    python -m utils.migrate_q1_hash --logs-dir logs_all --reset-db --yes

幂等：
    - index.jsonl 中已有 q1_hash 的行跳过；重复运行报告 0 行更新。
    - .bak 已存在则不覆盖。

一致性说明：
    q1_hash 优先由 req.json 里的**完整** Q1 计算；req.json 缺失时回退
    md5(chain_key)（chain_key 是 500 截断值，与完整 Q1 的 hash 可能不一致）。
    因此回填与清库重建应成对执行；对 req.json 已清理的历史目录，接受
    「该目录内部自洽、但与新写入数据口径略有差异」。
"""

import argparse
import json
import os
from utils.atomic_write import safe_replace
import shutil
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.log_paths import INDEX_FILENAME
from utils.message_common import (
    extract_messages,
    get_first_user_text,
    q1_hash_from_text,
)
from utils.req_index import _extract_q1_full_responses


def _load_json(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def _resolve_req_file(repo_root: Path, index_file: Path, req_file: str) -> Path | None:
    """index.jsonl 里的 req_file 是相对仓库根的路径；缺失时退回同目录同名文件。"""
    if not req_file:
        return None
    candidates = [
        repo_root / req_file,
        Path(req_file),
        index_file.parent / Path(req_file).name,
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def _compute_q1_hash_for_entry(repo_root: Path, index_file: Path, entry: dict) -> str:
    """按 provider 从 req.json 取完整 Q1 算 hash；失败时回退 chain_key。"""
    req_path = _resolve_req_file(repo_root, index_file, entry.get("req_file", ""))
    if req_path is not None:
        data = _load_json(req_path)
        if data is not None:
            provider = entry.get("provider", "")
            if provider == "responses":
                text = _extract_q1_full_responses(data.get("input"))
                if text:
                    return q1_hash_from_text(text)
            else:
                messages = extract_messages(data)
                if messages:
                    return q1_hash_from_text(get_first_user_text(messages))
    # 回退：用已存储的（截断）chain_key
    return q1_hash_from_text(entry.get("chain_key", ""))


def backfill_index_file(repo_root: Path, index_file: Path, dry_run: bool, workers: int) -> tuple[int, int]:
    """回填单个 index.jsonl。返回 (总行数, 更新行数)。

    两阶段：先解析所有行、挑出需回填的 entry；再用线程池并行读 req.json 算
    q1_hash（I/O 密集，读文件时 GIL 释放，多线程有效）。结果按原索引回填，
    保持行顺序不变。
    """
    total = 0
    # out_lines[i] 为最终写回的字符串；need[i] 记录待并行计算的 entry
    out_lines: list[str] = []
    pending: list[tuple[int, dict]] = []  # (out_lines 索引, entry)

    with open(index_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                out_lines.append(line)
                continue
            total += 1
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                out_lines.append(line)
                continue

            if entry.get("q1_hash"):
                out_lines.append(line)
                continue

            # 占位，稍后并行算出 hash 再填
            pending.append((len(out_lines), entry))
            out_lines.append(line)

    updated = len(pending)
    if updated:
        def _work(item: tuple[int, dict]) -> tuple[int, str]:
            idx, entry = item
            entry["q1_hash"] = _compute_q1_hash_for_entry(repo_root, index_file, entry)
            return idx, json.dumps(entry, ensure_ascii=False)

        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                results = pool.map(_work, pending)
        else:
            results = map(_work, pending)
        for idx, new_line in results:
            out_lines[idx] = new_line

    if updated and not dry_run:
        bak = index_file.with_suffix(index_file.suffix + ".bak")
        if not bak.exists():
            shutil.copy2(index_file, bak)
        tmp = str(index_file) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write("\n".join(out_lines) + ("\n" if out_lines else ""))
        safe_replace(tmp, str(index_file))

    return total, updated


def reset_db_scan(db_scan_root: str, root_dirs: list[str], dry_run: bool) -> int:
    """扫描 db_scan_root 下所有 session_cache.db，清除 root_dir 落在
    root_dirs 集合内的聚合行（DB 与 index.jsonl 解耦，需按 root_dir 过滤）。
    """
    norm_roots = {os.path.normpath(r) for r in root_dirs}
    db_paths = sorted(Path(db_scan_root).rglob("session_cache.db"))
    if not db_paths:
        print(f"[reset-db] {db_scan_root} 下未找到 session_cache.db")
        return 0

    affected = 0
    for db_path in db_paths:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        try:
            # 该 DB 里落在本次回填目录集合内的 root_dir
            db_roots = {
                os.path.normpath(r["root_dir"])
                for r in conn.execute(
                    "SELECT DISTINCT root_dir FROM index_progress"
                ).fetchall()
            }
            hit_roots = sorted(db_roots & norm_roots)
            if not hit_roots:
                continue

            placeholders = ",".join("?" for _ in hit_roots)
            db_affected = 0
            for tbl in ("sessions", "traces", "chain_index", "index_progress"):
                cnt = conn.execute(
                    f"SELECT COUNT(*) AS c FROM {tbl} WHERE root_dir IN ({placeholders})",
                    hit_roots,
                ).fetchone()["c"]
                db_affected += cnt
            print(f"[reset-db] {db_path}: {len(hit_roots)} 个 root_dir、{db_affected} 行将清除")
            affected += db_affected

            if dry_run:
                continue

            bak = str(db_path) + ".bak"
            if not os.path.exists(bak):
                shutil.copy2(str(db_path), bak)
            with conn:
                for tbl in ("sessions", "traces", "chain_index", "index_progress"):
                    conn.execute(
                        f"DELETE FROM {tbl} WHERE root_dir IN ({placeholders})",
                        hit_roots,
                    )
        finally:
            conn.close()

    return affected


def main() -> None:
    parser = argparse.ArgumentParser(description="回填 index.jsonl 的 q1_hash 并（可选）清库重建")
    parser.add_argument("--logs-dir", required=True, help="日志根目录（递归查找 index.jsonl）")
    parser.add_argument("--reset-db", action="store_true", help="清空对应 root_dir 的 DB 聚合表")
    parser.add_argument("--db-scan-root", default="logs", help="扫描 session_cache.db 的根目录（--reset-db 时用，默认 logs）")
    parser.add_argument("--workers", type=int, default=min(32, (os.cpu_count() or 4) * 4),
                        help="并行读 req.json 的线程数（I/O 密集，默认 CPU*4，上限 32）")
    parser.add_argument("--yes", action="store_true", help="实际执行写入（否则 dry-run）")
    args = parser.parse_args()

    dry_run = not args.yes
    workers = max(1, args.workers)
    repo_root = Path(__file__).resolve().parents[1]
    logs_dir = Path(args.logs_dir)

    index_files = sorted(logs_dir.rglob(INDEX_FILENAME))
    if not index_files:
        print(f"[warn] {logs_dir} 下未找到 {INDEX_FILENAME}")
        return

    print(f"[info] 找到 {len(index_files)} 个 index.jsonl（dry_run={dry_run}，workers={workers}）")

    grand_total = 0
    grand_updated = 0
    root_dirs: set[str] = set()
    for idx in index_files:
        total, updated = backfill_index_file(repo_root, idx, dry_run, workers)
        grand_total += total
        grand_updated += updated
        # root_dir 即 index.jsonl 所在目录（与运行时 build_index_path 一致）
        root_dirs.add(str(idx.parent))
        if updated:
            print(f"[backfill] {idx}: {updated}/{total} 行回填")

    print(f"[info] 合计 {grand_updated}/{grand_total} 行回填 q1_hash")

    if args.reset_db:
        affected = reset_db_scan(args.db_scan_root, sorted(root_dirs), dry_run)
        print(f"[info] DB 聚合表将清除 {affected} 行（清库后由运行时重建）")

    if dry_run:
        print("[info] dry-run 完成，加 --yes 实际执行")


if __name__ == "__main__":
    main()
