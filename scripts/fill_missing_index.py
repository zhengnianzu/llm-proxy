"""一次性脚本：为缺失 index.jsonl 的目录补建 index.jsonl（多进程并行版）。

用法: python3 scripts/fill_missing_index.py

为什么多进程：源在网络盘（SFS），单进程读 req+res 大部分时间等 IO（CPU ~13%）。
按目录内文件分片交给 N 个 worker 并行，每个 worker 写独立临时分片，最后合并。

健壮性：
  - 断点续跑：已存在的 index.jsonl 里的 req_file 集合会被跳过（重跑不重复写）
  - 单 worker 异常不拖垮整体（分片级 try/except + 统计）
  - 每个目录独立处理，一个失败不影响其他

只补 TARGETS 里的目录；已存在 index.jsonl 的目录若为「完整」（行数>=req 数）则跳过，
否则增量续跑（跳过已写行）。
"""
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = "/mnt/sfs_turbo/s3-asset-b-hd-cce-aifm-nlp-exp/zhengnianzu/test/raw/env-claude-99oR/"
# 需要补的目录（嵌套的 26071621 实际叶子在内层 26071621/26071621/）
TARGETS = ["26071018", "26071121", "26071621/26071621"]

N_WORKERS = 64  # 网络盘 IO 并行，可按实际调

from utils.message_common import (
    extract_res_usage,
    extract_messages,
    build_chain_key,
    get_first_user_text,
    count_real_user_turns,
)


def leaf_index_path(hour_rel: str) -> str:
    return os.path.join(ROOT, hour_rel, "index.jsonl")


def req_file_rel(hour_rel: str, fname: str) -> str:
    return os.path.join("logs_all", "env-claude-99oR", hour_rel, fname)


def build_entry(req_path: str, res_path: str, req_file_rel_s: str) -> dict:
    """从 req/res 重建一条 index 条目（13 字段，取不到留空/0）。"""
    ts = ""
    model = ""
    tok_in = 0
    tok_out = 0
    success = True
    provider = "openai"
    api_key = ""
    chain_key = ""
    q1_preview = ""
    start_turn = 0

    req_data = None
    try:
        with open(req_path, "r", encoding="utf-8") as f:
            req_data = json.load(f)
    except Exception:
        req_data = None
    if req_data:
        model = req_data.get("model", "") or ""
        msgs = extract_messages(req_data)
        if msgs:
            try:
                chain_key = build_chain_key(msgs) or ""
            except Exception:
                chain_key = ""
            try:
                q1_preview = get_first_user_text(msgs) or ""
            except Exception:
                q1_preview = ""
            try:
                start_turn = count_real_user_turns(msgs)
            except Exception:
                start_turn = 0
        api_key = (req_data.get("api_key") or req_data.get("key") or "") if isinstance(req_data, dict) else ""
    base = os.path.basename(req_path)
    ts = base[:-len("-req.json")] if base.endswith("-req.json") else base

    usage = None
    if res_path and os.path.isfile(res_path):
        try:
            with open(res_path, "r", encoding="utf-8") as f:
                res_data = json.load(f)
            usage = extract_res_usage(res_data)
        except Exception:
            usage = None
    if isinstance(usage, dict):
        tok_in = int(usage.get("token_in") or usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        tok_out = int(usage.get("token_out") or usage.get("completion_tokens") or usage.get("output_tokens") or 0)

    return {
        "ts": ts,
        "req_file": req_file_rel_s,
        "provider": provider,
        "model": model,
        "tok_in": tok_in,
        "tok_out": tok_out,
        "success": success,
        "api_key": api_key,
        "chain_key": chain_key,
        "q1_preview": q1_preview,
        "total_attempts": 1,
        "retried": False,
        "start_turn": start_turn,
    }


def _process_chunk(args):
    """worker 函数：处理一个文件分片，返回 (n_ok, n_nores, n_err, [json行])。"""
    hour_rel, chunk, done_set = args
    leaf_dir = os.path.join(ROOT, hour_rel)
    lines = []
    n_ok = n_nores = n_err = 0
    for fname in chunk:
        if fname in done_set:
            continue
        base = fname[:-len("-req.json")] if fname.endswith("-req.json") else fname
        req_path = os.path.join(leaf_dir, fname)
        res_path = os.path.join(leaf_dir, base + "-res.json")
        if not os.path.isfile(res_path):
            n_nores += 1
        try:
            entry = build_entry(req_path, res_path, req_file_rel(hour_rel, fname))
            lines.append(json.dumps(entry, ensure_ascii=False))
            n_ok += 1
        except Exception:
            n_err += 1
    return (n_ok, n_nores, n_err, lines)


def _existing_reqs(hour_rel: str) -> set:
    """读已存在的 index.jsonl，返回已覆盖的 req_file 集合（用于断点续跑）。"""
    idx = leaf_index_path(hour_rel)
    done = set()
    if not os.path.isfile(idx):
        return done
    try:
        with open(idx, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                    done.add(o.get("req_file", ""))
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    return done


def process_dir(hour_rel: str):
    import multiprocessing as mp

    leaf_dir = os.path.join(ROOT, hour_rel)
    if not os.path.isdir(leaf_dir):
        print(f"[skip] 目录不存在: {leaf_dir}", flush=True)
        return
    idx = leaf_index_path(hour_rel)

    reqs = sorted(
        f.name for f in os.scandir(leaf_dir)
        if f.name.endswith("-req.json") and f.is_file()
    )
    if not reqs:
        print(f"[skip] {hour_rel} 无 req.json", flush=True)
        return
    done = _existing_reqs(hour_rel)
    todo = [f for f in reqs if req_file_rel(hour_rel, f) not in done]
    print(f"[{hour_rel}] 共 {len(reqs)} 个 req，已写 {len(done)}，待写 {len(todo)}，"
          f"用 {N_WORKERS} worker 并行...", flush=True)
    if not todo:
        print(f"[{hour_rel}] 已完整，跳过", flush=True)
        return

    # 分片
    chunk_size = max(1, len(todo) // N_WORKERS)
    chunks = [todo[i:i + chunk_size] for i in range(0, len(todo), chunk_size)]

    total_ok = total_nores = total_err = 0
    all_lines = []
    with mp.Pool(N_WORKERS) as pool:
        for n_ok, n_nores, n_err, lines in pool.imap_unordered(
            _process_chunk, [(hour_rel, c, done) for c in chunks]
        ):
            total_ok += n_ok
            total_nores += n_nores
            total_err += n_err
            all_lines.extend(lines)
            print(f"[{hour_rel}] 完成分片，累计 {total_ok}/{len(todo)} ...", flush=True)

    # 合并：直接追加到现有 index.jsonl 末尾（SFS 网络盘不支持 rename/tmp 替换，
    # 只能用追加写；排序保证顺序稳定，已存在的行由断点续跑跳过）。
    all_lines.sort()
    with open(idx, "a", encoding="utf-8") as f:
        for ln in all_lines:
            f.write(ln + "\n")
    print(f"[{hour_rel}] 完成: 写入 {total_ok}，缺 res {total_nores}，失败 {total_err}，"
          f"共 {len(all_lines) + len(done)} 行", flush=True)


def main():
    for hour_rel in TARGETS:
        try:
            process_dir(hour_rel)
        except Exception as e:  # noqa: BLE001
            print(f"[{hour_rel}] 目录处理异常: {e}", flush=True)


if __name__ == "__main__":
    main()
