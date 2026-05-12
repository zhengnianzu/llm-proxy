#!/usr/bin/env python3
"""按天统计 Claude 模型调用情况（多进程版本）。

通过读取 index.jsonl 快速过滤 claude 模型，多进程并行解析 res 文件。
过滤 reasoning > 5000 字符的异常数据。

用法:
    python3 stats_daily.py                  # 默认扫描 keyxnGZ
    python3 stats_daily.py dir1 dir2 dir3   # 指定多个目录
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from pathlib import Path

REASONING_ANOMALY_THRESHOLD = 5000


def parse_res_file(args: tuple[str, str]) -> dict | None:
    """解析单个 res 文件，返回统计数据或 None。"""
    res_path_str, date_str = args
    try:
        with open(res_path_str) as f:
            payload = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError, OSError):
        return None

    reasoning_len = 0
    content_pieces: list[str] = []
    tool_count = 0

    for chunk in payload.get("chunks", []):
        chunk_type = chunk.get("type")

        if chunk_type == "content_block_start":
            block = chunk.get("content_block", {})
            btype = block.get("type")
            if btype == "thinking":
                # reasoning start 里可能有初始文本
                t = block.get("thinking", "")
                if t:
                    reasoning_len += len(t)
            elif btype == "text":
                t = block.get("text", "")
                if t:
                    content_pieces.append(t)
            elif btype == "tool_use":
                tool_count += 1
                name = block.get("name", "")
                if name:
                    content_pieces.append(name)

        elif chunk_type == "content_block_delta":
            delta = chunk.get("delta", {})
            dtype = delta.get("type")
            if dtype == "thinking_delta":
                t = delta.get("thinking", "")
                if t:
                    reasoning_len += len(t)
            elif dtype == "text_delta":
                t = delta.get("text", "")
                if t:
                    content_pieces.append(t)
            elif dtype == "input_json_delta":
                t = delta.get("partial_json", "")
                if t:
                    content_pieces.append(t)

    if reasoning_len > REASONING_ANOMALY_THRESHOLD:
        return {"anomaly": True}

    content_len = sum(len(p) for p in content_pieces)

    return {
        "date": date_str,
        "reasoning_len": reasoning_len,
        "content_len": content_len,
        "tool_count": tool_count,
    }


def load_tasks_from_index(directory: Path) -> list[tuple[str, str]]:
    """从 index.jsonl 读取 claude 模型的 res 文件路径列表。"""
    index_path = directory / "index.jsonl"
    tasks = []

    if not index_path.exists():
        return tasks

    with open(index_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            model = entry.get("model", "")
            if "claude" not in model.lower():
                continue

            ts = entry.get("ts", "")
            if not ts:
                continue

            date_str = ts[:10]
            res_path = directory / f"{ts}-res.json"
            tasks.append((str(res_path), date_str))

    return tasks


def main():
    import argparse

    parser = argparse.ArgumentParser(description="按天统计 Claude 模型调用情况")
    parser.add_argument("directories", nargs="*", default=["keyxnGZ"], help="数据目录（默认 keyxnGZ）")
    parser.add_argument("-w", "--workers", type=int, default=cpu_count(), help=f"并行进程数（默认 {cpu_count()}）")
    args = parser.parse_args()

    directories = [Path(d) for d in args.directories]

    for d in directories:
        if not d.exists():
            print(f"目录 {d} 不存在，跳过")

    directories = [d for d in directories if d.exists()]
    if not directories:
        print("无有效目录")
        return

    # 从 index.jsonl 收集任务
    all_tasks: list[tuple[str, str]] = []
    skipped_non_claude = 0

    for directory in directories:
        index_path = directory / "index.jsonl"
        if index_path.exists():
            tasks = load_tasks_from_index(directory)
            # 统计非 claude 数量
            with open(index_path) as f:
                total_lines = sum(1 for line in f if line.strip())
            skipped_non_claude += total_lines - len(tasks)
            all_tasks.extend(tasks)
        else:
            # fallback: 扫描目录
            res_files = sorted(directory.glob("*-res.json"))
            for res_path in res_files:
                prefix = res_path.name.replace("-res.json", "")
                headers_path = directory / f"{prefix}-headers.json"
                try:
                    with open(headers_path) as f:
                        headers = json.load(f)
                    if "claude" not in headers.get("Model-Id", "").lower():
                        skipped_non_claude += 1
                        continue
                except (json.JSONDecodeError, FileNotFoundError):
                    skipped_non_claude += 1
                    continue
                date_str = prefix[:10]
                all_tasks.append((str(res_path), date_str))

    num_workers = args.workers
    print(f"共 {len(all_tasks)} 个 Claude 请求待解析，使用 {num_workers} 进程...")

    # 多进程解析
    with Pool(processes=num_workers) as pool:
        results = pool.map(parse_res_file, all_tasks, chunksize=64)

    # 汇总
    records: list[dict] = []
    skipped_anomaly = 0

    for r in results:
        if r is None:
            continue
        if r.get("anomaly"):
            skipped_anomaly += 1
            continue
        records.append(r)

    if not records:
        print("无有效数据")
        return

    total = len(records)
    dates = sorted(set(r["date"] for r in records))
    zero_count = sum(1 for r in records if r["reasoning_len"] == 0)
    non_zero_count = total - zero_count

    all_reasoning = [r["reasoning_len"] for r in records]
    all_content = [r["content_len"] for r in records]
    non_zero_reasoning = [r["reasoning_len"] for r in records if r["reasoning_len"] > 0]
    non_zero_content = [r["content_len"] for r in records if r["reasoning_len"] > 0]

    gt300_all = sum(1 for l in all_reasoning if l > 300)
    gt600_all = sum(1 for l in all_reasoning if l > 600)
    gt300_nz = sum(1 for l in non_zero_reasoning if l > 300)
    gt600_nz = sum(1 for l in non_zero_reasoning if l > 600)

    total_tool_calls = sum(r["tool_count"] for r in records)
    has_tool_count = sum(1 for r in records if r["tool_count"] > 0)
    tool_counts_with_tool = [r["tool_count"] for r in records if r["tool_count"] > 0]
    avg_tool_per_call = sum(tool_counts_with_tool) / len(tool_counts_with_tool) if tool_counts_with_tool else 0

    # 生成 markdown
    lines: list[str] = []
    lines.append("# Claude 模型调用统计报告")
    lines.append("")
    dir_names = ", ".join(str(d) for d in directories)
    lines.append(f"- **数据目录**: {dir_names}")
    lines.append(f"- **统计时间范围**: {dates[0]} ~ {dates[-1]}（共 {len(dates)} 天）")
    lines.append(f"- **过滤条件**: reasoning > {REASONING_ANOMALY_THRESHOLD} 字符视为异常，已过滤 {skipped_anomaly} 条")
    lines.append(f"- **跳过非 Claude 模型**: {skipped_non_claude} 条")
    lines.append("")

    # 总体统计
    lines.append("## 总体统计")
    lines.append("")
    lines.append("| 指标 | 数值 |")
    lines.append("|------|------|")
    lines.append(f"| 总计调用轮次 | {total} |")
    lines.append(f"| reasoning 为零轮次 | {zero_count} ({zero_count/total:.1%}) |")
    lines.append(f"| reasoning 非零轮次 | {non_zero_count} ({non_zero_count/total:.1%}) |")
    lines.append(f"| 平均 reasoning 长度 | {sum(all_reasoning)/total:.0f} 字符 |")
    lines.append(f"| 平均 content 长度 | {sum(all_content)/total:.0f} 字符 |")
    lines.append(f"| reasoning > 300 | {gt300_all} ({gt300_all/total:.1%}) |")
    lines.append(f"| reasoning > 600 | {gt600_all} ({gt600_all/total:.1%}) |")
    lines.append("")

    # 非零 reasoning 统计
    lines.append("## 非零 Reasoning 统计")
    lines.append("")
    lines.append("| 指标 | 数值 |")
    lines.append("|------|------|")
    if non_zero_count > 0:
        lines.append(f"| 非零轮次数 | {non_zero_count} |")
        lines.append(f"| 平均 reasoning 长度 | {sum(non_zero_reasoning)/non_zero_count:.0f} 字符 |")
        lines.append(f"| 平均 content 长度 | {sum(non_zero_content)/non_zero_count:.0f} 字符 |")
        lines.append(f"| reasoning > 300 | {gt300_nz} ({gt300_nz/non_zero_count:.1%}) |")
        lines.append(f"| reasoning > 600 | {gt600_nz} ({gt600_nz/non_zero_count:.1%}) |")
    else:
        lines.append("| （无非零数据） | - |")
    lines.append("")

    # 工具调用统计
    lines.append("## 工具调用统计")
    lines.append("")
    lines.append("| 指标 | 数值 |")
    lines.append("|------|------|")
    lines.append(f"| 总工具调用次数 | {total_tool_calls} |")
    lines.append(f"| 包含工具调用的轮次 | {has_tool_count} ({has_tool_count/total:.1%}) |")
    lines.append(f"| 平均每轮工具调用数（仅含工具轮次） | {avg_tool_per_call:.2f} |")
    lines.append("")

    # 每日详细统计
    lines.append("## 每日详细统计")
    lines.append("")
    lines.append("| 日期 | 总次数 | 零值 | 平均reasoning | 非零avg_r | r>300占比 | r>600占比 | 非零r>300 | 非零r>600 | 工具调用次数 | 工具占比 |")
    lines.append("|------|--------|------|---------------|-----------|-----------|-----------|-----------|-----------|--------------|----------|")

    daily_grouped = defaultdict(list)
    for r in records:
        daily_grouped[r["date"]].append(r)

    for date in sorted(daily_grouped.keys()):
        day_records = daily_grouped[date]
        d_total = len(day_records)
        d_zero = sum(1 for r in day_records if r["reasoning_len"] == 0)

        d_all_r = [r["reasoning_len"] for r in day_records]
        d_nz_r = [r["reasoning_len"] for r in day_records if r["reasoning_len"] > 0]

        avg_r = sum(d_all_r) / d_total
        nz_avg_r = sum(d_nz_r) / len(d_nz_r) if d_nz_r else 0

        d_gt300 = sum(1 for l in d_all_r if l > 300)
        d_gt600 = sum(1 for l in d_all_r if l > 600)
        d_nz_gt300 = sum(1 for l in d_nz_r if l > 300)
        d_nz_gt600 = sum(1 for l in d_nz_r if l > 600)

        d_total_tools = sum(r["tool_count"] for r in day_records)
        d_has_tool = sum(1 for r in day_records if r["tool_count"] > 0)
        tool_pct = f"{d_has_tool/d_total:.1%}"

        nz_count = len(d_nz_r)
        nz_gt300_pct = f"{d_nz_gt300/nz_count:.1%}" if nz_count else "-"
        nz_gt600_pct = f"{d_nz_gt600/nz_count:.1%}" if nz_count else "-"

        lines.append(
            f"| {date} | {d_total} | {d_zero} | {avg_r:.0f} | {nz_avg_r:.0f} "
            f"| {d_gt300/d_total:.1%} | {d_gt600/d_total:.1%} "
            f"| {nz_gt300_pct} | {nz_gt600_pct} "
            f"| {d_total_tools} | {tool_pct} |"
        )

    # 合计行
    avg_r_all = sum(all_reasoning) / total
    nz_avg_r_all = sum(non_zero_reasoning) / non_zero_count if non_zero_count else 0

    lines.append(
        f"| **合计** | **{total}** | **{zero_count}** | **{avg_r_all:.0f}** | **{nz_avg_r_all:.0f}** "
        f"| **{gt300_all/total:.1%}** | **{gt600_all/total:.1%}** "
        f"| **{gt300_nz/non_zero_count:.1%}** | **{gt600_nz/non_zero_count:.1%}** "
        f"| **{total_tool_calls}** | **{has_tool_count/total:.1%}** |"
    )
    lines.append("")

    # 写入文件
    output_path = Path("stats_report.md")
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"报告已生成: {output_path}")

    # 同时打印到终端
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
