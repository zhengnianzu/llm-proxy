"""
analyze_sessions.py — 分析 session 格式对话日志

用法:
    python analyze_sessions.py --dir /path/to/test_session [--output report.json]

指标:
    - 对话轮次 (user 消息数，不含 tool_result)
    - messages 总条数
    - 工具调用次数
    - 工具调用成功率
    - 消耗时长 (首个文件时间戳 → 最后一个文件时间戳)
"""

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# 时间戳解析
# ---------------------------------------------------------------------------

FNAME_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})_\d{3}$")


def parse_folder_ts(name: str) -> Optional[datetime]:
    """解析文件夹/文件名时间戳，格式: 2026-03-11_17-55-49_475"""
    m = FNAME_TS_RE.match(name)
    if not m:
        return None
    date_part = m.group(1)
    time_part = f"{m.group(2)}:{m.group(3)}:{m.group(4)}"
    try:
        return datetime.strptime(f"{date_part} {time_part}", "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# 内容提取工具
# ---------------------------------------------------------------------------

def iter_content_blocks(content: Any):
    """统一遍历 content 字段（str / list）中的 block。"""
    if isinstance(content, str):
        yield {"type": "text", "text": content}
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                yield block


def count_tool_calls(messages: List[dict], response_content: Optional[List]) -> int:
    """统计 assistant 消息中的 tool_use 调用次数（含 response 里的 content）。"""
    count = 0
    for m in messages:
        if m.get("role") == "assistant":
            for blk in iter_content_blocks(m.get("content", [])):
                if blk.get("type") == "tool_use":
                    count += 1
    if response_content:
        for blk in (response_content if isinstance(response_content, list) else []):
            if isinstance(blk, dict) and blk.get("type") == "tool_use":
                count += 1
    return count


def count_tool_results(messages: List[dict]) -> Tuple[int, int]:
    """
    统计 user 消息中的 tool_result 块。
    返回 (success_count, error_count)
    """
    success, error = 0, 0
    for m in messages:
        if m.get("role") != "user":
            continue
        for blk in iter_content_blocks(m.get("content", [])):
            if blk.get("type") == "tool_result":
                if blk.get("is_error"):
                    error += 1
                else:
                    success += 1
    return success, error


def count_user_turns(messages: List[dict]) -> int:
    """
    统计真实的 user 发言轮次，排除纯 tool_result 消息。
    一条 user 消息若 content 全部是 tool_result，则不计入用户轮次。
    """
    count = 0
    for m in messages:
        if m.get("role") != "user":
            continue
        content = m.get("content", [])
        blocks = list(iter_content_blocks(content))
        has_non_tool = any(
            b.get("type") not in ("tool_result",) for b in blocks
        )
        if has_non_tool:
            count += 1
    return count


# ---------------------------------------------------------------------------
# 单个 session 分析
# ---------------------------------------------------------------------------

def analyze_session(folder: Path) -> Optional[Dict]:
    json_files = sorted(folder.glob("*.json"))
    if not json_files:
        return None

    # 起止时间：文件夹名=起始，最后一个文件名=结束
    start_ts = parse_folder_ts(folder.name)
    end_ts = parse_folder_ts(json_files[-1].stem)
    duration_s: Optional[float] = None
    if start_ts and end_ts and end_ts >= start_ts:
        duration_s = (end_ts - start_ts).total_seconds()

    # 统计 API call 次数 / 错误
    api_call_count = len(json_files)
    api_errors = 0
    models: List[str] = []

    # 先扫描所有文件，收集 api 错误 + 模型
    all_data: List[dict] = []
    for jf in json_files:
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            api_call_count -= 1
            continue
        all_data.append(data)
        response = data.get("response") or {}
        resp_status = response.get("status_code", 200) if isinstance(response, dict) else 200
        if isinstance(resp_status, int) and resp_status >= 400:
            api_errors += 1
        model = data.get("model", "")
        if model:
            models.append(model)

    if not all_data:
        return None

    # 关键：每个 API call 的 messages 包含完整历史，
    # 所以用「最完整的那次调用」来统计对话轮次/消息数/工具调用，避免重复计数。
    best_data = max(
        all_data,
        key=lambda d: len(d.get("messages") or []) + (
            1 if (d.get("response") or {}).get("content") else 0
        )
    )
    messages: List[dict] = best_data.get("messages") or []
    response = best_data.get("response") or {}
    resp_content = response.get("content") if isinstance(response, dict) else None

    # 合并最后一次 response 到消息列表
    full_messages = messages + ([{"role": "assistant", "content": resp_content}]
                                 if resp_content else [])

    user_turns = count_user_turns(full_messages)
    total_messages = len(full_messages)
    total_tool_calls = count_tool_calls(messages, resp_content)
    total_tool_success, total_tool_error = count_tool_results(full_messages)

    # 工具成功率
    total_results = total_tool_success + total_tool_error
    tool_success_rate: Optional[float] = None
    if total_results > 0:
        tool_success_rate = round(total_tool_success / total_results * 100, 1)

    model = models[-1] if models else ""

    return {
        "session": folder.name,
        "start_time": start_ts.strftime("%Y-%m-%d %H:%M:%S") if start_ts else None,
        "end_time": end_ts.strftime("%Y-%m-%d %H:%M:%S") if end_ts else None,
        "duration_s": duration_s,
        "api_call_count": api_call_count,
        "api_errors": api_errors,
        "user_turns": user_turns,
        "total_messages": total_messages,
        "tool_calls": total_tool_calls,
        "tool_success": total_tool_success,
        "tool_error": total_tool_error,
        "tool_success_rate": tool_success_rate,
        "model": model,
    }


# ---------------------------------------------------------------------------
# 汇总统计
# ---------------------------------------------------------------------------

def fmt_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "N/A"
    if seconds < 60:
        return f"{seconds:.0f}秒"
    if seconds < 3600:
        return f"{seconds/60:.1f}分钟"
    return f"{seconds/3600:.1f}小时"


def percentile(values: List[float], p: int) -> float:
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = int(len(sorted_v) * p / 100)
    idx = min(idx, len(sorted_v) - 1)
    return sorted_v[idx]


def build_report(sessions: List[Dict], top_n: int = 10) -> Tuple[str, Dict]:
    """返回 (markdown_text, summary_dict)"""
    total = len(sessions)
    lines: List[str] = []

    def ln(s: str = "") -> None:
        lines.append(s)

    from datetime import datetime as _dt
    generated_at = _dt.now().strftime("%Y-%m-%d %H:%M:%S")

    ln(f"# Session 分析报告")
    ln()
    ln(f"> 生成时间: {generated_at}　　总 session 数: **{total}**")
    ln()

    # ---- 1. 对话轮次 ----
    user_turns_vals = [s["user_turns"] for s in sessions]
    dist = Counter(user_turns_vals)
    ln("## 1. 对话轮次（User 发言数）")
    ln()
    ln(f"| 统计项 | 数值 |")
    ln(f"|--------|------|")
    ln(f"| 平均轮次 | {sum(user_turns_vals)/total:.2f} |")
    ln(f"| 最大轮次 | {max(user_turns_vals)} |")
    ln(f"| P50 | {percentile(user_turns_vals, 50):.0f} |")
    ln(f"| P90 | {percentile(user_turns_vals, 90):.0f} |")
    ln()
    ln("**轮次分布：**")
    ln()
    ln("| 轮次 | session 数 | 占比 | 分布 |")
    ln("|-----:|----------:|-----:|------|")
    max_cnt = max(dist.values())
    for k in sorted(dist):
        bar = "█" * max(1, round(dist[k] / max_cnt * 20))
        ln(f"| {k} | {dist[k]} | {dist[k]/total*100:.1f}% | {bar} |")
    ln()

    # ---- 2. 消息总数 ----
    msg_vals = [s["total_messages"] for s in sessions]
    buckets_m = [("1-2条", 0, 2), ("3-5条", 3, 5), ("6-10条", 6, 10),
                 ("11-20条", 11, 20), (">20条", 21, 10**9)]
    bucket_counts = Counter()
    for v in msg_vals:
        for label, lo, hi in buckets_m:
            if lo <= v <= hi:
                bucket_counts[label] += 1
                break

    ln("## 2. 消息总数（含 assistant 回复）")
    ln()
    ln(f"| 统计项 | 数值 |")
    ln(f"|--------|------|")
    ln(f"| 平均 | {sum(msg_vals)/total:.2f} |")
    ln(f"| 最大 | {max(msg_vals)} |")
    ln(f"| 最小 | {min(msg_vals)} |")
    ln()
    ln("**消息数分布：**")
    ln()
    ln("| 区间 | session 数 | 占比 |")
    ln("|------|----------:|-----:|")
    for label, _, __ in buckets_m:
        cnt = bucket_counts[label]
        ln(f"| {label} | {cnt} | {cnt/total*100:.1f}% |")
    ln()

    # ---- 3. API Call 次数 ----
    api_vals = [s["api_call_count"] for s in sessions]
    multi_api = [s for s in sessions if s["api_call_count"] > 1]
    api_err_sessions = sum(1 for s in sessions if s["api_errors"] > 0)

    ln("## 3. API Call 次数")
    ln()
    ln(f"| 统计项 | 数值 |")
    ln(f"|--------|------|")
    ln(f"| 平均 API calls / session | {sum(api_vals)/total:.2f} |")
    ln(f"| 最大 | {max(api_vals)} |")
    ln(f"| 单次 API call session | {sum(1 for v in api_vals if v==1)} |")
    ln(f"| 多次 API call session | {len(multi_api)} ({len(multi_api)/total*100:.1f}%) |")
    ln(f"| 含 API 错误(4xx/5xx) session | {api_err_sessions} ({api_err_sessions/total*100:.1f}%) |")
    ln()

    # ---- 4. 工具调用 ----
    with_tools = [s for s in sessions if s["tool_calls"] > 0]
    without_tools = [s for s in sessions if s["tool_calls"] == 0]
    total_calls = sum(s["tool_calls"] for s in sessions)
    total_succ = sum(s["tool_success"] for s in sessions)
    total_err = sum(s["tool_error"] for s in sessions)
    total_res = total_succ + total_err

    ln("## 4. 工具调用")
    ln()
    ln(f"| 统计项 | 数值 |")
    ln(f"|--------|------|")
    ln(f"| 有工具调用的 session | {len(with_tools)} ({len(with_tools)/total*100:.1f}%) |")
    ln(f"| 无工具调用的 session | {len(without_tools)} ({len(without_tools)/total*100:.1f}%) |")
    ln(f"| 工具调用总次数 | {total_calls} |")
    if with_tools:
        tc_vals = [s["tool_calls"] for s in with_tools]
        ln(f"| 平均调用次数（有工具 session） | {sum(tc_vals)/len(tc_vals):.1f} |")
        ln(f"| 最大调用次数 | {max(tc_vals)} |")
        ln(f"| P50 | {percentile(tc_vals, 50):.0f} |")
        ln(f"| P90 | {percentile(tc_vals, 90):.0f} |")
    if total_res > 0:
        ln(f"| 工具成功次数 | {total_succ} |")
        ln(f"| 工具失败次数 | {total_err} |")
        ln(f"| **工具成功率** | **{total_succ/total_res*100:.1f}%** |")
    ln()

    # ---- 5. 耗时 ----
    timed = [s for s in sessions if s["duration_s"] is not None and s["duration_s"] >= 0]
    multi_timed = [s for s in timed if s["api_call_count"] > 1]

    ln("## 5. 消耗时长")
    ln()
    ln("> 时长 = 首个 API call 时间戳 → 最后一个 API call 时间戳的差值（单次 session 时长为 0）")
    ln()
    ln(f"| 统计项 | 数值 |")
    ln(f"|--------|------|")
    ln(f"| 单次 API call（时长=0） | {sum(1 for s in sessions if s['api_call_count']==1)} |")
    ln(f"| 多轮 session（有时长） | {len(multi_timed)} |")

    if multi_timed:
        dur_vals = [s["duration_s"] for s in multi_timed]
        ln(f"| 平均耗时 | {fmt_duration(sum(dur_vals)/len(dur_vals))} |")
        ln(f"| 最长耗时 | {fmt_duration(max(dur_vals))} |")
        ln(f"| 最短耗时 | {fmt_duration(min(dur_vals))} |")
        ln(f"| P50 | {fmt_duration(percentile(dur_vals, 50))} |")
        ln(f"| P90 | {fmt_duration(percentile(dur_vals, 90))} |")
        ln()
        ln("**耗时分布（多轮 session）：**")
        ln()
        ln("| 区间 | session 数 | 占比 |")
        ln("|------|----------:|-----:|")
        buckets_t = [("<1分钟", 0, 60), ("1-5分钟", 60, 300),
                     ("5-15分钟", 300, 900), ("15-30分钟", 900, 1800), (">30分钟", 1800, 10**9)]
        for label, lo, hi in buckets_t:
            cnt = sum(1 for d in dur_vals if lo <= d < hi)
            if cnt:
                ln(f"| {label} | {cnt} | {cnt/len(dur_vals)*100:.1f}% |")
    ln()

    # ---- 6. 模型分布 ----
    model_dist = Counter(s["model"] for s in sessions)
    ln("## 6. 模型使用分布")
    ln()
    ln("| 模型 | session 数 | 占比 |")
    ln("|------|----------:|-----:|")
    for model, cnt in sorted(model_dist.items(), key=lambda x: -x[1]):
        ln(f"| `{model or '(未知)'}` | {cnt} | {cnt/total*100:.1f}% |")
    ln()

    # ---- 7. Top N 最长 session ----
    ln(f"## 7. 耗时最长 Top {top_n} Session")
    ln()
    ln("| Session | 时长 | 用户轮次 | 工具调用 | API Calls |")
    ln("|---------|-----:|---------:|---------:|----------:|")
    top_dur = sorted([s for s in sessions if s.get("duration_s")],
                     key=lambda x: -x["duration_s"])[:top_n]
    for s in top_dur:
        ln(f"| `{s['session']}` | {fmt_duration(s['duration_s'])} "
           f"| {s['user_turns']} | {s['tool_calls']} | {s['api_call_count']} |")
    ln()

    md_text = "\n".join(lines)

    # summary dict（用于 JSON 输出）
    summary = {
        "total_sessions": total,
        "generated_at": generated_at,
        "user_turns": {
            "avg": round(sum(user_turns_vals)/total, 2),
            "max": max(user_turns_vals),
            "p50": percentile(user_turns_vals, 50),
            "p90": percentile(user_turns_vals, 90),
            "distribution": dict(sorted(dist.items())),
        },
        "total_messages": {
            "avg": round(sum(msg_vals)/total, 2),
            "max": max(msg_vals),
            "min": min(msg_vals),
        },
        "api_calls": {
            "avg": round(sum(api_vals)/total, 2),
            "max": max(api_vals),
            "multi_call_sessions": len(multi_api),
            "error_sessions": api_err_sessions,
        },
        "tool_calls": {
            "sessions_with_tools": len(with_tools),
            "sessions_without_tools": len(without_tools),
            "total_calls": total_calls,
            "total_success": total_succ,
            "total_error": total_err,
            "overall_success_rate": round(total_succ/total_res*100, 1) if total_res > 0 else None,
        },
        "duration": {
            "multi_turn_sessions": len(multi_timed),
            "avg_s": round(sum(s["duration_s"] for s in multi_timed)/len(multi_timed), 1) if multi_timed else None,
            "max_s": max(s["duration_s"] for s in multi_timed) if multi_timed else None,
            "p50_s": percentile([s["duration_s"] for s in multi_timed], 50) if multi_timed else None,
            "p90_s": percentile([s["duration_s"] for s in multi_timed], 90) if multi_timed else None,
        },
        "models": dict(sorted(model_dist.items(), key=lambda x: -x[1])),
        "sessions": sessions,
    }
    return md_text, summary


def print_report(sessions: List[Dict], output_json: Optional[Path] = None,
                 output_md: Optional[Path] = None, top_n: int = 10) -> None:
    md_text, summary = build_report(sessions, top_n=top_n)

    # 始终在终端打印 markdown
    print(md_text)

    if output_md:
        output_md.write_text(md_text, encoding="utf-8")
        print(f"[done] Markdown 报告已写入: {output_md}")

    if output_json:
        output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[done] JSON 报告已写入: {output_json}")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="分析 session 格式对话日志")
    parser.add_argument("--dir", "-d", required=True, help="session 根目录（含各 session 文件夹）")
    parser.add_argument("--output", "-o", default=None, help="输出 JSON 报告路径（可选）")
    parser.add_argument("--markdown", "-m", default=None, help="输出 Markdown 报告路径（可选）")
    parser.add_argument("--top", "-t", type=int, default=10, help="Top N 最长 session 数量，默认 10")
    args = parser.parse_args()

    session_dir = Path(args.dir).resolve()
    if not session_dir.is_dir():
        sys.exit(f"[error] 目录不存在: {session_dir}")

    output_json = Path(args.output).resolve() if args.output else None
    output_md = Path(args.markdown).resolve() if args.markdown else None

    print(f"[info] 扫描目录: {session_dir}", file=sys.stderr)
    folders = sorted(f for f in session_dir.iterdir() if f.is_dir())
    print(f"[info] 发现 {len(folders)} 个 session 文件夹", file=sys.stderr)

    sessions = []
    for folder in folders:
        result = analyze_session(folder)
        if result:
            sessions.append(result)

    print(f"[info] 成功解析 {len(sessions)} 个 session", file=sys.stderr)
    print_report(sessions, output_json=output_json, output_md=output_md, top_n=args.top)


if __name__ == "__main__":
    main()
