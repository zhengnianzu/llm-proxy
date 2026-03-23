"""
analyze_sessions.py — 分析 session 格式对话日志

用法:
    python analyze_sessions.py <session目录>

输出（自动写入 <session目录>/stat/）:
    session_report.xlsx   — 每条 session 详情 + 分布统计（两个 sheet）
    session_report.md     — 汇总概览报告
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
    m = FNAME_TS_RE.match(name)
    if not m:
        return None
    try:
        return datetime.strptime(
            f"{m.group(1)} {m.group(2)}:{m.group(3)}:{m.group(4)}",
            "%Y-%m-%d %H:%M:%S",
        )
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# 工具错误关键字
# ---------------------------------------------------------------------------

_ERROR_PATTERNS: List[re.Pattern] = [
    re.compile(r"Traceback \(most recent call last\)", re.I),
    re.compile(
        r"\b(SyntaxError|NameError|TypeError|ValueError|AttributeError|"
        r"ImportError|ModuleNotFoundError|RuntimeError|KeyError|IndexError|"
        r"FileNotFoundError|PermissionError|OSError|IOError|ZeroDivisionError|"
        r"RecursionError|MemoryError)\s*:", re.I
    ),
    re.compile(r"permission denied", re.I),
    re.compile(r"operation not permitted", re.I),
    re.compile(r"access denied", re.I),
    re.compile(r"cannot execute", re.I),
    re.compile(r"no such file or directory", re.I),
    re.compile(r"file not found", re.I),
    re.compile(r"\btimed?\s*out\b", re.I),
    re.compile(r"\bkilled\b", re.I),
    re.compile(r"segmentation fault", re.I),
    re.compile(r"command not found", re.I),
    re.compile(r"\berror\b", re.I),
    re.compile(r"\bfailed\b", re.I),
    re.compile(r"\bexception\b", re.I),
    re.compile(r"\bfailure\b", re.I),
]


def _has_error_keywords(text: str) -> bool:
    return any(p.search(text) for p in _ERROR_PATTERNS)


def _collect_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for blk in content:
            if not isinstance(blk, dict):
                continue
            if blk.get("text"):
                parts.append(str(blk["text"]))
            inner = blk.get("content", "")
            if isinstance(inner, str):
                parts.append(inner)
            elif isinstance(inner, list):
                for sub in inner:
                    if isinstance(sub, dict) and sub.get("text"):
                        parts.append(str(sub["text"]))
        return "\n".join(parts)
    return ""


# ---------------------------------------------------------------------------
# 内容遍历
# ---------------------------------------------------------------------------

def iter_blocks(content: Any):
    if isinstance(content, str):
        yield {"type": "text", "text": content}
    elif isinstance(content, list):
        for b in content:
            if isinstance(b, dict):
                yield b


def count_user_turns(messages: List[dict]) -> int:
    count = 0
    for m in messages:
        if m.get("role") != "user":
            continue
        blocks = list(iter_blocks(m.get("content", [])))
        if any(b.get("type") != "tool_result" for b in blocks):
            count += 1
    return count


def count_tool_use(messages: List[dict], resp_content: Optional[List]) -> int:
    count = 0
    for m in messages:
        if m.get("role") == "assistant":
            for b in iter_blocks(m.get("content", [])):
                if b.get("type") == "tool_use":
                    count += 1
    if isinstance(resp_content, list):
        for b in resp_content:
            if isinstance(b, dict) and b.get("type") == "tool_use":
                count += 1
    return count


def analyze_tool_results(messages: List[dict]) -> Dict[str, int]:
    total = success = fail_flag = fail_kw = 0
    for m in messages:
        if m.get("role") != "user":
            continue
        for blk in iter_blocks(m.get("content", [])):
            if blk.get("type") != "tool_result":
                continue
            total += 1
            if blk.get("is_error"):
                fail_flag += 1
            else:
                text = _collect_text(blk.get("content", ""))
                if _has_error_keywords(text):
                    fail_kw += 1
                else:
                    success += 1
    fail_total = fail_flag + fail_kw
    return {
        "total": total,
        "success": total - fail_total,
        "fail_flag": fail_flag,
        "fail_kw": fail_kw,
        "fail_total": fail_total,
    }


# ---------------------------------------------------------------------------
# Q1 提取
# ---------------------------------------------------------------------------

def get_q1(messages: List[dict]) -> str:
    for msg in messages:
        if msg.get("role") != "user":
            continue
        c = msg.get("content", "")
        if isinstance(c, str):
            text = c
        elif isinstance(c, list):
            parts = [b.get("text") or b.get("id") or "" for b in c if isinstance(b, dict)]
            text = "\n".join(p for p in parts if p)
        else:
            text = str(c)
        while True:
            prev = text
            text = re.sub(r"^\s*\[[^\]]*\]\s*", "", text)
            text = re.sub(
                r"^Sender\s*(?:\([^)]*\))?:\s*```json\s*\{[\s\S]*?\}\s*```\s*",
                "", text, flags=re.IGNORECASE,
            )
            text = re.sub(r"^Sender\s*(?:\([^)]*\))?:[^\n]*\n?", "", text, flags=re.IGNORECASE)
            if text == prev:
                break
        return text.strip()
    return ""


# ---------------------------------------------------------------------------
# 单 session 分析
# ---------------------------------------------------------------------------

def analyze_session(folder: Path) -> Optional[Dict]:
    json_files = sorted(folder.glob("*.json"))
    if not json_files:
        return None

    start_ts = parse_folder_ts(folder.name)
    end_ts   = parse_folder_ts(json_files[-1].stem)
    duration_s: Optional[float] = None
    if start_ts and end_ts and end_ts >= start_ts:
        duration_s = (end_ts - start_ts).total_seconds()

    api_call_count = len(json_files)
    api_errors = 0
    models: List[str] = []
    all_data: List[dict] = []

    for jf in json_files:
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            api_call_count -= 1
            continue
        all_data.append(data)
        resp = data.get("response") or {}
        if isinstance(resp, dict) and isinstance(resp.get("status_code"), int):
            if resp["status_code"] >= 400:
                api_errors += 1
        if data.get("model"):
            models.append(data["model"])

    if not all_data:
        return None

    best_data = max(
        all_data,
        key=lambda d: len(d.get("messages") or []) + (
            1 if (d.get("response") or {}).get("content") else 0
        ),
    )
    messages: List[dict] = best_data.get("messages") or []
    resp_content = (best_data.get("response") or {}).get("content")
    full_messages = messages + (
        [{"role": "assistant", "content": resp_content}] if resp_content else []
    )

    q1             = get_q1(messages)
    user_turns     = count_user_turns(full_messages)
    total_messages = len(full_messages)
    tool_use_cnt   = count_tool_use(messages, resp_content)
    tr             = analyze_tool_results(full_messages)

    tool_result_cnt   = tr["total"]
    tool_success_rate = (
        round(tr["success"] / tool_result_cnt * 100, 1) if tool_result_cnt > 0 else None
    )

    return {
        "session":           folder.name,
        "start_time":        start_ts.strftime("%Y-%m-%d %H:%M:%S") if start_ts else None,
        "end_time":          end_ts.strftime("%Y-%m-%d %H:%M:%S")   if end_ts   else None,
        "duration_s":        duration_s,
        "api_call_count":    api_call_count,
        "api_errors":        api_errors,
        "user_turns":        user_turns,
        "total_messages":    total_messages,
        "tool_use_count":    tool_use_cnt,
        "tool_result_count": tool_result_cnt,
        "tool_success":      tr["success"],
        "tool_fail_flag":    tr["fail_flag"],
        "tool_fail_keyword": tr["fail_kw"],
        "tool_fail_total":   tr["fail_total"],
        "tool_success_rate": tool_success_rate,
        "model":             models[-1] if models else "",
        "q1":                q1,
        "completed":         None,
    }


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------

def fmt_duration(s: Optional[float]) -> str:
    if s is None:
        return "N/A"
    if s < 60:
        return f"{s:.0f}s"
    if s < 3600:
        return f"{s/60:.1f}min"
    return f"{s/3600:.1f}h"


def fmt_rate(r: Optional[float]) -> str:
    return f"{r:.1f}%" if r is not None else "-"


def pct(values: List[float], p: int) -> float:
    if not values:
        return 0.0
    sv = sorted(values)
    return sv[min(int(len(sv) * p / 100), len(sv) - 1)]


# ---------------------------------------------------------------------------
# Excel 导出
# ---------------------------------------------------------------------------

# 列定义：(字段key, 中文表头)
_DETAIL_COLS: List[Tuple[str, str]] = [
    ("q1",                "Q1首问"),
    ("session",           "Session"),
    ("start_time",        "开始时间"),
    ("end_time",          "结束时间"),
    ("duration_s",        "持续时长(s)"),
    ("api_call_count",    "请求次数"),
    ("api_errors",        "API错误次数"),
    ("user_turns",        "用户轮次"),
    ("total_messages",    "消息总数"),
    ("tool_use_count",    "tool_use次数"),
    ("tool_result_count", "tool_result次数"),
    ("tool_success",      "工具成功次数"),
    ("tool_fail_flag",    "失败(is_error标记)"),
    ("tool_fail_keyword", "失败(错误关键字)"),
    ("tool_fail_total",   "失败合计"),
    ("tool_success_rate", "工具成功率(%)"),
    ("model",             "模型"),
    ("completed",         "任务完成"),
]


def _bucket_df(pd, vals: List[float], buckets: List[Tuple], unit: str = ""):
    total = len(vals)
    rows = []
    for label, lo, hi in buckets:
        cnt = sum(1 for v in vals if v is not None and lo <= v < hi)
        rows.append({
            "区间": f"{label}{unit}",
            "数量": cnt,
            "占比(%)": round(cnt / total * 100, 1) if total else 0,
        })
    return pd.DataFrame(rows)


def write_excel(sessions: List[Dict], path: Path) -> None:
    import pandas as pd
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    # ── 构建详情 DataFrame ───────────────────────────────────────────────────
    rows = [{label: s.get(key) for key, label in _DETAIL_COLS} for s in sessions]
    df_detail = pd.DataFrame(rows)

    # ── 构建分布 DataFrames ──────────────────────────────────────────────────
    turns_vals = [s["user_turns"]       for s in sessions]
    msg_vals   = [s["total_messages"]   for s in sessions]
    api_vals   = [s["api_call_count"]   for s in sessions]
    tu_vals    = [s["tool_use_count"]   for s in sessions if s["tool_use_count"] > 0]
    rate_vals  = [s["tool_success_rate"] for s in sessions if s["tool_success_rate"] is not None]
    dur_vals   = [s["duration_s"]       for s in sessions
                  if s["api_call_count"] > 1 and s["duration_s"] is not None]

    dist_sections = [
        ("对话轮次分布",
         _bucket_df(pd, turns_vals,
                    [(1,1,2),(2,2,4),(4,4,8),(8,8,16),(16,16,10**9)], "轮")),
        ("消息总数分布",
         _bucket_df(pd, msg_vals,
                    [("1-2",0,3),("3-5",3,6),("6-10",6,11),
                     ("11-20",11,21),("21-50",21,51),(">50",51,10**9)], "条")),
        ("API Call次数分布",
         _bucket_df(pd, api_vals,
                    [(1,1,2),(2,2,4),("4-10",4,11),("11-30",11,31),(">30",31,10**9)], "次")),
        ("tool_use次数分布（有工具session）",
         _bucket_df(pd, tu_vals,
                    [("1-5",1,6),("6-15",6,16),("16-30",16,31),("31-50",31,51),(">50",51,10**9)], "次")),
        ("工具成功率分布",
         _bucket_df(pd, rate_vals,
                    [("0-50%",0,50),("50-80%",50,80),("80-95%",80,95),
                     ("95-99%",95,99),("100%",100,101)])),
        ("耗时分布（多轮session）",
         _bucket_df(pd, dur_vals,
                    [("<1min",0,60),("1-5min",60,300),("5-15min",300,900),
                     ("15-30min",900,1800),(">30min",1800,10**9)])),
    ]

    # ── 写入 Excel ───────────────────────────────────────────────────────────
    with pd.ExcelWriter(path, engine="openpyxl") as ew:
        # Sheet 1: 详情
        df_detail.to_excel(ew, sheet_name="Session详情", index=False)
        ws1 = ew.sheets["Session详情"]

        # 冻结首行首列
        ws1.freeze_panes = "B2"

        # 表头样式
        hdr_fill = PatternFill(fill_type="solid", fgColor="1F4E79")
        hdr_font = Font(bold=True, color="FFFFFF", size=10)
        hdr_align = Alignment(horizontal="center", vertical="center", wrap_text=True)
        for cell in ws1[1]:
            cell.fill  = hdr_fill
            cell.font  = hdr_font
            cell.alignment = hdr_align
        ws1.row_dimensions[1].height = 30

        # 隔行底色
        alt_fill = PatternFill(fill_type="solid", fgColor="EBF3FB")
        for row_idx in range(2, ws1.max_row + 1, 2):
            for cell in ws1[row_idx]:
                # 只在没有其他填充时加底色
                rgb = cell.fill.fgColor.rgb
                if rgb in ("00000000", "FFFFFFFF", "00FFFFFF"):
                    cell.fill = alt_fill

        # 成功率列转为小数（Excel百分比格式）
        rate_col_idx = next(
            i for i, (_, lbl) in enumerate(_DETAIL_COLS, 1) if lbl == "工具成功率(%)"
        )
        rate_letter = get_column_letter(rate_col_idx)
        for row in range(2, ws1.max_row + 1):
            cell = ws1[f"{rate_letter}{row}"]
            if cell.value is not None:
                try:
                    cell.value = float(cell.value) / 100
                    cell.number_format = "0.0%"
                except (TypeError, ValueError):
                    pass

        # Q1列文字对齐
        q1_col_idx = next(
            i for i, (_, lbl) in enumerate(_DETAIL_COLS, 1) if lbl == "Q1首问"
        )
        q1_letter = get_column_letter(q1_col_idx)
        for row in range(2, ws1.max_row + 1):
            ws1[f"{q1_letter}{row}"].alignment = Alignment(wrap_text=True, vertical="top")

        # 列宽自适应
        for col_idx, col_cells in enumerate(ws1.columns, start=1):
            col_letter = get_column_letter(col_idx)
            max_len = max(
                (len(str(c.value)) if c.value is not None else 0 for c in col_cells),
                default=8,
            )
            # Q1 列固定宽度 50，其余自适应
            if col_idx == q1_col_idx:
                ws1.column_dimensions[col_letter].width = 50
            else:
                ws1.column_dimensions[col_letter].width = min(max_len + 2, 40)

        # Sheet 2: 分布统计
        ws2 = ew.book.create_sheet("分布统计")
        title_font  = Font(bold=True, size=11, color="1F4E79")
        subhdr_fill = PatternFill(fill_type="solid", fgColor="D9E1F2")
        subhdr_font = Font(bold=True, size=10)

        cur_row = 1
        for section_title, df_s in dist_sections:
            # 区间标题
            title_cell = ws2.cell(cur_row, 1, section_title)
            title_cell.font = title_font
            cur_row += 1

            # 表头行
            for ci, col_name in enumerate(df_s.columns, 1):
                hc = ws2.cell(cur_row, ci, col_name)
                hc.fill = subhdr_fill
                hc.font = subhdr_font
                hc.alignment = Alignment(horizontal="center")
            cur_row += 1

            # 数据行
            for _, row in df_s.iterrows():
                for ci, val in enumerate(row, 1):
                    ws2.cell(cur_row, ci, val)
                cur_row += 1

            cur_row += 2  # 空行分隔

        # 列宽
        for col in ws2.columns:
            ws2.column_dimensions[get_column_letter(col[0].column)].width = 32


# ---------------------------------------------------------------------------
# Markdown 汇总报告
# ---------------------------------------------------------------------------

def build_md(sessions: List[Dict], top_n: int = 10) -> str:
    total = len(sessions)
    lines: List[str] = []

    def ln(s: str = "") -> None:
        lines.append(s)

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    ln("# Session 分析报告")
    ln()
    ln(f"> 生成时间: {generated_at}　　总 session 数: **{total}**")
    ln()
    ln("---")
    ln()

    # ── 1. 对话轮次 ──────────────────────────────────────────────────────────
    turns_vals = [s["user_turns"] for s in sessions]
    dist_turns = Counter(turns_vals)
    ln("## 1. 对话轮次（User 发言数）")
    ln()
    ln("| 统计项 | 数值 |")
    ln("|--------|------|")
    ln(f"| 平均 | {sum(turns_vals)/total:.2f} |")
    ln(f"| 最大 | {max(turns_vals)} |")
    ln(f"| P50  | {pct(turns_vals, 50):.0f} |")
    ln(f"| P90  | {pct(turns_vals, 90):.0f} |")
    ln()
    ln("**分布：**")
    ln()
    max_c = max(dist_turns.values())
    ln("| 轮次 | 数量 | 占比 | 分布 |")
    ln("|-----:|-----:|-----:|------|")
    for k in sorted(dist_turns):
        bar = "█" * max(1, round(dist_turns[k] / max_c * 15))
        ln(f"| {k} | {dist_turns[k]} | {dist_turns[k]/total*100:.1f}% | {bar} |")
    ln()

    # ── 2. 消息总数 ──────────────────────────────────────────────────────────
    msg_vals = [s["total_messages"] for s in sessions]
    ln("## 2. 消息总数（含 assistant 回复）")
    ln()
    ln("| 统计项 | 数值 |")
    ln("|--------|------|")
    ln(f"| 平均 | {sum(msg_vals)/total:.2f} |")
    ln(f"| 最大 | {max(msg_vals)} |")
    ln(f"| 最小 | {min(msg_vals)} |")
    ln(f"| P50  | {pct(msg_vals, 50):.0f} |")
    ln(f"| P90  | {pct(msg_vals, 90):.0f} |")
    ln()
    ln("**分布：**")
    ln()
    ln("| 区间 | 数量 | 占比 | 分布 |")
    ln("|------|-----:|-----:|------|")
    msg_buckets = [("1-2条",0,3),("3-5条",3,6),("6-10条",6,11),
                   ("11-20条",11,21),("21-50条",21,51),(">50条",51,10**9)]
    max_c2 = max(sum(1 for v in msg_vals if lo<=v<hi) for _,lo,hi in msg_buckets)
    for label, lo, hi in msg_buckets:
        cnt = sum(1 for v in msg_vals if lo <= v < hi)
        bar = "█" * max(1, round(cnt / max(max_c2,1) * 15)) if cnt else ""
        ln(f"| {label} | {cnt} | {cnt/total*100:.1f}% | {bar} |")
    ln()

    # ── 3. API Call 次数 ─────────────────────────────────────────────────────
    api_vals  = [s["api_call_count"] for s in sessions]
    multi_api = sum(1 for v in api_vals if v > 1)
    api_err   = sum(1 for s in sessions if s["api_errors"] > 0)
    ln("## 3. API Call 次数")
    ln()
    ln("| 统计项 | 数值 |")
    ln("|--------|------|")
    ln(f"| 平均   | {sum(api_vals)/total:.2f} |")
    ln(f"| 最大   | {max(api_vals)} |")
    ln(f"| 单次   | {sum(1 for v in api_vals if v==1)} ({sum(1 for v in api_vals if v==1)/total*100:.1f}%) |")
    ln(f"| 多次   | {multi_api} ({multi_api/total*100:.1f}%) |")
    ln(f"| 含 4xx/5xx 错误 | {api_err} ({api_err/total*100:.1f}%) |")
    ln()
    ln("**分布：**")
    ln()
    api_buckets = [("1次",1,2),("2-3次",2,4),("4-10次",4,11),("11-30次",11,31),(">30次",31,10**9)]
    ln("| 区间 | 数量 | 占比 | 分布 |")
    ln("|------|-----:|-----:|------|")
    max_c3 = max(sum(1 for v in api_vals if lo<=v<hi) for _,lo,hi in api_buckets)
    for label, lo, hi in api_buckets:
        cnt = sum(1 for v in api_vals if lo <= v < hi)
        bar = "█" * max(1, round(cnt / max(max_c3,1) * 15)) if cnt else ""
        ln(f"| {label} | {cnt} | {cnt/total*100:.1f}% | {bar} |")
    ln()

    # ── 4. 工具调用 ──────────────────────────────────────────────────────────
    with_tools = [s for s in sessions if s["tool_use_count"] > 0]
    total_tu   = sum(s["tool_use_count"]    for s in sessions)
    total_tr   = sum(s["tool_result_count"] for s in sessions)
    total_succ = sum(s["tool_success"]      for s in sessions)
    total_ff   = sum(s["tool_fail_flag"]    for s in sessions)
    total_fk   = sum(s["tool_fail_keyword"] for s in sessions)
    total_ft   = sum(s["tool_fail_total"]   for s in sessions)

    ln("## 4. 工具调用（tool_use / tool_result）")
    ln()
    ln("| 统计项 | 数值 |")
    ln("|--------|------|")
    ln(f"| 有工具调用的 session  | {len(with_tools)} ({len(with_tools)/total*100:.1f}%) |")
    ln(f"| 无工具调用的 session  | {total-len(with_tools)} ({(total-len(with_tools))/total*100:.1f}%) |")
    ln(f"| tool_use 总次数       | {total_tu} |")
    ln(f"| tool_result 总次数    | {total_tr} |")
    if with_tools:
        wtu = [s["tool_use_count"] for s in with_tools]
        ln(f"| 有工具 session 平均 tool_use | {sum(wtu)/len(wtu):.1f} |")
        ln(f"| 最大 tool_use  | {max(wtu)} |")
        ln(f"| P50 tool_use   | {pct(wtu, 50):.0f} |")
        ln(f"| P90 tool_use   | {pct(wtu, 90):.0f} |")
    ln()
    if total_tr > 0:
        ln("**成功率明细：**")
        ln()
        ln("| 判断方式 | 次数 | 占总 tool_result |")
        ln("|----------|-----:|-----------------:|")
        ln(f"| 成功（无 flag + 无错误关键字） | {total_succ} | {total_succ/total_tr*100:.1f}% |")
        ln(f"| 失败：is_error=True           | {total_ff}   | {total_ff/total_tr*100:.1f}% |")
        ln(f"| 失败：内容含错误关键字        | {total_fk}   | {total_fk/total_tr*100:.1f}% |")
        ln(f"| **失败合计**  | **{total_ft}** | **{total_ft/total_tr*100:.1f}%** |")
        ln(f"| **整体成功率** | — | **{total_succ/total_tr*100:.1f}%** |")
        ln()
    ln("**tool_use 次数分布（有工具 session）：**")
    ln()
    if with_tools:
        tu_vals = [s["tool_use_count"] for s in with_tools]
        tu_bkts = [("1-5次",1,6),("6-15次",6,16),("16-30次",16,31),("31-50次",31,51),(">50次",51,10**9)]
        max_c4 = max(sum(1 for v in tu_vals if lo<=v<hi) for _,lo,hi in tu_bkts)
        ln("| 区间 | 数量 | 占比 | 分布 |")
        ln("|------|-----:|-----:|------|")
        for label, lo, hi in tu_bkts:
            cnt = sum(1 for v in tu_vals if lo <= v < hi)
            bar = "█" * max(1, round(cnt / max(max_c4,1) * 15)) if cnt else ""
            ln(f"| {label} | {cnt} | {cnt/len(with_tools)*100:.1f}% | {bar} |")
        ln()
    rate_sessions = [s for s in sessions if s["tool_result_count"] > 0]
    if rate_sessions:
        ln("**工具成功率分布（有 tool_result 的 session）：**")
        ln()
        rv = [s["tool_success_rate"] for s in rate_sessions]
        rate_bkts = [("0-50%",0,50),("50-80%",50,80),("80-95%",80,95),("95-99%",95,99),("100%",100,101)]
        max_c5 = max(sum(1 for v in rv if lo<=v<hi) for _,lo,hi in rate_bkts)
        ln("| 区间 | 数量 | 占比 | 分布 |")
        ln("|------|-----:|-----:|------|")
        for label, lo, hi in rate_bkts:
            cnt = sum(1 for v in rv if v is not None and lo <= v < hi)
            bar = "█" * max(1, round(cnt / max(max_c5,1) * 15)) if cnt else ""
            ln(f"| {label} | {cnt} | {cnt/len(rate_sessions)*100:.1f}% | {bar} |")
        ln()

    # ── 5. 耗时 ──────────────────────────────────────────────────────────────
    multi_timed = [s for s in sessions if s["api_call_count"] > 1 and s["duration_s"] is not None]
    ln("## 5. 消耗时长")
    ln()
    ln("> 时长 = 首个 API call 时间戳 → 最后一个 API call 时间戳（单次 session 时长为 0）")
    ln()
    ln("| 统计项 | 数值 |")
    ln("|--------|------|")
    ln(f"| 单次 API call session | {sum(1 for s in sessions if s['api_call_count']==1)} |")
    ln(f"| 多轮 session（有时长） | {len(multi_timed)} |")
    if multi_timed:
        dv = [s["duration_s"] for s in multi_timed]
        ln(f"| 平均耗时 | {fmt_duration(sum(dv)/len(dv))} |")
        ln(f"| 最长     | {fmt_duration(max(dv))} |")
        ln(f"| 最短     | {fmt_duration(min(dv))} |")
        ln(f"| P50      | {fmt_duration(pct(dv, 50))} |")
        ln(f"| P90      | {fmt_duration(pct(dv, 90))} |")
        ln()
        dur_bkts = [("<1min",0,60),("1-5min",60,300),("5-15min",300,900),
                    ("15-30min",900,1800),(">30min",1800,10**9)]
        max_c6 = max(sum(1 for v in dv if lo<=v<hi) for _,lo,hi in dur_bkts)
        ln("**耗时分布（多轮 session）：**")
        ln()
        ln("| 区间 | 数量 | 占比 | 分布 |")
        ln("|------|-----:|-----:|------|")
        for label, lo, hi in dur_bkts:
            cnt = sum(1 for v in dv if lo <= v < hi)
            bar = "█" * max(1, round(cnt / max(max_c6,1) * 15)) if cnt else ""
            ln(f"| {label} | {cnt} | {cnt/len(dv)*100:.1f}% | {bar} |")
    ln()

    # ── 6. 模型分布 ──────────────────────────────────────────────────────────
    model_dist = Counter(s["model"] for s in sessions)
    ln("## 6. 模型使用分布")
    ln()
    ln("| 模型 | 数量 | 占比 |")
    ln("|------|-----:|-----:|")
    for mdl, cnt in sorted(model_dist.items(), key=lambda x: -x[1]):
        ln(f"| `{mdl or '(未知)'}` | {cnt} | {cnt/total*100:.1f}% |")
    ln()

    # ── 7. Top 10 最长 session ───────────────────────────────────────────────
    ln(f"## 7. 耗时最长 Top {top_n} Session")
    ln()
    ln("| Session | 时长 | 用户轮次 | tool_use | tool_result | 成功率 | 请求次数 |")
    ln("|---------|-----:|---------:|---------:|------------:|-------:|---------:|")
    top10 = sorted(
        [s for s in sessions if s.get("duration_s")],
        key=lambda x: -x["duration_s"],
    )[:top_n]
    for s in top10:
        ln(f"| `{s['session']}` | {fmt_duration(s['duration_s'])} "
           f"| {s['user_turns']} | {s['tool_use_count']} "
           f"| {s['tool_result_count']} | {fmt_rate(s['tool_success_rate'])} "
           f"| {s['api_call_count']} |")
    ln()

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="分析 session 格式对话日志，输出 Excel + Markdown 到 <session目录>/stat/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python analyze_sessions.py test_session\n"
            "  python analyze_sessions.py test_session --out /tmp/report\n"
            "  python analyze_sessions.py test_session --top 20"
        ),
    )
    parser.add_argument(
        "session_dir",
        metavar="SESSION_DIR",
        help="session 根目录（各 session 文件夹所在目录）",
    )
    parser.add_argument(
        "--out", "-o",
        default=None,
        metavar="OUTPUT_DIR",
        help="输出目录，默认为 <SESSION_DIR>/stat",
    )
    parser.add_argument(
        "--top", "-t",
        type=int,
        default=10,
        metavar="N",
        help="Markdown 报告中展示耗时最长的 Top N session，默认 10",
    )
    args = parser.parse_args()

    session_dir = Path(args.session_dir).resolve()
    if not session_dir.is_dir():
        parser.error(f"目录不存在: {session_dir}")

    stat_dir = Path(args.out).resolve() if args.out else session_dir / "stat"
    stat_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] 扫描目录: {session_dir}", file=sys.stderr)
    folders = sorted(f for f in session_dir.iterdir() if f.is_dir() and f.name != "stat")
    print(f"[info] 发现 {len(folders)} 个 session 文件夹", file=sys.stderr)

    sessions: List[Dict] = []
    for folder in folders:
        r = analyze_session(folder)
        if r:
            sessions.append(r)
    print(f"[info] 成功解析 {len(sessions)} 个 session", file=sys.stderr)

    xlsx_path = stat_dir / "session_report.xlsx"
    write_excel(sessions, xlsx_path)
    print(f"[done] Excel    → {xlsx_path}", file=sys.stderr)

    md_path = stat_dir / "session_report.md"
    md_path.write_text(build_md(sessions, top_n=args.top), encoding="utf-8")
    print(f"[done] Markdown → {md_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
