"""
token_stats.py — Token 用量统计与聚合

支持多日志目录参数，支持两种写法：
  python -m utils.token_stats -d logs_anthropic logs_session_anthropic
  python -m utils.token_stats -d logs_anthropic,logs_session_anthropic
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Literal, Optional
from pydantic import BaseModel
import pandas as pd

from utils.log_paths import INDEX_FILENAME


def check_date_range(file: Path, date_start: str, date_end: str):
    file_date = file.name.split('_')[0]
    return True if date_start <= file_date <= date_end else False


def find_first_key_value(obj, target_key, value_type):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == target_key and isinstance(value, value_type):
                if value_type == str:
                    return value
                elif value_type == int and value > 0:
                    return value
            if isinstance(value, (dict, list)):
                result = find_first_key_value(value, target_key, value_type)
                if result is not None:
                    return result
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, (dict, list)):
                result = find_first_key_value(item, target_key, value_type)
                if result is not None:
                    return result
    return None


class DataItem(BaseModel):
    model: str
    date_start: str
    date_end: str
    status: Literal['success', 'error']
    count: int = 0
    input_token_num: int = 0
    output_token_num: int = 0


class SummaryItem(BaseModel):
    count: int = 0
    status: Literal['success', 'error']
    total_input: int = 0
    total_output: int = 0


def find_files(dirs=None, filter_suffix="*-res.json"):
    if dirs:
        search_dirs = [Path(d) for d in dirs if Path(d).is_dir()]
    else:
        search_dirs = [Path(d) for d in os.listdir() if os.path.isdir(d) and d.startswith("logs_")]
    for _dir in search_dirs:
        for file in _dir.rglob(filter_suffix):
            yield file


def _load_index_file(index_path: str) -> Optional[list]:
    if not os.path.exists(index_path):
        return None
    entries = []
    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return entries


def _normalize_dirs(dirs) -> Optional[list[Path]]:
    if dirs is None:
        return None

    raw_items: list[str] = []
    if isinstance(dirs, (str, Path)):
        raw_items = [str(dirs)]
    else:
        for item in dirs:
            if item is None:
                continue
            raw_items.append(str(item))

    normalized: list[Path] = []
    seen: set[str] = set()
    for item in raw_items:
        for part in item.split(","):
            part = part.strip()
            if not part:
                continue
            key = os.path.normpath(part)
            if key in seen:
                continue
            seen.add(key)
            normalized.append(Path(part))
    return normalized


def _collect_index_files(dirs=None) -> list[Path]:
    index_files: list[Path] = []
    search_dirs = _normalize_dirs(dirs)
    root_dirs = search_dirs or [
        Path(d) for d in os.listdir()
        if os.path.isdir(d) and d.startswith("logs_")
    ]
    for root in root_dirs:
        root_index = root / INDEX_FILENAME
        if root_index.is_file():
            index_files.append(root_index)
    return index_files


def _aggregate_index_entries(entries, model_filter, date_start, date_end, status, model_count_data, channel_key_filter=""):
    for entry in entries:
        entry_date = entry.get("ts", "")[:10]
        if not (date_start <= entry_date <= date_end):
            continue

        if channel_key_filter:
            entry_channel = entry.get("channel_key", "") or ""
            if entry_channel != channel_key_filter:
                continue

        _model = entry.get("model", "") or ""
        if not _model:
            continue

        if model_filter and _model.lower() not in model_filter.lower():
            continue

        tok_in  = entry.get("tok_in", 0) or 0
        tok_out = entry.get("tok_out", 0) or 0

        if "valid" in entry:
            is_success = bool(entry["valid"]) and tok_out > 0
        else:
            is_success = bool(entry.get("success", False)) and tok_out > 0

        if _model not in model_count_data:
            model_count_data[_model] = {
                "success": DataItem(model=_model, date_start=date_start, date_end=date_end, status="success"),
                "error":   DataItem(model=_model, date_start=date_start, date_end=date_end, status="error"),
            }

        if status in ["全部", "成功"] and is_success:
            model_count_data[_model]["success"].count += 1
            model_count_data[_model]["success"].input_token_num  += tok_in
            model_count_data[_model]["success"].output_token_num += tok_out
        elif status in ["全部", "失败"] and not is_success:
            model_count_data[_model]["error"].count += 1
            model_count_data[_model]["error"].input_token_num += tok_in


def statistic_tokens(model: str = '', date_start: str = '2000-01-01', date_end: str = '9999-12-31',
                     status: str = '全部', channel_key: str = '', dirs=None, **kwargs) -> dict:
    """
    统计token数。优先从 index.jsonl 快速加载，不存在时降级扫描 res 文件。
    :param model: 过滤模型，忽略大小写，多个模型用,拼接
    :param date_start: 过滤日期-开启，格式YYYY-MM-DD
    :param date_end: 过滤日期-结束，格式YYYY-MM-DD
    :param status: 过滤状态: 全部、成功、失败
    :param channel_key: 过滤渠道 key，为空时不过滤
    """
    model_count_data = dict()
    normalized_dirs = _normalize_dirs(dirs)

    index_files = _collect_index_files(normalized_dirs)
    if index_files:
        for index_file in index_files:
            entries = _load_index_file(str(index_file))
            if entries is not None:
                _aggregate_index_entries(entries, model, date_start, date_end, status, model_count_data, channel_key)

        if normalized_dirs is None:
            return _build_result(model_count_data, date_start, date_end)
        indexed_dirs = {index_file.parent.resolve() for index_file in index_files}
        normalized_dirs = [d for d in normalized_dirs if d.resolve() not in indexed_dirs]
        if not normalized_dirs:
            return _build_result(model_count_data, date_start, date_end)

    scan_dirs = [str(d) for d in normalized_dirs] if normalized_dirs is not None else None
    for file in find_files(dirs=scan_dirs):
        if not check_date_range(file, date_start, date_end):
            continue
        try:
            with open(file, 'r', encoding='utf8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"{file} json load error: {e}", file=sys.stderr)
            continue
        _model = find_first_key_value(data, 'model', str)
        if not isinstance(_model, str):
            continue
        if len(model) > 0 and _model.lower() not in model.lower():
            continue

        tok_in = 0
        for key_in in ['input_tokens', 'prompt_tokens']:
            v = find_first_key_value(data, key_in, int)
            if v:
                tok_in = v; break
        tok_out = 0
        for key_out in ['output_tokens', 'completion_tokens']:
            v = find_first_key_value(data, key_out, int)
            if v:
                tok_out = v; break

        if _model not in model_count_data:
            model_count_data[_model] = {
                "success": DataItem(model=_model, date_start=date_start, date_end=date_end, status="success"),
                "error":   DataItem(model=_model, date_start=date_start, date_end=date_end, status="error"),
            }
        if status in ["全部", "成功"] and tok_out > 0:
            model_count_data[_model]["success"].count += 1
            model_count_data[_model]["success"].input_token_num  += tok_in
            model_count_data[_model]["success"].output_token_num += tok_out
        elif status in ["全部", "失败"] and tok_out == 0:
            model_count_data[_model]["error"].count += 1
            model_count_data[_model]["error"].input_token_num += tok_in

    return _build_result(model_count_data, date_start, date_end)


def _build_result(model_count_data: dict, date_start: str, date_end: str) -> dict:
    res_data = []
    for val in model_count_data.values():
        if val['success'].count > 0:
            res_data.append(val['success'].model_dump())
        if val['error'].count > 0:
            res_data.append(val['error'].model_dump())

    summary_success = SummaryItem(status="success")
    summary_error   = SummaryItem(status="error")
    for d in res_data:
        if d['status'] == 'success':
            summary_success.count += d['count']
            summary_success.total_input  += d['input_token_num']
            summary_success.total_output += d['output_token_num']
        else:
            summary_error.count += d['count']
            summary_error.total_input += d['input_token_num']

    rows = [
        (k, v['success'].count, v['success'].input_token_num, v['success'].output_token_num,
         v['success'].input_token_num + v['success'].output_token_num)
        for k, v in model_count_data.items() if v['success'].count > 0
    ]
    df = pd.DataFrame(rows, columns=["模型", "调用次数", "输入Token", "输出Token", "总Token"])
    df = df.sort_values("总Token", ascending=False).reset_index(drop=True)

    print(f"""
================================================================================
API 调用统计摘要
================================================================================
总调用次数: {summary_success.count}
总输入 Tokens: {summary_success.total_input}
总输出 Tokens: {summary_success.total_output}
总 Tokens: {summary_success.total_input + summary_success.total_output}

按模型统计:
{df.to_string(index=False)}
================================================================================""")

    return {"data": res_data, "summary": [summary_success.model_dump(), summary_error.model_dump()]}


def _mask_api_key(key: str) -> str:
    if not key or len(key) <= 8:
        return key or "(empty)"
    return f"{key[:4]}...{key[-4:]}"


def statistic_keys(date_start: str = '2000-01-01', date_end: str = '9999-12-31', dirs=None) -> dict:
    """
    按 API Key 维度聚合统计。
    返回格式：{"keys": [{"key": "sk-a...xz9f", "count": N, "tok_in": N, "tok_out": N, "sessions": N}]}
    """
    normalized_dirs = _normalize_dirs(dirs)
    index_files = _collect_index_files(normalized_dirs)

    key_data: dict = {}

    for index_file in index_files:
        entries = _load_index_file(str(index_file))
        if entries is None:
            continue
        for entry in entries:
            entry_date = entry.get("ts", "")[:10]
            if not (date_start <= entry_date <= date_end):
                continue
            raw_key = entry.get("api_key", "") or ""
            if not raw_key:
                raw_key = "(unknown)"
            tok_in = entry.get("tok_in", 0) or 0
            tok_out = entry.get("tok_out", 0) or 0
            if raw_key not in key_data:
                key_data[raw_key] = {"count": 0, "tok_in": 0, "tok_out": 0, "sessions": set()}
            key_data[raw_key]["count"] += 1
            key_data[raw_key]["tok_in"] += tok_in
            key_data[raw_key]["tok_out"] += tok_out
            sess_id = entry.get("q1_hash") or entry.get("chain_key", "")
            if sess_id:
                key_data[raw_key]["sessions"].add(sess_id)

    keys_list = [
        {"key": _mask_api_key(k), "count": v["count"], "tok_in": v["tok_in"], "tok_out": v["tok_out"], "sessions": len(v["sessions"]) or v["count"]}
        for k, v in key_data.items()
    ]
    keys_list.sort(key=lambda x: x["count"], reverse=True)
    return {"keys": keys_list}


def statistic_channels(date_start: str = '2000-01-01', date_end: str = '9999-12-31', dirs=None) -> dict:
    """
    按渠道 Key 维度聚合统计。
    返回格式：{"channels": [{"key": "sk-a...xz9f", "count": N, "tok_in": N, "tok_out": N, "sessions": N}]}
    """
    normalized_dirs = _normalize_dirs(dirs)
    index_files = _collect_index_files(normalized_dirs)

    ch_data: dict = {}

    for index_file in index_files:
        entries = _load_index_file(str(index_file))
        if entries is None:
            continue
        for entry in entries:
            entry_date = entry.get("ts", "")[:10]
            if not (date_start <= entry_date <= date_end):
                continue
            raw_key = entry.get("channel_key", "") or ""
            if not raw_key:
                raw_key = "(default)"
            tok_in = entry.get("tok_in", 0) or 0
            tok_out = entry.get("tok_out", 0) or 0
            if raw_key not in ch_data:
                ch_data[raw_key] = {"count": 0, "tok_in": 0, "tok_out": 0, "sessions": set()}
            ch_data[raw_key]["count"] += 1
            ch_data[raw_key]["tok_in"] += tok_in
            ch_data[raw_key]["tok_out"] += tok_out
            sess_id = entry.get("q1_hash") or entry.get("chain_key", "")
            if sess_id:
                ch_data[raw_key]["sessions"].add(sess_id)

    ch_list = [
        {"key": _mask_api_key(k), "count": v["count"], "tok_in": v["tok_in"], "tok_out": v["tok_out"], "sessions": len(v["sessions"]) or v["count"]}
        for k, v in ch_data.items()
    ]
    ch_list.sort(key=lambda x: x["count"], reverse=True)
    return {"channels": ch_list}


def list_known_channel_keys(dirs=None) -> list[str]:
    """扫描 index 获取所有出现过的 channel_key 值。"""
    normalized_dirs = _normalize_dirs(dirs)
    index_files = _collect_index_files(normalized_dirs)
    keys: set[str] = set()
    for index_file in index_files:
        entries = _load_index_file(str(index_file))
        if entries is None:
            continue
        for entry in entries:
            ck = entry.get("channel_key", "")
            if ck:
                keys.add(ck)
    return sorted(keys)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirs", "-d", nargs="+", default=None, help="指定日志目录（可多个，支持空格分隔或逗号分隔），不指定则扫描当前目录下 logs_ 开头的目录")
    parser.add_argument("--model", "-m", type=str, default="", help="过滤模型，忽略大小写，多个模型用,拼接")
    parser.add_argument("--date_start", "-s", type=str, default="2000-01-01", help="过滤日期-开启，格式YYYY-MM-DD，默认2000-01-01")
    parser.add_argument("--date_end", "-e", type=str, default="9999-12-31", help="过滤日期-结束，格式YYYY-MM-DD，默认9999-12-31")
    parser.add_argument("--status", "-t", type=str, default="全部", help="过滤状态: 全部、成功、失败")
    args = parser.parse_args()
    statistic_tokens(**args.__dict__)
