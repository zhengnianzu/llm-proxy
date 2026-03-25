import os
import sys
import json
import argparse
from pathlib import Path
from typing import Literal, Optional
from pydantic import BaseModel
import pandas as pd


def check_date_range(file: Path, date_start: str, date_end: str):
    """判断日志是否符合筛选日期"""
    file_date = file.name.split('_')[0]
    return True if date_start <= file_date <= date_end else False


def find_first_key_value(obj, target_key, value_type):
    """
    深度优先递归遍历 JSON 对象（dict/list），
    返回第一个键为 target_key 且值为 target_type 类型的 (key, value) 对，
    若未找到则返回 None。
    """
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
    """对应 /statistic 返回的 data 列表中的单个记录"""
    model: str
    date_start: str
    date_end: str
    status: Literal['success', 'error']
    count: int = 0
    input_token_num: int = 0
    output_token_num: int = 0


class SummaryItem(BaseModel):
    """对应 /statistic 返回的 summary 列表中的单个汇总项"""
    count: int = 0
    status: Literal['success', 'error']
    total_input: int = 0
    total_output: int = 0


def find_files(dirs=None, filter_suffix="*-res.json"):
    """扫描日志目录。dirs 为 None 时自动扫描当前目录下所有 logs_ 开头的子目录。"""
    if dirs:
        search_dirs = [Path(d) for d in dirs if Path(d).is_dir()]
    else:
        search_dirs = [Path(d) for d in os.listdir() if os.path.isdir(d) and d.startswith("logs_")]
    for _dir in search_dirs:
        for file in _dir.rglob(filter_suffix):
            yield file


def _load_index_file(index_path: str) -> Optional[list]:
    """从 index.jsonl 加载条目。文件不存在时返回 None（触发降级扫描）。"""
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


def _aggregate_index_entries(entries, model_filter, date_start, date_end, status, model_count_data):
    """将 index 条目聚合到 model_count_data 字典中。"""
    for entry in entries:
        # 日期过滤：ts 前10位为 YYYY-MM-DD
        entry_date = entry.get("ts", "")[:10]
        if not (date_start <= entry_date <= date_end):
            continue

        _model = entry.get("model", "") or ""
        if not _model:
            continue

        # 模型过滤
        if model_filter and _model.lower() not in model_filter.lower():
            continue

        tok_in  = entry.get("tok_in", 0) or 0
        tok_out = entry.get("tok_out", 0) or 0

        # 判断成功/失败：Anthropic 用 valid，OpenAI 用 success
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


def statistic_tokens(model: str = '', date_start: str = '2000-01-01', date_end: str = '9999-12-31', status: str = '全部', dirs=None, **kwargs) -> dict:
    """
    统计token数。优先从 index.jsonl 快速加载，不存在时降级扫描 res 文件。
    :param model: 过滤模型，忽略大小写，多个模型用,拼接
    :param date_start: 过滤日期-开启，格式YYYY-MM-DD
    :param date_end: 过滤日期-结束，格式YYYY-MM-DD
    :param status: 过滤状态: 全部、成功、失败
    """
    model_count_data = dict()

    # ---- 快速路径：index.jsonl ----
    # 仅在未指定 dirs 时启用（指定 dirs 通常是命令行精确扫描）
    if dirs is None:
        anthropic_index = os.path.join("logs_anthropic", "index.jsonl")
        openai_index    = os.path.join("logs_openai",    "index.jsonl")
        ant_entries = _load_index_file(anthropic_index)
        oai_entries = _load_index_file(openai_index)

        # 至少一个 index 存在，则走快速路径
        if ant_entries is not None or oai_entries is not None:
            if ant_entries is not None:
                _aggregate_index_entries(ant_entries, model, date_start, date_end, status, model_count_data)
            if oai_entries is not None:
                _aggregate_index_entries(oai_entries, model, date_start, date_end, status, model_count_data)

            # session 目录没有 index，仍走扫描
            session_dirs = ["logs_session_anthropic", "logs_session_openai"]
            for file in find_files(dirs=[d for d in session_dirs if os.path.isdir(d)]):
                if not check_date_range(file, date_start, date_end):
                    continue
                try:
                    with open(file, 'r', encoding='utf8') as f:
                        data = json.load(f)
                except Exception:
                    continue
                _model = find_first_key_value(data, 'model', str)
                if not isinstance(_model, str):
                    continue
                if model and _model.lower() not in model.lower():
                    continue
                tok_in  = 0
                tok_out = 0
                for key_in in ['input_tokens', 'prompt_tokens']:
                    v = find_first_key_value(data, key_in, int)
                    if v:
                        tok_in = v; break
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

    # ---- 降级路径：扫描所有 res 文件（原始逻辑）----
    for file in find_files(dirs=dirs):
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
        model_count_data[_model] = model_count_data[_model]  # noop, keep reference

    return _build_result(model_count_data, date_start, date_end)


def _build_result(model_count_data: dict, date_start: str, date_end: str) -> dict:
    """将聚合数据转换为标准响应格式并打印摘要。"""
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="", help="过滤模型，忽略大小写，多个模型用,拼接")
    parser.add_argument("--date_start", "-s", type=str, default="2000-01-01", help="过滤日期-开启，格式YYYY-MM-DD，默认2000-01-01")
    parser.add_argument("--date_end", "-e", type=str, default="9999-12-31", help="过滤日期-结束，格式YYYY-MM-DD，默认9999-12-31")
    parser.add_argument("--status", "-t", type=str, default="全部", help="过滤状态: 全部、成功、失败")
    parser.add_argument("--dirs", "-d", nargs="+", default=None, help="指定日志目录（可多个），不指定则扫描当前目录下 logs_ 开头的目录")
    args = parser.parse_args()

    # args.date_start = '2026-03-10'
    # args.date_end = '2026-03-10'
    # print(args.__dict__)
    statistic_tokens(**args.__dict__)
