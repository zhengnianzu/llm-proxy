import os
import sys
import json
import argparse
from pathlib import Path
from typing import Literal
from pydantic import BaseModel


def check_date_range(file: Path, date_start: str, date_end: str):
    """判断日志是否符合筛选日期"""
    file_date = file.name.split('_')[0]
    # print(f"{file_date=}")
    return True if date_start <= file_date <= date_end else False


def find_first_key_value(obj, target_key, value_type):
    """
    深度优先递归遍历 JSON 对象（dict/list），
    返回第一个键为 target_key 且值为 target_type 类型的 (key, value) 对，
    若未找到则返回 None。
    """
    if isinstance(obj, dict):
        for key, value in obj.items():
            # print(f"{key=}", file=sys.stderr)
            # print(f"{value=}", file=sys.stderr)
            # print(f"{type(value)=}", file=sys.stderr)
            # print(f"{value_type=}", file=sys.stderr)
            # print("-"*66, file=sys.stderr)
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
            # print(f"{item=}", file=sys.stderr)
            if isinstance(item, (dict, list)):
                result = find_first_key_value(item, target_key, value_type)
                if result is not None:
                    return result
    return None


class DataItem(BaseModel):
    """对应 /statistic 返回的 data 列表中的单个记录"""
    model: str  # 模型名称，例如 "test1"
    date_start: str  # 起始日期，格式为 YYYY-MM-DD
    date_end: str  # 结束日期，格式为 YYYY-MM-DD
    status: Literal['success', 'error']  # 状态，限定为 success 或 error
    count: int = 0  # 请求次数
    input_token_num: int = 0  # 输入 token 数
    output_token_num: int = 0  # 输出 token 数


class SummaryItem(BaseModel):
    """对应 /statistic 返回的 summary 列表中的单个汇总项"""
    count: int = 0  # 该状态下的总记录数
    status: Literal['success', 'error']  # 状态，与 data 中的 status 对应
    total_input: int = 0  # 该状态下的总输入 token 数
    total_output: int = 0  # 该状态下的总输出 token 数


def find_files(filter_suffix="*-res.json"):
    for _dir in os.listdir():
        if os.path.isdir(_dir) and _dir.startswith("logs_"):
            for file in Path(_dir).rglob(filter_suffix):
                # print(file)
                yield file


def statistic_tokens(model: str = '', date_start: str = '', date_end: str = '', status: str = '全部', **kwargs) -> dict:
    """
    统计token数
    :param model: 过滤模型，忽略大小写，多个模型用,拼接
    :param date_start: 过滤日期-开启，格式YYYY-MM-DD
    :param date_end: 过滤日期-结束，格式YYYY-MM-DD
    :param status: 过滤状态: 全部、成功、失败
    :return: dict
    """
    model_count_data = dict()
    res_data = list()

    for file in find_files():
        # 筛选符合日期的日志文件
        if not check_date_range(file, date_start, date_end):
            continue
        with open(file, 'r', encoding='utf8') as f:
            raw_data = f.read()
        try:
            data = json.loads(raw_data)
        except Exception as e:
            print(f"{file} json load error: {e}", file=sys.stderr)
            continue
        # print(f"{data=}")
        _model = find_first_key_value(data, 'model', str)
        if not isinstance(_model, str):
            continue
        # 筛选符合模型要求的日志
        if len(model) > 0 and _model.lower() not in model.lower():
            continue
        if _model in model_count_data:
            info_success = model_count_data[_model]['success']
            info_error = model_count_data[_model]['error']
        else:
            info_success = DataItem(model=_model, date_start=date_start, date_end=date_end, status="success")
            info_error = DataItem(model=_model, date_start=date_start, date_end=date_end, status="error")

        token_key_in = ['input_tokens', 'prompt_tokens']
        # 输入token数
        for key_in in token_key_in:
            token_in = find_first_key_value(data, key_in, int)
            if token_in:
                break
        else:
            token_in = 0
        # 输出token数
        token_key_out = ['output_tokens', 'completion_tokens']
        for key_out in token_key_out:
            token_out = find_first_key_value(data, key_out, int)
            if token_out:
                break
        else:
            token_out = 0

        # 局部汇总
        if status in ["全部", "成功"] and token_out > 0:  # 成功的情况
            info_success.count += 1
            info_success.input_token_num += token_in
            info_success.output_token_num += token_out
        elif status in ["全部", "失败"] and token_out == 0:  # 失败的情况
            info_error.count += 1
            info_error.input_token_num += token_in

        # 汇总到全局
        model_count_data[_model] = {'success': info_success, 'error': info_error}

    for key, val in model_count_data.items():
        if val['success'].count > 0:
            res_data.append(val['success'].model_dump())
        if val['error'].count > 0:
            res_data.append(val['error'].model_dump())

    # 全局汇总到摘要
    summary_success = SummaryItem(status="success")
    summary_error = SummaryItem(status="error")
    for data in res_data:
        if data['status'] == 'success':
            summary_success.count += data['count']
            summary_success.total_input += data['input_token_num']
            summary_success.total_output += data['output_token_num']
        elif data['status'] == 'error':
            summary_error.count += data['count']
            summary_error.total_input += data['input_token_num']
            summary_error.total_output += data['output_token_num']

    res = {"data": res_data, "summary": [summary_success.model_dump(), summary_error.model_dump()]}
    # print(res)

    # fake data
    # res = {
    #     "data": [
    #         {"model": "test1", "date_start": "2025-03-01", "date_end": "2025-03-31", "status": "success", "count": 1000,
    #          "input_token_num": 1200, "output_token_num": 450},
    #         # {"model": "test1", "date_start": "2025-03-01", "date_end": "2025-03-31", "status": "error", "count": 10,
    #         #  "input_token_num": 120, "output_token_num": 0},
    #         {"model": "test2", "date_start": "2026-03-01", "date_end": "2026-03-09", "status": "success", "count": 800,
    #          "input_token_num": 1500, "output_token_num": 777},
    #         # {"model": "test2", "date_start": "2026-03-01", "date_end": "2026-03-09", "status": "error", "count": 8,
    #         #  "input_token_num": 150, "output_token_num": 0},
    #     ],
    #     "summary": [
    #         {"count": 1800, "status": "success", "total_input": 2700, "total_output": 1227},
    #         # {"count": 10, "status": "error", "total_input": 120, "total_output": 0},
    #         # {"count": 1010, "status": "error", "total_input": 3120, "total_output": 1200},
    #     ]
    # }

    print_log = f"""
================================================================================
API 调用统计摘要
================================================================================

总调用次数: {summary_success.count}
总输入 Tokens: {summary_success.total_input}
总输出 Tokens: {summary_success.total_output}
总 Tokens: {summary_success.total_input + summary_success.total_output}

按模型统计:
--------------------------------------------------------------------------------
"""
    # claude-sonnet-4-20250514: 调用 10 次, 输入 500, 输出 1,200, 总计 1,700 tokens
    for key, val in model_count_data.items():
        if val['success'].count > 0:
            _count = val['success'].count
            _input_token = val['success'].input_token_num
            _output_token = val['success'].output_token_num
            _all_token = _input_token + _output_token
            print_log += f"""{key}: 调用 {_count} 次, 输入 {_input_token} , 输出 {_output_token}, 总计 {_all_token} tokens"""
            print_log += '\n'
    print_log += "--------------------------------------------------------------------------------"

    print(print_log)
    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="", help="过滤模型，忽略大小写，多个模型用,拼接")
    parser.add_argument("--date_start", "-s", type=str, default="", help="过滤日期-开启，格式YYYY-MM-DD")
    parser.add_argument("--date_end", "-e", type=str, default="", help="过滤日期-结束，格式YYYY-MM-DD")
    parser.add_argument("--status", "-t", type=str, default="全部", help="过滤状态: 全部、成功、失败")
    args = parser.parse_args()

    # args.date_start = '2026-03-10'
    # args.date_end = '2026-03-10'
    # print(args.__dict__)
    statistic_tokens(**args.__dict__)
