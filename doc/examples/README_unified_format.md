# 导出 / 质检 数据格式统一方案

目标：**让导出（export）直接产出「质检模型」的记录格式**，只保留一种 JSON 结构，
既有质检的分析字段，又不丢导出独有的信息。

- 统一后的示例数据见同目录 [`unified_session_analysis.json`](./unified_session_analysis.json)。
- 配套的样例轨迹（每条 session 的 `-req.json`）见 [`trajectories/`](./trajectories/README.md)，
  演示「查看」页如何从磁盘读取轨迹并渲染 `response` 回复。
- 数据集在磁盘上的组织方式（必须项 / 路径拼接 / `<session>` 文件夹含义）见
  [`README_Dataset.md`](./README_Dataset.md)。
- 顶层沿用 `session_analysis.json` 的封装（`save_analysis_json`，`utils/eval/eval.py:1053`），
  仅把 `version` 从 `1` 升到 `2` 以区分含导出字段的新记录。

## 统一前：现状有两套不兼容的格式

### A. 导出格式 `session_index.jsonl`（聚合器直接产物）
JSONL，一行一条 session，末行是 `{"_meta": true, ...}` 汇总。字段：

| 字段 | 说明 |
|---|---|
| `_key` | session 主键 = 首请求时间戳（下划线格式） |
| `api_key` | 完整 sk-… key |
| `q1` | 首个用户问题 |
| `first_ts` / `last_ts` | 首/末请求时间戳（下划线格式） |
| `models` | 用到的模型列表 |
| `latest_file` | 消息最多的那次请求文件名 |
| `msg_count` | 上述文件的消息数 |
| `trace_list` | 每次 API 调用一条：`{filename, model, msg_count, ts}` |

只有索引，**无任何分析/质检字段**。

### B. 质检格式 `session_analysis.json`（analyze 产物）
单个 JSON 对象：`{version, generated_at, session_count, sessions:[...]}`。
每条 session 记录（`utils/eval/reformat.py:164`）含全部分析 + 质检字段：
`session, start_time, end_time, duration_s, api_call_count, api_errors,
user_turns, total_messages, tool_use_count, tool_result_count, tool_success,
tool_fail_flag, tool_fail_keyword, tool_fail_total, tool_success_rate, model, q1,
latest_file, log_dir, tool_use_detail, tool_success_detail, tool_fail_detail,
skills_used, completed, completed_note`。

但它**丢掉了导出独有的三个字段**：`api_key`、`models`(list)、`trace_list`。

## 统一策略：兼容超集

统一记录 = **质检记录形状 + 导出三字段**，且**同时保留新旧键名**，让 ~10 个下游读取方
（前端 `index.html`、report/xlsx/html、chat-log-viewer、thinking_reflection 等，硬编码了
`_key`/`first_ts`/`models` 或 `session`/`model`）**无需改动**。

- **规范键（新）**：`session`、`start_time`/`end_time`、`total_messages`、`api_call_count`、
  `models`(list)、`api_key`、`q1`、`latest_file`、`trace_list`。
- **兼容别名（旧）**：`_key`、`first_ts`、`last_ts`、`msg_count`；质检侧历史的 scalar `model`
  也作为别名保留（值取 `models[0]`），但**统一 schema 以 `models` 列表为准**。
- **评估字段**：`duration_s`/`api_errors`/`user_turns`/`tool_*`/`completed`/`completed_note`/
  `log_dir` 等，**仅「导出+质检」填充**；纯导出缺省/空 —— 这就是导出 vs 导出+质检的唯一区别。

组装逻辑落在 `utils/session_store.py:to_unified_record`（纯导出）与
`utils/eval/reformat.py:_process_one`/`_rebuild_from_cache`（导出+质检）。

## 字段映射（导出 → 统一格式）

| 导出字段 | 统一格式字段 | 处理 |
|---|---|---|
| `_key` | `session`（+ 保留 `_key` 别名） | 改名对齐质检 |
| `first_ts` | `start_time`（+ 保留 `first_ts` 别名） | 对外时间格式统一为 `YYYY-MM-DD HH:MM:SS` |
| `last_ts` | `end_time`（+ 保留 `last_ts` 别名） | 同上 |
| `models`(list) | `models`(list) | **只保留 list，不再引入 scalar `model`**（质检侧 scalar `model` 仅作历史别名） |
| `msg_count` | `total_messages`（+ 保留 `msg_count` 别名） | 语义对齐 |
| `len(trace_list)` | `api_call_count` | 由 trace 数推导 |
| `api_key` | `api_key` | **新增回质检记录** |
| `trace_list` | `trace_list` | **新增回质检记录** |
| — | 其余分析/质检字段 | 仅导出+质检时由 analyze 计算填入 |

## 时间格式统一

现状混用三种：下划线（`2026-07-15_20-11-39_889`）、空格（`2026-07-15 20:58:20`）、
unix 浮点（reflection 的 `processed_at`）。统一格式内**对外字段一律用 `YYYY-MM-DD HH:MM:SS`**，
`trace_list` 内部的 `ts` 保留下划线（它对应真实文件名，改了会对不上磁盘文件）。

## 质检码（completed）取值

`completed` 为 `0`（无异常）或逗号分隔的错误码串；`completed_note` 为对应中文描述。
规则见 `utils/eval/quality_rules.py`：

- `E001` 乱码(行均字符过少)
- `E002` 200空响应
- `E003` 工具调用过少(<3次)
- `E004` write成功率低于30%
