# 样例轨迹（sample trajectories）

配合同目录 [`unified_session_analysis.json`](./unified_session_analysis.json) 的样例轨迹文件，
用于演示「Session 管理 → 查看」页面如何从磁盘读取轨迹并渲染对话。

## 目录布局

轨迹文件按 **`<session>/<trace_list.filename>`** 组织，`<session>` 取自 analysis 里每条
session 的 `session` 字段，文件名严格匹配该 session `trace_list[].filename`
（改名会和磁盘对不上，见统一格式文档「时间格式统一」一节）：

```
trajectories/
├── 2026-07-28_09-10-05_120/                 # session 1：CSV 按省份汇总（多轮 + 工具）
│   ├── 2026-07-28_09-10-05_120-req.json      #   第 1 次调用：回复=thinking+text（追问文件/列名）
│   ├── 2026-07-28_09-12-31_640-req.json      #   第 2 次调用：exec 读文件后，回复=thinking+text+tool_use
│   └── 2026-07-28_09-18-40_887-req.json      #   末次调用：回复=thinking+text（收尾给出汇总表）
└── 2026-07-28_09-40-22_003/                 # session 2：天气查询失败（质检码 E002,E003）
    ├── 2026-07-28_09-40-22_003-req.json      #   第 1 次调用：回复=text+tool_use（web_fetch）
    └── 2026-07-28_09-41-05_500-req.json      #   末次调用：web_fetch 报错 → 200 空响应（content=[]）
```

## 轨迹文件结构

每个 `-req.json` 是一次 API 调用的完整快照，顶层四个键：

| 键 | 说明 |
|---|---|
| `model` | 实际路由到的模型 |
| `messages` | **请求侧**上下文（Anthropic 格式：`user`/`assistant`，content 为 block 列表） |
| `header` | 请求头快照（样例留空 `{}`） |
| `response` | **本次调用的返回**（Anthropic 原生：`{content:[blocks], status_code, ...}`） |

关键点：**本次调用的模型回复只存在 `response` 里，不会回写进 `messages`**。它会作为
assistant 消息出现在*下一个*轨迹文件的 `messages` 中——所以一个 session 的最后一次调用的
回复，只能从该文件自己的 `response` 读到。查看页据此把 `response` 拼成一条带
「↓ 来自 res 文件的回复」标记的 assistant 气泡追加到末尾
（`src/thinking_reflection/merger.py:append_response_message`）。

`2026-07-28_09-41-05_500-req.json` 特意造了一个 **200 空响应**（`content: []`），
对应质检码 `E002`：这类空内容**不会**渲染出空气泡（append 时跳过），是预期行为。

## 在「查看」页预览

这些文件放在 `doc/examples/` 下，是**文档 fixture，不是线上数据集**，默认不会出现在 UI。
要临时预览，可让一条质检记录的 `local_copy_dir` 指向本目录并把 `session_analysis.json`
放进去（注意 `_is_valid_dataset` 要求路径含 `analysis`）：

```bash
# 例：造一个临时 analysis 目录指向样例
mkdir -p /tmp/demo_analysis
cp doc/examples/unified_session_analysis.json /tmp/demo_analysis/session_analysis.json
cp -r doc/examples/trajectories/* /tmp/demo_analysis/
# 再把某条 export 记录的 local_copy_dir 指到 /tmp/demo_analysis 即可在「查看」页看到两条 session
```
