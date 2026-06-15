# Session 导出与质检 — 设计文档

## 概览

Session 导出/质检模块提供 Web 界面，用于将 LLM 代理日志从本地 mtime 目录导出到结构化的 session 文件，并可选地进行质量检测（reformat + 分析 + 报告）。两种模式共享统一的后端任务引擎和前端轮询机制。

**入口**: `/keys/export` （与 Key 管理共享认证）

## 架构

```
┌─────────────────────────────────────────────────────────────┐
│  前端 (templates/export.html)                                │
│  Vue 2 + Bootstrap 5 (无构建)                                │
│                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────┐ │
│  │ Key 列表  │  │ 导出Modal │  │ 进度日志  │  │ 报告面板     │ │
│  │ + 记录历史 │  │ mtime选择 │  │ 实时轮询  │  │ Markdown 内联│ │
│  └──────────┘  └──────────┘  └──────────┘  └─────────────┘ │
│        │              │              │              │        │
│        │   POST /run  │  GET /status │  GET /report │        │
└────────┼──────────────┼──────────────┼──────────────┼────────┘
         ▼              ▼              ▼              ▼
┌─────────────────────────────────────────────────────────────┐
│  后端 (utils/export_routes.py)                               │
│                                                             │
│  register_export_routes(app, logs_dir)                      │
│    ├─ GET  /api/export/config        服务配置                │
│    ├─ GET  /api/export/keys          Key列表 + 历史记录       │
│    ├─ POST /api/export/run           创建任务(导出/质检)      │
│    ├─ POST /api/export/eval          手动质检(基于已有导出)    │
│    ├─ GET  /api/export/status/:id    任务状态 + 日志          │
│    ├─ GET  /api/export/records       记录列表                │
│    ├─ GET  /api/export/obs/ls        OBS 目录浏览             │
│    └─ GET  /api/export/eval/report/:id  质检报告(Markdown)    │
│                                                             │
│  _run_task(record_id, env_dir, env_key_name,                │
│            obs_prefix, now_tag, mode)                        │
│    → 统一任务执行引擎，mode="export" | "eval"                │
└─────────────────────────────────────────────────────────────┘
         │                          │
         ▼                          ▼
┌─────────────────┐    ┌──────────────────────┐
│ export_store.py  │    │ eval/                 │
│ SQLite 持久化     │    │  reformat.py  重组日志 │
│ export_records   │    │  eval.py      质检分析 │
│  + mode 字段     │    │  quality_rules.py     │
└─────────────────┘    └──────────────────────┘
```

## 两种模式

导出和质检是 **互斥** 的两种操作模式（OR 关系），由 DB 记录的 `mode` 字段区分：

### 导出模式 (mode="export")

用途：将原始 session 三元组文件（request/response/trace）复制到本地目录并同步到 OBS。

```
用户选 Key + mtime目录 → POST /api/export/run (auto_eval=false)
  → 创建 record (mode="export")
  → _run_task 在后台线程执行:
      for each mtime_dir:
        1. export_session_index()  — 生成 session_index.jsonl
        2. sync_session_index()    — 按 key 过滤，本地复制 + OBS 上传
      → 更新 status = success/failed
```

本地输出: `logs_session/{env_key}/{slot}/ex-{now_tag}/`
OBS 路径: `{obs_prefix}/session/{env_key}/{slot}/ex-{now_tag}/`

**关键**: OBS 上传发生在 `sync_session_index()` 内部，逐 mtime 目录上传。

### 质检模式 (mode="eval")

用途：将三元组合并为完整对话，分析消息/工具/质量指标，生成 Markdown 报告。

```
触发方式 A: 勾选"质检模式" → POST /api/export/run (auto_eval=true)
触发方式 B: 在已有导出记录上点 [质检] → POST /api/export/eval
  → 创建 record (mode="eval")
  → _run_task 在后台线程执行:
      for each mtime_dir:
        1. _load_session_index()     — 读取已有索引
        2. reformat_and_analyze()    — 合并三元组 + 分析 (多进程)
      post-loop:
        3. evaluate_sessions()       — 聚合统计 + 生成 session_report.md
        4. _run_upload_cmd()         — 一次性上传整个目录到 OBS
      → 更新 status = success/failed
```

本地输出: `logs_session_analysis/{env_key}/{slot}/ex-{now_tag}/`
OBS 路径: `{obs_prefix}/session_analysis/{env_key}/{slot}/ex-{now_tag}/`

**关键**: OBS 上传在循环外统一做一次（因为报告需要所有 session 数据汇总后才能生成）。

### 手动质检 (POST /api/export/eval)

从已有的导出记录创建 **新的** eval 记录，复制 `api_key`、`key_slot`、`mtime_dirs`：

```python
new_record_id = create_record(
    api_key=rec["api_key"], key_slot=rec["key_slot"],
    mtime_dirs=rec["mtime_dirs"], mode="eval",
)
```

这样每次质检都有独立的状态追踪和日志，不会污染原始导出记录。

## 统一任务引擎 (_run_task)

`_run_task` 是导出和质检的统一入口，替代了之前独立的 `_do_export` + `_run_eval` 两个函数：

```
_run_task(record_id, env_dir, env_key_name, obs_prefix, now_tag, mode, force=False)
  │
  ├─ 根据 mode 计算 local_base 和 obs_dst
  │    export → logs_session/...      + session/...
  │    eval   → logs_session_analysis/... + session_analysis/...
  │
  ├─ 遍历 mtime_dirs（唯一分歧点）:
  │    export: export_session_index() + sync_session_index()
  │    eval:   _load_session_index() + reformat_and_analyze()
  │
  ├─ 后处理（仅 eval）:
  │    写 session_index.jsonl
  │    evaluate_sessions() → 生成 session_report.md
  │    _run_upload_cmd() → 上传到 OBS
  │
  └─ 统一状态更新:
       update_status(record_id, "success"/"failed", ...)
```

## 数据库 (export_store.py)

SQLite, 路径: `{service_log_dir}/export_session_record.db`

### export_records 表

| 字段 | 类型 | 说明 |
|------|------|------|
| id | INTEGER PK | 自增 |
| api_key | TEXT | 筛选的 API key（空=全量） |
| key_slot | TEXT | `"all"` 或 `"key-xxxx"` |
| mtime_dirs | TEXT (JSON) | `["26061500", "26061415"]` |
| mode | TEXT | `"export"` 或 `"eval"` |
| status | TEXT | `pending` → `running` → `success`/`failed` |
| error_message | TEXT | 失败原因 |
| total_sessions | INTEGER | 匹配/处理的 session 数 |
| files_uploaded | INTEGER | 上传文件数（export 模式） |
| files_skipped | INTEGER | 跳过文件数（export 模式） |
| eval_report_path | TEXT | 报告文件路径（eval 模式） |
| progress_log | TEXT (JSON) | 实时日志 `[{"ts":"...", "msg":"..."}]` |
| created_at | TEXT | 创建时间 |
| started_at | TEXT | 开始执行时间 |
| finished_at | TEXT | 完成时间 |

旧字段 `eval_status` 保留用于兼容旧记录，新代码不再写入。

## 质检报告 (Markdown)

报告由 `utils/eval/eval.py` 的 `write_markdown()` 生成，输出为 `session_report.md`，前端通过 `<pre>` 标签内联展示（monospace 下 markdown 表格天然对齐）。

### 报告结构

```markdown
# Session 质检报告

> 总 Sessions: 10 | 工具调用: 58 | 成功: 52 (89.7%)
> 模型: claude-3-opus: 8, gpt-4: 2 | 多轮: 6 | API错误: 1

## Session 详情
| Session | Q1 | 时长(s) | 轮次 | 消息 | 工具调用 | 成功率 | 模型 | 质检 |
（精简 9 列，Q1 截断 80 字符）

## 分布统计
### 对话轮次分布 / 消息总数分布 / API Call次数 / ...
（6 个分布表）

## 技能统计
（技能名称 + 使用次数）
```

### 报告 API

```
GET /api/export/eval/report/{record_id}
→ {"report_md": "# Session 质检报告\n...", "record_id": 123}
```

认证: 使用 session cookie 检查（非 XHR header 检查），因为需要支持直接访问。

## 前端交互

### 状态轮询

统一使用 `_startPoll()`，每 2 秒轮询 `/api/export/status/{id}`：

```javascript
var done = (d.status === 'success' || d.status === 'failed');
if (done) {
    clearInterval(self.pollTimer);
    self.fetchKeys();  // 刷新列表
}
```

无论导出还是质检，每个任务都是独立的 record，完成条件相同。

### 记录行渲染（按 mode 区分）

```
export + success → 显示 "N sessions / M files" + [质检] 按钮
eval + success   → 显示 "N sessions" + [报告] 按钮 + "质检" badge
running          → 显示 "导出中" 或 "质检中"（根据 mode）
```

### 操作流

```
[导出] 按钮 → 打开 Modal → 选 mtime + 是否勾选质检 → POST /run
  ├─ 不勾选 → mode=export → 轮询 → 完成后显示 [质检] 按钮
  └─ 勾选   → mode=eval   → 轮询 → 完成后显示 [报告] 按钮

[质检] 按钮 → POST /eval → 创建新 eval 记录 → 轮询 → [报告]
[报告] 按钮 → fetch /report → 报告面板内联显示 Markdown
[详情] 按钮 → fetch /status → 日志面板显示进度
```

## 认证

与 Key 管理共享 `key_state.yaml` 密码认证：

- 页面访问: `request.session["key_authenticated"]`
- API 调用: `_require_key_api()` 检查 `X-Requested-With: XMLHttpRequest` + session
- 报告端点: 仅检查 session（不要求 XHR header）

## OBS 上传

使用 `tools/obs_upload.sh` 包装 `obsutil cp -f -r`：

- 不使用 `-flat`，利用 obsutil 递归上传自动携带本地目录名的特性
- `obs_dst` 保持完整路径（含 `ex-{now_tag}/`），用于 DB 存储和前端 OBS 浏览
- 上传时将 `obs_dst` 去掉最后一层，本地路径不带尾 `/`，obsutil 自动附加目录名还原完整路径
- 导出模式: 在 `sync_session_index()` 内部逐 mtime 上传
- 质检模式: 在循环外统一上传整个分析目录

## 文件清单

| 文件 | 职责 |
|------|------|
| `utils/export_routes.py` | 路由注册 + _run_task 统一引擎 |
| `utils/export_store.py` | SQLite CRUD（export_records 表） |
| `utils/export_sync.py` | session_index 生成 + 文件同步 |
| `utils/eval/reformat.py` | 三元组合并 + 单 session 分析（多进程） |
| `utils/eval/eval.py` | 统计聚合 + Markdown 报告生成 |
| `utils/eval/quality_rules.py` | 质量检测规则 |
| `utils/obs_sync.py` | OBS 上传工具函数 |
| `tools/obs_upload.sh` | obsutil cp 包装脚本 |
| `templates/export.html` | 前端页面（Vue 2 单文件） |
