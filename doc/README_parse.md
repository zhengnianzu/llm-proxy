# Session 解析（Thinking Reflection）— 设计文档

## 概览

解析模块对质检（导出+质检）产物中的 **thinking / reasoning 推理块**做一次「反思（Reflection）」LLM 回放，
产出被清洗/复原后的推理文本，并把结果合并回原始 trajectory，导出为 `*--thinking.json`（可选上传 OBS）。

代码术语为 **reflection**，前端界面标签为 **解析**。

- **入口**：`/thinking`（**管理**，数据集管理）与 `/thinking/tasks`（**解析任务**，运行/任务监控）
- **权限**：`thinking`（`admin` 或用户具备 `thinking` 权限），见 `app.py:149-150,268`
- **注册**：`register_reflection_routes(app, templates)`（`app.py:1830`，导入于 `app.py:55`）
- **包**：`src/thinking_reflection/`

> 上游是质检（eval）产物，下游是 reflection 专属的 OBS 导出。参见 [README_EXPORT.md](README_EXPORT.md)（Session 导出/质检）。

## 「解析」是什么

核心手法在 `consumer.py:reflect`：构造一段伪三消息对话——上一条 assistant 轮里放入
**原始 `thinking` 文本 + 其 `signature`**，再跟一条用户指令（形如「为何这么简单的回复却想了这么久？」）。
模型被 `tool_choice` 强制走 reflect 工具调用，从而**复原/重述**该段先前推理；工具输出解析回文本。

两种方法（`method`），各自一套 prompt/tool：

| method | 工具名 | 输出形态 | 组装函数 |
|---|---|---|---|
| `bulk` | `reflect_on_prior_reasoning_bulk` | 单段连续 `sentences` 字符串 | `bulk_to_reasoning` |
| `sentence` | `reflect_on_prior_reasoning` | 逐句数组 `prior_reasoning_sentences`，按轮次/序号重排 | `sentences_to_reasoning` |

## 端到端流程

```
质检产物(session_analysis.json + trajectory *.json)
        │  import_run (importer.py)
        ▼
extract_signatures (extractor.py) ── 递归找非空 signature 块
        │  每个 signature → dataset_tasks(pending)
        │  每个文件      → dataset_trajectories
        ▼
WorkerManager 线程池 (worker_manager.py) ── 认领 pending
        │  reflect (consumer.py) → 调用本地 proxy 的 Reflection 端点
        │  存 processed_text / usage / 响应元数据；done | failed
        ▼
merge (merger.py) ── 在每个 block_path 注入 "reflect" 对象
        │  {status, text, run_id, model, error, retry_count, processed_at}
        ▼
export (service.export) → <export_root>/<run_id>/<session_id>/<stem>--thinking.json
        │  可选 upload_run_to_obs → OBS reflection/ 前缀
```

1. **来源**：质检导出记录，其 `local_copy_dir` 路径需包含 `analysis`（即 eval 产出的 `logs_session_analysis/...` 树）。在 `service.create_run` / `import_tasks` 强校验（`service.py:160`）。
2. **导入**（`importer.py:import_run`）：优先读 `session_analysis.json` 索引枚举 session（`_load_session_index`），否则 `rglob`。对每个 trajectory `*.json`（排除 `session_analysis.json`、`session_index.json`、`failure_report.json`、`manifest.json` 及 `*--thinking.json`）跑 `extract_signatures`。
3. **抽取**（`extractor.py:extract_signatures`）：递归遍历 JSON，凡含非空字符串 `signature` 的 dict 即产出 `{block_path(形如 $.messages[..]…), message_index, signature, original_thinking}`（thinking 取自 `thinking` 或 `reasoning_content`），跳过鉴权/头部键。
4. **入库**：每个 signature → 一条 `dataset_tasks`（`pending`），并把 original_thinking + signature 落到 `logs_thinking/<key>/<ts>/` 的每任务明细 JSON。
5. **消费**（`worker_manager.py` + `consumer.py:reflect`）：worker 线程认领任务，调 Reflection 端点，存复原文本与元数据；重试耗尽转 `failed`。
6. **合并**（`merger.py:merge`）：深拷贝原始 trajectory，在各 `block_path` 注入 `reflect` 对象。
7. **导出**（`service.export`）：写合并 JSON，并可选上传 OBS。

## 路由 / 端点

来源 `src/thinking_reflection/routes.py`。

**页面（Jinja 模板）**

| 路由 | handler | 模板 | active_page | 导航 |
|---|---|---|---|---|
| `GET /thinking` | `page` | `thinking.html` | `thinking` | 管理 |
| `GET /thinking/tasks` | `tasks_page` | `thinking_tasks.html` | `thinking_tasks` | 解析任务 |
| `GET /thinking/dataset/{record_id}` | `dataset_page` | `thinking_dataset.html` | — | — |
| `GET /thinking/failed` | `failed_page` | `thinking_failed.html` | — | — |

**API（`/api/reflection/` 前缀）**

- **数据集**：`GET /datasets?key_slot=`、`GET /datasets-all`、`GET /datasets/{id}/analysis`、`GET /datasets/{id}/sessions`、`GET /datasets/{id}/session-trajectory`
- **配置**：`GET /config`（keys、source_key_slots、prompt 可用性、export_root、reflection_base_url、obs_base）
- **Run 生命周期**：`POST /runs`、`GET /runs`、`GET /runs/{id}`、`GET /runs/{id}/snapshot`、`GET /runs/{id}/trajectories`、`PATCH /runs/{id}/config`、`POST /runs/{id}/start|pause|stop`、`DELETE /runs/{id}`、`POST /runs/{id}/export`、`POST /runs/{id}/upload-obs`、`GET /runs/{id}/logs?since_id=`（轮询）
- **任务**：`GET /tasks?run_id=`、`GET /tasks/{uuid}`、`GET /tasks/{uuid}/attempts`、`POST /tasks/{uuid}/retry`、`POST /tasks/retry-failed`、`POST /tasks/rerun-done`
- **数据集生命周期**：`POST /import-tasks`、`POST /datasets/register`（注册外部 OBS 数据集）、`POST /datasets/{id}/download-obs`
- **批量**：`POST /batch-start|batch-pause|batch-cancel|batch-retry|batch-rerun`、`GET /tasks-summary`
- **其他**：`POST /test`（单条即席解析）、`GET /trajectories/{trajectory_id}/merged`

## 数据模型 / 存储

**SQLite**：`logs_session/thinking/thinking.db`（`config.get_service_log_dir()/thinking/thinking.db`），`SCHEMA_VERSION = 2`，WAL。表见 `db.py:init`：

| 表 | 说明 |
|---|---|
| `reflection_runs` | 每次运行一行（run_id `run_<hex12>`、source/quality 关联、endpoint、key、model、method、worker_count、max_retries、export_root、obs_root、status、`config_snapshot`、prompt 名/sha256、`launch_type`、`parent_run_id`、snapshot 计数、时间戳） |
| `dataset_trajectories` | (export_id, trajectory_path, trajectory_id[uuid5], session_id, source_root) |
| `dataset_tasks` | **解析单元**。uuid5(`export_id:session:path:block_path`)；signature/len、`detail_path`、`latest_status`(pending/processing/done/failed)、latest_run_id、latest_processed_text/model/response_id/stop_reason/usage_json/sentence_count、retry_count/max_retries、last_error。唯一键 (export_id, session_id, trajectory_path, block_path) |
| `run_trajectory_outputs` | 每 run/trajectory 的导出文件路径 |
| `task_attempts` | 每次尝试一行（status、耗时、error、response_id、usage） |
| `run_logs` | 每 run 流式日志行 |

**文件产物**

- **每任务明细**：`logs_thinking/<source_key>/<ts>/<sid>_<task_uuid>.json`（`db.task_detail_dir`）——original_thinking、signature，处理后追加 `processed_text` + `tool_input` + `raw_response`。用于把大文本移出 DB。
- **导出结果**：`<export_root>/<run_id>/<session_id>/<stem>--thinking.json`，默认 `export_root = logs_session_eval/reflection`（env 可覆盖）。可选镜像到 OBS：`<obs_base>/reflection/<source_key>/<run_id>/`。

## 配置 / 运行

`config.py:load_config`（`ReflectionConfig`）：

| 项 | 来源 / 默认 |
|---|---|
| `runtime_dir` | `<service_log_dir>/thinking` |
| `db_path` | `runtime_dir/thinking.db` |
| `prompt_dir` | env `REFLECTION_PROMPT_DIR`（默认 `src/thinking_reflection/prompt`） |
| `export_root` | env `REFLECTION_EXPORT_ROOT`（默认 `logs_session_eval/reflection`） |
| `reflection_base_url` | `http://127.0.0.1:<PROXY_PORT|4000>`——解析调用回环经本代理 |
| Reflection API key | 取自 `utils.key_store`（`reflection_api_key_id`，须 `active`） |
| 鉴权 | `MONITOR_AUTH_ENABLED` / `MONITOR_USERNAME`（`routes.context`） |

Prompt 文件 `bulk.json` / `sentence.json` 由 `prompt_loader.load_prompt` 校验并 sha256 固定
（须含 `instruction`、`unrelated_thinking`、`tool`，且工具名与 `input_schema` 精确匹配）。

**编排**（`worker_manager.py:WorkerManager`）：进程内 daemon 线程，非外部队列。
`start(run_id)` 起 `worker_count` 个线程（每任务 `reflect`）；每个数据集同时只允许一个活跃 run，
由 `_active_by_export` 锁保证（否则抛「数据集已有活跃 Run」）。`stop(cancel=)` 置 `threading.Event`。
服务初始化时 `db.reset_processing` 把卡住的 `processing` 重置为 pending 并暂停孤儿 run；
run 清空后 `_finish_if_empty` 自动收尾为 `completed` / `completed_with_failures`。

**启动一次解析**：导入任务（`POST /import-tasks` 或经 `create_run`）→ 启动
（`POST /runs/{id}/start` 或 `POST /batch-start`，复用导入期的占位草稿 run）。
失败重跑 `retry-failed` / `batch-retry`；已完成重跑 `rerun-done`。
**监控**：轮询 `GET /runs/{id}/logs`（含日志与 run 状态）与 `GET /tasks-summary`。

## 前端页面

复用 `templates/_layout.html`（侧边栏 60-66 行两项：`/thinking`→管理、`/thinking/tasks`→解析任务）。Vue 2，无构建。

- `templates/thinking.html`（标题「Session 管理」，**管理**页）：质检数据集表——名称 / Session 数 / 解析状态 / OBS / 解析OBS / 操作。动作：刷新、注册外部数据集、导入任务库、导出结果、OBS 文件浏览抽屉。
- `templates/thinking_tasks.html`（标题「解析任务」，**解析任务**页）：顶部统计卡（导入总数/已完成/进行中/失败，来自 `/tasks-summary`）；批量工具条（启动/暂停/取消）；按数据集分行（done/pending+processing/failed、run_status 徽标、补跑），可展开运行历史；按任务分行（状态徽标、sentence_count 或 last_error、重试）；抽屉：运行日志（实时轮询 + 进度条）、Run 详情、单条测试（调 `/test`）。
- `templates/thinking_dataset.html`：按数据集浏览 session/trajectory。
- `templates/thinking_failed.html`：失败清单。

## 与质检 / 导出的关系

- **上游（质检 / eval）**：解析消费质检产物。`session_analysis.json` 由 eval 写出
  （`utils/eval/eval.py:save_analysis_json`）。解析仅接受 `local_copy_dir` 含 `analysis` 的记录，
  经 `utils.export_store`（`list_records_for_datasets` / `list_records_by_key` / `get_record_resolved`）读取；
  `service.analysis` 亦在 UI 中展示 eval 的 `analysis_json` sessions。即：**eval 产出质检 session 目录 + `session_analysis.json` → 解析将其作为数据集导入**。
- **下游（导出）**：解析有独立导出步骤（`service.export` → `result_exporter.upload_run_to_obs`），
  产出 `--thinking.json` 合并文件并推送到 OBS 的 `reflection/` 专属前缀
  （见 `templates/obs.html` 「解析任务路径」= `<base>/reflection/`）。这与通用 Session 导出/质检
  （入口 `/keys/export`，[README_EXPORT.md](README_EXPORT.md)）相互独立，但共享 `export_store` 记录与
  `utils/obs_utils` OBS 工具。

## 关键文件

- `src/thinking_reflection/{routes,service,importer,extractor,consumer,merger,worker_manager,db,config,prompt_loader,result_exporter}.py`
- `src/thinking_reflection/prompt/{bulk,sentence}.json`
- `templates/{_layout,thinking,thinking_tasks,thinking_dataset,thinking_failed}.html`
- `utils/eval/eval.py`、`utils/export_store.py`、`app.py:1830`
