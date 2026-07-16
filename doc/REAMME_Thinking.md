# Thinking Reflection UI Design

## 1. 结论

建议采用“在 LLM Proxy 主界面中新增一级 Thinking 解析工作区”的方案：

- 不单独新建一套完全独立的界面。
- 不把全部功能塞入现有 Session 统计或 Session 导出页面的普通 Tab。
- 在 LLM Proxy 主侧边栏中增加“任务”分组，并增加“Thinking 解析”入口。
- Reflection 工作区内部再按任务流程拆分二级页面或 Tab。
- 复用现有 Session 列表、Trajectory 展示、thinking 渲染、登录和鉴权能力。
- 页面严格复用 LLM Proxy 现有 `_layout.html`、设计 token、表格、按钮、徽标、Modal、Drawer 和响应式行为，不建立第二套视觉系统。

LLM Proxy 当前主界面按“监控 / 历史 / 管理”组织导航。Thinking 解析是一条长时间运行的后台任务流程，适合新增独立的“任务”分组。`chat-log-viewer` 不是主界面，只复用它已有的对话与 thinking 可视化能力。

## 2. 选择该方案的原因

单独建设新 UI 会重复实现以下能力：

- 登录、Key 上下文和权限控制。
- Session、Trajectory 和消息可视化。
- 原始 JSON 与解析后 JSON 的查询和联动跳转。
- UI 样式、配置与部署流程。

同时，Reflection 又不只是 Session 的一个展示属性。它包含数据选择、导入、运行配置、后台执行、失败恢复、结果导出和历史审计，因此不适合作为 Session 统计或 Session 导出页面里的简单 Tab。

推荐产品层级：

```text
LLM Proxy
├── 监控
│   ├── 监控总览
│   ├── 查询统计
│   └── Session 统计
├── 历史
│   ├── 对话历史
│   └── 失败记录
├── 任务（新增）
│   └── Thinking 解析
│       ├── 运行批次
│       ├── 数据导入
│       ├── 解析任务
│       ├── 失败任务
│       └── 结果导出
└── 管理
    ├── Key 管理
    ├── 渠道管理
    ├── Session 导出
    ├── 备份管理
    └── 用户管理
```

其中，“任务”是主侧边栏的新分组，“Thinking 解析”是该分组下的一级页面。运行批次、数据导入、解析任务、失败任务和结果导出使用页面内二级 Tab。这样既不会把五个高度关联的页面全部铺到主侧边栏，也不会混入代理请求本身的“失败记录”。

## 3. Thinking 解析主页面

Thinking 解析主页面采用表格方式展示当前 Key 的导出数据集及其转换状态。主页面负责选择数据集、创建任务、监控进度和进入结果详情，不直接展示单个样本的完整对话。

```text
Thinking 解析                         当前 Key: key-xxxx

[全部] [未转换] [转换中] [已完成] [有失败]    [刷新]

导出ID | 导出时间 | Sessions | 质检状态 | 转换进度 | 成功 | 失败 | Worker | 输出 | 操作
128    | 07-15   | 120      | 已质检   | 78%      | 650 | 14  | 运行中 | 本地+OBS | 查看
127    | 07-14   | 42       | 未质检   | 未开始   | 0   | 0   | -      | -        | 创建任务
126    | 07-13   | 80       | 质检失败 | 已完成   | 510 | 3   | 已停止 | 本地     | 查看/重试
```

### 3.1 数据集表格

表格数据来自当前代理实例的 `export_records`，并按照当前 `source_key` 对应的 `key_slot` 筛选。每行代表一条导出数据集，不代表一个 Key，也不代表数据集内的单个样本。

建议列：

```text
source_export_id
created_at
mtime_dirs / 数据范围
total_sessions
local_copy_dir
obs_dst
quality_status
latest_run_id
reflection_status
total_signature_tasks
pending_count
processing_count
done_count
failed_count
progress_percent
worker_status
updated_at
```

支持筛选：全部、未转换、等待中、转换中、已完成、部分失败、失败、已质检和未质检。运行中的行每 2 至 5 秒刷新进度，刷新时保持筛选、分页和选中行。

### 3.2 质检标识

数据集行必须显示质检状态：

```text
未质检
质检中
已质检
质检失败
```

状态来自与 export 记录关联的最新 eval 记录。`已质检` 表示存在成功的 eval 记录并且 `session_analysis.json` 可以解析，不能用 Reflection 是否完成代替质检状态。

### 3.3 行操作

根据数据集状态提供以下操作：

```text
[创建任务]       尚未创建 Reflection Run
[单条测试]       使用当前配置测试一条 signature
[启动/暂停/停止] 管理该数据集的 worker
[查看]           进入数据集样本详情页
[重试失败]       重试该数据集的失败任务
[本地文件]       打开或复制本地输出路径
[OBS]            浏览 OBS 输出目录
[运行记录]       查看该数据集的历史 Run
```

表格中的失败数量可点击，点击后进入该数据集详情页并自动应用“有失败”筛选。质检状态可点击，进入质检报告或在详情页查看按样本合并后的分析字段。

### 3.4 数据集详情页

点击表格中的“查看”后，进入一个导出数据集的详情页。该页面才复用 chat view 模式，用于在转换中或转换完成后浏览数据集内的不同样本：

```text
左侧：当前数据集的 Session/Trajectory 样本列表
中间：当前样本的转换后对话，支持 Original / Reflect
右侧：当前样本状态、具体问题和质检分析
```

左侧样本列表展示 Session 名称、首问摘要、signature 数、成功/失败/处理中数量和质检标识；不展示其他 Key 或其他导出数据集。

如果当前样本存在质检数据，右侧将 `session_analysis.json` 中与该 Session 对应的分析字段合并进详情，例如：

```text
user_turns / api_call_count / duration_s
tool_use_count / tool_fail_total / tool_success_rate
completed / tool_fail_detail
模型、消息数及其他已有分析字段
```

左侧样本显示“已质检”标识；没有匹配分析数据时显示“未质检”，不能把 Reflection 转换完成误标为质检完成。

下方按序号列出具体问题 1、2、3……，每条包含：

```text
session_id
trajectory_path
block_path / message_index
错误类型与错误信息
retry_count / max_retries
最后尝试时间
[定位到内容] [查看尝试日志] [重试]
```

点击“定位到内容”时，中间区域滚动并高亮对应 thinking block。点击中间区域的失败标识时，右侧展开对应问题。右侧支持重试单条和重试当前样本全部失败项。

### 3.5 批次和配置入口

数据集表格是 Thinking 解析主页面。数据集详情页顶部工具栏承载：

- 返回数据集/Run 列表。
- 当前数据集和 Run 信息。
- 单条测试及测试日志。
- 启动、暂停和停止 worker。
- Run 历史选择。
- Tasks、Failed、Exports 的高级表格入口。

Runs、Tasks、Failed、Exports 作为数据集或 Run 的高级运维和历史审计入口，不替代主表格与数据集样本详情页。

## 4. 新建解析任务向导

点击“新建解析任务”后，建议使用分步向导。

### 4.1 选择数据

- 选择代理服务中的当前 Key。
- 从当前代理实例的 Session 导出数据库加载该 Key 对应的导出记录。
- 区分可导出、已导出、已导入和已解析状态。
- 支持单选、多选、按时间范围选择和全选筛选结果。

现有实现已经将记录保存到：

```text
{service_log_dir}/export_session_record.db
```

表名为 `export_records`，可直接复用现有接口：

```text
GET /api/export/records?key_slot=<current_key_slot>
GET /api/export/status/{record_id}
```

加载规则：

1. 当前页面从登录/选择状态取得 `source_key`，后端换算为当前导出逻辑使用的 `key_slot`，例如 `key-xxxx`。
2. 后端调用现有 `list_records_by_key(key_slot)`，不要让前端直接读取 SQLite 文件。
3. 默认只将 `mode=export`、`status=success` 且 `local_copy_dir` 有效的记录作为可导入数据。
4. `running` 记录显示进度但不能导入；`failed` 记录显示错误并提供回到 Session 导出页面的入口。
5. `mode=eval` 是现有质检记录，不应被误认为 Reflection 的输入导出记录。
6. 通过 `source_export_id=export_records.id` 关联 Reflection Run，判断已导入和已解析状态。

`export_session_record.db` 按代理运行实例的 `service_log_dir` 存放，并不是全局唯一数据库。Thinking 页面应通过当前 FastAPI 进程已经初始化的 export store/API 加载当前实例记录，不应硬编码某个固定 DB 路径，也不应跨实例扫描文件。

### 4.1.1 质检记录关联

现有质检流程会为手动质检新建独立的 `mode=eval` 记录。质检完成后：

- `session_analysis.json` 的内容保存到 eval 记录的 `analysis_json` 字段。
- `eval_report_path` 保存质检报告路径。
- `analysis_json` 可能是 `file://...` 外置文件标记，必须通过现有 `get_record_resolved()` 解析，不能把字段值直接当 JSON。

当前 `export_records` 缺少 export 记录与 eval 记录的可靠关联。建议新增：

```text
source_export_id INTEGER NULL
```

规则如下：

1. 普通 `mode=export` 记录的 `source_export_id` 为空。
2. 通过 `POST /api/export/eval` 创建质检记录时，将原导出记录 ID 写入 `source_export_id`。
3. 数据集页面通过 `mode=eval AND source_export_id=<export_id>` 查找最新成功质检记录。
4. 找到成功且 `analysis_json` 可解析的记录，数据集和对应样本显示“已质检”。
5. eval 正在运行时显示“质检中”；失败时显示“质检失败”并可查看错误；不存在关联记录时显示“未质检”。
6. 解析 `session_analysis.json` 后按 `session` 字段与左侧样本的 session/folder 标识匹配，将分析数据合并到右侧详情。

为兼容历史数据，可临时用 `key_slot + mtime_dirs` 查找可能的 eval 记录，但必须标记为弱匹配；新记录一律使用 `source_export_id`，避免同一个 Key、同一时间目录多次导出或质检时关联错误。

### 4.2 配置导出

- 本地导出根目录默认使用 `logs_session_eval`。
- 配置默认 OBS 上传根路径，并允许在创建 Run 时确认或修改目标子路径。
- 配置是否重新导出。
- 配置已存在文件的处理方式。
- 显示预计 Session、Trajectory 和文件数量。

建议默认值和路径规则：

```text
local_export_root = logs_session_eval
obs_base          = load_obs_base()
local_run_dir     = logs_session_eval/reflection/<source_key_slot>/<run_id>/
obs_run_dir       = <obs_base>/reflection/<source_key_slot>/<run_id>/
```

`obs_base` 应复用现有 OBS 配置读取顺序：`.cli_state.yaml` 中的 `sync_config` 优先，找不到时回退到 `settings/obs_base.yaml`。页面首次加载时由后端返回默认值；若未配置，必须明确显示“仅保存本地，不上传 OBS”，不能静默假定上传成功。

Run 创建后应保存解析后的本地路径和 OBS 目标路径快照。全局默认路径后来发生变化时，不应改变历史 Run 的路径。

浏览器不应直接提供任意服务器文件系统选择能力。后端应配置允许的导出根目录，前端只能在允许范围内选择或创建子目录。

### 4.3 配置 Reflection

- Reflection API URL。
- API Key。
- 模型。
- 调用方法，例如 `bulk` 或 `sentence`。
- worker 数量。
- 最大重试次数。
- 请求超时与并发限制。

这里需要明确区分两个 Key：

- `source_key`：用于在代理仓中筛选和导出 Session。
- `reflection_api_key`：worker 调用模型接口使用的 Key。

两者默认可以相同，但数据模型中不应使用同一个字段，以支持“处理 A Key 的日志，但通过 B Key 调用解析服务”。界面默认只展示脱敏值，后端不应将完整 Key 返回给浏览器。

本任务中两者明确不同：

- `source_key` 来自当前 Session 导出上下文，只用于加载和筛选导出记录。
- `reflection_api_key_id` 从 Key 管理中的启用 Key 列表选择，用于调用 Reflection URL。

Thinking 页面可以复用 `GET /api/keys` 获取 Key 的 `id`、名称、脱敏值和状态，只允许选择 `status=active` 的数据库 Key。Run 中保存 `reflection_api_key_id` 和脱敏快照，不保存明文 Key；worker 执行时在服务端通过 Key ID 读取完整 Key。前端不应调用 `/api/keys/{id}/reveal` 获取后再提交。

选择器旁提供“前往 Key 管理”链接，用于创建或启用 Key；从 Key 管理返回后可以刷新候选列表。若所选 Key 在 Run 启动前被禁用或删除，启动应失败并给出明确提示。

### 4.4 确认并启动

启动前展示配置摘要：

- Session 数量。
- Trajectory 数量。
- Thinking task 数量。
- 跳过数量及原因。
- 输出目录。
- 模型、worker 数和重试策略。

确认后创建独立 Run，再由后台 worker 异步执行。不要让浏览器请求一直等待整个批次完成。

启动前增加“单条测试”操作：从所选导出记录中选择一条包含 signature 的 task，使用当前 URL、模型和 `reflection_api_key_id` 发起真实测试，但不进入批量队列。测试结果和日志应保存在 Run 草稿中，至少记录：

```text
test_id
run_id / draft_id
task_uuid 或 trajectory + block_path
status
started_at / finished_at / elapsed_ms
endpoint（脱敏）
reflection_api_key_id（不保存明文）
model / method
request_id / response_id
usage
processed_text 预览
error_type / error_message
```

界面提供“测试日志”抽屉，按时间顺序展示准备请求、发起调用、收到响应、解析响应或失败的事件。signature、完整 Key 和大段原始请求默认不写日志、不直接展示。测试成功后仍需用户明确点击“启动批量任务”。

## 5. 需求与页面映射

| 需求 | 推荐位置 | 设计 |
|---|---|---|
| 当前 Key 的 Session 导出列表 | `Reflection > Imports` | 显示可导出、已导出、已导入和已解析状态 |
| 启动 worker，配置 Key、URL 和模型 | 新建任务向导、Run 详情 | 配置保存为 Run 快照，避免历史任务受后续配置变化影响 |
| 选择导出路径并导入任务 | 新建任务向导 | 只允许选择服务端配置的安全根目录及其子目录 |
| 监控成功、失败和重试 | `Runs`、`Failed` | 展示进度、吞吐、错误摘要，支持单个和批量重试 |
| 查看 thinking 是否完成解析 | `Tasks`、Session 状态列 | 展示未发现、待处理、处理中、已完成、部分失败和失败 |
| 查看解析 JSON 并增加 reflect | Run 或 Session 详情 | 提供“查看结果”和“打开可视化”操作 |
| 多个任务的执行记录 | `Runs` | 每次启动生成独立 Run 和不可变配置快照 |

## 5.1 已确认的关键决策

1. 当前 Key 的导出记录从当前代理实例的 `export_session_record.db` / `export_records` 加载，通过现有导出 API 按 `key_slot` 查询。
2. 本地导出根目录默认是 `logs_session_eval`；OBS 默认路径从现有 `load_obs_base()` 读取，并保存到 Run 配置快照。
3. `source_key` 与 `reflection_api_key_id` 是两个不同概念；后者从 Key 管理的 active DB Key 中选择，完整 Key 仅由服务端读取。
4. 批量运行前支持单条真实测试，并持久化脱敏的测试日志、耗时、usage、响应标识及错误。
5. Thinking 解析主页面采用数据集表格，每行是一条当前 Key 的导出记录；点击“查看”后进入数据集详情页，详情页左侧展示样本，中间展示转换后对话，右侧展示进度、问题和 `session_analysis.json` 质检分析。
6. Thinking 转换只能读取质检成功且产物完整的 eval 目录；未质检、质检中、质检失败或产物缺失时禁止创建 Run。

## 6. 页面详细设计

### 6.1 Runs

Runs 表格建议包含：

```text
run_id
source_key
status
model
session_count
task_count
pending_count
processing_count
done_count
failed_count
created_at
started_at
finished_at
```

Run 状态建议包含：

```text
draft
queued
running
paused
completed
completed_with_failures
cancelled
failed
```

Run 详情应显示配置快照、进度、错误摘要、关联 Session、任务列表和导出结果。

### 6.2 Imports

Imports 用于维护代理仓 Session 与 Reflection 服务之间的数据交接。

建议字段：

```text
source_key
session_id
source_export_id
source_status
export_path
exported_at
import_status
imported_at
latest_run_id
reflection_status
```

“已导出”和“已导入”必须分开：导出成功只表示文件已经生成，导入成功表示 Reflection 服务已经登记 trajectory 并提取任务。

### 6.3 Tasks

Tasks 页面用于操作单个 signature 解析任务。

建议字段：

```text
uuid
run_id
status
session_id
trajectory_path
block_path
message_index
retry_count
max_retries
signature_len
processed_text_len
model
last_attempt_at
updated_at
```

支持：

- 按状态筛选。
- 按 Run、Session、模型和时间筛选。
- 查看原始位置、处理结果和错误历史。
- 单个重试和批量重试。
- 跳转到对应 Trajectory。

原始 signature 默认隐藏，避免在列表、日志或错误提示中直接暴露大段 signature。

### 6.4 Failed

Failed 页面专门处理达到自动重试上限的任务。

建议展示：

- 最后错误。
- 尝试次数。
- 首次和最后失败时间。
- 使用的 URL、模型和方法，但 API Key 必须脱敏。
- 单条重试、批量重试和导出失败报告。

手动重试应创建新的 attempt 记录，不应覆盖旧错误历史。

### 6.5 Exports

Exports 页面负责生成和维护处理后的评审产物。

支持：

- 导出单个 Trajectory。
- 导出一个 Run 的全部 Trajectory。
- 导出处理后的 Session Report。
- 导出失败任务报告。
- 查看产物路径、创建时间和对应 Run。
- 打开解析后 JSON 的可视化页面。

## 7. 数据模型调整

现有 `trajectories` 和 `thinking_tasks` 可以表达当前任务状态，但不足以表达多个执行批次及每次尝试的完整记录。建议增加以下实体。

### 7.1 reflection_runs

```text
run_id
source_key
source_export_id
quality_record_id
reflection_endpoint
reflection_api_key_id
reflection_model
worker_count
max_retries
export_root
obs_root
status
config_snapshot
total_count
pending_count
processing_count
done_count
failed_count
created_at
started_at
finished_at
```

`config_snapshot` 保存创建 Run 时的不可变配置。敏感 Key 不应以明文存入快照，可以保存凭据引用或脱敏标识。

### 7.2 run_trajectories

```text
run_id
trajectory_id
source_export_id
import_status
output_path
exported_at
```

该表解决同一个 Trajectory 被不同 Run 重复处理或使用不同模型处理的问题。

### 7.3 task_attempts

```text
attempt_id
run_id
task_uuid
attempt_no
status
started_at
finished_at
error
response_id
usage
```

不要只依赖 `thinking_tasks.retry_count`。单一计数无法回答每次尝试何时发生、使用什么配置、错误是否变化。`task_attempts` 用于支撑运行历史和问题审计。

整体关系：

```text
Proxy Key
   │
   ├── Session exports
   │       │
   │       └── Reflection Run
   │              ├── Trajectories
   │              │      └── Thinking Tasks
   │              │              └── Attempts
   │              └── Exported JSON
   │                     └── Existing Trajectory Viewer
```

## 8. 解析后 JSON 设计

保留原始 JSON 不变，生成新的 `<stem>--thinking.json`。每个 thinking block 建议增加结构化的 `reflect` 字段：

```json
{
  "type": "thinking",
  "thinking": "...",
  "signature": "...",
  "reflect": {
    "status": "done",
    "text": "...",
    "run_id": "run_42",
    "model": "claude-opus-4-8",
    "processed_at": "2026-07-15T10:30:00+08:00",
    "retry_count": 1
  }
}
```

失败或尚未完成时也保留统一结构：

```json
{
  "reflect": {
    "status": "failed",
    "text": null,
    "run_id": "run_42",
    "error": "request timeout",
    "retry_count": 3
  }
}
```

如果导出文件会共享给非运维人员，应评估是否保留完整 `signature` 和详细错误信息，必要时提供脱敏导出选项。

## 9. 与现有对话可视化复用

解析完成后，不需要创建新的 JSON Viewer。可以复用 `chat-log-viewer` 的消息和 thinking block 渲染逻辑，并将其整合到 LLM Proxy 的对话历史详情中；也可以先将处理后的文件跳转到独立查看路由，例如：

```text
/?view=trajectory&source=reflection&run_id=run_42&trajectory_id=xxx
```

`chat-log-viewer` 已经支持 thinking block 渲染，但它在这里是可复用组件或查看路由，不是产品主导航。thinking 区域需要增加：

- `Original / Reflect` 切换。
- 原始 thinking 与 reflect 文本上下对照。
- 显示解析状态、模型、Run 和重试次数。
- 从 Session 或 Run 页面返回的导航上下文。

## 10. API 边界建议

代理仓负责：

- 当前 Key 上下文。
- Session 查询和导出记录。
- 统一 UI、登录和可视化入口。

Reflection 服务负责：

- 导入 Trajectory。
- 提取 signature。
- 管理 Run、Task 和 Attempt。
- worker 生命周期和任务执行。
- 合并并导出处理后的 JSON。

代理仓可通过反向代理或服务端 API 调用 Reflection 服务。浏览器不应直接持有 reflection API key，也不应直接负责启动本机进程。

建议 API 分组：

```text
POST /api/reflection/runs
GET  /api/reflection/runs
GET  /api/reflection/runs/{run_id}
POST /api/reflection/runs/{run_id}/start
POST /api/reflection/runs/{run_id}/pause
POST /api/reflection/runs/{run_id}/stop

GET  /api/reflection/imports
POST /api/reflection/imports

GET  /api/reflection/tasks
POST /api/reflection/tasks/{uuid}/retry
POST /api/reflection/tasks/retry-failed

GET  /api/reflection/exports
POST /api/reflection/runs/{run_id}/export
GET  /api/reflection/trajectories/{trajectory_id}/merged
```

worker 不应直接绑定浏览器会话。用户关闭页面后，Run 必须继续执行，并可在之后重新进入页面查看状态。

## 11. 实施顺序

建议分阶段实现：

1. 增加 `reflection_runs`、`run_trajectories` 和 `task_attempts`，先完善后端批次语义。
2. 打通代理仓导出记录到 Reflection 导入的接口。
3. 实现 Runs 首页、创建向导和 worker 状态轮询。
4. 实现 Tasks、Failed、单个重试和批量重试。
5. 实现处理后 JSON 导出及现有 Trajectory 页面的 `reflect` 展示。
6. 最后补充吞吐图表、日志详情、配置模板和批量运维功能。

第一版优先完成闭环：选择 Session、创建 Run、启动 worker、观察进度、处理失败、导出 JSON、跳转可视化。图表和高级筛选可以后续补充。

## 12. Python 代码目录建议

Thinking 功能建议放在代理仓根目录的 `src/thinking_reflection/`，不要继续把大量业务逻辑加入 `app.py`，也不要放入 `chat-log-viewer`。`app.py` 只负责注册路由和启动/关闭生命周期。

```text
src/
└── thinking_reflection/
    ├── __init__.py
    ├── config.py
    ├── models.py
    ├── db.py
    ├── migrations.py
    ├── routes.py
    ├── schemas.py
    ├── service.py
    ├── export_source.py
    ├── quality_source.py
    ├── importer.py
    ├── extractor.py
    ├── consumer.py
    ├── prompt_loader.py
    ├── prompt/
    │   ├── bulk.txt
    │   └── sentence.txt
    ├── worker_manager.py
    ├── worker.py
    ├── retry.py
    ├── merger.py
    ├── result_exporter.py
    ├── obs_uploader.py
    ├── viewer.py
    └── cli.py
```

各脚本职责：

| 文件 | 职责 |
|---|---|
| `config.py` | `logs_session_eval`、OBS 默认路径、`service_log_dir/thinking/thinking.db`、并发、超时和重试配置 |
| `models.py` | Run、RunTrajectory、ThinkingTask、TaskAttempt、TestRun 等领域对象 |
| `db.py` | SQLite 连接、事务及 CRUD，不承载流程编排 |
| `migrations.py` | Thinking 表建表与 `export_records.source_export_id` 等兼容迁移 |
| `routes.py` | `/api/reflection/*` 路由，解析请求并调用 service |
| `schemas.py` | FastAPI 请求和响应模型，避免直接暴露数据库 Row |
| `service.py` | 创建 Run、启动/暂停/停止、聚合表格状态等应用层编排 |
| `export_source.py` | 通过现有 export store 读取当前 Key 的成功导出记录及本地路径 |
| `quality_source.py` | 关联 eval 记录，解析 `analysis_json/session_analysis.json` 并按 Session 合并 |
| `importer.py` | 扫描导出数据集、登记 Session/Trajectory，保证重复导入幂等 |
| `extractor.py` | 从 trajectory JSON 提取 signature 与 block path |
| `consumer.py` | 使用服务端 Key ID 和已加载 Prompt 调用 Reflection URL，返回统一结果；不得内嵌 Prompt 文本 |
| `prompt_loader.py` | 从本地 prompt 目录加载、校验和计算版本指纹，不提供业务默认 Prompt |
| `prompt/` | Reflection Prompt 的本地运行时目录，不纳入 Git 管理 |
| `worker_manager.py` | 管理后台 worker 生命周期、Run 级停止事件和状态恢复 |
| `worker.py` | 原子领取 task、调用 consumer、写入成功或失败结果 |
| `retry.py` | 自动重试、手动重试和失败任务批量重置规则 |
| `merger.py` | 将 processed text 以结构化 `reflect` 字段合并到原始 JSON 副本 |
| `result_exporter.py` | 输出 `--thinking.json`、数据集清单和失败报告 |
| `obs_uploader.py` | 将完成产物上传至 Run 快照中的 OBS 目录 |
| `viewer.py` | 为主表格和数据集详情页组装数据，不包含 HTML 字符串 |
| `cli.py` | 运维、恢复和离线调试入口，与 Web 调用同一 service |

前端与模板建议：

```text
templates/thinking.html                 # 数据集表格主页面
templates/thinking_dataset.html         # 数据集样本详情页
static/js/thinking/index.js             # 表格、筛选、轮询和行操作
static/js/thinking/dataset.js           # 样本列表、转换内容与问题联动
static/css/thinking.css
```

可从现有 `signature_service` 迁移 `extract.py`、`consumer.py`、`workers.py`、`merge.py` 和 `export.py` 的核心逻辑，但应先去除 CLI/独立 Web UI 假设，再通过上述模块接入代理仓。现有 `utils/export_store.py`、`utils/export_routes.py`、`utils/key_store.py` 和 `utils/obs_utils.py` 继续作为已有系统边界，不复制第二套实现。

### 12.1 Prompt 目录与加载规则

Reflection Prompt 必须与 Python 业务代码分离，放在：

```text
src/thinking_reflection/prompt/
```

该目录属于部署环境的本地敏感配置，不由 Git 管理。仓库 `.gitignore` 增加：

```gitignore
/src/thinking_reflection/prompt/
```

不提交真实 Prompt，也不在 `consumer.py`、环境变量示例、测试快照或日志中复制 Prompt 内容。部署时由运维在目标机器创建目录并写入对应文件。

建议按 method 使用独立文件：

```text
prompt/bulk.txt
prompt/sentence.txt
```

加载规则

1. `prompt_loader.py` 根据 Run 的 `method` 选择文件。
2. 路径必须解析在配置的 prompt 根目录以内，禁止请求参数传入任意文件路径。
3. 文件缺失、为空或不可读时，单条测试和 Run 启动均立即失败，不提供代码内置兜底 Prompt。
4. Run 创建/启动时计算 Prompt 内容的 SHA-256，仅保存 `prompt_name`、`prompt_sha256` 和加载时间，不把 Prompt 正文写入数据库。
5. worker 启动时加载一次并使用该 Run 的内存快照；运行期间文件变化不影响已启动 Run。
6. 新 Run 重新加载文件，因此历史 Run 可以通过哈希追溯使用的 Prompt 版本。
7. API 和前端只展示文件名、是否可用和哈希短值，不提供 Prompt 正文读取接口。

Run 配置快照增加：

```text
prompt_name
prompt_sha256
prompt_loaded_at
```

测试环境应在临时目录动态创建 Prompt fixture，测试结束后删除，避免测试依赖开发者机器上的真实 Prompt。

### 12.2 实例隔离的数据库与日志目录

Thinking 的数据库、worker 日志、单条测试日志和运行时状态应放在当前代理实例的 `service_log_dir` 下：

```text
<service_log_dir>/thinking/
├── thinking.db
├── worker/
│   └── <run_id>.log
├── test/
│   └── <test_id>.log
├── runs/
│   └── <run_id>/
│       ├── manifest.json
│       └── failure_report.json
└── tmp/
```

结合当前 `get_service_log_dir()`，实际形态类似：

```text
logs/port<port>/<env-key>/thinking/thinking.db
logs/port<port>/<env-key>/thinking/worker/<run_id>.log
```

如果后续统一把实例目录命名调整为 `port-<port>`，则对应形态为：

```text
logs/port-<port>/<env-key>/thinking/...
```

实现时不要自行拼接 `logs/port...`，必须调用现有 `get_service_log_dir()`，再追加 `thinking/`。这样目录命名规则变化时 Thinking 模块无需修改。

建议配置：

```python
service_log_dir = Path(get_service_log_dir())
thinking_runtime_dir = service_log_dir / "thinking"
thinking_db_path = thinking_runtime_dir / "thinking.db"
```

目录与数据隔离规则：

1. 每个代理端口和 env/key 实例使用独立 `thinking.db`，不得跨实例共用 SQLite 连接。
2. 数据库中仍保存 `source_export_id` 和 `quality_record_id`；它们只引用同一实例的 `export_session_record.db`。
3. worker 和单条测试日志按 Run/Test 分文件，数据库仅保存日志路径和结构化摘要，避免大文本撑大 SQLite。
4. 服务启动时创建所需目录并初始化数据库；目录不可写时禁止启动 Thinking worker并展示明确错误。
5. `tmp/` 只存原子导出和上传中间文件，成功后清理；不能将最终产物只保存在临时目录。
6. `thinking/` 位于已有 `logs/` 忽略范围内，不纳入 Git 管理。

最终转换 JSON 的业务输出仍写入配置的 `logs_session_eval/reflection/...` 并可上传 OBS；`service_log_dir/thinking/` 保存的是实例运行状态、数据库、日志和审计信息，两者职责不同。

### 12.3 UI 一致性约束

Thinking 页面继承现有 `templates/_layout.html`，侧边栏只增加“任务 / Thinking 解析”入口。具体实现遵守：

- 使用 `static/css/tokens.css` 中已有颜色、间距、圆角、字体和暗色主题变量。
- 主表格复用现有 `.data-table`、badge、分页、空状态和 loading 形式。
- 创建任务与配置使用现有 Modal；运行日志、失败详情和 OBS 信息使用现有 Drawer。
- 使用与 Session 导出一致的状态颜色和措辞；Thinking 特有状态只扩展 badge，不重新设计组件。
- 沿用现有登录 session、角色和 permission 判断，新增独立 `thinking` permission。
- 移动端沿用主布局的 sidebar 和横向表格滚动策略。
- 不嵌入 `chat-log-viewer` 的整页 HTML；只迁移或抽取消息渲染能力到数据集详情页，避免两套导航和样式叠加。

## 13. 单条数据集的完整流转

这里的“一条数据”指 Thinking 主表格中的一条导出数据集记录，即一个 `source_export_id`。数据集内部包含多个 Session/Trajectory，每个 Trajectory 又可能包含多个 signature task。

### 13.1 阶段一：Session 导出

```text
用户选择 source_key + mtime_dirs
  -> POST /api/export/run (auto_eval=false)
  -> export_records 新建 mode=export, status=pending
  -> 后台 _run_task 将状态改为 running
  -> export_session_index() 生成/读取 session 索引
  -> sync_session_index() 按 source_key 过滤并复制 Session 文件
  -> 本地输出目录与 OBS 目标写入 export_records
  -> 完成后 status=success
```

现有代码默认写入 `logs_session/...`。本方案要求 Thinking 使用的默认本地根目录为 `logs_session_eval`，实现时需要明确是修改现有导出任务的可配置 `local_root`，还是在创建 Thinking Run 时从现有 `local_copy_dir` 复制/整理到 `logs_session_eval/reflection/...`。推荐后者，避免改变已有 Session 导出的行为。

只有满足以下条件的 export 记录才能发起质检，但还不能直接创建 Thinking Run：

```text
mode = export
status = success
local_copy_dir 存在且可读
```

### 13.2 阶段二：质检

```text
用户对 export 记录发起质检
  -> POST /api/export/eval {record_id: source_export_id}
  -> 新建 mode=eval 记录，并写 source_export_id
  -> reformat_and_analyze() 整理每个 Session 的 trajectory
  -> evaluate_sessions() 生成：
       session_report.md
       session_report.html
       session_analysis.json
  -> analysis_json 内容保存到 eval 记录
  -> 可选上传 OBS session_analysis 路径
  -> eval status=success/failed
```

质检是 Thinking 转换的强制前置条件，流程必须串行，不允许 Thinking 直接消费普通导出目录，也不允许质检与 Thinking 转换并行。

能够创建 Thinking Run 的输入必须是关联 eval 记录的质检目录，并同时满足：

```text
eval.mode = eval
eval.source_export_id = export.id
eval.status = success
eval.local_copy_dir 存在且可读
eval.local_copy_dir/session_report.xlsx 存在
eval.local_copy_dir/session_analysis.json 存在
eval.local_copy_dir/<Session>/*.json 至少存在一个
```

质检阶段的 `reformat_and_analyze()` 会把原始 request/headers/response 三元组整理成 Thinking 服务需要的合并 trajectory JSON，`evaluate_sessions()` 再生成 `session_report.xlsx` 和 `session_analysis.json`。因此 Thinking importer 的数据根目录必须取：

```text
quality_record.local_copy_dir
```

而不是：

```text
source_export_record.local_copy_dir
```

主表格规则：

- `未质检`：禁用“创建任务”，只显示“开始质检”。
- `质检中`：禁用“创建任务”，展示质检进度。
- `质检失败`：禁用“创建任务”，提供查看错误和重新质检。
- `已质检`但目录校验失败：显示“质检产物不完整”，禁止创建任务。
- 只有质检成功且目录产物校验通过时，启用“单条测试”和“创建任务”。

### 13.3 阶段三：创建 Thinking Run 与导入

```text
用户在主表格点击“创建任务”
  -> 后端根据 source_export_id 查找最新成功的 eval 记录
  -> 校验质检目录和必要产物
  -> 选择 reflection_api_key_id、URL、模型、worker 和重试策略
  -> 可先执行单条测试
  -> POST /api/reflection/runs
  -> 创建 reflection_runs 配置快照
  -> importer 读取 quality_record.local_copy_dir
  -> 读取 session_report.xlsx 作为 Session 索引
  -> 遍历数据集内 Session/Trajectory JSON
  -> 写 run_trajectories
  -> extractor 为每个 signature 写 thinking_tasks(status=pending)
```

幂等键建议为：

```text
run_trajectory: UNIQUE(run_id, trajectory_path)
thinking_task:  UNIQUE(run_id, trajectory_id, block_path)
```

没有 signature 的 Trajectory 仍应登记，但 task 数为 0，并显示“未发现 thinking”，以便数据集 Session 总数与导出记录保持一致。

### 13.4 阶段四：Thinking 转换

```text
启动 Run
  -> worker_manager 创建 Run 级 worker 池
  -> worker 原子领取 pending task
  -> pending -> processing
  -> 通过 reflection_api_key_id 在服务端读取完整 Key
  -> consumer 调用配置的 URL/模型
  -> 成功：processed_text 写库，task -> done
  -> 失败且未达上限：记录 attempt，task -> pending
  -> 达到上限：记录 attempt，task -> failed
```

每次调用都写 `task_attempts`，主表格按 task 状态实时聚合：

```text
progress = (done + failed) / total_signature_tasks
```

`failed` 是已结束但未成功，必须单独显示，不能计入成功。Run 最终状态为 `completed` 或 `completed_with_failures`。

### 13.5 阶段五：动态查看与结果导出

转换过程中不必等待整个数据集结束：

```text
主表格
  -> 聚合当前 Run 的 pending/processing/done/failed

数据集详情页
  -> 左侧按 Session 聚合 task 数与质检标识
  -> 中间使用 raw_json + 当前 task 结果动态 merge
  -> 右侧合并 task 错误、attempt 日志和 session_analysis 数据
```

Run 完成或用户主动导出时：

```text
merger 读取原始 trajectory + task results
  -> 为对应 thinking block 添加 reflect
  -> result_exporter 写入 logs_session_eval/reflection/<key_slot>/<run_id>/
  -> 生成数据集 manifest 和失败报告
  -> obs_uploader 上传到 <obs_base>/reflection/<key_slot>/<run_id>/
  -> 写 run_trajectories.output_path / exported_at
```

原始 Session 导出文件保持不变。动态查看和最终文件必须使用同一 `merger`，避免界面显示内容与实际导出 JSON 不一致。

### 13.6 状态与关联主线

```text
export_records(id=source_export_id, mode=export)
  ├── export_records(mode=eval, source_export_id)
  │     ├── session_report.xlsx + <Session>/*.json -> Thinking 强制输入
  │     ├── session_analysis.json -> Session 质检详情
  │     └── reflection_runs(source_export_id, quality_record_id)
  │           └── run_trajectories
  │                 └── thinking_tasks
  │                       └── task_attempts
  └── 未通过质检时不得创建 reflection_runs
```

主表格以 export 记录为基表，左连接最新 eval 和最新 Reflection Run；Reflection Run 还必须保存 `quality_record_id`，确保运行时和历史审计都能准确知道使用了哪一次质检产物。数据集详情页再按 Session 聚合 trajectory、task 和 analysis。

严格状态主线为：

```text
export pending -> running -> success
  -> quality pending -> running -> success + artifacts_valid
    -> reflection draft -> queued -> running
      -> completed / completed_with_failures
```

任一前置阶段失败时，只能在该阶段重试，不能越过质检直接进入 Thinking 转换。

## 14. 最终建议

采用“LLM Proxy 主侧边栏新增任务分组 + Thinking 解析页面内部二级 Tab”的方案：

- LLM Proxy 的 Session 统计、Session 导出和对话历史继续负责原始数据入口。
- Reflection 负责导入、运行配置、后台执行、重试、产物和运行历史。
- `chat-log-viewer` 仅复用消息、Trajectory 和 thinking 的可视化能力，不作为主产品层级。
- 两者通过 `session_id`、`trajectory_id`、`run_id` 和 `source_export_id` 建立可追踪关联。

该方案保持产品体验统一，同时为长时间后台任务和多批次执行记录提供独立、清晰的操作空间。
