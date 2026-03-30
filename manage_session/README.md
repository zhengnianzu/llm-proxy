# manage_session

LLM 测试任务与执行轨迹的匹配管理系统。

将批次任务文件（NDJSON）与执行轨迹 index（JSON / xlsx）做精确匹配，追踪每个任务是否已有对应轨迹，并通过 Web UI 展示完成率和 topic 分布。

---

## 目录结构

```
manage_session/
├── tasks/                  # 任务批次文件（NDJSON，每行一个 query）
├── raw_index/              # 执行轨迹 index 文件
│   └── <session_name>/
│       ├── index.json      # JSON 格式 index
│       └── *.xlsx          # xlsx 格式 index（读取 Q1首问 列）
├── manifests/              # 元数据缓存（自动维护，勿手动修改）
│   ├── tasks.json
│   └── indexes.json
├── pair_cache/             # task × index 匹配结果缓存（自动维护）
├── views/                  # 状态视图（自动生成）
│   ├── batch_status/
│   └── global_summary.json
├── web/
│   ├── server.py           # FastAPI Web UI
│   └── templates/          # Jinja2 模板
├── core/
│   ├── config.py           # 配置加载与路径管理
│   ├── manifest.py         # 自动扫描与元数据注册
│   ├── matcher.py          # 精确匹配引擎
│   ├── cache.py            # 匹配缓存管理
│   └── views.py            # 状态视图生成
├── cli.py                  # CLI 入口
├── config.yaml             # 配置文件
└── requirements.txt
```

---

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 准备数据

将任务文件放入 `tasks/`，将轨迹 index 放入 `raw_index/<session_name>/`：

```
tasks/
  260325-7840-blue-linux.json      # NDJSON，每行含 {"query": "...", "topic": "...", ...}

raw_index/
  test_session2/
    index.json                     # JSON 数组，每条含 {"q1": "...", ...}
  another_session/
    data.xlsx                      # xlsx，含 "Q1首问" 列
```

文件放好后无需任何注册命令，系统会自动扫描。

### 运行匹配

```bash
# 增量匹配（跳过已有缓存的组合）
python cli.py match

# 全量重算（清空缓存后重跑）
python cli.py match --full
```

### 启动 Web UI

```bash
python web/server.py
# 或指定端口
python web/server.py --port 8081
```

访问 http://localhost:8081

---

## CLI 命令

| 命令 | 说明 |
|------|------|
| `python cli.py match` | 自动扫描并增量匹配 |
| `python cli.py match --full` | 清空缓存后全量重匹配 |
| `python cli.py status` | 列出所有批次完成率 |
| `python cli.py status --task <task_id>` | 查看某批次详情 |
| `python cli.py summary` | 打印全局汇总 |
| `python cli.py list-caches` | 列出所有匹配缓存 |
| `python cli.py clear-cache` | 清空所有缓存（需确认） |

---

## Web UI 页面

| 路径 | 说明 |
|------|------|
| `/` | 全局汇总：KPI 指标、批次列表、index 列表 |
| `/batch/<task_id>` | 批次详情：匹配来源、topic 分布（已匹配 / 未匹配 / 合计） |
| `/report-html/<index_id>` | 查看 xlsx index 同目录下的 `session_report.html` |
| `/api/summary` | 全局汇总 JSON |
| `/api/batch/<task_id>` | 批次状态 JSON |

---

## 数据格式

### 任务文件（tasks/*.json）

NDJSON 格式，每行一个对象，必须含 `query` 字段：

```json
{"query": "如何查看系统日志", "topic": "运维", "env_name": "linux", ...}
```

### JSON index（raw_index/**/*.json）

JSON 数组，每条必须含 `q1` 字段：

```json
[{"q1": "如何查看系统日志", ...}, ...]
```

### xlsx index（raw_index/**/*.xlsx）

表格文件，必须有名为 `Q1首问` 的列。如果同目录下存在 `session_report.html`，Web UI 会自动显示"查看报告"链接。

---

## 配置

编辑 `config.yaml` 修改目录路径或 Web 端口：

```yaml
dirs:
  tasks: tasks
  raw_index: raw_index
  manifests: manifests
  pair_cache: pair_cache
  views: views

web:
  host: 127.0.0.1
  port: 8081

matching:
  task_query_field: query   # task 里的查询字段名
  index_query_field: q1     # index 里的查询字段名
  match_type: exact
```

也可通过环境变量指定配置文件路径：

```bash
MANAGE_SESSION_CONFIG=/path/to/config.yaml python cli.py match
```

---

## 增量匹配原理

1. 扫描 `tasks/` 和 `raw_index/`，对每个文件计算 MD5
2. 与 `manifests/` 中的记录对比，仅处理新增或变化的文件
3. 对每对 (task, index) 查找 `pair_cache/`，命中缓存则跳过
4. 未命中则执行精确匹配（`task.query == index.q1`），结果写入缓存
5. 更新 `views/` 下的批次状态和全局汇总
