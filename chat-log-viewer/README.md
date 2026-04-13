# Chat Log Viewer

基于 FastAPI 的 LLM 对话日志查看与分析工具集，支持原始日志浏览、会话导出、统计分析、多报告合并及云端同步。

---

## 项目结构

```
chat-log-viewer/
├── src/                   # Python 包：所有核心模块
│   ├── __init__.py
│   ├── cli.py             # 统一管理 CLI：管理 server / sync / client 的启动、停止、日志查看
│   ├── server.py          # 主 Web 服务：浏览原始日志 / 会话目录
│   ├── client.py          # OBS 下载客户端：从 OBS 增量下载会话目录
│   ├── sync_sessions.py   # 守护进程：增量导出 + 上传到 OBS
│   └── utils/
│       ├── __init__.py
│       ├── message_utils.py   # 消息解析工具函数
│       └── triplet_collector.py  # 三元组文件收集 + index.jsonl 读取
├── cli                    # 可执行脚本：统一 CLI 入口
├── server                 # 可执行脚本：server 服务快捷入口
├── sync                   # 可执行脚本：sync 服务快捷入口
├── client                 # 可执行脚本：client 服务快捷入口
├── env.sh                 # Shell 环境加载器：提供 server/sync/client/cli 命令函数
├── report.py              # 报告汇总服务：展示各 key 的分析报告
├── analyze_sessions.py    # CLI：分析会话目录，生成 xlsx/html/md 报告
├── export_sessions.py     # CLI：将原始三元组日志转换为会话目录
├── merge_reports.py       # CLI：合并多份 session_report.xlsx
├── configs/
│   ├── server.yaml        # server 服务配置
│   ├── sync_config.yaml   # sync 服务配置
│   └── client.yaml        # client 服务配置
├── logs/                  # 运行日志和 PID 文件
├── static/
│   └── index.html         # 单页前端应用（SPA）
├── templates/             # Jinja2 模板（HTML / Markdown 报告）
└── update_dir/
    └── obsutil/setup.sh   # obsutil 安装脚本
```

---

## 数据格式

### 原始三元组日志（triplet）

每次 API 调用产生三个文件，存放在 `logs_anthropic/` 目录下：

```
<timestamp>-req.json      # 请求体
<timestamp>-headers.json  # 请求头
<timestamp>-res.json      # 响应体
```

### index.jsonl

增量追加的索引文件，每行一条 JSON 记录，包含 `req_file` 字段（相对路径），用于快速增量扫描。

### 会话目录（session dir）

由 `export_sessions.py` 导出，结构如下：

```
<output_dir>/
    <session_id>/
        <timestamp>-req.json
        <timestamp>-res.json
        ...
    index.json    # 会话索引，记录每个会话的元信息
```

---

## 各模块功能说明

### cli — 服务管理入口

统一管理 `server`、`sync` 和 `client` 的启动、停止、重启、状态和日志查看。

**快速开始（推荐）：**

```bash
# 1. 加载 shell 环境（在项目根目录执行）
source env.sh

# 2. 注册默认实例（只需执行一次）
server config configs/server.yaml --name main
sync config configs/sync_config.yaml --name sync-main
client config configs/client.yaml --name client-main

# 3. 使用简化命令管理服务
server start
server status
server logs -n 100
server list

sync start
sync status
sync logs -n 100
sync list

client start
client status
client logs -n 100
client list

# 4. 全局查看所有实例
cli list
```

**完整用法：**

```bash
# 方式 1：使用 shell 函数（需先 source env.sh）
server start [--config configs/server.yaml] [--name main]
server stop [--config ... | --name main]
server restart [--config ... | --name main]
server status [--config ... | --name main]
server logs [--lines 50] [--follow] [--config ... | --name main]
server list [--running]
server config <path> [--name main]    # 注册/更新实例，并设为默认
server config --clear                 # 清除默认实例
server config --clear --name main     # 删除实例记录

sync start [--config configs/sync_config.yaml] [--name sync-main] [--once]
sync stop [--config ... | --name sync-main]
sync restart [--config ... | --name sync-main]
sync status [--config ... | --name sync-main]
sync logs [--lines 50] [--follow] [--config ... | --name sync-main]
sync list [--running]
sync config <path> [--name sync-main]

client start [--config configs/client.yaml] [--name client-main]
client stop [--config ... | --name client-main]
client restart [--config ... | --name client-main]
client status [--config ... | --name client-main]
client logs [--lines 50] [--follow] [--config ... | --name client-main]
client list [--running]
client config <path> [--name client-main]

cli list [--service server|sync|client] [--running]

# 方式 2：使用可执行脚本
./server start
./sync start
./client start

# 方式 3：使用 Python 模块
python3 -m src.cli server start --config configs/server.yaml
python3 -m src.cli sync start --config configs/sync_config.yaml
python3 -m src.cli client start --config configs/client.yaml
python3 -m src.cli list --running
```

**运行产物规则：**

- 每个实例都有独立 PID 和日志文件，文件名带实例标识。
- `server` 默认形态：
  - `logs/server-<instance>-port8080.pid`
  - `logs/server-<instance>-port8080.log`
- `sync` 默认形态：
  - `logs/sync-<instance>-export-interval3600.pid`
  - `logs/sync-<instance>-export-interval3600.log`
- `client` 默认形态：
  - `logs/client-<instance>-once.pid`
  - `logs/client-<instance>-once.log`

**配置管理说明：**

- 使用 `config <path> [--name ...]` 可注册实例并保存为默认实例。
- 实例主键默认是配置文件路径，`--name` 只是便于查找和操作。
- CLI 状态保存在 `.cli_state.yaml`，里面会记录各服务的多个实例。
- 不传 `--config`/`--name` 时，`start/stop/status/logs` 默认作用于该服务的默认实例。
- `list` 可快速查看当前登记的实例和运行中的 PID。
- `sync`/`client` 会拦截明显冲突的输出目录，避免两个实例同时写同一目录。

**env.sh 说明：**

`env.sh` 提供了便捷的 shell 函数，自动在项目根目录执行命令：

```bash
source env.sh
# 之后可直接使用 server、sync、client、cli 命令
# 无需关心当前工作目录，命令会自动在项目根目录执行
```

---

### server — 日志浏览服务

启动 FastAPI Web 服务，提供单页前端界面用于浏览 API 调用日志。

**支持两种模式（不可混用）：**

- **扫描模式**：直接扫描目录下的 JSON 文件
- **会话模式**：读取预导出的会话目录（含 `index.json`）

**用法：**

```bash
# 推荐：使用 CLI 管理（通过 YAML 配置）
server config configs/server.yaml
server start

# 或直接运行 Python 模块
python3 -m src.server --dir /path/to/logs
python3 -m src.server --dir /path/a --dir /path/b

# 扫描模式：指定父目录，其所有子目录均作为扫描根
python3 -m src.server --dirs /path/to/parents

# 会话模式：指定预导出的会话目录
python3 -m src.server --session-dir /path/to/session_dir

# 会话模式：指定父目录，其所有子目录均作为会话根
python3 -m src.server --session-dirs /path/to/session_parents

# 通用参数
python3 -m src.server --dir /path/to/logs --port 9000 --host 127.0.0.1
```

**API 端点：**

| 端点 | 说明 |
|------|------|
| `GET /` | 返回前端 SPA 页面 |
| `GET /api/config` | 返回当前模式和目录列表 |
| `GET /api/list?dir=<key>` | 返回指定目录的日志条目列表 |
| `GET /api/file?rel_path=<path>&dir=<key>` | 返回单条日志的完整消息内容 |
| `GET /api/refresh?dir=<key>` | 增量刷新指定目录的缓存 |

**前端功能：**

- 左侧面板：会话列表 + 搜索框 + 目录切换下拉框
- 右侧面板：渲染对话消息（支持 tool_use / tool_result 块）

---

### report.py — 报告汇总服务

扫描 `report/` 目录下各 key 子目录中的 `session_report.xlsx`，汇总统计指标并展示总览页。每次访问主页时自动检测目录变化并重新生成 `overview.html`。

**用法：**

```bash
python report.py <report_dir> [--port 8080] [--host 0.0.0.0]

# 示例
python report.py ./report
python report.py /data/reports --port 9000 --host 127.0.0.1
```

**report_dir 结构：**

```
report/
    key1/
        session_report.html
        session_report.xlsx
    key2/
        session_report.html
        session_report.xlsx
    overview.html    # 自动生成
```

**展示指标：**

- 每个 key 的 session 数量
- 任务完成率（成功率）
- 工具成功率
- 链接到各 key 的子报告及全量合并报告

---

### analyze_sessions.py — 会话分析

对会话目录进行统计分析，输出详细报告。

**用法：**

```bash
python analyze_sessions.py <session_dir> [--out <output_dir>]

# 示例
python analyze_sessions.py ./sessions/key1
python analyze_sessions.py ./sessions/key1 --out ./reports/key1
```

**输出文件（默认写入 `<session_dir>/stat/`）：**

| 文件 | 说明 |
|------|------|
| `session_report.xlsx` | 详情 + 分布统计（多 sheet） |
| `session_report.html` | HTML 可视化报告 |
| `session_report.md` | Markdown 报告 |

**统计指标：**

- 总 session 数、任务完成率
- API 调用次数、错误次数
- 用户轮次、消息总数
- tool_use / tool_result 次数及工具成功率
- 对话时长分布、API call 次数分布等

---

### export_sessions.py — 会话导出

将 `logs_anthropic/` 下的原始三元组日志按 Q1（首条用户消息）分组，导出为会话目录格式。

**用法：**

```bash
python export_sessions.py <logs_dir> --out <output_dir>

# 增量导出（基于上次导出状态）
python export_sessions.py <logs_dir> --out <output_dir> --base-output <prev_output_dir>

# 示例
python export_sessions.py ./logs_anthropic --out ./sessions/key1
```

**主要参数：**

| 参数 | 说明 |
|------|------|
| `logs_dir` | 原始三元组日志根目录 |
| `--out` | 输出会话目录 |
| `--base-output` | 上次导出的基准目录，用于增量模式 |

**输出：**

- 每个会话一个子目录，内含对应的 req/res JSON 文件
- `index.json`：会话索引，记录 Q1、文件列表、消息数等元信息

---

### merge_reports.py — 报告合并

将多份 `session_report.xlsx` 合并为统一报告。

**用法：**

```bash
python merge_reports.py <xlsx1> <xlsx2> ... --out <output_dir>

# 示例
python merge_reports.py reports/*.xlsx --out /tmp/merged
python merge_reports.py key1/session_report.xlsx key2/session_report.xlsx --out merged/
```

**输出文件：**

| 文件 | 说明 |
|------|------|
| `merged_session_report.xlsx` | 合并详情（含来源列）+ 分布统计 + 来源汇总 |
| `merged_session_report.html` | 合并后的 HTML 报告 |
| `merged_session_report.md` | 合并后的 Markdown 报告 |

---

### sync — 云端同步（上传）

守护进程模式，将本地会话目录增量上传至华为云 OBS。

**用法：**

```bash
# 推荐：使用 CLI 管理（通过 YAML 配置）
sync config configs/sync_config.yaml
sync start

# 单次运行模式（测试/cron）
sync start --once

# 或直接运行 Python 模块
python3 -m src.sync_sessions \
    --src ./sessions \
    --session-dir ./sessions \
    --obs-session obs://bucket/sessions \
    --interval 3600
```

**YAML 配置示例（configs/sync_config.yaml）：**

```yaml
src: ./sessions                    # 本地会话目录
session_dir: ./sessions            # 会话目录（同 src）
obs_session: obs://bucket/sessions # OBS 目标路径
interval_seconds: 3600             # 同步间隔（秒）
upload_erase: false                # 上传后是否删除本地文件
upload_script: obsutil             # 上传脚本（obsutil 或自定义）
upload_workers: 4                  # 并发上传线程数
log_level: INFO
```

**依赖：** 需安装 `obsutil`（见下方安装说明）。

---

### client — OBS 下载客户端

从华为云 OBS 增量下载 session 目录或 raw 文件到本地，支持守护进程模式。

**用法：**

```bash
# 推荐：使用 CLI 管理（通过 YAML 配置）
client config configs/client.yaml
client start

# 或直接运行 Python 模块
python3 -m src.client \
    --mode raw \
    --obs-path obs://bucket/sessions/ \
    --output ./local_sessions
```

**YAML 配置示例（configs/client.yaml）：**

```yaml
mode: raw                          # session | raw，默认 session
obs_path: obs://bucket/sessions/   # OBS 源路径，必须以 obs:// 开头
output: ./local_sessions           # 本地输出目录
base_output: ./local_sessions      # 可选：用已有输出目录的 index 状态做增量同步
download_script: tools/obs_download.sh  # 可选，默认 obs_download.sh
workers: 4                         # 可选，默认 4
interval: 10                       # 可选；不填则只执行一次
```

**字段说明：**

1. `mode`
   `session` 模式按索引里的 `folder` 字段下载整个 session 目录；`raw` 模式按 `req_file/path/file/rel_path` 下载对应原始请求族文件。
2. `obs_path`
   OBS 根路径，代码会自动补齐结尾 `/`。
3. `output`
   本地输出目录，下载结果和本地状态文件都会写到这里。
4. `base_output`
   可选。指定后会使用该目录已有的 index 状态作为增量起点，只处理后续新增内容。
5. `download_script`
   下载脚本签名为 `<script> <obs_path> <local_path>`；当前仓库默认脚本是 `tools/obs_download.sh`。
6. `interval`
   不设置时只执行一次；设置后按秒轮询。

**工作原理：**

1. 从 OBS 下载 `index.jsonl`；失败时回退尝试 `index.json`
2. 根据 `mode` 解析索引，提取待下载目标
3. 基于本地状态文件做增量同步，避免重复处理旧索引行
4. 并发下载新增目标
5. 设置 `interval` 时循环执行同步

**依赖：** 需安装 `obsutil`（见下方安装说明）。

---

### obsutil 安装

```bash
bash update_dir/obsutil/setup.sh
```

该脚本将 obsutil 二进制文件复制到 `/obsutil` 并写入 `/etc/profile`，使其在全局可用。

---

## 工具函数

### src/utils/message_utils.py

| 函数 | 说明 |
|------|------|
| `load_json(path)` | 加载 JSON 文件 |
| `extract_messages(data)` | 从请求数据中提取消息列表 |
| `get_first_user_text(messages)` | 获取第一条用户消息的文本内容 |
| `count_user_messages(messages)` | 统计用户消息数量 |
| `parse_response(data)` | 解析响应数据，提取 assistant content |

### src/utils/triplet_collector.py

| 函数 | 说明 |
|------|------|
| `collect_triplets(root, start_line)` | 收集三元组文件路径（优先读 index.jsonl）|
| `read_session_index(session_dir, start_line)` | 增量读取会话目录的 index.jsonl |

---

## 依赖安装

```bash
pip install fastapi uvicorn[standard] jinja2 pandas openpyxl
```

> `sync_sessions.py` 额外依赖：`pyyaml`
>
> ```bash
> pip install pyyaml
> ```

---

## 典型工作流

### 0. 环境准备

```bash
# 加载 shell 环境（推荐）
cd /path/to/chat-log-viewer
source env.sh

# 设置默认配置
server config configs/server.yaml
sync config configs/sync_config.yaml
client config configs/client.yaml
```

### 1. 浏览原始日志

```bash
# 使用 CLI 管理
server start
# 打开 http://localhost:8080

# 或直接运行
python3 -m src.server --dir ./logs_anthropic --port 8080
```

### 2. 导出并分析会话

```bash
# 导出
python export_sessions.py ./logs_anthropic --out ./sessions/key1

# 分析
python analyze_sessions.py ./sessions/key1

# 查看报告
open ./sessions/key1/stat/session_report.html
```

### 3. 浏览会话视图

```bash
# 修改 configs/server.yaml 为会话模式，然后
server restart

# 或直接运行
python3 -m src.server --session-dir ./sessions/key1 --port 8080
```

### 4. 合并多 key 报告并查看汇总

```bash
# 合并报告
python merge_reports.py sessions/key1/stat/session_report.xlsx sessions/key2/stat/session_report.xlsx --out report/

# 启动报告服务
python report.py ./report --port 8081
# 打开 http://localhost:8081
```

### 5. 开启云端自动同步

```bash
# 上传到 OBS
sync start

# 从 OBS 下载
client start

# 查看运行状态
sync status
client status

# 查看日志
sync logs -n 100
client logs -n 100
```
