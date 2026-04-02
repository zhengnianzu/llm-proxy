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

# 2. 设置默认配置（只需执行一次）
server config configs/server.yaml
sync config configs/sync_config.yaml
client config configs/client.yaml

# 3. 使用简化命令管理服务
server start
server status
server logs -n 100
server stop

sync start
sync status
sync logs -n 100
sync stop

client start
client status
client logs -n 100
client stop
```

**完整用法：**

```bash
# 方式 1：使用 shell 函数（需先 source env.sh）
server start [--config configs/server.yaml]
server stop
server restart
server status
server logs [--lines 50]
server config <path>           # 设置默认配置
server config --clear          # 清除默认配置

sync start [--config configs/sync_config.yaml] [--once]
sync stop
sync restart
sync status
sync logs [--lines 50]
sync config <path>

client start [--config configs/client.yaml]
client stop
client restart
client status
client logs [--lines 50]
client config <path>

# 方式 2：使用可执行脚本
./server start
./sync start
./client start

# 方式 3：使用 Python 模块
python3 -m src.cli server start --config configs/server.yaml
python3 -m src.cli sync start --config configs/sync_config.yaml
python3 -m src.cli client start --config configs/client.yaml
```

**运行产物规则：**

- `server` 日志文件按端口区分：
  - `logs/server-port8080.log`
  - `logs/server-port9000.log`
- `server` PID 文件也按端口区分：
  - `logs/server-port8080.pid`
  - `logs/server-port9000.pid`
- `sync` 日志文件按同步间隔区分：
  - `logs/sync-interval3600.log`
- `sync` PID 文件：
  - `logs/sync.pid`
- `client` 日志和 PID 文件：
  - `logs/client.log`
  - `logs/client.pid`

**配置管理说明：**

- 使用 `config` 子命令可保存默认配置路径，后续操作无需重复传 `--config`
- 配置信息保存在 `.cli_state.yaml` 文件中
- 如需临时覆盖，仍可继续传 `--config` 参数
- 使用 `config --clear` 可清除已保存的默认配置

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

从华为云 OBS 增量下载会话目录到本地，支持守护进程模式。

**用法：**

```bash
# 推荐：使用 CLI 管理（通过 YAML 配置）
client config configs/client.yaml
client start

# 或直接运行 Python 模块
python3 -m src.client \
    --obs-path obs://bucket/sessions \
    --output ./local_sessions \
    --interval 3600
```

**YAML 配置示例（configs/client.yaml）：**

```yaml
obs_path: obs://bucket/sessions    # OBS 源路径
output: ./local_sessions           # 本地输出目录
download_script: obsutil           # 下载脚本（obsutil 或自定义）
workers: 4                         # 并发下载线程数
interval: 3600                     # 同步间隔（秒）
log_level: INFO
```

**工作原理：**

1. 从 OBS 下载 `index.jsonl`（或 `index.json`）获取会话列表
2. 比对本地已有会话，识别新增或更新的会话
3. 并发下载新增/更新的会话文件夹
4. 支持基于 `latest_file` 时间戳的增量同步
5. 守护进程模式下定期执行同步

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
