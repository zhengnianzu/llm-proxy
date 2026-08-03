## 项目介绍

本项目对 Anthropic `/v1/messages`和 OpenAI `/v1/chat/completions` 请求进行了封装，解决了:

1. 日志自动记录
2. token数一键统计

## 配置

- 复制模板

```shell
cp .env.example .env
```

- 配置说明
```text
这两个key配置后强制使用给定值，当值为空时，才选择用户传入的模型
UPSTREAM_API_KEY=sk-xxx
MODEL_ID=gpt-5

# API_KEY 用于接口鉴权，支持多个 key，用逗号分隔
# 不配置则跳过鉴权，任何请求都能访问
API_KEY=key1,key2,key3
```

```shell
# 转发前的地址
UPSTREAM_URL=https://yibuapi.com/v1  # 后缀写到v1
UPSTREAM_API_KEY="填入AK"  # 密钥
MODEL_ID=

# 转发后的地址
PROXY_HOST=127.0.0.1
PROXY_PORT=4000

# 进程数：默认 1（单 worker）。多核机器可调大以吃满 CPU，详见「多 Worker（多进程）」章节
PROXY_WORKERS=1

# 其他参数
SSL_VERIFY=false
BAN_EXPLORE=false
BAN_STREAM=false

# 监控后台登录保护
# 配置这两个值后，会保护 /、/query、/history、/failures、/statistic、/metrics/*、/logs/*
MONITOR_USERNAME=admin
MONITOR_PASSWORD=请改成强密码

# Session Cookie 配置
MONITOR_SESSION_SECRET=请改成一串随机长字符串
MONITOR_COOKIE_SECURE=false
MONITOR_SESSION_MAX_AGE=43200

# 代理相关
TRUST_ENV=true  # 为true时使用下面全局环境变量，密码特殊字符记得转码
HTTP_PROXY=http://华为账号:华为密码@proxyhk.huawei.com:8080
HTTPS_PROXY=http://华为账号:华为密码@proxyhk.huawei.com:8080
NO_PROXY=localhost,127.0.0.1,*.huawei.com,*.local,*.lan,10.70.85.106
```

如果你想让日志目录能区分不同 `.env` 启动的实例，不需要手动配置额外变量。
使用 `./app start --env .env.xxx` 时，CLI 会自动把 env 文件名转换成启动时环境变量 `LOG_TASK_TAG`。

例如：

```text
./app start --env .env.prod
```

会自动生成类似：

```text
LOG_TASK_TAG=env-prod
```

因此日志目录会从：

```text
logs_all/nokey/26040713
```

变成：

```text
logs_all/env-prod-wy92/26040713
```

日志目录采用嵌套结构 `{base}/{tag}-{key_prefix}/{YYMMDDHH}`，按环境和小时自动分组，减少根目录下的文件夹数量。

如果不是通过 `./app` CLI 启动，而是直接 `python app.py`，那就不会自动带这个标记，目录名仍保持原样（只有 key_prefix 部分）。

### 监控后台登录保护

如果这个服务会被外网或其他人访问，建议开启监控后台登录保护。

启用方式：

```text
MONITOR_USERNAME=你的用户名
MONITOR_PASSWORD=你的密码
MONITOR_SESSION_SECRET=随机长字符串
```

启用后：

```text
1. 访问 /、/query、/history、/failures 会先跳转到 /login
2. /statistic、/metrics/*、/logs/*、/docs、/redoc、/openapi.json 也会被保护
3. /v1/messages 和 /chat/completions 仍然继续使用原来的 API_KEY 鉴权，不受这套网页登录影响
```

可选配置：

```text
MONITOR_COOKIE_SECURE=true      # 通过 HTTPS 暴露时建议开启
MONITOR_SESSION_MAX_AGE=43200   # 登录态有效期，单位秒，默认 12 小时
MONITOR_AUTH_ENABLED=true       # 显式开启；默认情况下只要配置了用户名和密码就会自动开启
```

安全建议：

```text
1. 不要只依赖登录页，最好同时加 Nginx/Caddy 反向代理和 HTTPS
2. 对公网暴露时，建议再配 IP 白名单
3. MONITOR_SESSION_SECRET 不要和弱口令一起使用，更不要直接提交到仓库
```

## 环境

```shell
pip install -r requirements.txt
```

## 启动与管理

推荐使用根目录下的 CLI，而不是手动执行 `bash server.sh start .env`。

先给 CLI 执行权限：

```shell
chmod +x app
```

### 1. 配置默认 env

```shell
./app config .env
```

这会把当前默认使用的环境文件写入 `.cli_state.yaml` 的 `source_env` 字段。

查看当前配置：

```shell
./app config
```

### 2. 启动 / 停止 / 重启

```shell
./app start
./app stop
./app restart
```

这些命令默认作用于当前 `source_env`。

### 3. 指定某个 env 操作

```shell
./app start --env .env.test
./app stop --env .env.test
./app restart --env .env.prod
./app logs --env .env.prod -f
```

### 4. 查看日志

```shell
./app logs
./app logs -f
./app logs --env .env.test -n 200
```

### 5. 查看服务状态

```shell
./app status
./app list
```

`status` 会显示当前默认 `source_env`，并列出所有已记录服务。  
`list` 会直接列出 `.cli_state.yaml` 中记录的所有 env 服务。

## 多个 .env 同时运行

CLI 支持多个 `.env` 同时启动，只要它们的 `PROXY_PORT` 不冲突。

例如：

```shell
./app start --env .env
./app start --env .env.test
./app start --env .env.prod
```

每个 env 都会单独记录到 `.cli_state.yaml` 的 `services` 中，字段包括：

```text
env_path
pid
host
port
pid_file
log_file
started_at
```

典型状态示例：

```yaml
source_env: .env
services:
  .env:
    pid: 1234
    host: 127.0.0.1
    port: 4000
    pid_file: logs/app-port4000.pid
    log_file: logs/app-port4000.log
  .env.test:
    pid: 2345
    host: 127.0.0.1
    port: 4001
    pid_file: logs/app-port4001.pid
    log_file: logs/app-port4001.log
```

说明：

```text
1. 不同 env 本质上是不同的 app.py 进程
2. CLI 按 env 文件区分服务
3. stop / restart / logs 可通过 --env 精确作用到某个服务
```

## 多 Worker（多进程）

单 worker 是单事件循环、单核，高并发（约 1000+）下 CPU 会跑满成为瓶颈。多核机器可以用多 worker 吃满 CPU。

### 开启方式

在 `.env` 里设置进程数（不设或设为 1 时行为与旧版完全一致）：

```text
PROXY_WORKERS=4
```

底层用 `uvicorn.run(..., workers=N)` fork 出 N 个 worker 进程共同监听同一个 `PROXY_PORT`，由内核分发连接。经验值：设为 CPU 核数的一半到全部，例如 16 核先试 `4`~`8`，压测看 CPU 和吞吐再调。

### 多 worker 下的行为说明

改造后多个 worker 之间已做好协调，无需额外配置：

```text
1. 后台 metrics 扫描线程只在抢到文件锁的那个进程里跑（scanner.lock），
   避免 N 个进程重复写 rpm.log / rate.log / scanner_state.json
2. 请求日志文件名带进程号后缀，多 worker 并发落盘不会互相覆盖
3. /metrics/index-stats 的首次/总体/有效计数改为读磁盘 index 聚合，
   反映所有 worker 的全局真值（不再只统计单进程）
4. /metrics/live 的实时在途请求聚合所有存活 worker 的心跳，
   死掉的 worker 会被自动跳过
5. 导出任务全局串行执行，取消操作跨 worker 生效
```

### 停止 / 重启

多 worker 下一个服务是「1 个 master + N 个 worker」共 N+1 个进程。`./app stop`、`./app restart`
（以及 `server.sh stop/restart`）会按**进程组**停止，master 和所有 worker 一起退出，不会残留
占端口的孤儿进程 —— 服务启动时用独立 session（master 是进程组组长），停止时对整个进程组发信号。

### 注意

- 多 worker 只在直接 `python app.py` 或 CLI 启动时生效；`PROXY_WORKERS` 通过 `.env` 传入。
- 每个 worker 是独立进程、独立内存，共享状态都落在磁盘/SQLite 上，因此调大 worker 数不会导致统计错乱。
- 内存占用大致随 worker 数线性增长，调大前留意机器内存。

## Session 导出与同步

`sess` 命令用于将日志中的请求聚合为 session，导出 `session_index.jsonl`，并可选上传到 OBS。

### 数据流

```text
app.py 写入请求日志
  → logs_all/{env-key}/{mtime}/index.jsonl          原始请求索引
  → logs_all/{env-key}/{mtime}/.session_cache.jsonl  session 聚合缓存（自动生成）
  → logs_all/{env-key}/{mtime}/session_index.jsonl   导出的 session 索引（sess export 生成）
  → OBS obs_base/session/{env-key}/{mtime}/          云端同步（sess export --sync）
```

### 快捷命令

加载 `source env.sh` 后可直接使用 `sess`：

```shell
sess list                    # 列出所有 mtime 目录及导出状态
sess config 26060309         # 切换到指定 mtime 目录
sess export                  # 导出当前 mtime 的 session_index.jsonl
sess export --sync           # 导出并上传三元组到 OBS
sess logs                    # 查看 export 运行日志
```

等价于 `./app sess list`、`./app sess config 26060309` 等。

### 典型工作流

```shell
# 1. 查看当前 env 下有哪些 mtime 目录
sess list --env .env.xunxing

# 输出示例:
# [sess] env-xunxing-zyKA (6 dirs, 5 new)
#   26052819: - (has index.jsonl)
#   26052910: - (has index.jsonl)
# * 26060309: 1 sessions, avg 2 msg, exported=2026-06-04 14:52:22
#   26060317: - (empty)
#   26060414: - (has index.jsonl)

# 2. 选择要处理的 mtime
sess config 26060120 --env .env.xunxing

# 3. 导出 session_index.jsonl
sess export --env .env.xunxing

# 4. 导出并同步到 OBS（需要先 sync config 配置 obs_base）
sess export --sync --env .env.xunxing

# 5. 查看运行日志
sess logs --env .env.xunxing
```

### session_index.jsonl 格式

每行一个 JSON，表示一个 session：

```json
{
  "q1": "用户第一条消息",
  "models": ["claude-sonnet-4-6", "gpt-5.4"],
  "latest_file": "2026-06-02_19-18-18_530-req.json",
  "msg_count": 2,
  "api_key": "",
  "first_ts": "2026-06-01_20-21-41_177",
  "last_ts": "2026-06-02_19-18-18_530",
  "trace_list": [
    {"filename": "...-req.json", "model": "...", "msg_count": 2, "ts": "..."}
  ],
  "_key": "2026-06-01_20-21-41_177"
}
```

末尾一行 `_meta`：

```json
{"_meta": true, "total_sessions": 10, "avg_msg_count": 5, "source_mtime": 1780451536.3, "updated_at": "2026-06-04T10:40:16"}
```

### 云端路径结构

配置 `sync config settings/obs_base.yaml` 后，`sess export --sync` 上传到：

```text
obs_base/session/{env-key}/{mtime}/
  ├── session_index.jsonl
  ├── {ts}-req.json        (latest_file 三元组)
  ├── {ts}-headers.json
  └── {ts}-res.json
```

对比 `sync start` 的 raw 日志上传路径：

```text
obs_base/raw/{env-key}/{mtime}/
  ├── index.jsonl
  ├── .session_cache.jsonl
  ├── {ts}-req.json        (全量三元组)
  ├── {ts}-headers.json
  └── {ts}-res.json
```

### 状态管理

每个 env 的 sess 状态保存在 `logs/port{N}/{env-slug}/sessions.json`：

```json
{
  "current_mtime": "26060309",
  "mtimes": {
    "26060309": {
      "total_sessions": 1,
      "avg_msg_count": 2,
      "exported_at": "2026-06-04 14:52:22",
      "synced_at": "2026-06-04 14:53:00",
      "obs_dst": "obs://bucket/session/env-xxx/26060309/"
    }
  }
}
```

### 与 export_sessions.py / analyze_sessions.py 配合

分析流水线：`export_sessions.py`（转换为 session 文件夹格式）→ `analyze_sessions.py`（分析生成报告）。

`export_sessions.py` 数据源优先级：
1. `session_index.jsonl`（已聚合的 session 索引，快速路径）
2. `index.jsonl`（原始请求索引，需重新聚合）
3. 目录扫描（最后兜底）

```shell
cd chat-log-viewer

# 默认：优先使用 session_index.jsonl（快速路径）
python export_sessions.py --src ../logs_all/env-xunxing-zyKA/26060309 --out /tmp/sessions

# 分析导出后的 session 文件夹
python analyze_sessions.py --dir /tmp/sessions
```

## 测试

```
python test/test-api.py
```

## 离线批量导出（reformat）

`tools/offline_reformat_export.py` 是**网页版 reformat 导出的离线跑版本**：对当前所有 key，逐个 key 做「合并导出（reformat）」，成功后写入与网页同一个 `export_session_record.db`，刷新导出页即在对应 key 卡片下显示为**成功**。

用途：不经过网页/队列，直接在服务器上一次性把所有 key 的 session 合并导出（并可上传 OBS），记录与网页版逐字段对齐（`mode=reformat`、`total_sessions`、`local_copy_dir`、`obs_dst`、`progress_log`、状态判定等）。

### 做什么

严格复刻 `utils/export_routes.py::_run_task_inner` 的 `mode="reformat"` 分支，逐 key 执行：

1. `build_stats_multi` 取到每个 key 的完整 api_key + 全部 mtime 目录；
2. `create_record(mode="reformat")` → `export_session_index` → `_load_session_index` 按 key 过滤 → `reformat_and_analyze(analyze=False)` 合并三元组落本地 → 写 `session_index.jsonl` → 整目录上传 OBS → `update_status("success", …)`。

产物落在 `logs_session_analysis/{env}/{key-slot}/ex-{时间戳}/`，与网页版一致。

### 用法

在项目根目录（`/mnt/llm-proxy-main`）下运行：

```shell
# 所有 key，全部 mtime，上传 OBS（跟网页版完全一致）
python3 tools/offline_reformat_export.py

# 先看计划、不执行（强烈建议全量前先跑一次）
python3 tools/offline_reformat_export.py --dry-run

# 只本地导出，不上传 OBS
python3 tools/offline_reformat_export.py --no-obs

# 只导某个 key —— 三种写法（可多次、可混用）：
python3 tools/offline_reformat_export.py --key sk-xxxxxxxx   # 完整 api_key
python3 tools/offline_reformat_export.py --key key-Kjfu      # key slot（网页卡片上的分组名）
python3 tools/offline_reformat_export.py --key Kjfu          # 后 4 位

# 只导匹配的 mtime 目录（子串匹配，可多次）
python3 tools/offline_reformat_export.py --mtime 260803

# qualified 阈值（同网页，默认 5）
python3 tools/offline_reformat_export.py --threshold 5
```

`--key` 的后三种写法按「api_key 最后 4 位相同」匹配，与网页卡片 `key-XXXX` 的分组口径一致。

### 说明

- **零配置/零凭据**：自动扫 `logs/port*/` 定位服务日志目录（含各 `.db`），读 `logs/app-meta-port*.json` 推出 `ENV_DIR`，与正在运行的服务指向同一套 DB / 同一 logs 根，无需任何环境变量。也可用 `--service-log-dir` / `--env-dir` 手动指定。
- **串行执行**：逐 key 顺序跑（与网页队列 `EXPORT_CONCURRENCY=1` 一致），不会和线上导出撞 OBS / 本地拷贝。
- **全量耗时**：默认对全部 key 上传 OBS，量大耗时长。建议先 `--dry-run` 看计划，或先 `--key` 单个试跑确认 OBS 通畅，再全量。

## 离线批量回填（new-api index）

`scripts/backfill_all.py` 是**数据管理页「回填」的命令行版**：对页面登记的所有源（活跃 env_dir + 历史路径）逐个执行 new-api 富 index（`index.db`）回填，等价于在页面上对每个 new-api 源依次点「同步」+「构建索引」。

用途：不经过网页/后台队列，直接在服务器上把所有源的小时叶子索引一次性补齐（历史预览、导出、质检的前置）。适合 cron 定时补跑或手动排障，**无需服务在运行**。

### 与网页版的关系

- **状态、日志与页面完全同源**：叶子构建状态/计数写进同一个 `log_dir.db`（数据管理页、导出页读同一张表）；生命周期事件写进同一份 `{SERVICE_LOG_DIR}/backfill.log`（页面「构建日志」弹窗读同一文件）。跑完刷新数据管理页即见最新「已回填/总数」。
- **底层同一套**：复用 `utils/newapi_backfill.py` 的 `sync_leaves`（同步清单）+ `_run`（逐叶构建），与页面「同步」「构建索引」按钮走同一代码。
- **关键差异**：网页版由全局调度线程**异步**串行执行（点一下即入队返回）；本脚本在**当前进程内同步**串行执行，一个源跑完再跑下一个，跑完即退出（终端会一直占用直到全部完成——它不是后台进程；构建时 `workers=N` 是叶子内 enrich 的进程池子进程，属正常）。

### 做什么

逐源执行两步（等价页面「同步」+「构建索引」）：

1. `sync_leaves(root)`：扫叶子清单写入 `log_dir.db`，让 DB 先有权威的叶子总数/已建数；
2. `_run(root, workers, force)`：逐叶 `build_leaf`（ingest + enrich 补 meta + 聚合 sessions），状态/计数写 `log_dir.db`，事件写 `backfill.log`。

只对 **new-api** 格式的源回填，非 new-api / 不存在的源自动跳过。构建时每隔几秒打印实时进度：**当前正在构建哪个叶子目录、已处理多少条、叶子级总进度**。

### 用法

在项目根目录（`/mnt/llm-proxy-main`）下运行：

```shell
# 增量回填所有 new-api 源（跳过已完成且无新增的叶子）
python -m scripts.backfill_all

# 先看计划、不执行（不写 DB、不构建；建议全量前先跑一次）
python -m scripts.backfill_all --dry-run

# 全量重建（清掉旧 index.db 重建，用于口径变更 / 修数据）
python -m scripts.backfill_all --force

# 按「标识」筛选源（匹配 name / root_id / 路径子串，可多次，取并集）
python -m scripts.backfill_all --source jumper-003        # 按名称（可能命中多个同前缀源）
python -m scripts.backfill_all --source 438181a9          # 按 root_id（唯一，精确定位单一源）
python -m scripts.backfill_all --source 438181a9 --source proxy-004

# 按目录直接指定源（可多次）
python -m scripts.backfill_all --root /data/logs_all/xxx --root /data/hist/yyy

# 只同步叶子清单到 DB、不真正构建（等价页面「同步」按钮）
python -m scripts.backfill_all --sync-only

# 覆盖叶子内 enrich 进程池并行度（默认 min(8, CPU)）
python -m scripts.backfill_all --workers 4

# 调整/关闭实时进度打印间隔（秒，默认 5，0 关闭）
python -m scripts.backfill_all --progress-interval 2
```

构建进度打印形如：

```text
构建中 [叶子 20/66] 当前=26080100（计数中…）
构建中 [叶子 20/66] 当前=26080100 已处理 400/72268 条
构建中 [叶子 21/66] 当前=26080101 已处理 1200/58133 条
```

### 说明

- **状态同步与服务同源**：`log_dir.db` / `backfill.log` 均在 `SERVICE_LOG_DIR` 下，由 `PROXY_PORT`、`LOG_TASK_TAG`、`UPSTREAM_API_KEY` 等环境变量决定。**务必在与服务相同的工作目录、相同环境变量下运行**，否则会解析到另一套目录、写到别的 `log_dir.db`，页面读不到。
- **`--source` 口径**：命中 name（页面展示名）/ root_id（路径 md5 前 8 位，唯一）/ 路径子串任一即可，多个 `--source` 取并集。name 可能重复（如 `jumper-003` 与 `jumper-003-latest`），要精确定位单一源用 **root_id**；任一 `--source` 匹配不到源即报错退出。`--dry-run` 输出里每行都带 `root_id`，照抄即可。
- **增量 vs 全量**：默认（不带 `--force`）只处理需要构建的叶子（无 `index.db` / `index.jsonl` 有新增 / 有待补 meta / sessions 脏），跑过的不重复。`--force` 清掉旧 `index.db` 全量重建，用于修数据或口径变更。
- **叶子卡住可续跑**：单个叶子若疑似 NFS 读死（`stall_timeout` 内无进展）会跳过并标 error，不拖垮整个源；网络恢复后再跑一次（增量模式会自动重扫补跑这些叶子）。
- **退出码**：0 全部成功；1 有源在回填中报错（root 级崩溃或有失败/卡住的叶子）。

## 统计token数

1. 基于web界面

```text
在代理服务启动后，访问 http://127.0.0.1:4000/
```

2. 基于函数调用

- 命令行调用

```shell
python print_stats_summary.py --date_start=2026-01-01 --date_end=2026-03-09
```

- 参数解释

```text
model: 过滤模型，忽略大小写，多个模型用,拼接
date_start: 过滤日期-开启，格式YYYY-MM-DD
date_end: 过滤日期-结束，格式YYYY-MM-DD
status: 过滤状态: 全部、成功、失败
```

### Token 统计计算规则

代理层记录每次请求时，从上游 API 返回的原始 `usage` 中提取 token 数据，写入 `index.jsonl`。

#### index.jsonl 记录字段

| 字段 | 含义 | 来源 |
|------|------|------|
| `tok_in` | 输入 token（不含缓存） | Anthropic: `input_tokens`；OpenAI: `prompt_tokens` |
| `tok_out` | 输出 token | Anthropic: `output_tokens`；OpenAI: `completion_tokens` |
| `cache_in` | 缓存 token（读取 + 写入） | Anthropic: `cache_read_input_tokens` + `cache_creation_input_tokens`；OpenAI: 0 |
| `usage` | 上游返回的原始 usage 字典 | 完整保留，用于精确计费和审计 |

#### Anthropic（Claude）usage 字段

| 字段 | 示例 | 含义 | 计费归类 |
|------|------|------|----------|
| `input_tokens` | 3 | 非缓存输入 token | 输入 · 标准价 |
| `cache_read_input_tokens` | 71800 | 缓存命中读取 | 输入 · 0.1× 价 |
| `cache_creation_input_tokens` | 529 | 缓存写入 | 输入 · 1.25× 价 |
| `output_tokens` | 251 | 输出 token（含 thinking） | 输出 · 标准价 |
| `output_tokens_details.thinking_tokens` | 114 | 其中思考 token | 已含在 output_tokens |
| `cache_creation.ephemeral_5m_input_tokens` | 529 | 5 分钟 TTL 缓存写入明细 | 已含在 cache_creation_input_tokens |
| `cache_creation.ephemeral_1h_input_tokens` | 0 | 1 小时 TTL 缓存写入明细 | 已含在 cache_creation_input_tokens |
| `iterations[]` | — | 多轮 tool_use 每轮明细 | 已含在顶层汇总 |
| `server_tool_use.web_search_requests` | 1 | 服务端 web_search 调用次数 | 按次独立计费 |
| `server_tool_use.web_fetch_requests` | 1 | 服务端 web_fetch 调用次数 | 按次独立计费 |
| `inference_geo` | "global" | 推理地区 | 不计费 |
| `service_tier` | "standard" | 服务层级 | 不计费 |

**Anthropic 实际输入 token 总量** = `input_tokens` + `cache_read_input_tokens` + `cache_creation_input_tokens`

**Anthropic 输出 token 总量** = `output_tokens`

#### OpenAI / DeepSeek usage 字段

| 字段 | 示例 | 含义 | 计费归类 |
|------|------|------|----------|
| `prompt_tokens` | 14 | 输入 token 总量（含缓存） | 输入 · 标准价 |
| `prompt_tokens_details.cached_tokens` | 0 | 其中缓存命中 | 已含在 prompt_tokens · 0.5× 价 |
| `prompt_cache_hit_tokens` | 0 | DeepSeek 缓存命中 | 同上（DeepSeek 专用字段） |
| `prompt_cache_miss_tokens` | 14 | DeepSeek 缓存未命中 | 已含在 prompt_tokens · 标准价 |
| `completion_tokens` | 26 | 输出 token 总量（含推理） | 输出 · 标准价 |
| `completion_tokens_details.reasoning_tokens` | 24 | 其中推理/思考 token | 已含在 completion_tokens |
| `prompt_tokens_details.cached_creation_tokens` | — | 缓存写入（如有） | 已含在 prompt_tokens |
| `prompt_tokens_details.image_tokens` | — | 图片输入 token | 已含在 prompt_tokens |
| `prompt_tokens_details.audio_tokens` | — | 音频输入 token | 已含在 prompt_tokens |
| `completion_tokens_details.image_tokens` | — | 图片输出 token | 已含在 completion_tokens |
| `completion_tokens_details.audio_tokens` | — | 音频输出 token | 已含在 completion_tokens |
| `total_tokens` | 41 | prompt + completion 总和 | 汇总，不单独计费 |

**OpenAI 输入 token 总量** = `prompt_tokens`（已包含缓存部分）

**OpenAI 输出 token 总量** = `completion_tokens`

#### 上游平台注入字段

上游代理平台（如 One API / New API）可能在 usage 中注入额外字段：

| 字段 | 含义 |
|------|------|
| `claude_cache_creation_5_m_tokens` | 平台转换的 5 分钟缓存写入数 |
| `claude_cache_creation_1_h_tokens` | 平台转换的 1 小时缓存写入数 |
| `cache_read_tokens` | 平台转换的缓存读取数 |
| `usage_semantic` / `usage_source` | 平台元数据标识 |
| `speed` | 生成速度 |

这些字段仅供参考，不参与本代理的 token 统计计算。

#### 差异说明

Anthropic 和 OpenAI 在 token 统计上的主要差异：

```text
                    Anthropic                    OpenAI / DeepSeek
输入 token 字段     input_tokens                 prompt_tokens
                   （不含缓存 token）              （已含缓存 token）

缓存 token         cache_read_input_tokens       prompt_tokens_details.cached_tokens
                   cache_creation_input_tokens    prompt_cache_hit_tokens (DeepSeek)

输出 token 字段     output_tokens                completion_tokens

思考 token         output_tokens_details         completion_tokens_details
                   .thinking_tokens              .reasoning_tokens
                  （已含在 output_tokens 中）     （已含在 completion_tokens 中）
```

关键区别：Anthropic 的 `input_tokens` **不含**缓存 token，需要额外加上 `cache_read` 和 `cache_creation` 才是完整输入量；OpenAI 的 `prompt_tokens` **已包含**缓存 token。

## 多轮对话可视化

```text
在代理服务启动后，访问：

- 监控总览: `http://127.0.0.1:4000/`
- 查询统计: `http://127.0.0.1:4000/query`
- 对话历史记录: `http://127.0.0.1:4000/history`
- 失败历史记录: `http://127.0.0.1:4000/failures`

如果开启了监控后台登录保护，上面这些页面会先跳转到：

- 登录页: `http://127.0.0.1:4000/login`
```
