# 增量索引设计说明

本项目使用两套增量索引来加速 Web 页面的统计查询，避免每次请求都全量扫描日志文件。

## 通用设计模式

两套索引共享相同的缓存架构：

```
┌─────────────────────────────────────────────────────┐
│  Web API 请求                                        │
│     ↓                                                │
│  ① 进程内存缓存 (TTL 10s)  ── 命中 → 直接返回       │
│     ↓ 未命中                                         │
│  ② 持久化索引文件           ── 加载到内存             │
│     ↓                                                │
│  ③ 增量扫描源数据文件                                │
│     ├─ frozen 目录 → 跳过 (不 stat, 不读取)          │
│     ├─ mtime+size 无变化 → 跳过                      │
│     └─ 有变化 → offset 增量读取 / 全量重扫            │
│     ↓                                                │
│  ④ 更新内存缓存 + 写回持久化文件                     │
└─────────────────────────────────────────────────────┘
```

### 关键机制

| 机制 | 说明 |
|------|------|
| **进程内存缓存** | `_mem_index` + `_mem_index_ts`，TTL 10 秒。TTL 内直接返回内存数据，耗时 ~0.001ms |
| **持久化索引文件** | 进程重启后从磁盘加载，避免全量重扫。耗时 ~2ms |
| **frozen 目录** | 历史目录（非当前 `STARTUP_DATE_TAG`）扫描一次后标记 `frozen=true`，后续不再 stat 文件系统 |
| **活跃目录检测** | `STARTUP_DATE_TAG`（如 `26070211`）标识当前进程写入的 mtime 目录，该目录不会被 frozen |
| **变化检测** | 比较 `stat().st_mtime` + `stat().st_size`，两者都没变则跳过 |
| **offset 增量读取** | 记录上次读到的字节偏移 `scan_offset`，文件只增不减时仅读追加部分 |
| **线程安全** | `threading.Lock()` + double-check locking |
| **force 刷新** | 跳过 TTL + frozen，全量检查所有目录，用于用户手动点击"刷新" |

### 性能对比

| 场景 | 耗时 |
|------|------|
| 首次全量扫描（冷启动） | ~110ms |
| 进程重启后从持久化文件加载 | ~2ms |
| TTL 过期后增量检查（全 frozen） | ~0.6ms |
| TTL 内内存缓存命中 | ~0.001ms |
| 活跃目录 offset 增量读取 | ~0.2ms |

---

## 1. Token 用量索引 — `utils/token_index.py`

### 用途

为 `/query`（查询统计）页面提供数据，替代原来 4 个 API 各自全量读取所有 `index.jsonl` 的方式。

### 数据源

```
logs_all/{env}/{mtime}/index.jsonl     ← 每行一条请求记录
```

`index.jsonl` 条目格式：
```json
{
  "ts": "2026-06-01_15-27-33_792",
  "model": "claude-sonnet-4-6",
  "tok_in": 3647, "tok_out": 17,
  "success": true, "valid": true,
  "api_key": "test1",
  "chain_key": "Say hi in one sentence.",
  "channel_key": "sk-upstream1"
}
```

### 持久化文件

`logs_all/.token_index.jsonl`（JSONL 格式，第一行 meta）

```jsonl
{"_meta": true, "version": 1, "updated_at": 1782900000.0}
{"dir": "env-5Nc1/26061009", "index_mtime": 1781160873.0, "index_size": 2255522, "scan_offset": 2255522, "frozen": true, "entry_count": 2383, "models": {"claude-sonnet-4-6|2026-06-10": {"s_count": 50, "s_tok_in": 25000, "s_tok_out": 15000, "e_count": 2, "e_tok_in": 100}}, "keys": {"test1|2026-06-10": {"count": 30, "tok_in": 15000, "tok_out": 9000, "sessions": 5}}, "channels": {"(default)|2026-06-10": {"count": 30, "tok_in": 15000, "tok_out": 9000, "sessions": 5}}, "channel_keys_set": ["sk-upstream1"], "dates": ["2026-06-10", "2026-06-11"]}
```

### 预聚合维度

每个目录行按日期分桶，存储三个维度的聚合数据：

| 维度 | Bucket Key 格式 | 聚合字段 |
|------|------------------|----------|
| **models** | `{model}\|{date}` | `s_count`, `s_tok_in`, `s_tok_out`（成功）；`e_count`, `e_tok_in`（失败） |
| **keys** | `{api_key}\|{date}` | `count`, `tok_in`, `tok_out`, `sessions` |
| **channels** | `{channel_key}\|{date}` | `count`, `tok_in`, `tok_out`, `sessions` |

查询时按日期范围过滤 bucket key 中的日期部分，实现日期过滤。

### 查询接口

| 函数 | 对应 API | 说明 |
|------|----------|------|
| `query_token_stats(model, date_start, date_end, status, channel_key)` | `GET /api/statistic` | 按模型聚合 token 用量 |
| `query_key_stats(date_start, date_end)` | `GET /api/statistic/keys` | 按 API Key 聚合用量 |
| `query_channel_stats(date_start, date_end)` | `GET /api/statistic/channels` | 按渠道聚合用量 |
| `query_channel_keys()` | `GET /api/statistic/channel-keys` | 列出所有已知 channel_key |

### 刷新流程

```python
refresh_token_index(force=False)
    # 1. 内存缓存命中 (TTL 10s) → 直接返回
    # 2. 遍历 logs_all/{env}/{mtime}/
    #    - frozen 且非活跃目录 → 跳过
    #    - stat(index.jsonl) mtime+size 无变化 → 跳过，标记 frozen
    #    - 文件变大 → offset 增量读取新追加部分，合并到已有聚合
    #    - 文件变小/force → 全量重扫该文件
    # 3. 更新内存缓存 + 持久化到 .token_index.jsonl
```

---

## 2. Session 统计索引 — `utils/stats_index.py`

### 用途

为 `/sessions`（Session 管理）、`/keys/export`（Session 导出）、`/logs/dirs`（日志目录列表）提供数据。

### 数据源

```
logs_all/env-{KEY}/{mtime}/.session_cache.jsonl    ← session 聚合缓存
logs_all/env-{KEY}/{mtime}/*-req.json              ← 请求文件（用于 req_count 计数）
```

> **⚠️ 已迁移到 DB**：session 聚合数据现在的**主真相是 `session_cache.db`**（见下方
> [完整数据流](#完整数据流请求落盘--db--导出)），`.session_cache.jsonl` 是**废弃的旧中间层**，
> 运行时**不再生成**（仅 `export_session_index` 导出时会顺带留一份）。
> `_scan_session_cache()` 读取时已改为**优先查 DB**、文件仅作降级。
> 历史遗留：多处代码曾以「`.session_cache.jsonl` 是否存在」判断目录有无 session，
> 导致**活跃目录**（当前进程正在写、尚未导出、故无该文件）被误判为空。已修复的点见完整数据流章节末尾。

### 持久化文件

`logs_all/env-{KEY}/.stats_index.json`（JSON 格式，每个 env 一个文件）

```json
{
  "_version": 3,
  "dirs": {
    "26070100": {
      "cache_mtime": 1719820000.0,
      "cache_size": 123456,
      "req_count": 4200,
      "req_count_mtime": 1719820000.0,
      "frozen": true,
      "sessions": {
        "sk-abcd|2026-06-28": {"total": 5, "qualified": 3}
      }
    }
  },
  "updated_at": 1719820100.0
}
```

### 预聚合维度

每个目录记录 session 统计，按 `{api_key}|{date}` 分桶：

| 字段 | 说明 |
|------|------|
| `total` | 该 key 在该日期的 session 总数 |
| `qualified` | 消息数 >= threshold（默认 5）的 session 数 |
| `req_count` | 该 mtime 目录下的 `-req.json` 文件数 |

### 查询接口

| 函数 | 消费方 | 说明 |
|------|--------|------|
| `refresh_index(env_dir, threshold, force)` | 所有下游 | 增量刷新，返回索引数据 |
| `build_stats_from_index(index, threshold)` | `/sessions/stats`, `/api/export/keys` | 构建 api_key × date 统计矩阵 |
| `get_dir_counts(index)` | `/logs/dirs` | 提取每个 mtime 目录的请求数 |
| `get_date_to_mtime_map(index)` | Session 导出 | 构建 date → [mtime_dir, ...] 映射 |

### 刷新流程

```python
refresh_index(env_dir, threshold, force=False)
    # 1. 内存缓存命中 (TTL 10s, 绑定 env_dir) → 直接返回，_changed_buckets=[]
    # 2. 遍历 env_dir/{mtime}/
    #    - frozen 且非活跃目录 → 跳过
    #    - .session_cache.jsonl 的 mtime+size 无变化 → 跳过，标记 frozen
    #    - 有变化 → 全量扫描该 .session_cache.jsonl，diff old/new sessions → _changed_buckets
    #    - 无 cache 文件 → 计数 *-req.json 文件数
    # 3. 更新内存缓存 + 持久化到 .stats_index.json
    # 4. 返回 index（含 _changed_buckets 供 build_stats_from_index 增量使用）
```

### 增量聚合机制

`build_stats_from_index` 维护 key × date / key × mtime 两个维度的预聚合表，
持久化到 `{env_dir}/.session_key_cache.json`，支持 bucket 级定向更新。

#### 变更追踪

`refresh_index` 在扫描活跃目录时，对比旧/新 `sessions` dict：

```python
_changed_buckets = [(dir_name, bucket_key, old_counts, new_counts), ...]
```

- `old_counts=None` → 新增 bucket
- `new_counts=None` → 删除 bucket
- 两者都有 → 数值变化

典型场景：50 个 frozen 目录 + 1 个活跃目录，只有活跃目录中变化的 key bucket 产生 diff。

#### 持久化文件

`{env_dir}/.session_key_cache.json`：

```json
{
  "_version": 1,
  "updated_at": 1720000000.0,
  "table": {"sk-abcd": {"2026-07-08": {"total": 5, "qualified": 3}}},
  "mtime_table": {"sk-abcd": {"26070809": {"total": 5, "qualified": 3}}},
  "totals": {"total": 100, "qualified": 60}
}
```

#### 三条执行路径

| 路径 | 条件 | 操作 | 耗时 |
|------|------|------|------|
| **A 无变更命中** | `_changed_buckets` 为空，内存缓存有效 | 从内存 key cache 构建 rows | ~0.01ms |
| **B 定向增量** | `_changed_buckets` 非空 | 逐 bucket 做加减法更新 table/mtime_table，持久化，构建 rows | ~0.1ms |
| **C 全量** | 首次 / 进程重启 / env 切换 | 先尝试加载磁盘缓存；否则全量计算并持久化 | ~2ms（磁盘）/ ~50ms（全量） |

启动恢复：进程重启后从 `.session_key_cache.json` 直接加载，跳过全量遍历。

### `/api/export/keys` 性能优化 — key 元数据 + records 缓存

`/api/export/keys` 端点原本每次请求都做 **N+1 次 SQLite 查询**：
- 1 次 `list_keys()` 查全表获取 key 名称/创建时间
- N 次 `list_records_by_key(slot)` 每个 key 查一次导出记录

从 version 2 开始，这些低频变更数据也缓存到 `.session_key_cache.json`，通过轻量检测按需刷新。

#### 扩展的缓存结构（version 2）

```json
{
  "_version": 2,
  "table": {...},
  "mtime_table": {...},
  "totals": {...},
  "key_meta": {
    "db_keys_hash": "a1b2c3...",
    "mapping": {
      "sk-abcd": {
        "key_name": "测试Key",
        "key_slot": "key-abcd",
        "created_at": "2026-07-01 10:00:00"
      }
    }
  },
  "key_records": {
    "db_max_id": 42,
    "db_count": 100,
    "records": {
      "key-abcd": [
        {
          "id": 42,
          "status": "success",
          "mode": "export",
          "created_at": "...",
          "total_sessions": 10,
          "files_uploaded": 5,
          "obs_dst": "obs://...",
          "error_message": ""
        }
      ]
    }
  }
}
```

#### 变更检测机制

| 字段 | 检测方式 | 刷新条件 |
|------|----------|----------|
| `key_meta` | 对 `list_keys()` 结果计算 MD5 hash（拼接 key + name + created_at） | hash 变化 → 重做后缀匹配，更新 mapping |
| `key_records` | `SELECT MAX(id), COUNT(*) FROM export_records` | max_id 或 count 变化 → 一次性查所有 records（排除 progress_log），按 key_slot 分组 |

**records 字段精简**：前端卡片只用 `id, status, mode, created_at, total_sessions, files_uploaded, obs_dst, error_message` 8 个字段，缓存时排除 `progress_log`（按需通过 `/api/export/status/{id}` 单独加载）。

#### 执行路径对比

| 场景 | 之前 | 之后 |
|------|------|------|
| 无变化（常态） | 1 × list_keys + N × list_records_by_key | 1 × list_keys（hash比较）+ 1 × SELECT MAX/COUNT |
| 新增 record | 同上 | 同 + 1 × list_records_all_slim（一次查全部分组） |
| key 配置变更 | 同上 | 同 + 重做后缀匹配 |

SQLite 查询从 **N+1 次** 降到 **2 次**（常态）或 **3 次**（有变化时）。

#### 相关函数

| 函数 | 位置 | 说明 |
|------|------|------|
| `get_records_summary()` | `export_store.py` | 返回 `(max_id, count)`，快速判断 export_records 是否有变化 |
| `list_records_all_slim()` | `export_store.py` | 一次性查所有 records（排除大字段），按 key_slot 分组 |
| `update_key_meta(cache, db_keys_list)` | `stats_index.py` | 检查 hash，按需更新 key 元数据映射 |
| `update_key_records(cache, env_dir)` | `stats_index.py` | 检查 max_id/count，按需刷新 records 缓存并持久化 |
| `get_current_key_cache()` | `stats_index.py` | 返回内存中的 key cache 引用 |

---

## 目录结构总览

```
logs_all/
├── .token_index.jsonl              ← Token 索引（全局，跨 env）
├── env-5Nc1/
│   ├── .stats_index.json           ← Session 索引（每个 env 一个）
│   ├── .session_key_cache.json     ← Key 聚合缓存（增量更新）
│   ├── 26061009/
│   │   ├── index.jsonl             ← Token 索引数据源
│   │   ├── .session_cache.jsonl    ← Session 索引数据源
│   │   ├── *-req.json              ← 请求文件
│   │   └── ...
│   └── 26070115/                   ← 活跃目录（= STARTUP_DATE_TAG）
│       └── ...
├── env-xunxing-zyKA/
│   ├── .stats_index.json
│   └── ...
├── logs_session/                   ← 跳过（startswith("logs_")）
└── logs_session_analysis/          ← 跳过
```

---

## 完整数据流：请求落盘 → DB → 导出

上面两套「索引」是**加速 Web 统计查询**的旁路缓存。下面是主干数据流——从真实请求到最终导出/上传 OBS 的端到端链路。**单一数据真相是 `session_cache.db`**。

```
① 真实请求         ② index.jsonl        ③ 增量消费→DB          ④ 导出            ⑤ 上传 OBS
   (代理转发)  ──▶   (append 日志)  ──▶   session_cache.db  ──▶  session_index  ──▶  三元组文件
   + 三元组文件      每请求一行           (会话聚合)             .jsonl            + index → OBS
```

### ① 请求落盘（`app.py` + `utils/req_index.py`，每请求实时）

代理每处理完一个请求，写入 `logs_all/<env-key>/<STARTUP_DATE_TAG>/`（如 `logs_all/env-ann-5i8w/26071922/`）：

- **原始三元组文件**：`<ts>-req.json` / `-res.json` / `-headers.json`
- **一行索引**：`append_index()` 往该目录 `index.jsonl` **追加一行**，含
  `ts / req_file / api_key / model / chain_key / q1_preview / msg_count / user_turns / success / usage …`

目录名 = `log_paths._env_key_segment()` = `<LOG_TASK_TAG>-<UPSTREAM_API_KEY 后4位>`，
**进程启动时固定**；时间戳段 `STARTUP_DATE_TAG`（如 `26071922`）也是启动时刻，即「活跃目录」。

### ② → ③ index.jsonl 增量消费 → 会话聚合入库（`utils/log_routes.py`）

- **`_refresh_state(kind, root_dir)`**（`_REFRESH_TTL=10s` 节流）：按 `index_progress` 表记录的
  `byte_offset` 只读 `index.jsonl` 的**新增字节**（`_read_new_index_entries`），实现增量。
- **`_process_req_row()`** 做**会话聚合**：
  - `lookup_key = api_key || chain_key`（chain_key 由消息内容生成，同会话连续请求 chain 前缀递增）
  - 经 `chain_index` 表查 lookup_key → `session_key`；查不到则**新建 session**（session_key = 首个 ts），
    查到则**更新**（`latest_file` 取消息最多/带响应的那次，`msg_count / models / last_ts` 等）
  - 新会话检测：user_turns 回落到 ≤1 时判为新会话，另起 `##session_N` 后缀

写入 **`session_cache.db`**（`utils/session_store.py`，WAL 模式）：

| 表 | 内容 |
|----|------|
| `sessions` | 一行一个会话：`session_key, root_dir, api_key, q1, first_ts, last_ts, models, latest_file, msg_count, max_real_turns, best_req_count` |
| `traces` | 一个会话的所有请求轨迹（每 req 一行） |
| `chain_index` | `lookup_key → session_key`（会话归并的关键） |
| `index_progress` | `root_dir → byte_offset`（index.jsonl 消费进度，支持增量） |

### ④ 导出 session_index.jsonl（`utils/export_sync.py:export_session_index`）

对**单个时间戳目录**操作：

1. `_refresh_session_cache()` → 先跑一次 `_refresh_state` 把 index.jsonl 最新内容补进 DB
2. **优先 `session_store.export_sessions(root)` 从 DB 读**（`_read_session_cache`），降级读旧文件
3. 变更判据 `source_mtime`：**有 `.session_cache.jsonl` 文件用文件 mtime，无文件用 DB 条数**
   （负值编码，条数变化即触发重导）；未变则 skip
4. 每个 session 写一行，末尾附 `_meta`（total_sessions / avg_msg_count），落 **`session_index.jsonl`**

### ⑤ 上传 OBS / 质检（`utils/export_routes.py` + `sync_session_index`）

由 Web「Session 导出」/「质检」按钮驱动，对 `mtime_dirs` 里每个目录：

- **导出模式**：`_load_session_index()` 读回 → 按 `api_key` 精确过滤（slot = `key-<后4位>`）→
  收集每个 session 的 `latest_file` 三元组 → 复制到 `logs_session/…/ex-<tag>/` →
  `obsutil` 整目录上传到 `obs://…/session/<env>/<slot>/ex-<tag>/`；`.sync_state.json` 记已传 key 做增量
- **质检模式**：过滤后 `reformat_and_analyze` + `evaluate_sessions` 生成 `session_analysis.json`，
  上传到 `…/session_analysis/…`。前端「Session 导出」卡片跳转的 `.../session_analysis/` 即看此结果。

### 各产物定位小结

| 产物 | 角色 | 生命周期 |
|------|------|----------|
| `<ts>-req/res/headers.json` | 原始请求三元组 | 每请求实时写 |
| `index.jsonl` | **append-only 源**，每请求一行 | 每请求实时写 |
| `session_cache.db` | **会话聚合主真相** | 由 index.jsonl 增量消费 |
| `.session_cache.jsonl` | **废弃旧中间层** | 运行时不再生成 |
| `session_index.jsonl` | 导出快照 | 导出时生成 |
| `.token_index.jsonl` / `.stats_index.json` / `.session_key_cache.json` | Web 统计加速缓存（旁路） | 按需增量刷新 |

### 「文件 → DB 迁移」遗留 bug（活跃目录暴露）

以下三处都假设 `.session_cache.jsonl` 文件必然存在，在**活跃目录**（当前进程正在写、
尚未导出、故无该文件）上返回空/0，导致运行期间新建的 key 在 restart 前无法显示/导出。均已修复：

| 位置 | 症状 | 修复 |
|------|------|------|
| `stats_index.refresh_index` | 导出页 key **列表**不含活跃目录新 key | 无文件时查 DB（`get_session_count_by_root`），有数据则从 DB 聚合，以 `db_count` 作变更判据 |
| `export_sync.export_session_index` | **导出/质检**活跃目录得 0 sessions（“无 session 数据”） | 无文件时查 DB 决定是否继续；`source_mtime` 无文件时用 DB 条数 |
| `debug_logs`（另一条线，非 session 流） | 8084 SSE 写 debug 日志每次全量读+重写 index → 阻塞事件循环卡顿 | `_append_debug_index` 改纯 append(O(1))，读取端按 filename 合并；`write_debug_async` 用 `asyncio.to_thread` 移出事件循环 |

