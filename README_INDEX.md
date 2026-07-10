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
