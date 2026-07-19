# .session_cache.jsonl 问题分析与增量改造方案

## 问题描述

### 当前行为

`utils/log_routes.py` 中的 `_save_state_to_disk`（第 174 行）每次处理完 `index.jsonl` 新增行后，将内存中**所有 session** 全量序列化写入 `.session_cache.jsonl`，采用 tmp → `os.replace` 原子写入。

```
_refresh_state
  └─ _read_new_index_entries  (增量读 index.jsonl)
  └─ _process_req_row × N
  └─ _save_state_to_disk      ← 全量重写，无论改了多少
```

### 实际数据规模

| 目录 | `.session_cache.jsonl` 大小 |
|------|---------------------------|
| `26071621/` | 2.6 GB |
| `26071419/` | 1.1 GB |
| `26071718/` | 127 MB（仍在增长） |

当前活跃目录（`26071718/`）的 `index.jsonl` 已达 983 MB（174 万行），每 10 秒刷新一次，只要有新请求就触发对 127 MB 文件的全量重写。

### 为什么会卡

1. `_save_state_to_disk` 是**同步 I/O**，运行在 uvicorn 主事件循环线程
2. 调用链被 `_CACHE_LOCK` 保护，持锁期间所有其他 HTTP 请求（包括网页加载）全部排队等待
3. 单 worker 进程（`server.sh` 直接 `python app.py`），没有其他 worker 接管请求
4. 全量重写 127 MB = 每次触发都要完整序列化 + 写磁盘，耗时数秒

### 为什么全量重写是浪费的

通过代码分析（`_process_req_row`，第 273–364 行），session 的字段变化规律如下：

| 字段 | 变化规律 |
|------|---------|
| `_key`（first_ts） | 创建后**永不变更** |
| `q1`、`api_key`、`first_ts` | 创建后**永不变更** |
| `trace_list` | **纯 append**，只增不改 |
| `models` | 只增不删 |
| `last_ts`、`latest_file`、`msg_count` | 可更新，但对话结束后冻结 |

一个对话结束后，对应的 session 彻底冻结，不再被修改。生产环境中绝大多数历史 session 都处于冻结状态，每次全量重写是在反复序列化这些不变的数据。

---

## 解决方案：双文件增量格式

### 核心思路

- **主文件 `.session_cache.jsonl`**：只存冷冻结的 session，写入后不再修改
- **增量文件 `.session_cache.delta`**：append-only，只追加热 session 的变更行
- **周期性压实（compact）**：delta 超过阈值时，后台线程将完整状态合并写入主文件并清空 delta

### 文件格式

**主文件**（格式与现有兼容，新增 `format_version: 2`）：

```jsonl
{"_meta": true, "byte_offset": 1234567, "line_count": 5000, "known_keys": ["key1"], "format_version": 2}
{"_key": "2026-07-18_10-00-01_000", "q1": "...", "models": ["..."], "latest_file": "...", "msg_count": 42, "api_key": "...", "first_ts": "...", "last_ts": "...", "trace_list": [...], "_frozen": true}
...
```

**增量文件 `.session_cache.delta`**（四种行类型）：

```jsonl
{"_type": "meta_update", "byte_offset": 1234999, "line_count": 5010, "known_keys": ["key1", "key2"]}
{"_type": "session_create", "_key": "2026-07-19_09-00-01_000", "q1": "...", "api_key": "...", "first_ts": "...", "last_ts": "...", "models": ["..."], "latest_file": "...", "msg_count": 1, "trace_list": [...]}
{"_type": "trace_append", "_key": "2026-07-19_09-00-01_000", "trace": {"filename": "...", "model": "...", "msg_count": 5, "ts": "..."}}
{"_type": "session_update", "_key": "2026-07-19_09-00-01_000", "last_ts": "...", "latest_file": "...", "msg_count": 5, "models": ["..."]}
```

### 写入策略

**热 session 判断**：`last_ts` 距当前时间 < 300 秒（5 分钟）为热 session。

```
每次 _refresh_state 处理新行后：
  对每个 dirty session：
    if 热 session  → 写 delta（append，极快）
    if 刚冷却     → 写入主文件末尾（append），标记 _in_main = True
  更新 meta → 写 meta_update 行到 delta
  if delta_size > 20 MB → 触发后台 compact
```

**compact 流程**（后台线程，不阻塞主循环）：

```
1. 将完整内存 state 写入 .session_cache.jsonl.tmp
2. os.replace(.tmp → .session_cache.jsonl)
3. 清空 .session_cache.delta（open("w").close()）
4. 重置 _delta_size = 0，_in_main_keys = all session keys
```

**load 流程**：

```
1. 读主文件，重建 sessions + chain_map（与现在相同）
2. 若 delta 文件存在，逐行 apply：
   - session_create → 加入 sessions 和 chain_map
   - trace_append   → append 到对应 session 的 trace_list
   - session_update → 更新可变字段
   - meta_update    → 更新 byte_offset/line_count/known_keys
```

### 向后兼容

- 旧格式主文件（无 `format_version`）：load 时走现有路径，不读 delta，首次 compact 后自动升级到新格式
- delta 文件不存在：直接跳过，仅用主文件

---

## 需要修改的代码

全部改动集中在 **`utils/log_routes.py`** 一个文件：

| 位置 | 改动内容 |
|------|---------|
| 顶部常量（第 30 行附近） | 新增 `_DELTA_FILE`、`_HOT_SESSION_WINDOW`、`_COMPACT_THRESHOLD`、`_CACHE_FORMAT_VERSION` |
| `_build_state`（第 153 行） | 新增 `_dirty_keys: set()`、`_in_main_keys: set()`、`_delta_size: 0` |
| `_save_state_to_disk`（第 174 行） | 拆分为 `_append_delta` + `_freeze_cold_sessions`；原函数保留用于 compact |
| `_load_state_from_disk`（第 202 行） | load 完主文件后继续 apply delta 文件 |
| `_process_req_row`（第 273 行） | 创建/更新 session 时将 `session_key` 加入 `state["_dirty_keys"]` |
| `_refresh_state`（第 370 行） | 将 `_save_state_to_disk(state)` 替换为新的增量写入调用 |

---

## 预期效果

| 指标 | 当前 | 改造后 |
|------|------|-------|
| 每次刷新写入量 | 全量（127 MB+） | 只写新增 delta（几 KB） |
| 写入方式 | tmp + os.replace（需要完整序列化） | append（直接追加一行） |
| 持锁时间 | 数秒（序列化 + 写 127 MB） | 毫秒级（append 几行） |
| 主文件增长 | 无限增长 | 只追加冷 session，compact 后重置 |
| 进程重启恢复 | 读全量主文件 | 读主文件 + apply delta（通常 delta 很小） |

---

## 验证方法

1. 重启服务，观察 `.session_cache.delta` 出现并持续 append（`ls -lh`、`wc -l`）
2. 访问 `/logs/aggregate`，确认返回数据与原来一致
3. 等待 5 分钟（`_HOT_SESSION_WINDOW`），观察冷 session 写入主文件
4. 让 delta 超过 20 MB，观察 compact 触发：delta 被清空，主文件更新
5. kill 进程重启，从主文件 + delta 重建 state，对比重启前后 `/logs/aggregate` 的 session 列表应无差异

---

## 相关文件

- `utils/log_routes.py`：核心实现
- `utils/stats_index.py`：类似的 session 聚合逻辑（`_scan_session_cache` 读取 `.session_cache.jsonl`），compact 后需确保该函数仍能正确读取新格式主文件
