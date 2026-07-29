# new-api 富 index（index.db）设计方案

## 背景与问题

历史/导出视图把「叶子目录（{天}/{小时}）」聚合成会话。两种日志格式：

| 格式 | 谁写 index.jsonl | index 行字段 | 聚合成本 |
|------|------------------|--------------|----------|
| native | 本项目 `req_index.append_index` | `ts,model,api_key,req_file,` **`q1_hash,msg_count,user_turns,chain_key`** | 低：`_process_req_row` 快速路径直接用 index 行，**不开原文件** |
| new-api | new-api 上游 | `ts,model,api_key,req_file,usage`（**无聚合字段**） | 高：必须逐个 `parse_combined_file` 打开 ~1MB 合并文件算 q1_hash 等 |

实测单个 new-api 小时叶子 `260728/26072806`：157,823 行 / ~157,825 个合并文件 / ~78GB。首次打开时 `newapi_backfill.aggregate_leaf` **从 offset 0 全量解析全部文件、单进程（1 叶子=1 worker）**，十几分钟 → 「打开特别费劲」。

### 结构性病灶

1. **new-api index 缺聚合字段** → 聚合必须开 78GB 原文件。
2. **大叶子回填非增量**：`aggregate_leaf` 忽略 `index_progress`，永远从 0 全量重算；增量的 `consume_leaf` 只接在 <500 行小叶子。
3. **DB 碎片化**：`session_cache.db` 由 `init_db(SERVICE_LOG_DIR)` 落在**每个 port/env 各一份**（`logs/port{PORT}/{env-seg}/session_cache.db`），另加 CLI 默认 `data/session_cache.db`——共 5~6 份互不相通。换实例打开同一叶子，很可能它那份库是空的 → 又全量回填。
4. 导出产物 `session_index.jsonl`（叶子内，共享盘，含 session 级聚合）**无人读回**：`_leaf_needs_backfill` 只看 DB。

## 设计目标

- 视图请求**永不**触发全量原文件解析。
- 补 meta 的贵操作变成**日志管理页批量触发、可并行、可断点续跑**。
- 产物 co-located 在叶子目录（共享盘）→ **跨实例共享**，消除 DB 碎片化。
- new-api 补齐 meta 后**复用 native 的快速路径**，统一聚合引擎。
- 端到端**全增量**：摄取增量、补 meta 增量、聚合增量。

## 核心思路

给 new-api 补一个上游没写的「富 index」sidecar：**每叶子一个 `index.db`（SQLite）**，与 `index.jsonl` 并排放在叶子目录里（和现在写 `session_index.jsonl` 到叶子里同一先例）。

`index.db` = request 级富 index。补齐 `q1_hash/msg_count/user_turns/q1_preview/chain_key` 后，session 可随时廉价重算，且能喂进 native 同款聚合逻辑。

## 三段式流水线（各自独立进度、各自增量）

### Stage 1 — 摄取（便宜，随时增量）
从 `index.jsonl` 按 `byte_offset` 只读新增行 → 插入 `index.db.requests`，meta 字段留空（NULL）。进度存 `index.db.meta.ingest_offset`（复用 native 的 `_read_new_index_entries` 字节偏移机制）。

### Stage 2 — 补 meta（贵，日志管理批量触发）
`SELECT req_file FROM requests WHERE q1_hash IS NULL` → `parse_combined_file` 开原文件算 `q1_hash/msg_count/user_turns/q1_preview/chain_key/success` → `UPDATE`。
- **可自由分片并行**（按 rowid 段分多进程），摆脱「1 叶子=1 worker」。
- **天然幂等/断点续跑**：进度即 `WHERE q1_hash IS NULL` 的存在性。
- 跑完永久固化；改聚合口径无需再碰原文件。

### Stage 3 — 聚合（便宜）
读 `index.db`（meta 齐全）→ 内存按 `api_key‖q1_hash` 分组出 session（纯内存、秒级、零文件 I/O）。复用 `_process_req_row` / `consume_leaf` 的会话切分逻辑（`user_turns<=1` 回落另起会话）。结果可写入现有 `session_cache.db`，或从 index.db 现算 + 轻缓存。

## Schema（每叶子 `index.db`）

```sql
CREATE TABLE requests (
    req_file    TEXT PRIMARY KEY,   -- 与 index.jsonl 行一一对应（basename 或相对路径）
    ts          TEXT,
    model       TEXT,
    api_key     TEXT,
    usage_json  TEXT,               -- Stage 1 原样搬运
    success     INTEGER,
    -- Stage 2 补：
    q1_hash     TEXT,               -- NULL = 待补
    msg_count   INTEGER,
    user_turns  INTEGER,
    q1_preview  TEXT,
    chain_key   TEXT
);
CREATE INDEX requests_pending ON requests(q1_hash) WHERE q1_hash IS NULL;  -- 快速找待补
CREATE INDEX requests_lookup  ON requests(api_key, q1_hash);               -- Stage 3 聚合

CREATE TABLE meta (
    ingest_offset INTEGER DEFAULT 0,   -- Stage 1 已摄取到的 index.jsonl 字节偏移
    updated_at    TEXT
);
```

- 粒度按叶子（小时），单库约 30MB，与 `iter_index_dirs` 一致。
- native 叶子可选不建 index.db（其 index.jsonl 已富）；如统一，native 的 Stage 2 为 no-op。

## 与现有存储的关系

- `index.db`（request 级富 index，共享盘/叶子内）→ 最底层、跨实例共享、可重算一切。
- `session_cache.db`（session 级，per-port）→ 退成 index.db 之上的可选缓存；短期并存。
- `session_index.jsonl`（session 级导出快照，叶子内）→ 仍由导出从聚合结果生成；可选用其 `trace_list` 给 index.db 做部分预填（减少 Stage 2 开文件数）。

## 接入点

- **日志管理页**（`logs_routes.py` `/api/logs-admin/backfill` + `templates/logs_admin.html` 已有「回填」按钮与状态轮询）→ 扩成「构建/补齐 index.db」批量动作，复用 `newapi_backfill` 的队列/状态框架。
- **meta 计算函数现成**：`message_common.{compute_q1_hash,count_real_user_turns,build_chain_key,get_first_user_text}` + `newapi_format.parse_combined_file`。
- **`_refresh_state` 的 newapi 分支**（`log_routes.py:406`）→ 改为读 index.db 聚合；有待补行时**不同步解析**，只在 payload 带「N 条待补 meta」徽标（复用 `_attach_backfill_status` 模式）。

## 待决策

1. native 是否也统一走 index.db（默认：先只上 new-api，风险小）。
2. session_cache.db 去留（默认：短期并存，中期让 new-api 只依赖 index.db）。
3. 共享盘并发写：Stage 2 单写者纪律（`utils/leader_lock.py` 已有 leader 锁）或每叶子文件锁，视图只读。
4. 已完成叶子冷启动：是否用 `session_index.jsonl` 预填 index.db（优化项，非必需）。

## 全增量对照（改造后）

| 环节 | 现状 new-api 大叶子 | 改造后 |
|------|:---:|:---:|
| 摄取 index 新增行 | ❌ 全量从 0 | ✅ 按 ingest_offset 增量 |
| 补 meta 免重复解析 | ❌ 每次重开文件 | ✅ 仅 `q1_hash IS NULL` 行，一次固化 |
| 视图触发解析 | ❌ 同步全量 78GB | ✅ 只读 index.db，待补仅徽标提示 |
| 跨实例共享 | ❌ per-port DB | ✅ 叶子内 index.db |
