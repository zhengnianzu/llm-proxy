# 日志目录结构说明

## 服务日志目录: `logs/port{PORT}/env-{KEY}/`

每个 env 实例在 `logs/port{PORT}/env-{KEY}/` 下维护运行时状态文件。

### 应用运行时

| 文件 | 说明 |
|------|------|
| `app.pid` | 代理服务进程 PID |
| `app.log` | 代理服务运行日志 |
| `app-meta.json` | 运行时元信息，记录当前和上一个 logs_dir 路径 |

### 日志同步 (sync)

| 文件 | 说明 |
|------|------|
| `sync.pid` | sync 守护进程 PID |
| `sync.log` | sync 守护进程运行日志 |
| `scanner_state.json` | 增量扫描状态，记录每个 `index.jsonl` 已读取的字节偏移量和 mtime，避免重复扫描 |

### Session 导出 (sess)

| 文件 | 说明 |
|------|------|
| `sess.log` | `sess export` 的运行日志 |
| `sessions.json` | 每个 mtime 的导出/上传状态，包括各 slot (nokey/key-xxxx) 的上传记录和 OBS 路径 |

`sessions.json` 示例:
```json
{
  "current_mtime": "26060515",
  "mtimes": {
    "26060515": {
      "total_sessions": 4,
      "avg_msg_count": 2,
      "exported_at": "2026-06-08 17:42:29",
      "slots": {
        "nokey": {
          "synced_at": "2026-06-08 17:42:10",
          "sync_uploaded": 12,
          "obs_dst": "obs://bucket/session/env-key/26060515/nokey/ex-26060817/"
        },
        "key-2325": {
          "synced_at": "2026-06-08 17:42:35",
          "sync_uploaded": 3,
          "obs_dst": "obs://bucket/session/env-key/26060515/key-2325/ex-26060817/"
        }
      }
    }
  }
}
```

### Session 网页展示

| 文件 | 说明 |
|------|------|
| `.sessions_status.json` | `/sessions/stats` 接口的缓存，按 api_key x 日期的 session 统计矩阵，供网页展示 |

### API Key 管理

| 文件 | 说明 |
|------|------|
| `keys.db` | SQLite 数据库，存储 API Key 记录 |
| `keys.db-shm` / `keys.db-wal` | SQLite WAL 模式的共享内存和预写日志 |
| `key_state.yaml` | Key 管理配置（邀请码、密码、key 长度等） |

### 监控日志

| 文件 | 说明 |
|------|------|
| `rate.log` | 速率限制日志 |
| `rpm.log` | RPM（每分钟请求数）监控日志 |

### 调试

| 文件 | 说明 |
|------|------|
| `debug/{mtime}/` | 按 mtime 分目录存放重试调试信息，如 `*_attempt{N}_claude_http_503.txt` |

---

## 请求日志目录: `logs_all/env-{KEY}/{mtime}/`

每次服务启动按时间生成 mtime 目录，存放请求日志和 session 缓存。

| 文件 | 说明 |
|------|------|
| `index.jsonl` | 请求索引，每行一条请求记录 |
| `{timestamp}-req.json` | 请求体 |
| `{timestamp}-headers.json` | 请求头 |
| `{timestamp}-res.json` | 响应体 |
| `.session_cache.jsonl` | session 聚合缓存（由 index.jsonl 增量生成，网页展示依赖，勿删） |
| `session_index.jsonl` | session 索引（由 sess export 生成，可通过 `sess clear` 清理） |
| `.sync_export_state.json` | 上传状态记录，按 slot 跟踪已上传的 session（可通过 `sess clear` 清理） |

`.sync_export_state.json` 示例:
```json
{
  "slots": {
    "nokey": { "uploaded_keys": ["2026-05-30_17-32-55_613", "..."] },
    "key-2325": { "uploaded_keys": ["2026-06-05_16-34-34_688"] }
  }
}
```

---

## Session 导出目录: `logs_session/env-{KEY}/{mtime}/`

`sess export --sync` 将三元组文件复制到此目录后上传到 OBS，按 key 维度和时间维度组织。

```
logs_session/env-{KEY}/{mtime}/
  nokey/                  # 全量上传
    ex-26060817/          # 第一次上传（增量）
    ex-26060818/          # 第二次上传（只有新增的 session）
  key-2325/               # 按 api_key 后 4 位分组
    ex-26060817/
    ex-26060818/
```

每个 `ex-*` 目录内包含:
- 三元组文件（`*-req.json`, `*-headers.json`, `*-res.json`）
- `session_index.jsonl`（有 key 过滤时只含过滤后的条目）

OBS 上传路径与本地结构一致: `obs_base/session/env-{KEY}/{mtime}/{nokey|key-xxxx}/ex-{time}/`
