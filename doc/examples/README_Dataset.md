# 数据集（Dataset）格式说明

本文件说明「Session 管理 → 查看」页面读取的数据集在磁盘上的组织方式：
**哪些是必须项、单条轨迹路径如何拼接、`<session>` 文件夹是什么意义**。

配套样例见同目录 [`unified_session_analysis.json`](./unified_session_analysis.json) +
[`trajectories/`](./trajectories/README.md)。字段的完整超集定义见
[`README_unified_format.md`](./README_unified_format.md)，本文只讲「能被查看页正确加载」所需的最小约束。

---

## 1. 一个数据集在磁盘上的样子

一个数据集 = **一个根目录**（数据库记录里的 `local_copy_dir` 指向它），
根目录下放一个 `session_analysis.json` 和若干个 `<session>/` 子目录：

```
<local_copy_dir>/                     # 根目录，= DB 记录的 local_copy_dir
├── session_analysis.json             # 会话清单（决定“查看”页左侧列表）
├── <session-A>/                      # 一个 session 一个文件夹，文件夹名 = session 字段
│   ├── <ts1>-req.json                #   该 session 的一次 API 调用（一条轨迹）
│   ├── <ts2>-req.json
│   └── <ts3>-req.json
├── <session-B>/
│   └── <ts>-req.json
└── .session_cache.json               # 自动生成的缓存，勿手动维护（见 §6）
```

> 记录（`local_copy_dir`）本身不是本文档的 JSON 文件，而是数据库里的一列。
> 一条记录要能作为数据集出现在管理列表，DB 侧必须满足两个硬条件
> （`_is_valid_dataset`）：**`status == "success"`**，且 **`local_copy_dir` 路径里含
> `analysis` 子串**（大小写不敏感）。否则该记录不会出现在「Session 管理」列表里。

---

## 2. 必须项 vs 可选项

### 2.1 根目录

| 项 | 必须？ | 说明 |
|---|---|---|
| `session_analysis.json` | **强烈建议** | 有它 → 走 `session_analysis` 模式，左侧列表按它渲染（q1、模型、质检码等齐全）。**没有它** → 退化为 `filesystem` 模式：直接扫根目录下每个子目录当一个 session，q1 从第一条轨迹的首个 user 消息里现抓，其余字段留空。 |
| `<session>/` 子目录 | **必须**（至少 1 个） | 没有任何 session 目录，列表为空。 |
| `.session_cache.json` | 否 | 运行时自动生成/失效，不要手写。 |

### 2.2 `session_analysis.json` 内每条 session

`session_analysis.json` 顶层结构：`{"sessions": [ {…}, {…} ]}`
（也兼容顶层直接是一个数组）。每条 session 里：

| 字段 | 必须？ | 作用 |
|---|---|---|
| `session` | **必须** | **唯一硬依赖**：它同时是磁盘上 `<session>/` 文件夹名。取不到就找不到轨迹文件夹（该行 `trajectory_count=0`）。 |
| `q1` | 可选 | 左侧列表标题（首个用户问题）。缺省空串。 |
| `model` | 可选 | 展示用。缺省空串。 |
| `start_time` | 可选 | 展示用。缺省空串。 |
| `duration_s` | 可选 | 展示用。缺省 0。 |
| `api_call_count` | 可选 | 展示用。缺省 0。 |
| `completed` | 可选 | 质检码（`0` 或逗号分隔错误码，见 `README_unified_format.md`）。缺省 0。 |
| `completed_note` | 可选 | 质检码中文描述。缺省空串。 |
| `trace_list` | **必须（有序）** | 每次调用一条 `{filename, model, msg_count, ts}` 的清单，**按调用顺序排列**（末个 = 一连串 API 调用的最后一次）。查看页据它确定 `trajectory_files` 的顺序与默认展示的 `latest_file`（见 §4 的定位优先级）。导出链路必产出、非空；缺省即空列表，此时查看页退化到按文件名排序（顺序不再权威）。 |
| `latest_file` | 可选 | 默认展示的那次调用文件名。**优先由 `trace_list` 末个推导**；仅在无 `trace_list` 时才用本字段，再无则回退到目录排序末个 `*.json`（见 §4）。 |
| 其余分析字段 | 可选 | `tool_*`、`api_errors` 等仅统计/文档用途，查看页加载轨迹时**不依赖**它们。 |

> 一句话：**`session` 决定「找哪个文件夹」，`trace_list` 决定「文件夹里的顺序和默认展示哪个」**。
> 二者是查看页真正依赖的字段；其余全部有缺省值、只影响展示或供导出侧统计。
> `trace_list[].filename` 必须与磁盘上的 `*.json` 文件名一致——查看页以 trace_list 顺序为准，
> 磁盘上多出来的文件（trace_list 未提及）按文件名排序补到末尾，trace_list 提到但磁盘缺失的跳过（见 §4）。

### 2.3 轨迹文件（`<session>/*.json`）

一条轨迹是一次 API 调用的快照。要能被渲染，唯一实质要求是**含 `messages` 数组**：

| 顶层键 | 必须？ | 作用 |
|---|---|---|
| `messages` | **必须** | 请求侧对话上下文，查看页渲染的主体。缺省时回退读 `request.messages`；再没有则该轨迹显示“无消息内容”。 |
| `response` | 可选 | 本次调用的模型返回。非空时会被拼成一条带「↓ 来自 res 文件的回复」标记的 assistant 气泡追加到末尾；`content` 为空（`[]`/`""`，如 200 空响应）则跳过、不渲染空气泡。兼容 **Anthropic 原生**（`{content:[…], status_code}`）与 **OpenAI**（`{choices:[{message:…}]}`）两种格式。 |
| `model` | 可选 | 展示用。 |
| `header` | 可选 | 请求头快照，可为 `{}`。 |

`messages` 为 Anthropic 格式：`role` 取 `user`/`assistant`，`content` 为字符串或
block 列表（`text` / `thinking` / `tool_use` / `tool_result`）。详见
[`trajectories/README.md`](./trajectories/README.md#轨迹文件结构)。

---

## 3. `<session>` 文件夹是什么意义

**一个 `<session>` 文件夹 = 一次完整会话（一个用户会话/一段连续对话）。**

- 文件夹名**必须等于** `session_analysis.json` 里对应记录的 `session` 字段——
  这就是二者的绑定关系，也是查看页从「列表某一行」定位到「磁盘哪个目录」的依据。
- 文件夹里的每个 `*.json` 是这次会话中的**一次 API 调用**（一条轨迹）。
  一个会话通常有多次调用（多轮对话 / 多次工具往返），所以一个 session 目录下常有多个文件。
- 同一 session 内，**上一次调用的模型回复会作为 assistant 消息出现在下一个轨迹文件的
  `messages` 里**；而每个文件*自己那次*的返回只存在它的 `response` 里
  （所以最后一次调用的回复只能从 `response` 读到——这正是查看页要追加 `response` 的原因）。

---

## 4. 单条轨迹路径是怎么拼接的

查看页打开某条轨迹时，后端 `session_trajectory(record_id, session_id, file_name)`
的路径拼接**只有一步**：

```
轨迹文件绝对路径 = local_copy_dir / session_id / file_name
```

其中：
- `local_copy_dir` —— 数据库记录里的根目录；
- `session_id` —— 即列表行的 `session`（= 文件夹名）；
- `file_name` —— 该 session 目录下的某个 `*.json` 文件名。

而**列表里每条 session 的 `trajectory_files`（有序）** 是这样定出来的
（`_build_sessions_from_analysis`）：

第一步，`glob` 出该 session 目录下的候选文件并排除噪声：

```
root / <session> / *.json
  ├─ 排除 _EXCLUDED_FILES：session_analysis.json / session_index.json(.jsonl) /
  │    failure_report.json / manifest.json / .session_cache.json(.jsonl) /
  │    session_report.html / .md / .xlsx
  └─ 排除 *--thinking.json（反思导出产物）
```

第二步，**用 `trace_list` 确定顺序和 `latest_file`**，优先级如下
（`_order_by_trace_list`）：

1. **有 `trace_list`（权威）**：`trajectory_files` 按 `trace_list[].filename` 的**调用顺序**排列
   （交集——只保留磁盘存在的）；磁盘上多出、trace_list 未提及的文件按文件名排序**补到末尾**（不丢文件）；
   `latest_file` = `trace_list` 里**最后一个磁盘存在**的 `filename`。
2. **无 `trace_list`、但有 `latest_file` 字段**：`trajectory_files` 按文件名排序，`latest_file` 用声明值（存在才用）。
3. **两者皆无**（如 filesystem 退化模式）：按文件名（时间戳前缀，字典序即时间序）排序，`latest_file` = 排序末个。

前端点开某个 session 默认加载 `latest_file`（`s.latest_file || files[files.length-1]`），
即一连串 API 调用的最后一次；再按需请求其它文件——每次都用上面那条
`local_copy_dir/session_id/file_name` 拼路径。

> 安全约束：`session_id` / `file_name` 直接参与路径拼接，请勿包含 `..`、`/` 等；
> 文件名建议沿用 `YYYY-MM-DD_HH-MM-SS_mmm-req.json` 这类与时间戳对应的命名。

---

## 5. 双层嵌套自动穿透

用 `obsutil cp` 上传目录时，常把目录名本身也复制进去，导致**双层同名嵌套**：
`<local_copy_dir>/<local_copy_dir 同名子目录>/session_analysis.json`。

`dataset_sessions` 会自动处理：若根目录下**没有** `session_analysis.json`、且**恰好只有一个**
非隐藏子目录、而那个子目录里**有** `session_analysis.json`，则自动下钻一层，把它当作真正的
根目录。所以下面两种布局都能正确加载：

```
# 正常
<root>/session_analysis.json + <root>/<session>/...

# 双层嵌套（也能识别）
<root>/<root-name>/session_analysis.json + <root>/<root-name>/<session>/...
```

---

## 6. 缓存（.session_cache.json）

- **缓存的只是 session 列表**（每行的 q1、文件名、质检码等），**不缓存轨迹正文**。
- 失效判据是 `source_mtime`：`session_analysis.json` 的 mtime（analysis 模式）
  或根目录 mtime（filesystem 模式）变了就重建。
- 因此改动**轨迹文件内容**（比如补 `response`）**无需清缓存**，刷新即生效；
  只有增删 session、改 `session_analysis.json` 才会触发列表重建。
- 前端「刷新」按钮可带 `force=true` 强制重建列表缓存。

---

## 7. 最小可用清单（Checklist）

要让一个目录能作为数据集在「查看」页正常显示回复：

- [ ] 根目录路径里含 `analysis`，且对应 DB 记录 `status=success`
- [ ] 根目录下有 `session_analysis.json`，`sessions[]` 里每条都有 **`session`** 字段
- [ ] 每个 `session` 值都有一个同名 `<session>/` 文件夹
- [ ] 每个文件夹里至少有一个非排除的 `*.json`，且含 **`messages`** 数组
- [ ] `sessions[]` 里每条带 **`trace_list`**（有序，末个=最后一次调用），`filename` 与磁盘文件一致——决定轨迹顺序和默认展示项
- [ ] 想显示“本次调用的回复”，则该 `*.json` 带非空 `response.content`
