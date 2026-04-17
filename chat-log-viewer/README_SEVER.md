# Chat Log Viewer Server Design

## 1. 背景与目标

当前 `chat-log-viewer/src/server.py` 在 session 模式下主要依赖 `index.json` 提供会话列表，只能完成：

- 浏览 session 列表
- 选择单条对话并查看消息内容

但 `analyze_sessions.py` 已经输出了更适合人工抽检和异常定位的分析结果，例如：

- `user_turns`
- `api_call_count`
- `duration_s`
- `tool_use_count`
- `tool_fail_total`
- `tool_success_rate`
- `completed`
- `tool_fail_detail`

目标是把 `chat-log-viewer` 从一个“日志查看器”升级为一个“轻量分析工作台”，满足以下需求：

- 基于分析结果筛选异常 session
- 支持高轮次、低成功率、工具失败等抽检条件
- 保留现有按 session 查看原始对话的能力
- 支持分析图表与报告文件入口
- 明确 `session_dir` 和 `report_dir` 的映射关系


## 2. 产品结构

整体采用单服务、单前端、多视图结构，不拆成独立系统。

建议主界面包含 3 个视图：

- `Sessions`
- `Analysis`
- `Reports`

职责划分：

- `Sessions`
  - 用于人工抽检具体 case
  - 支持多条件筛选、排序、快捷异常过滤
  - 支持查看单条 session 对话内容

- `Analysis`
  - 用于查看聚合统计和图表
  - 通过图表联动 session 列表
  - 用于发现异常分布和高风险样本

- `Reports`
  - 用于展示已有离线报告
  - 包括 HTML、Markdown、XLSX


## 3. 配置设计

### 3.1 设计原则

在 session 模式下，`session_dir` 和 `report_dir` 必须有明确映射关系。

不能只依赖默认规则 `session_dir/stat`，因为存在以下情况：

- 分析结果输出到自定义目录
- 多个 session 数据源同时存在
- 报告目录与 session 目录不在同一路径下


### 3.2 推荐配置

推荐新增 `session_sources` 作为统一配置入口：

```yaml
mode: session

session_sources:
  - label: exp-a
    session_dir: /data/sessions/exp-a
    report_dir: /data/sessions/exp-a/stat

  - label: exp-b
    session_dir: /data/sessions/exp-b
    report_dir: /data/reports/exp-b

host: 0.0.0.0
port: 8080
log_level: INFO
```

字段说明：

- `label`
  - 数据源展示名称
  - 可选，默认取目录名

- `session_dir`
  - 会话根目录
  - 必填

- `report_dir`
  - 报告与分析缓存目录
  - 可选


### 3.3 兼容旧配置

保留旧字段兼容：

```yaml
mode: session

session_dir:
  - /data/sessions/exp-a
  - /data/sessions/exp-b

report_dir:
  - /data/sessions/exp-a/stat
  - /data/reports/exp-b
```

兼容规则：

1. 若配置了 `session_sources`，优先使用
2. 否则使用 `session_dir[i] -> report_dir[i]`
3. 若未配置 `report_dir[i]`，则默认使用 `session_dir[i] / "stat"`

约束建议：

- `session_sources` 与旧版 `session_dir/session_dirs` 不混用
- `report_dir` 数量少于 `session_dir` 时，缺省项按默认规则推断


## 4. 后端数据模型

### 4.1 Source 抽象

在 `src/server.py` 中引入统一的 source 数据结构：

```python
{
    "key": "session-<hash>",
    "label": "exp-a",
    "session_dir": Path("/data/sessions/exp-a"),
    "report_dir": Path("/data/sessions/exp-a/stat"),
}
```

用途：

- 支撑顶部数据源切换
- 统一会话目录与分析目录的访问
- 为 `Sessions` / `Analysis` / `Reports` 共用


### 4.2 建议缓存结构

建议服务端维护以下缓存：

- `source_registry`
  - source 元信息列表

- `session_list_cache`
  - 每个 source 的 session 列表
  - 数据来自 `index.json` 与 `session_analysis.json` 的 merge

- `analysis_summary_cache`
  - 每个 source 的图表聚合数据

- `report_meta_cache`
  - 每个 source 的报告文件存在性信息


## 5. 数据来源与 merge 规则

### 5.1 数据来源

基础列表来自：

- `<session_dir>/index.json`

分析增强数据来自：

- `<report_dir>/session_analysis.json`

报告文件来自：

- `<report_dir>/session_report.html`
- `<report_dir>/session_report.md`
- `<report_dir>/session_report.xlsx`


### 5.2 merge 主键

建议使用 session 文件夹名作为主键，即 `session_analysis.json` 中的：

- `session`

与 `index.json` 中的：

- `folder`

进行对应。


### 5.3 merge 后 session item 结构

建议 `/api/list` 返回的每条 item 至少包含以下字段：

```json
{
  "id": 12,
  "session": "2026-04-16-123456",
  "label": "用户首问",
  "label_short": "用户首问截断",
  "msg_count": 42,
  "model": "gpt-5",
  "rel_path": "2026-04-16-123456/2026-04-16-123500.json",

  "user_turns": 11,
  "api_call_count": 6,
  "duration_s": 322.0,
  "api_errors": 0,
  "tool_use_count": 13,
  "tool_result_count": 12,
  "tool_success": 8,
  "tool_fail_flag": 2,
  "tool_fail_keyword": 2,
  "tool_fail_total": 4,
  "tool_success_rate": 66.7,
  "completed": "未完成",
  "completed_note": "xxx",

  "tool_use_detail": {},
  "tool_fail_detail": {},
  "analysis_available": true,

  "dir_key": "session-xxx",
  "dir_label": "exp-a"
}
```

兼容要求：

- 没有分析数据时必须仍可用
- `analysis_available = false`
- 分析字段允许为 `null`、空字典或默认值


## 6. 后端接口设计

### 6.1 `/api/config`

用途：

- 返回所有数据源
- 返回当前视图能力
- 返回报告与分析是否可用

建议返回：

```json
{
  "mode": "session",
  "dirs": [
    {
      "key": "session-xxx",
      "label": "exp-a",
      "session_dir": "/data/sessions/exp-a",
      "report_dir": "/data/sessions/exp-a/stat",
      "has_analysis": true,
      "has_report_html": true,
      "has_report_md": true,
      "has_report_xlsx": true
    }
  ],
  "active_dir": "session-xxx",
  "views": ["sessions", "analysis", "reports"]
}
```


### 6.2 `/api/list?dir=...`

用途：

- 返回 merge 后的 session 列表

建议返回：

```json
{
  "items": [],
  "meta": {
    "total": 226,
    "analysis_enabled": true
  }
}
```


### 6.3 `/api/file?rel_path=...&dir=...`

用途：

- 保持当前行为
- 加载单条 session 对话内容

本接口不做协议变更。


### 6.4 `/api/analysis/summary?dir=...`

用途：

- 返回分析页所需的图表与 KPI 数据

建议返回：

```json
{
  "total_sessions": 226,
  "analysis_sessions": 220,
  "kpis": {
    "high_turn_sessions": 50,
    "failed_sessions": 37,
    "avg_tool_success_rate": 81.2
  },
  "charts": {
    "user_turns_hist": [],
    "tool_use_hist": [],
    "tool_success_rate_hist": [],
    "duration_hist": [],
    "model_dist": [],
    "completed_dist": [],
    "tool_fail_top": []
  }
}
```


### 6.5 `/api/report/meta?dir=...`

用途：

- 返回当前数据源报告文件可用性

建议返回：

```json
{
  "html": {
    "exists": true,
    "url": "/report/view?dir=session-xxx"
  },
  "md": {
    "exists": true,
    "url": "/report/raw/md?dir=session-xxx"
  },
  "xlsx": {
    "exists": true,
    "url": "/report/raw/xlsx?dir=session-xxx"
  }
}
```


### 6.6 报告接口

建议补充：

- `/report/view?dir=...`
- `/report/raw/md?dir=...`
- `/report/raw/xlsx?dir=...`

用途：

- HTML 预览
- Markdown 原文查看
- XLSX 下载


## 7. 前端页面结构

前端建议继续使用单页结构，但从单视图升级为多视图工作台。

总体结构：

- 顶部导航栏
- 主内容区
- 支持 `Sessions / Analysis / Reports` 切换

顶部导航建议包含：

- 产品标题
- Tab 视图切换
- source 切换下拉框
- 刷新按钮
- 统计 chip


## 8. Sessions 视图设计

### 8.1 布局

采用三栏结构：

- 左栏：筛选器 + session 列表
- 中栏：对话详情
- 右栏：属性面板，可收起


### 8.2 筛选器能力

支持规则式筛选，而不是仅靠搜索框。

建议支持字段：

- `q1`
- `model`
- `user_turns`
- `api_call_count`
- `duration_s`
- `tool_use_count`
- `tool_fail_total`
- `tool_success_rate`
- `completed`

建议操作符：

- 文本型
  - `contains`
  - `not_contains`
  - `=`

- 数值型
  - `>`
  - `>=`
  - `<`
  - `<=`
  - `=`
  - `between`

- 枚举型
  - `is`
  - `is_not`
  - `in`

排序建议支持：

- `user_turns desc`
- `tool_fail_total desc`
- `tool_success_rate asc`
- `duration_s desc`
- `api_call_count desc`


### 8.3 快捷预设

建议内置以下异常抽检预设：

- `高轮次异常`
  - `user_turns >= 8`
  - `tool_fail_total >= 1`

- `低成功率`
  - `tool_success_rate <= 70`

- `重工具调用`
  - `tool_use_count >= 10`

- `超长会话`
  - `duration_s >= P90`
  - 第一阶段也可使用固定阈值

- `API 异常`
  - `api_errors >= 1`


### 8.4 session 列表项展示

左栏列表项不应只显示标题和路径，建议增加简要指标摘要。

建议展示：

- 标题
- model
- 用户轮次
- 工具调用数
- 工具失败数
- 成功率
- 时长
- 完成状态

目标：

- 异常 session 在列表中即可被快速识别


### 8.5 右侧属性面板

右侧面板默认展开，可收起。

建议展示：

- 基础信息
  - session id
  - model
  - 路径

- 过程指标
  - `user_turns`
  - `api_call_count`
  - `duration_s`

- 工具指标
  - `tool_use_count`
  - `tool_fail_total`
  - `tool_success_rate`

- 质量信息
  - `completed`
  - `completed_note`

- 工具明细
  - `tool_use_detail`
  - `tool_fail_detail`

建议支持点击属性值生成筛选条件。


## 9. Analysis 视图设计

`Analysis` 视图只负责聚合统计和异常定位，不显示完整消息流。

建议模块：

- KPI 卡片
- 直方图
- 分布图
- Top 榜单

首批图表建议：

- `user_turns` 分布
- `tool_use_count` 分布
- `tool_success_rate` 分布
- `duration_s` 分布
- `completed` 状态分布
- `tool_fail_detail` Top N

联动规则：

- 点击图表区间后跳转到 `Sessions`
- 自动生成对应筛选条件

示例：

- 点击“成功率 0-50%”
- 自动切换到 `Sessions`
- 应用 `tool_success_rate <= 50`


## 10. Reports 视图设计

`Reports` 视图只展示已有离线报告文件，不参与分析计算。

建议内容：

- `session_report.html`
- `session_report.md`
- `session_report.xlsx`

行为建议：

- HTML 支持内嵌预览
- Markdown 支持原文展示
- XLSX 支持下载

若当前 source 无报告文件：

- 页面应明确显示缺失状态
- 不应报错


## 11. 视觉风格设计

界面风格可以参考 BiasNavi 的“研究分析工具感”，但不直接照搬其具体页面。

参考方向：

- 强调分析工作台气质
- 模块化卡片布局
- 图表与筛选区域层次清晰
- 减少传统后台表格风格

建议视觉规则：

- 顶部深色导航栏
- 页面使用浅灰蓝背景
- 卡片化布局
- 统一圆角和边框
- 异常、警告、正常状态颜色明确

建议颜色变量：

```css
:root {
  --bg: #f3f6fb;
  --surface: #ffffff;
  --surface-alt: #eef3f9;
  --text: #1f2a37;
  --muted: #6b7280;
  --primary: #0f6c7b;
  --primary-soft: #d8eef2;
  --success: #1f9d68;
  --warn: #d97706;
  --danger: #d64545;
  --border: #dbe3ec;
}
```


## 12. 实施阶段

### Phase 1：配置与数据接线

- 支持 `session_sources`
- 支持 `report_dir`
- 建立 source 抽象
- 读取 `session_analysis.json`
- `/api/list` merge 分析字段
- `/api/config` 返回分析与报告状态


### Phase 2：Sessions 视图增强

- 顶部导航骨架
- 左栏规则筛选器
- session 列表项指标摘要
- 右侧属性面板


### Phase 3：Analysis 视图

- `/api/analysis/summary`
- KPI 卡片
- 基础图表
- 图表点击联动筛选


### Phase 4：Reports 视图

- 报告文件元信息
- HTML 预览
- Markdown 查看
- XLSX 下载


### Phase 5：视觉统一

- 整体主题升级
- BiasNavi 风格收敛
- 卡片、图表、导航、侧边栏统一样式


## 13. 最小可交付版本

若需要尽快交付，可先实现最小版本：

1. 配置支持 `report_dir`
2. `/api/list` 合并分析字段
3. `Sessions` 视图支持筛选与排序
4. 右侧属性面板
5. 顶部导航占位
6. `Analysis` 视图先提供 4 张基础图

这一版已经可以满足“人工抽检异常 session”的核心需求。


## 14. 风险与兼容要求

必须保证以下兼容性：

- 没有 `session_analysis.json` 时，viewer 仍可正常使用
- 没有报告文件时，`Reports` 视图仅显示缺失状态
- 现有 `/api/file` 保持兼容
- 原有基于 `index.json` 的 session 查看流程不受影响

设计目标不是替换现有 viewer，而是在其上叠加分析能力。
