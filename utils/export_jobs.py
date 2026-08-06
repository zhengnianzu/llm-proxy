"""
utils/export_jobs.py — 导出任务参数契约层

把「导出任务执行参数 → DB 行 → 反序列化」集中一处，消除 app（入队方）与
export_worker（执行方）对字段约定的口头耦合。

背景：导出任务原本是 app 进程内的内存闭包（捕获 obs_prefix/now_tag/mode/force 等），
无法跨进程。剥离出独立 worker 进程后，参数改为在入队时持久化进
export_records.task_json 列，worker 靠 record_id + task_json 重建执行上下文。

- persist_params(): app 侧在 create_record 之后、_enqueue_task 之前调用。
- load_params(): worker 侧领取任务后调用；缺失（旧记录 / 半途升级）返回 None，
  worker 应把该记录标 failed 并提示用户重建。
"""

import json
import logging
import os
from typing import Optional

from utils.export_store import set_task_json, get_record

logger = logging.getLogger(__name__)

# 前端可见「排队 / 执行中」；worker 可领取；终态。仅作口径文档与潜在复用。
ENQUEUE_STATES = ("queued", "pending", "running")
CLAIMABLE_STATES = ("queued",)
TERMINAL_STATES = ("success", "failed", "cancelled")

# kind 区分 worker 该调哪个执行器：
#   "export"        -> _run_task(...)         （导出 / eval / retry，走完整流程）
#   "upload_retry"  -> _run_upload_only(...)  （仅重传已在本地的产物，参数取自记录）
KIND_EXPORT = "export"
KIND_UPLOAD_RETRY = "upload_retry"


def persist_params(
    record_id: int,
    *,
    mode: str,
    obs_prefix: str,
    force: bool,
    now_tag: str,
    env_dir: str,
    env_key_name: str,
    workers: int = 0,
    dir_workers: int = 0,
) -> None:
    """把导出/eval/retry 任务的执行参数序列化写入 export_records.task_json。

    now_tag 在入队端点算好并持久化（前端建任务时已用它展示 obs_dst 草稿），
    worker 执行时直接复用、不重算，避免 OBS 目标路径漂移。

    env_dir / env_key_name 也必须持久化：app 侧由 register_export_routes 收到的
    logs_dir（get_log_dir("logs_all") → 形如 logs_all/<env>）反推，worker 进程无法
    从 get_service_log_dir()（logs/port<P>/<segment>，base 与日期段都不同）复原出
    同一路径。原为闭包捕获，剥离后必须随任务一并落库。绝对路径化，规避 worker
    与 app 工作目录差异。
    """
    payload = {
        "kind": KIND_EXPORT,
        "mode": mode,
        "obs_prefix": obs_prefix or "",
        "force": bool(force),
        "now_tag": now_tag,
        "env_dir": os.path.abspath(str(env_dir)),
        "env_key_name": str(env_key_name),
        "workers": int(workers or 0),
        "dir_workers": int(dir_workers or 0),
    }
    set_task_json(record_id, json.dumps(payload, ensure_ascii=False))


def persist_upload_retry(record_id: int) -> None:
    """标记该记录为「仅重传」任务；local_copy_dir / obs_dst 已在记录行内，worker 直接取。"""
    set_task_json(record_id, json.dumps({"kind": KIND_UPLOAD_RETRY}, ensure_ascii=False))


def load_params(record_id: int) -> Optional[dict]:
    """读回执行参数；缺失或解析失败返回 None（老记录 / 升级瞬间在途任务）。"""
    rec = get_record(record_id)
    if not rec:
        return None
    raw = rec.get("task_json") or ""
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        logger.warning("export task_json 解析失败 (record=%s)", record_id)
        return None
    if not isinstance(data, dict):
        return None
    kind = data.get("kind") or (KIND_EXPORT if "mode" in data else None)
    if kind == KIND_UPLOAD_RETRY:
        return {"kind": KIND_UPLOAD_RETRY}
    if kind != KIND_EXPORT or "mode" not in data:
        return None
    # 归一化，容忍缺省字段。env_dir/env_key_name 缺失（旧记录）返回空串，
    # 由 worker 侧判定后走兜底或标 failed。
    return {
        "kind": KIND_EXPORT,
        "mode": data.get("mode", "export"),
        "obs_prefix": data.get("obs_prefix", "") or "",
        "force": bool(data.get("force", False)),
        "now_tag": data.get("now_tag", ""),
        "env_dir": data.get("env_dir", "") or "",
        "env_key_name": data.get("env_key_name", "") or "",
        "workers": int(data.get("workers", 0) or 0),
        "dir_workers": int(data.get("dir_workers", 0) or 0),
    }
