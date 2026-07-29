from __future__ import annotations

import copy
import json
import threading
import time
from pathlib import Path
from typing import Any

from utils.export_store import (
    create_record,
    get_record,
    get_record_resolved,
    list_records,
    list_records_by_key,
    list_records_for_datasets,
    set_in_manage,
    set_in_manage_bulk,
    set_manage_name,
    update_status,
)
from utils.key_store import get_key_full, list_keys, mask_key
from utils.obs_utils import run_download_cmd

from . import db
from .config import ReflectionConfig
from .consumer import reflect
from .extractor import extract_signatures
from .importer import import_run, preview_run, validate_quality_dir
from .merger import merge, append_response_message
from .prompt_loader import load_prompt
from .worker_manager import WorkerManager


class ReflectionService:
    def __init__(self, config: ReflectionConfig):
        self.config = config
        db.init(config.db_path)
        db.reset_processing(config.db_path)
        self.workers = WorkerManager(config.db_path, config.prompt_dir)

    @staticmethod
    def _pick_latest_run(runs: list[dict]) -> dict | None:
        """Pick the run that best represents current dataset progress.

        Priority: running > queued > completed/completed_with_failures > paused >
                  failed/cancelled > draft. Within a tier take the newest created_at.
        `runs` is expected to be sorted by created_at DESC already.
        """
        if not runs:
            return None
        priority = {
            "running": 0, "queued": 1,
            "completed": 2, "completed_with_failures": 2,
            "paused": 3,
            "failed": 4, "cancelled": 4,
            "draft": 5,
        }
        return min(runs, key=lambda r: (priority.get(r.get("status"), 6),
                                        -(r.get("created_at") or 0)))

    def _dataset_status(self, export_id: int, runs: list[dict]) -> tuple[str, dict[str, int]]:
        """Return (status_enum, counts) computed from dataset_tasks + runs.

        Truth is dataset_tasks.latest_status. Run status is only consulted to
        distinguish 'running' from 'imported' when there ARE still pending tasks.

        Enum:
          - not_imported: no dataset_tasks rows for this export_id
          - completed:    no pending, no processing, no failed
          - has_failed:   failed > 0 (regardless of what runs report)
          - running:      pending/processing > 0 AND some run is running/queued
          - imported:     pending/processing > 0 AND no run is active
        """
        with db.connect(self.config.db_path) as conn:
            counts = db.dataset_counts(conn, export_id)
        total = sum(counts.values())
        if total == 0:
            return "not_imported", counts
        if counts["failed"] > 0:
            return "has_failed", counts
        outstanding = counts["pending"] + counts["processing"]
        if outstanding == 0:
            return "completed", counts
        # only trust "running" when the DB still has work to do
        if any(r.get("status") in ("running", "queued") for r in runs):
            return "running", counts
        return "imported", counts

    def source_key_slots(self) -> list[dict]:
        grouped: dict[str, list[dict]] = {}
        for record in list_records_for_datasets(limit=1000):
            if "analysis" not in (record.get("local_copy_dir") or "").lower():
                continue
            grouped.setdefault(record["key_slot"], []).append(record)
        slots = [{
            "key_slot": key_slot,
            "record_count": len(records),
            "latest_created_at": records[0].get("created_at", ""),
        } for key_slot, records in grouped.items()]
        slots.sort(key=lambda item: item["latest_created_at"], reverse=True)
        return slots

    def datasets(self, key_slot: str) -> list[dict]:
        result = []
        for record in list_records_by_key(key_slot, limit=100):
            if "analysis" not in (record.get("local_copy_dir") or "").lower():
                continue
            artifacts_valid = record["status"] == "success"
            quality_error = ""
            if artifacts_valid:
                d = Path(record["local_copy_dir"])
                if not d.is_dir():
                    artifacts_valid = False
                    quality_error = f"质检目录不存在: {d}"
            runs = db.list_runs(self.config.db_path, record["id"])
            latest = self._pick_latest_run(runs)
            status, counts = self._dataset_status(record["id"], runs)
            total = sum(counts.values())
            result.append({
                **record,
                "created_at": record["created_at"], "total_sessions": record["total_sessions"],
                "local_copy_dir": record["local_copy_dir"], "obs_dst": record["obs_dst"],
                "artifacts_valid": artifacts_valid,
                "error": quality_error or record.get("error_message", ""),
                "latest_run": latest,
                "dataset_status": status,
                "dataset_counts": counts,
                "has_tasks": total > 0,
            })
        return result

    @staticmethod
    def _is_valid_dataset(record: dict) -> bool:
        """是否为可作为数据集的质检记录：成功 且 local_copy_dir 含 analysis。"""
        if record.get("status") != "success":
            return False
        return "analysis" in (record.get("local_copy_dir") or "").lower()

    def datasets_all(self) -> list[dict]:
        """Session 管理列表：只返回被手动添加进管理（in_manage=1）的数据集。"""
        result = []
        for record in list_records_for_datasets(limit=1000, in_manage=True):
            if not self._is_valid_dataset(record):
                continue
            artifacts_valid = True
            quality_error = ""
            d = Path(record["local_copy_dir"])
            if not d.is_dir():
                artifacts_valid = False
                quality_error = f"质检目录不存在: {d}"
            runs = db.list_runs(self.config.db_path, record["id"])
            latest = self._pick_latest_run(runs)
            status, counts = self._dataset_status(record["id"], runs)
            total = sum(counts.values())
            result.append({
                **record,
                "artifacts_valid": artifacts_valid,
                "error": quality_error or record.get("error_message", ""),
                "latest_run": latest,
                "dataset_status": status,
                "dataset_counts": counts,
                "has_tasks": total > 0,
            })
        return result

    def available_datasets(self) -> list[dict]:
        """可添加池：有效但尚未加入管理列表（in_manage=0）的数据集（轻量，不查 runs）。"""
        result = []
        for record in list_records_for_datasets(limit=1000, in_manage=False):
            if not self._is_valid_dataset(record):
                continue
            result.append({
                "id": record["id"],
                "key_slot": record["key_slot"],
                "mode": record.get("mode", "export"),
                "created_at": record.get("created_at", ""),
                "total_sessions": record.get("total_sessions", 0),
                "obs_dst": record.get("obs_dst", ""),
            })
        return result

    def add_to_manage(self, body: dict) -> dict:
        """将选中的导出结果加入 Session 管理列表（支持多选）。

        可选 body["name"]：仅在单条导入时作为自定义显示名（留空则回退原 key_slot）。
        """
        raw_ids = body.get("ids") or []
        ids = []
        for x in raw_ids:
            try:
                rid = int(x)
            except (TypeError, ValueError):
                continue
            record = get_record(rid)
            if record and self._is_valid_dataset(record):
                ids.append(rid)
        if not ids:
            raise ValueError("没有可添加的有效数据集")
        set_in_manage_bulk(ids, True)
        name = str(body.get("name") or "").strip()
        if name and len(ids) == 1:
            set_manage_name(ids[0], name)
        return {"status": "ok", "added": len(ids)}

    def remove_from_manage(self, record_id: int) -> dict:
        """从管理列表移除（仅解除关联，不删 export 记录与本地/OBS 数据）。"""
        record = get_record(record_id)
        if not record:
            raise ValueError("记录不存在")
        set_in_manage(record_id, False)
        return {"status": "ok"}

    def create_run(self, body: dict) -> dict:
        source_id = int(body["source_export_id"])
        source = get_record_resolved(source_id)
        if not source or source["status"] != "success" or "analysis" not in (source.get("local_copy_dir") or "").lower():
            raise ValueError("只能使用 local_copy_dir 包含 analysis 的成功质检记录")
        quality_root = Path(source["local_copy_dir"])
        validate_quality_dir(quality_root)
        key_id = int(body["reflection_api_key_id"])
        key = get_key_full(key_id)
        if not key or key["status"] != "active": raise ValueError("Reflection Key 已禁用或不存在")
        method = body.get("method", "bulk")
        prompt = load_prompt(self.config.prompt_dir, method)
        worker_count = max(1, min(int(body.get("worker_count", 4)), 32))
        max_retries = max(1, min(int(body.get("max_retries", 3)), 10))
        endpoint = self.config.reflection_base_url
        model = str(body.get("reflection_model", "")).strip()
        if not model:
            raise ValueError("Reflection 模型不能为空")
        snapshot = {"source_key": source["key_slot"], "quality_dir": quality_root.as_posix(),
                    "endpoint": endpoint, "model": model, "method": method,
                    "worker_count": worker_count, "max_retries": max_retries,
                    "stream": bool(body.get("stream", False)),
                    "max_tokens": max(1, min(int(body.get("max_tokens", 16384)), 65536)),
                    "prompt_name": prompt.name, "prompt_sha256": prompt.sha256}
        run_id = db.create_run(self.config.db_path, {
            "source_key": source["key_slot"], "source_export_id": source_id, "quality_record_id": source["id"],
            "reflection_endpoint": endpoint, "reflection_api_key_id": key_id, "reflection_key_mask": mask_key(key["key"]),
            "reflection_model": model, "method": method, "worker_count": worker_count, "max_retries": max_retries,
            "export_root": (self.config.export_root / source["key_slot"]).as_posix(), "snapshot": snapshot,
            "prompt_name": prompt.name, "prompt_sha256": prompt.sha256, "prompt_loaded_at": prompt.loaded_at,
        })
        imported = import_run(self.config.db_path, source_id, quality_root, max_retries,
                              db.task_detail_dir(source["key_slot"], source.get("created_at", "")))
        return {**db.get_run(self.config.db_path, run_id), **imported}

    @staticmethod
    def _load_raw_json(row) -> dict:
        source_root = (row["source_root"] or "").strip()
        if not source_root:
            raise ValueError(f"trajectory 缺少 source_root: {row['trajectory_path']}")
        p = Path(source_root) / row["trajectory_path"]
        if not p.is_file():
            raise ValueError(f"轨迹文件不存在: {p}")
        return json.loads(p.read_text(encoding="utf-8"))

    @staticmethod
    def _load_task_detail(task: dict) -> dict:
        dp = task.get("detail_path") or ""
        if dp:
            try:
                return json.loads(Path(dp).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
        return {
            "original_thinking": task.get("original_thinking"),
            "signature": task.get("signature"),
            "processed_text": task.get("latest_processed_text") or task.get("processed_text"),
            "tool_input": json.loads(task["tool_input_json"]) if task.get("tool_input_json") else None,
        }

    def tasks(self, run_id: str, status: str | None = None,
              offset: int = 0, limit: int = 50) -> dict:
        export_id = db.resolve_export_id(self.config.db_path, run_id)
        if export_id is None:
            return {"items": [], "total": 0}
        with db.connect(self.config.db_path) as conn:
            base_sql = "FROM dataset_tasks WHERE export_id=?"
            params: list = [export_id]
            if status:
                base_sql += " AND latest_status=?"
                params.append(status)
            total = conn.execute("SELECT COUNT(*) " + base_sql, params).fetchone()[0]
            rows = conn.execute(
                "SELECT * " + base_sql + " ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                params + [limit, offset],
            ).fetchall()
            result = []
            for row in rows:
                item = dict(row)
                for k in ("signature", "original_thinking", "latest_processed_text"):
                    item.pop(k, None)
                # UI-compat aliases
                item["status"] = item.get("latest_status")
                item["run_id"] = item.get("latest_run_id") or ""
                item["model"] = item.get("latest_model")
                item["response_id"] = item.get("latest_response_id")
                item["stop_reason"] = item.get("latest_stop_reason")
                item["sentence_count"] = item.get("latest_sentence_count")
                item["usage_json"] = item.get("latest_usage_json")
                result.append(item)
            return {"items": result, "total": total}

    def retry(self, task_uuid: str) -> None:
        with db.connect(self.config.db_path) as conn:
            row = conn.execute("SELECT export_id FROM dataset_tasks WHERE uuid=?", (task_uuid,)).fetchone()
            if not row: raise ValueError("Task 不存在")
            conn.execute(
                "UPDATE dataset_tasks SET latest_status='pending',last_error=NULL,updated_at=? WHERE uuid=?",
                (time.time(), task_uuid),
            )

    def trajectory(self, run_id: str, trajectory_id: str) -> dict:
        export_id = db.resolve_export_id(self.config.db_path, run_id)
        if export_id is None:
            raise ValueError("Run 不存在")
        with db.connect(self.config.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM dataset_trajectories WHERE trajectory_id=? AND export_id=?",
                (trajectory_id, export_id),
            ).fetchone()
            if not row: raise ValueError("Trajectory 不存在")
            tasks = [dict(x) for x in conn.execute(
                "SELECT * FROM dataset_tasks WHERE export_id=? AND trajectory_id=? ORDER BY block_path",
                (export_id, trajectory_id))]
        for task in tasks:
            detail = self._load_task_detail(task)
            task["processed_text"] = detail.get("processed_text")
        raw = self._load_raw_json(row)
        merged = merge(raw, tasks, run_id)
        append_response_message(merged, raw)
        return {"trajectory": {k: row[k] for k in ("trajectory_id", "session_id", "trajectory_path")},
                "merged": merged,
                "tasks": [{k: v for k, v in task.items() if k not in {"signature", "original_thinking"}} for task in tasks]}

    def export(self, run_id: str) -> dict:
        run = db.get_run(self.config.db_path, run_id)
        if not run: raise ValueError("Run 不存在")
        export_id = int(run["source_export_id"])
        root = Path(run["export_root"]) / run_id
        outputs = []
        with db.connect(self.config.db_path) as conn:
            trajectories = conn.execute(
                "SELECT * FROM dataset_trajectories WHERE export_id=?", (export_id,),
            ).fetchall()
            for row in trajectories:
                tasks = [dict(x) for x in conn.execute(
                    "SELECT * FROM dataset_tasks WHERE export_id=? AND trajectory_id=?",
                    (export_id, row["trajectory_id"]))]
                for task in tasks:
                    detail = self._load_task_detail(task)
                    task["processed_text"] = detail.get("processed_text")
                out = root / row["session_id"] / (Path(row["trajectory_path"]).stem + "--thinking.json")
                out.parent.mkdir(parents=True, exist_ok=True)
                raw = self._load_raw_json(row)
                merged = merge(raw, tasks, run_id)
                out.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                db.record_run_output(self.config.db_path, run_id, export_id,
                                     row["trajectory_id"], out.as_posix())
                outputs.append(out.as_posix())
        from .result_exporter import upload_run_to_obs
        try:
            obs_result = upload_run_to_obs(self.config.db_path, run_id)
        except ValueError:
            obs_result = None
        result = {"root": root.as_posix(), "count": len(outputs), "outputs": outputs}
        if obs_result:
            result["obs"] = obs_result
        return result

    @staticmethod
    def active_keys() -> list[dict]:
        return [{"id": key["id"], "name": key["name"], "key": key["key"]} for key in list_keys() if key["status"] == "active"]

    def get_task(self, task_uuid: str) -> dict:
        task = db.get_task(self.config.db_path, task_uuid)
        if not task:
            raise ValueError("Task 不存在")
        detail = self._load_task_detail(task)
        task.update(detail)
        return task

    def task_attempts(self, task_uuid: str) -> list[dict]:
        return db.list_attempts(self.config.db_path, task_uuid)

    def retry_all_failed(self, run_id: str) -> dict:
        export_id = db.resolve_export_id(self.config.db_path, run_id)
        if export_id is None:
            return {"run_id": run_id, "retried": 0}
        count = db.retry_all_failed(self.config.db_path, export_id)
        return {"run_id": run_id, "retried": count}

    def rerun_all_done(self, run_id: str) -> dict:
        export_id = db.resolve_export_id(self.config.db_path, run_id)
        if export_id is None:
            return {"run_id": run_id, "reset": 0}
        count = db.reset_all_done(self.config.db_path, export_id)
        return {"run_id": run_id, "reset": count}

    def delete_run(self, run_id: str) -> None:
        self.workers.stop(run_id)
        db.delete_run(self.config.db_path, run_id)

    def trajectory_list(self, run_id: str, offset: int = 0, limit: int = 0,
                        search: str = "") -> dict:
        export_id = db.resolve_export_id(self.config.db_path, run_id)
        if export_id is None:
            return {"items": [], "total": 0}
        needle = (search or "").strip().lower()
        if needle:
            # 跨全部轨迹按首句 q1 / session_id 过滤：一次取全、构建 q1 映射后过滤再分页
            items = db.list_trajectories(self.config.db_path, export_id, run_id=run_id,
                                         offset=0, limit=0)
            q1map: dict = {}
            try:
                for s in self.dataset_sessions(export_id, 0, 10 ** 9).get("items", []):
                    q1map[s.get("session_id", "")] = s.get("q1", "") or ""
            except Exception:
                q1map = {}
            items = [
                it for it in items
                if needle in (it.get("session_id", "") or "").lower()
                or needle in (q1map.get(it.get("session_id", ""), "") or "").lower()
            ]
            total = len(items)
            paged = items[offset:offset + limit] if limit > 0 else items[offset:]
            return {"items": paged, "total": total}
        total = db.count_trajectories(self.config.db_path, export_id)
        items = db.list_trajectories(self.config.db_path, export_id, run_id=run_id,
                                     offset=offset, limit=limit)
        return {"items": items, "total": total}

    def test(self, body: dict) -> dict:
        key_id = int(body["reflection_api_key_id"])
        key = get_key_full(key_id)
        if not key or key["status"] != "active":
            raise ValueError("Reflection Key 已禁用或不存在")
        method = body.get("method", "bulk")
        prompt = load_prompt(self.config.prompt_dir, method)
        user_sig = str(body.get("signature", "")).strip()
        user_thinking = str(body.get("thinking", "")).strip() or None
        if user_sig:
            sample = {"signature": user_sig, "original_thinking": user_thinking}
        else:
            source_id = body.get("source_export_id")
            if not source_id:
                raise ValueError("未提供 signature 时必须选择数据集")
            source = get_record_resolved(int(source_id))
            if not source or source["status"] != "success":
                raise ValueError("源记录不可用")
            quality_root = Path(source["local_copy_dir"])
            validate_quality_dir(quality_root)
            sample = self._find_sample(quality_root)
            if not sample:
                raise ValueError("数据集中未找到包含 signature 的样本")
        endpoint = self.config.reflection_base_url
        model = str(body.get("reflection_model", "")).strip()
        result = reflect(
            endpoint=endpoint,
            api_key=key["key"],
            model=model,
            instruction=prompt.instruction,
            tool=prompt.tool,
            unrelated_thinking=prompt.unrelated_thinking,
            signature=sample["signature"],
            thinking=sample.get("original_thinking"),
            method=method,
            stream=bool(body.get("stream", False)),
            max_tokens=max(1, min(int(body.get("max_tokens", 16384)), 65536)),
            timeout=max(60, min(int(body.get("timeout", 600)), 1200)),
        )
        return {
            "text": result.text, "sentence_count": result.sentence_count,
            "model": result.model, "response_id": result.response_id,
            "stop_reason": result.stop_reason, "usage": result.usage,
            "tool_input": result.tool_input,
            "endpoint": endpoint, "request_model": model,
            "signature_used": sample["signature"][:80],
        }

    @staticmethod
    def _find_sample(quality_root: Path) -> dict | None:
        for json_file in sorted(quality_root.rglob("*.json")):
            if json_file.name.startswith("."):
                continue
            try:
                data = json.loads(json_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            for sig in extract_signatures(data):
                if sig.get("signature"):
                    return sig
        return None

    def analysis(self, record_id: int) -> dict:
        eval_rec = latest_quality_record(record_id)
        if not eval_rec:
            return {"eval_status": "未质检", "sessions": []}
        analysis_json = eval_rec.get("analysis_json") or ""
        if analysis_json.startswith("file://"):
            try:
                analysis_json = Path(analysis_json[7:]).read_text(encoding="utf-8")
            except OSError:
                analysis_json = "{}"
        try:
            data = json.loads(analysis_json) if analysis_json else {}
        except json.JSONDecodeError:
            data = {}
        return {
            "eval_status": {"success": "已质检", "failed": "质检失败"}.get(eval_rec["status"], "质检中"),
            "eval_record_id": eval_rec["id"],
            "sessions": data.get("sessions", []) if isinstance(data, dict) else [],
        }

    _EXCLUDED_FILES = {"session_analysis.json", "session_index.json", "session_index.jsonl",
                       "failure_report.json", "manifest.json",
                       ".session_cache.json", ".session_cache.jsonl",
                       "session_report.html", "session_report.md", "session_report.xlsx"}

    def dataset_sessions(self, record_id: int, offset: int = 0, limit: int = 50,
                         force: bool = False, search: str = "") -> dict:
        record = get_record(record_id)
        if not record:
            raise ValueError("记录不存在")
        root = Path(record["local_copy_dir"])
        cache_path = root / ".session_cache.json"

        needle = (search or "").strip().lower()

        def _finalize(sessions: list) -> dict:
            # 按首句 q1 / session_id 过滤（覆盖全部会话后再分页），与 chat-viewer 一致
            if needle:
                sessions = [
                    s for s in sessions
                    if needle in (s.get("q1", "") or "").lower()
                    or needle in (s.get("session_id", "") or "").lower()
                ]
            return {"items": sessions[offset:offset + limit], "total": len(sessions)}

        # 自动穿透：如果 root 下没有 session_analysis.json，但只有一个子目录（同名嵌套），
        # 则往下一层查找（obsutil cp 上传目录时会把目录名本身复制进去造成双层嵌套）
        analysis_path = root / "session_analysis.json"
        if not analysis_path.is_file() and root.is_dir():
            subdirs = [d for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")]
            if len(subdirs) == 1 and (subdirs[0] / "session_analysis.json").is_file():
                root = subdirs[0]
                analysis_path = root / "session_analysis.json"
                cache_path = root / ".session_cache.json"

        if analysis_path.is_file():
            source, source_mtime = "session_analysis", analysis_path.stat().st_mtime
        else:
            source, source_mtime = "filesystem", root.stat().st_mtime if root.is_dir() else 0

        if not force and cache_path.is_file():
            try:
                cache = json.loads(cache_path.read_text(encoding="utf-8"))
                meta = cache.get("_meta", {})
                if meta.get("source_mtime") == source_mtime:
                    sessions = cache.get("sessions", [])
                    return _finalize(sessions)
            except (json.JSONDecodeError, OSError):
                pass

        if source == "session_analysis":
            sessions = self._build_sessions_from_analysis(root, analysis_path)
        else:
            sessions = self._build_sessions_from_filesystem(root)

        from datetime import datetime
        cache_data = {
            "_meta": {
                "source": source,
                "source_mtime": source_mtime,
                "generated_at": datetime.now().isoformat(),
                "total": len(sessions),
            },
            "sessions": sessions,
        }
        try:
            tmp = cache_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(cache_data, ensure_ascii=False, indent=1), encoding="utf-8")
            tmp.replace(cache_path)
        except OSError:
            pass

        return _finalize(sessions)

    @staticmethod
    def _order_by_trace_list(disk_names: list, trace_list) -> tuple:
        """按 trace_list 的调用顺序给磁盘轨迹文件排序，返回 (ordered_names, latest_file)。

        trace_list 是权威的调用顺序（末个 = 一连串 API 调用的最后一次）。规则：
        - 交集排序：先按 trace_list 里 filename 的顺序取出磁盘上存在的文件；
        - 补漏：磁盘上有、但 trace_list 没提到的文件，按文件名排序追加到末尾（不丢文件）；
        - latest_file 取 trace_list 里最后一个「磁盘存在」的 filename；trace_list 全部缺失时
          回退到排序后的末个磁盘文件。
        trace_list 为空/非法时返回 (None, None)，由调用方走 glob 排序的退化路径。
        """
        if not isinstance(trace_list, list) or not trace_list:
            return None, None
        disk_set = set(disk_names)
        seen = set()
        ordered = []
        latest_from_trace = ""
        for tr in trace_list:
            fn = tr.get("filename", "") if isinstance(tr, dict) else ""
            if fn in disk_set and fn not in seen:
                ordered.append(fn)
                seen.add(fn)
            if fn in disk_set:
                latest_from_trace = fn  # trace_list 里最后一个磁盘存在的文件
        # 补漏：磁盘多出来、trace_list 未提及的文件，按文件名排序追加
        extras = sorted(n for n in disk_names if n not in seen)
        ordered.extend(extras)
        latest_file = latest_from_trace or (ordered[-1] if ordered else "")
        return ordered, latest_file

    def _build_sessions_from_analysis(self, root: Path, analysis_path: Path) -> list:
        payload = json.loads(analysis_path.read_text(encoding="utf-8"))
        raw_sessions = payload.get("sessions", []) if isinstance(payload, dict) else payload
        if not isinstance(raw_sessions, list):
            raw_sessions = []
        items = []
        for s in raw_sessions:
            sid = s.get("session", "")
            session_dir = root / sid
            traj_files = sorted(session_dir.glob("*.json")) if session_dir.is_dir() else []
            traj_files = [f for f in traj_files
                          if f.name not in self._EXCLUDED_FILES and not f.name.endswith("--thinking.json")]
            glob_names = [f.name for f in traj_files]
            # 定位轨迹顺序与 latest_file：trace_list 是权威调用顺序，优先用它。
            names, latest_file = self._order_by_trace_list(glob_names, s.get("trace_list"))
            if names is None:
                # 无 trace_list：退化到按文件名（时间戳前缀）排序；latest_file 优先 analysis 显式声明，
                # 否则取排序末个（字典序即时间序，末个即最后一次调用）。
                names = glob_names
                declared_latest = s.get("latest_file") or ""
                latest_file = declared_latest if declared_latest in names else (names[-1] if names else "")
            items.append({
                "session_id": sid,
                "q1": s.get("q1", ""),
                "model": s.get("model", ""),
                "start_time": s.get("start_time", ""),
                "duration_s": s.get("duration_s", 0),
                "api_call_count": s.get("api_call_count", 0),
                "completed": s.get("completed", 0),
                "completed_note": s.get("completed_note", ""),
                "trajectory_count": len(names),
                "trajectory_files": names,
                "latest_file": latest_file,
            })
        return items

    def _build_sessions_from_filesystem(self, root: Path) -> list:
        if not root.is_dir():
            return []
        items = []
        for session_dir in sorted(root.iterdir()):
            if not session_dir.is_dir() or session_dir.name.startswith("."):
                continue
            traj_files = sorted(session_dir.glob("*.json"))
            traj_files = [f for f in traj_files
                          if f.name not in self._EXCLUDED_FILES and not f.name.endswith("--thinking.json")]
            if not traj_files:
                continue
            q1 = ""
            try:
                first = json.loads(traj_files[0].read_text(encoding="utf-8"))
                msgs = first.get("messages") or (first.get("request", {}) or {}).get("messages") or []
                for m in msgs:
                    if m.get("role") == "user":
                        c = m.get("content", "")
                        if isinstance(c, str):
                            q1 = c[:200]
                        elif isinstance(c, list):
                            for part in c:
                                if isinstance(part, dict) and part.get("type") == "text":
                                    q1 = (part.get("text") or "")[:200]
                                    break
                        break
            except (json.JSONDecodeError, OSError):
                pass
            items.append({
                "session_id": session_dir.name,
                "q1": q1,
                "model": "",
                "start_time": "",
                "duration_s": 0,
                "api_call_count": 0,
                "completed": 0,
                "completed_note": "",
                "trajectory_count": len(traj_files),
                "trajectory_files": [f.name for f in traj_files],
                # 无 analysis，直接取排序后最后一个（= 一连串调用的最后一次）
                "latest_file": traj_files[-1].name,
            })
        return items

    def session_trajectory(self, record_id: int, session_id: str, file_name: str) -> dict:
        record = get_record(record_id)
        if not record:
            raise ValueError("记录不存在")
        root = Path(record["local_copy_dir"])
        traj_path = root / session_id / file_name
        if not traj_path.is_file():
            raise ValueError(f"文件不存在: {traj_path}")
        raw = json.loads(traj_path.read_text(encoding="utf-8"))
        relative = f"{session_id}/{file_name}"
        tasks = []
        with db.connect(self.config.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM dataset_tasks "
                "WHERE export_id=? AND trajectory_path=? AND latest_status='done' "
                "ORDER BY block_path",
                (record_id, relative),
            ).fetchall()
            for row in rows:
                task = dict(row)
                detail = self._load_task_detail(task)
                task["processed_text"] = detail.get("processed_text")
                tasks.append(task)
        run_id = tasks[0].get("latest_run_id") or "" if tasks else ""
        # tasks 为空时 merge 不会执行（merged=raw），此处 deepcopy 一份避免就地修改 raw
        merged = merge(raw, tasks, run_id) if tasks else copy.deepcopy(raw)
        append_response_message(merged, raw)
        return {
            "merged": merged,
            "tasks": [{k: v for k, v in t.items() if k not in {"signature", "original_thinking"}} for t in tasks],
            "has_reflect": len(tasks) > 0,
        }

    # ------------------------------------------------------------------
    # 注册外部数据集：登记一个已存在于 OBS 的数据集（来自其他平台导出）
    # ------------------------------------------------------------------

    def register_external(self, body: dict) -> dict:
        obs_dst = str(body.get("obs_dst") or "").strip()
        if not obs_dst or not obs_dst.startswith("obs://"):
            raise ValueError("OBS 路径必须以 obs:// 开头")
        if not obs_dst.endswith("/"):
            obs_dst += "/"
        name = str(body.get("name") or "").strip()
        if not name:
            raise ValueError("名称不能为空")
        try:
            total_sessions = max(0, int(body.get("total_sessions") or 0))
        except (TypeError, ValueError):
            total_sessions = 0

        # local_copy_dir 必须落在 analysis 根下（datasets_all 过滤要求路径含 "analysis"），
        # 目录此时可以尚不存在，"导入任务库" 会按 obs_dst 自动下载到此处。
        import re
        slug_base = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-") or "dataset"
        slug = f"{slug_base}-{int(time.time())}"
        local_copy_dir = (Path("logs_session_analysis") / "external" / slug).as_posix()

        record_id = create_record(
            api_key="",
            key_slot=name,
            mtime_dirs="[]",
            obs_dst=obs_dst,
            local_copy_dir=local_copy_dir,
            mode="external",
        )
        update_status(record_id, "success", total_sessions=total_sessions)
        set_in_manage(record_id, True)  # 注册即加入管理列表
        return {"status": "ok", "id": record_id}

    # ------------------------------------------------------------------
    # 导入任务库：从 export record 提取 signature 写入 dataset_tasks
    # ------------------------------------------------------------------

    def preview_tasks(self, body: dict) -> dict:
        """只读预统计：不写库，返回该导出可解析出的 signature 任务规模。

        与 import_tasks 共用 quality_root 定位逻辑，但本地目录不存在时**不**自动
        从 OBS 下载——保持统计快、纯只读；缺目录直接提示先下载/导入。
        """
        source_id = int(body["source_export_id"])
        source = get_record_resolved(source_id)
        if not source:
            raise ValueError("记录不存在")

        with db.connect(self.config.db_path) as conn:
            existing = conn.execute(
                "SELECT COUNT(*) FROM dataset_tasks WHERE export_id=?", (source_id,)
            ).fetchone()[0]

        quality_root = Path(source.get("local_copy_dir") or "")
        if not quality_root.is_dir():
            raise ValueError("本地质检目录不存在，请先下载/导入后再统计")

        stats = preview_run(quality_root)
        stats["already_imported"] = int(existing)
        return {"status": "ok", **stats}

    def import_tasks(self, body: dict) -> dict:
        source_id = int(body["source_export_id"])
        source = get_record_resolved(source_id)
        if not source:
            raise ValueError("记录不存在")

        with db.connect(self.config.db_path) as conn:
            existing = conn.execute(
                "SELECT COUNT(*) FROM dataset_tasks WHERE export_id=?", (source_id,)
            ).fetchone()[0]
        if existing:
            return {"status": "already_imported", "tasks": existing}

        quality_root = Path(source.get("local_copy_dir") or "")
        if not quality_root.is_dir():
            obs_dst = (source.get("obs_dst") or "").strip()
            if not obs_dst:
                raise ValueError("本地目录不存在且无 OBS 路径，无法下载")
            local_dir = str(quality_root)
            Path(local_dir).mkdir(parents=True, exist_ok=True)
            ok, msg = run_download_cmd(obs_dst, local_dir, timeout=600)
            if not ok:
                raise ValueError(f"OBS 自动下载失败: {msg}")

        max_retries = max(1, min(int(body.get("max_retries", 3)), 10))
        imported = import_run(self.config.db_path, source_id, quality_root, max_retries,
                              db.task_detail_dir(source["key_slot"], source.get("created_at", "")))
        return {"status": "imported", **imported}

    # ------------------------------------------------------------------
    # OBS 下载
    # ------------------------------------------------------------------

    @staticmethod
    def download_obs(record_id: int) -> dict:
        record = get_record(record_id)
        if not record:
            raise ValueError("记录不存在")
        obs_dst = (record.get("obs_dst") or "").strip()
        local_dir = (record.get("local_copy_dir") or "").strip()
        if not obs_dst:
            raise ValueError("该记录没有 OBS 路径")
        if not local_dir:
            raise ValueError("该记录没有 local_copy_dir")
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        ok, msg = run_download_cmd(obs_dst, local_dir, timeout=600)
        if not ok:
            raise ValueError(f"OBS 下载失败: {msg}")
        return {"status": "ok", "local_path": local_dir, "message": msg}

    # ------------------------------------------------------------------
    # 更新 run 配置
    # ------------------------------------------------------------------

    def update_run_config(self, run_id: str, body: dict) -> dict:
        run = db.get_run(self.config.db_path, run_id)
        if not run:
            raise ValueError("Run 不存在")
        if run["status"] in ("running",):
            raise ValueError("运行中的 Run 不能修改配置，请先暂停")
        updates: dict[str, Any] = {}
        if "reflection_endpoint" in body:
            endpoint = str(body["reflection_endpoint"]).strip().rstrip("/")
            if not endpoint:
                raise ValueError("Endpoint 不能为空")
            updates["reflection_endpoint"] = endpoint
        if "reflection_api_key_id" in body:
            key_id = int(body["reflection_api_key_id"])
            key = get_key_full(key_id)
            if not key or key["status"] != "active":
                raise ValueError("Reflection Key 已禁用或不存在")
            updates["reflection_api_key_id"] = key_id
            updates["reflection_key_mask"] = mask_key(key["key"])
        if "reflection_model" in body:
            model = str(body["reflection_model"]).strip()
            if not model:
                raise ValueError("模型不能为空")
            updates["reflection_model"] = model
        if "method" in body:
            method = body["method"]
            load_prompt(self.config.prompt_dir, method)
            updates["method"] = method
        if "worker_count" in body:
            updates["worker_count"] = max(1, min(int(body["worker_count"]), 32))
        if "max_retries" in body:
            updates["max_retries"] = max(1, min(int(body["max_retries"]), 10))
        if updates:
            snapshot = db.get_run_snapshot(self.config.db_path, run_id)
            for k, v in updates.items():
                if k == "reflection_model":
                    snapshot["model"] = v
                elif k in snapshot:
                    snapshot[k] = v
            if "max_tokens" in body:
                snapshot["max_tokens"] = max(1, min(int(body["max_tokens"]), 65536))
            updates["config_snapshot"] = json.dumps(snapshot, ensure_ascii=False)
            db.update_run_config(self.config.db_path, run_id, updates)
        return {"status": "updated", "run_id": run_id}

    # ------------------------------------------------------------------
    # 批量操作（启动 / 暂停 / 取消 / 失败重试）
    # ------------------------------------------------------------------

    @staticmethod
    def _is_placeholder(run: dict) -> bool:
        """A placeholder run is the empty draft row created by import-tasks."""
        return (run.get("status") == "draft"
                and not (run.get("reflection_endpoint") or "").strip()
                and not (run.get("reflection_model") or "").strip()
                and not (run.get("method") or "").strip())

    def _apply_config_to_run(self, run_id: str, config: dict) -> None:
        """Push a batch config onto an existing (placeholder) run row."""
        body = dict(config or {})
        if not body:
            return
        body.setdefault("reflection_endpoint", self.config.reflection_base_url)
        self.update_run_config(run_id, body)

    def _build_run_from_config(self, source: dict, config: dict,
                               counts: dict[str, int], launch_type: str,
                               parent_run_id: str | None = None) -> dict:
        """Assemble the run_values dict for db.create_run from a bare config.

        Used when there is no prior run to copy fields from (fresh dataset that
        was just imported).
        """
        method = str(config.get("method", "bulk")).strip() or "bulk"
        prompt = load_prompt(self.config.prompt_dir, method)
        key_id = int(config.get("reflection_api_key_id") or 0)
        key = get_key_full(key_id) if key_id else None
        if not key or key["status"] != "active":
            raise ValueError("Reflection Key 已禁用或不存在")
        model = str(config.get("reflection_model", "")).strip()
        if not model:
            raise ValueError("Reflection 模型不能为空")
        endpoint = (str(config.get("reflection_endpoint") or "").strip()
                    or self.config.reflection_base_url).rstrip("/")
        worker_count = max(1, min(int(config.get("worker_count", 4)), 32))
        max_retries = max(1, min(int(config.get("max_retries", 3)), 10))
        snapshot = {"source_key": source["key_slot"],
                    "quality_dir": (source.get("local_copy_dir") or ""),
                    "endpoint": endpoint, "model": model, "method": method,
                    "worker_count": worker_count, "max_retries": max_retries,
                    "stream": bool(config.get("stream", False)),
                    "max_tokens": max(1, min(int(config.get("max_tokens", 16384)), 65536)),
                    "prompt_name": prompt.name, "prompt_sha256": prompt.sha256}
        return {
            "source_key": source["key_slot"],
            "source_export_id": int(source["id"]),
            "quality_record_id": int(source["id"]),
            "reflection_endpoint": endpoint,
            "reflection_api_key_id": key_id,
            "reflection_key_mask": mask_key(key["key"]),
            "reflection_model": model,
            "method": method,
            "worker_count": worker_count,
            "max_retries": max_retries,
            "export_root": (self.config.export_root / source["key_slot"]).as_posix(),
            "obs_root": "",
            "snapshot": snapshot,
            "prompt_name": prompt.name,
            "prompt_sha256": prompt.sha256,
            "prompt_loaded_at": prompt.loaded_at,
            "launch_type": launch_type,
            "parent_run_id": parent_run_id,
            "snapshot_total": sum(counts.values()),
            "snapshot_pending": counts["pending"],
        }

    def batch_start(self, body: dict) -> dict:
        export_ids = body.get("source_export_ids", [])
        config = body.get("config")
        if not export_ids:
            return {"started": [], "count": 0}
        started = []
        for eid in export_ids:
            eid = int(eid)
            try:
                with db.connect(self.config.db_path) as conn:
                    counts = db.dataset_counts(conn, eid)
                if sum(counts.values()) == 0:
                    # not imported yet — nothing to start
                    continue
                if counts["pending"] == 0:
                    continue
                runs = db.list_runs(self.config.db_path, eid)
                # Reuse the import-time placeholder draft in-place instead of forking a new run.
                placeholder = next((r for r in runs if self._is_placeholder(r)), None)
                if placeholder and config:
                    try:
                        self._apply_config_to_run(placeholder["run_id"], config)
                        self.workers.start(placeholder["run_id"])
                        started.append(placeholder["run_id"])
                        continue
                    except ValueError:
                        continue
                latest = self._pick_latest_run(runs)
                if latest is None:
                    if not config:
                        continue
                    source = get_record_resolved(eid)
                    if not source:
                        continue
                    run_values = self._build_run_from_config(source, config, counts, "start")
                    new_run_id = db.create_run(self.config.db_path, run_values)
                    try:
                        self.workers.start(new_run_id)
                    except ValueError:
                        continue
                    started.append(new_run_id)
                    continue
                run_values = {
                    "source_key": latest["source_key"],
                    "source_export_id": eid,
                    "quality_record_id": latest["quality_record_id"],
                    "reflection_endpoint": latest["reflection_endpoint"],
                    "reflection_api_key_id": latest["reflection_api_key_id"],
                    "reflection_key_mask": latest["reflection_key_mask"],
                    "reflection_model": latest["reflection_model"],
                    "method": latest["method"],
                    "worker_count": latest["worker_count"],
                    "max_retries": latest["max_retries"],
                    "export_root": latest["export_root"],
                    "obs_root": latest.get("obs_root", ""),
                    "snapshot": db.get_run_snapshot(self.config.db_path, latest["run_id"]),
                    "prompt_name": latest["prompt_name"],
                    "prompt_sha256": latest["prompt_sha256"],
                    "prompt_loaded_at": latest["prompt_loaded_at"],
                    "launch_type": "start",
                    "parent_run_id": latest["run_id"],
                    "snapshot_total": sum(counts.values()),
                    "snapshot_pending": counts["pending"],
                }
                if config:
                    for k in ("reflection_endpoint", "reflection_api_key_id", "reflection_model",
                              "method", "worker_count", "max_retries"):
                        if k in config:
                            run_values[k] = config[k]
                    if "reflection_api_key_id" in config:
                        key = get_key_full(int(config["reflection_api_key_id"]))
                        if key:
                            run_values["reflection_key_mask"] = mask_key(key["key"])
                if not run_values.get("reflection_endpoint"):
                    run_values["reflection_endpoint"] = self.config.reflection_base_url
                new_run_id = db.create_run(self.config.db_path, run_values)
                try:
                    self.workers.start(new_run_id)
                except ValueError:
                    # dataset already has an active run — leave the new row in draft
                    continue
                started.append(new_run_id)
            except (ValueError, KeyError):
                pass
        return {"started": started, "count": len(started)}

    def batch_pause(self, body: dict) -> dict:
        run_ids = self._resolve_run_ids(body)
        paused = []
        for rid in run_ids:
            try:
                self.workers.stop(rid)
                paused.append(rid)
            except Exception:
                pass
        return {"paused": paused, "count": len(paused)}

    def batch_cancel(self, body: dict) -> dict:
        run_ids = self._resolve_run_ids(body)
        cancelled = []
        for rid in run_ids:
            try:
                self.workers.stop(rid, cancel=True)
                run = db.get_run(self.config.db_path, rid)
                if run:
                    with db.connect(self.config.db_path) as conn:
                        conn.execute(
                            "UPDATE dataset_tasks SET latest_status='pending',updated_at=? "
                            "WHERE export_id=? AND latest_status='processing'",
                            (time.time(), int(run["source_export_id"])),
                        )
                cancelled.append(rid)
            except Exception:
                pass
        return {"cancelled": cancelled, "count": len(cancelled)}

    def batch_retry(self, body: dict) -> dict:
        export_ids = body.get("source_export_ids", [])
        total = 0
        for eid in export_ids:
            total += db.retry_all_failed(self.config.db_path, int(eid))
        return {"retried": total}

    def batch_rerun(self, body: dict) -> dict:
        export_ids = body.get("source_export_ids", [])
        config = body.get("config")
        if not export_ids:
            return {"started": [], "count": 0}
        started = []
        for eid in export_ids:
            eid = int(eid)
            try:
                reset_count = db.retry_all_failed(self.config.db_path, eid)
                if reset_count == 0:
                    continue
                runs = db.list_runs(self.config.db_path, eid)
                latest = self._pick_latest_run(runs)
                if not latest:
                    continue
                with db.connect(self.config.db_path) as conn:
                    counts = db.dataset_counts(conn, eid)
                run_values = {
                    "source_key": latest["source_key"],
                    "source_export_id": eid,
                    "quality_record_id": latest["quality_record_id"],
                    "reflection_endpoint": latest["reflection_endpoint"],
                    "reflection_api_key_id": latest["reflection_api_key_id"],
                    "reflection_key_mask": latest["reflection_key_mask"],
                    "reflection_model": latest["reflection_model"],
                    "method": latest["method"],
                    "worker_count": latest["worker_count"],
                    "max_retries": latest["max_retries"],
                    "export_root": latest["export_root"],
                    "obs_root": latest.get("obs_root", ""),
                    "snapshot": db.get_run_snapshot(self.config.db_path, latest["run_id"]),
                    "prompt_name": latest["prompt_name"],
                    "prompt_sha256": latest["prompt_sha256"],
                    "prompt_loaded_at": latest["prompt_loaded_at"],
                    "launch_type": "rerun",
                    "parent_run_id": latest["run_id"],
                    "snapshot_total": sum(counts.values()),
                    "snapshot_pending": counts["pending"],
                }
                if config:
                    for k in ("reflection_endpoint", "reflection_api_key_id", "reflection_model",
                              "method", "worker_count", "max_retries"):
                        if k in config:
                            run_values[k] = config[k]
                    if "reflection_api_key_id" in config:
                        key = get_key_full(int(config["reflection_api_key_id"]))
                        if key:
                            run_values["reflection_key_mask"] = mask_key(key["key"])
                if not run_values.get("reflection_endpoint"):
                    run_values["reflection_endpoint"] = self.config.reflection_base_url
                new_run_id = db.create_run(self.config.db_path, run_values)
                try:
                    self.workers.start(new_run_id)
                except ValueError:
                    continue
                started.append(new_run_id)
            except (ValueError, KeyError):
                pass
        return {"started": started, "count": len(started)}

    def _resolve_run_ids(self, body: dict) -> list[str]:
        """从请求体中解析出 run_ids（前端传 source_export_ids，后端查对应 run）。"""
        export_ids = body.get("source_export_ids", [])
        run_ids = body.get("run_ids", [])
        if export_ids:
            for eid in export_ids:
                runs = db.list_runs(self.config.db_path, int(eid))
                for r in runs:
                    if r["run_id"] not in run_ids:
                        run_ids.append(r["run_id"])
        return run_ids

    def all_tasks_summary(self) -> dict:
        """统计所有 run 的全局 task 汇总。"""
        with db.connect(self.config.db_path) as conn:
            rows = conn.execute(
                "SELECT latest_status, COUNT(*) as cnt FROM dataset_tasks GROUP BY latest_status"
            ).fetchall()
        counts = {"total": 0, "done": 0, "pending": 0, "processing": 0, "failed": 0}
        for row in rows:
            counts[row["latest_status"]] = row["cnt"]
            counts["total"] += row["cnt"]
        counts.update(self.workers.live_stats())
        return counts

