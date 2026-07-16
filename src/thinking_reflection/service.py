from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

from utils.export_store import (
    get_record,
    get_record_resolved,
    list_records,
    list_records_by_key,
)
from utils.key_store import get_key_full, list_keys, mask_key
from utils.obs_utils import run_download_cmd

from . import db
from .config import ReflectionConfig
from .consumer import reflect
from .extractor import extract_signatures
from .importer import import_run, validate_quality_dir
from .merger import merge
from .prompt_loader import load_prompt
from .worker_manager import WorkerManager


class ReflectionService:
    def __init__(self, config: ReflectionConfig):
        self.config = config
        db.init(config.db_path)
        db.reset_processing(config.db_path)
        self.workers = WorkerManager(config.db_path, config.prompt_dir)

    @staticmethod
    def source_key_slots() -> list[dict]:
        grouped: dict[str, list[dict]] = {}
        for record in list_records(limit=1000):
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
            latest = runs[0] if runs else None
            result.append({
                **record,
                "created_at": record["created_at"], "total_sessions": record["total_sessions"],
                "local_copy_dir": record["local_copy_dir"], "obs_dst": record["obs_dst"],
                "artifacts_valid": artifacts_valid,
                "error": quality_error or record.get("error_message", ""),
                "latest_run": latest,
            })
        return result

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
        imported = import_run(self.config.db_path, run_id, quality_root, max_retries,
                              db.task_detail_dir(source["key_slot"], source.get("created_at", "")))
        return {**db.get_run(self.config.db_path, run_id), **imported}

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
            "processed_text": task.get("processed_text"),
            "tool_input": json.loads(task["tool_input_json"]) if task.get("tool_input_json") else None,
        }

    def tasks(self, run_id: str, status: str | None = None) -> list[dict]:
        with db.connect(self.config.db_path) as conn:
            sql, params = "SELECT * FROM thinking_tasks WHERE run_id=?", [run_id]
            if status: sql, params = sql + " AND status=?", [run_id, status]
            rows = conn.execute(sql + " ORDER BY updated_at DESC LIMIT 1000", params).fetchall()
            result = []
            for row in rows:
                item = dict(row)
                for k in ("signature", "original_thinking", "processed_text", "tool_input_json"):
                    item.pop(k, None)
                result.append(item)
            return result

    def retry(self, task_uuid: str) -> None:
        with db.connect(self.config.db_path) as conn:
            row = conn.execute("SELECT run_id FROM thinking_tasks WHERE uuid=?", (task_uuid,)).fetchone()
            if not row: raise ValueError("Task 不存在")
            conn.execute("UPDATE thinking_tasks SET status='pending',last_error=NULL,updated_at=? WHERE uuid=?", (time.time(), task_uuid))
            conn.execute("UPDATE reflection_runs SET status='paused',updated_at=? WHERE run_id=?", (time.time(), row["run_id"]))

    def trajectory(self, run_id: str, trajectory_id: str) -> dict:
        with db.connect(self.config.db_path) as conn:
            row = conn.execute("SELECT * FROM run_trajectories WHERE run_id=? AND trajectory_id=?", (run_id, trajectory_id)).fetchone()
            if not row: raise ValueError("Trajectory 不存在")
            tasks = [dict(x) for x in conn.execute("SELECT * FROM thinking_tasks WHERE run_id=? AND trajectory_id=? ORDER BY block_path", (run_id, trajectory_id))]
        for task in tasks:
            detail = self._load_task_detail(task)
            task["processed_text"] = detail.get("processed_text")
        return {"trajectory": {k: row[k] for k in ("trajectory_id", "session_id", "trajectory_path", "output_path")},
                "merged": merge(json.loads(row["raw_json"]), tasks, run_id),
                "tasks": [{k: v for k, v in task.items() if k not in {"signature", "original_thinking"}} for task in tasks]}

    def export(self, run_id: str) -> dict:
        run = db.get_run(self.config.db_path, run_id)
        if not run: raise ValueError("Run 不存在")
        root = Path(run["export_root"]) / run_id
        outputs = []
        with db.connect(self.config.db_path) as conn:
            trajectories = conn.execute("SELECT * FROM run_trajectories WHERE run_id=?", (run_id,)).fetchall()
            for row in trajectories:
                tasks = [dict(x) for x in conn.execute("SELECT * FROM thinking_tasks WHERE run_id=? AND trajectory_id=?", (run_id, row["trajectory_id"]))]
                for task in tasks:
                    detail = self._load_task_detail(task)
                    task["processed_text"] = detail.get("processed_text")
                out = root / row["session_id"] / (Path(row["trajectory_path"]).stem + "--thinking.json")
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(merge(json.loads(row["raw_json"]), tasks, run_id), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                conn.execute("UPDATE run_trajectories SET output_path=?,exported_at=? WHERE id=?", (out.as_posix(), time.time(), row["id"]))
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
        count = db.retry_all_failed(self.config.db_path, run_id)
        return {"run_id": run_id, "retried": count}

    def delete_run(self, run_id: str) -> None:
        self.workers.stop(run_id)
        db.delete_run(self.config.db_path, run_id)

    def trajectory_list(self, run_id: str, offset: int = 0, limit: int = 0) -> dict:
        total = db.count_trajectories(self.config.db_path, run_id)
        items = db.list_trajectories(self.config.db_path, run_id, offset, limit)
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

    # ------------------------------------------------------------------
    # 导入任务库：从 export record 提取 signature 写入 thinking_tasks
    # ------------------------------------------------------------------

    def import_tasks(self, body: dict) -> dict:
        source_id = int(body["source_export_id"])
        source = get_record_resolved(source_id)
        if not source:
            raise ValueError("记录不存在")

        existing = db.list_runs(self.config.db_path, source_id)
        if existing:
            run = existing[0]
            return {"run_id": run["run_id"], "status": "already_imported", "tasks": run.get("total_count", 0)}

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
        run_id = db.create_run(self.config.db_path, {
            "source_key": source["key_slot"], "source_export_id": source_id,
            "quality_record_id": source["id"],
            "reflection_endpoint": "", "reflection_api_key_id": 0,
            "reflection_key_mask": "", "reflection_model": "", "method": "",
            "worker_count": 1, "max_retries": max_retries,
            "export_root": (self.config.export_root / source["key_slot"]).as_posix(),
            "snapshot": {"source_key": source["key_slot"], "quality_dir": quality_root.as_posix()},
            "prompt_name": "", "prompt_sha256": "", "prompt_loaded_at": "",
        })
        imported = import_run(self.config.db_path, run_id, quality_root, max_retries,
                              db.task_detail_dir(source["key_slot"], source.get("created_at", "")))
        return {"run_id": run_id, "status": "imported", **imported}

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

    def batch_start(self, body: dict) -> dict:
        export_ids = body.get("source_export_ids", [])
        config = body.get("config")
        if not export_ids:
            return {"started": [], "count": 0}
        started = []
        for eid in export_ids:
            eid = int(eid)
            try:
                task_group_id = db.get_task_group_id(self.config.db_path, eid)
                if not task_group_id:
                    continue
                runs = db.list_runs(self.config.db_path, eid)
                latest = runs[0] if runs else None
                if not latest:
                    continue
                with db.connect(self.config.db_path) as conn:
                    counts = db.run_counts(conn, task_group_id)
                if counts["pending"] == 0:
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
                    "task_group_id": task_group_id,
                    "snapshot_total": counts["pending"] + counts["processing"] + counts["done"] + counts["failed"],
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
                new_run_id = db.create_run(self.config.db_path, run_values)
                self.workers.start(new_run_id, task_group_id)
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
                tg = (run.get("task_group_id") or rid) if run else rid
                with db.connect(self.config.db_path) as conn:
                    conn.execute(
                        "UPDATE thinking_tasks SET status='pending',updated_at=? WHERE run_id=? AND status='processing'",
                        (time.time(), tg),
                    )
                cancelled.append(rid)
            except Exception:
                pass
        return {"cancelled": cancelled, "count": len(cancelled)}

    def batch_retry(self, body: dict) -> dict:
        export_ids = body.get("source_export_ids", [])
        total = 0
        for eid in export_ids:
            tg = db.get_task_group_id(self.config.db_path, int(eid))
            if tg:
                total += db.retry_all_failed(self.config.db_path, tg)
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
                task_group_id = db.get_task_group_id(self.config.db_path, eid)
                if not task_group_id:
                    continue
                reset_count = db.retry_all_failed(self.config.db_path, task_group_id)
                if reset_count == 0:
                    continue
                runs = db.list_runs(self.config.db_path, eid)
                latest = runs[0] if runs else None
                if not latest:
                    continue
                with db.connect(self.config.db_path) as conn:
                    counts = db.run_counts(conn, task_group_id)
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
                    "task_group_id": task_group_id,
                    "snapshot_total": counts["pending"] + counts["processing"] + counts["done"] + counts["failed"],
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
                new_run_id = db.create_run(self.config.db_path, run_values)
                self.workers.start(new_run_id, task_group_id)
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
                "SELECT status, COUNT(*) as cnt FROM thinking_tasks GROUP BY status"
            ).fetchall()
        counts = {"total": 0, "done": 0, "pending": 0, "processing": 0, "failed": 0}
        for row in rows:
            counts[row["status"]] = row["cnt"]
            counts["total"] += row["cnt"]
        return counts

