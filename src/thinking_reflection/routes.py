from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates

from . import db
from .config import load_config
from .service import ReflectionService

_test_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="reflection-test")


def register_reflection_routes(app: FastAPI, templates: Jinja2Templates) -> ReflectionService:
    service = ReflectionService(load_config())

    def _is_auth_enabled():
        import os
        explicit = os.getenv("MONITOR_AUTH_ENABLED")
        if explicit is not None:
            return explicit.strip().lower() in {"1", "true", "yes", "on"}
        return bool(os.getenv("MONITOR_USERNAME", "").strip())

    def context(request: Request, active_page: str = "thinking") -> dict:
        if _is_auth_enabled():
            role = request.session.get("monitor_role", "user")
            user_name = request.session.get("monitor_user", "")
            perms = [x.strip() for x in (request.session.get("monitor_permissions") or "").split(",") if x.strip()]
        else:
            role, user_name, perms = "admin", "", []
        return {"active_page": active_page, "user_role": role, "user_name": user_name, "user_permissions": perms}

    def call(fn, *args, **kwargs):
        try: return fn(*args, **kwargs)
        except (ValueError, KeyError) as exc: raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/thinking")
    def page(request: Request): return templates.TemplateResponse(request, "thinking.html", context=context(request))

    @app.get("/thinking/dataset/{record_id}")
    def dataset_page(request: Request, record_id: int):
        return templates.TemplateResponse(request, "thinking_dataset.html", context={**context(request), "record_id": record_id})

    @app.get("/thinking/tasks")
    def tasks_page(request: Request):
        return templates.TemplateResponse(request, "thinking_tasks.html", context=context(request, "thinking_tasks"))

    @app.get("/thinking/failed")
    def failed_page(request: Request):
        return templates.TemplateResponse(request, "thinking_failed.html", context=context(request, "thinking_failed"))

    @app.get("/api/reflection/datasets")
    def datasets(key_slot: str): return service.datasets(key_slot)

    @app.get("/api/reflection/datasets-all")
    def datasets_all(): return service.datasets_all()

    @app.get("/api/reflection/datasets-available")
    def datasets_available(): return service.available_datasets()

    @app.post("/api/reflection/datasets/add")
    async def datasets_add(request: Request): return call(service.add_to_manage, await request.json())

    @app.delete("/api/reflection/datasets/{record_id}/manage")
    def datasets_remove(record_id: int): return call(service.remove_from_manage, record_id)

    @app.get("/api/reflection/datasets/{record_id}/analysis")
    def analysis(record_id: int): return call(service.analysis, record_id)

    @app.get("/api/reflection/datasets/{record_id}/sessions")
    def dataset_sessions(record_id: int, offset: int = 0, limit: int = 50, force: bool = False):
        return call(service.dataset_sessions, record_id, offset, limit, force)

    @app.get("/api/reflection/datasets/{record_id}/session-trajectory")
    def session_trajectory(record_id: int, session_id: str, file_name: str):
        return call(service.session_trajectory, record_id, session_id, file_name)

    @app.get("/api/reflection/config")
    def config():
        from utils.obs_utils import load_obs_base
        from .worker_manager import DEFAULT_MAX_GLOBAL_WORKERS
        prompt = {method: (service.config.prompt_dir / f"{method}.json").is_file() for method in ("bulk", "sentence")}
        source_key_slots = service.source_key_slots()
        max_global_workers = int(db.get_setting(
            service.config.db_path, "max_global_workers", str(DEFAULT_MAX_GLOBAL_WORKERS)))
        return {
            "keys": service.active_keys(),
            "source_key_slots": source_key_slots,
            "default_source_key_slot": source_key_slots[0]["key_slot"] if source_key_slots else "",
            "prompt_available": prompt,
            "export_root": service.config.export_root.as_posix(),
            "reflection_base_url": service.config.reflection_base_url,
            "obs_base": load_obs_base(),
            "max_global_workers": max_global_workers,
        }

    @app.post("/api/reflection/settings")
    async def update_settings(request: Request):
        body = await request.json()
        raw = body.get("max_global_workers")
        try:
            value = int(raw)
        except (TypeError, ValueError):
            raise HTTPException(400, "max_global_workers 必须是整数")
        # 单个 run 的 worker_count 上限是 32；全局上限必须 >= 32，否则满额 run 永远排不进队列。
        if value < 32:
            raise HTTPException(400, "全局最大 Workers 不能小于 32（单 Run 上限）")
        db.set_setting(service.config.db_path, "max_global_workers", value)
        return {"max_global_workers": value}

    @app.post("/api/reflection/runs")
    async def create_run(request: Request): return call(service.create_run, await request.json())

    @app.get("/api/reflection/runs")
    def runs(source_export_id: int | None = None): return db.list_runs(service.config.db_path, source_export_id)

    @app.get("/api/reflection/runs/{run_id}")
    def run(run_id: str):
        result = db.get_run(service.config.db_path, run_id)
        if not result: raise HTTPException(404, "Run 不存在")
        return result

    @app.get("/api/reflection/runs/{run_id}/snapshot")
    def run_snapshot(run_id: str): return call(db.get_run_snapshot, service.config.db_path, run_id)

    @app.get("/api/reflection/runs/{run_id}/trajectories")
    def run_trajectories(run_id: str, offset: int = 0, limit: int = 0):
        return service.trajectory_list(run_id, offset, limit)

    @app.patch("/api/reflection/runs/{run_id}/config")
    async def update_run_config(run_id: str, request: Request):
        return call(service.update_run_config, run_id, await request.json())

    @app.post("/api/reflection/runs/{run_id}/start")
    def start(run_id: str):
        run = db.get_run(service.config.db_path, run_id)
        if not run: raise HTTPException(404, "Run 不存在")
        call(service.workers.start, run_id)
        updated = db.get_run(service.config.db_path, run_id)
        return {"status": updated["status"] if updated else "running"}

    @app.post("/api/reflection/runs/{run_id}/pause")
    def pause(run_id: str): service.workers.stop(run_id); return {"status": "paused"}

    @app.post("/api/reflection/runs/{run_id}/stop")
    def stop(run_id: str): service.workers.stop(run_id, cancel=True); return {"status": "cancelled"}

    @app.delete("/api/reflection/runs/{run_id}")
    def delete_run(run_id: str): call(service.delete_run, run_id); return {"status": "deleted"}

    @app.post("/api/reflection/runs/{run_id}/upload-obs")
    async def upload_obs(request: Request, run_id: str):
        from .result_exporter import upload_run_to_obs
        return call(upload_run_to_obs, service.config.db_path, run_id)

    @app.get("/api/reflection/tasks")
    def tasks(run_id: str, status: str | None = None, offset: int = 0, limit: int = 50):
        return service.tasks(run_id, status, offset, limit)

    @app.get("/api/reflection/tasks/{task_uuid}")
    def task_detail(task_uuid: str): return call(service.get_task, task_uuid)

    @app.get("/api/reflection/tasks/{task_uuid}/attempts")
    def task_attempts(task_uuid: str): return service.task_attempts(task_uuid)

    @app.post("/api/reflection/tasks/{task_uuid}/retry")
    def retry(task_uuid: str): call(service.retry, task_uuid); return {"status": "pending"}

    @app.post("/api/reflection/tasks/retry-failed")
    async def retry_failed(request: Request):
        body = await request.json()
        return call(service.retry_all_failed, body.get("run_id", ""))

    @app.post("/api/reflection/tasks/rerun-done")
    async def rerun_done(request: Request):
        body = await request.json()
        return call(service.rerun_all_done, body.get("run_id", ""))

    @app.post("/api/reflection/test")
    async def test(request: Request):
        body = await request.json()
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(_test_pool, service.test, body)
            return result
        except (ValueError, KeyError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            return JSONResponse(status_code=500, content={"detail": str(exc)})

    @app.get("/api/reflection/trajectories/{trajectory_id}/merged")
    def trajectory(trajectory_id: str, run_id: str): return call(service.trajectory, run_id, trajectory_id)

    @app.post("/api/reflection/runs/{run_id}/export")
    def export(run_id: str): return call(service.export, run_id)

    @app.get("/api/reflection/runs/{run_id}/logs")
    def run_logs(run_id: str, since_id: int = 0, limit: int = 200):
        logs = db.get_run_logs(service.config.db_path, run_id, since_id, limit)
        run = db.get_run(service.config.db_path, run_id)
        return {"logs": logs, "run_status": run["status"] if run else "unknown", "run": run}

    # ------------------------------------------------------------------
    # 新增：导入任务库 / OBS 下载 / 批量操作
    # ------------------------------------------------------------------

    @app.post("/api/reflection/preview-tasks")
    async def preview_tasks(request: Request): return call(service.preview_tasks, await request.json())

    @app.post("/api/reflection/import-tasks")
    async def import_tasks(request: Request): return call(service.import_tasks, await request.json())

    @app.post("/api/reflection/datasets/register")
    async def register_external(request: Request): return call(service.register_external, await request.json())

    @app.post("/api/reflection/datasets/{record_id}/download-obs")
    async def download_obs(record_id: int):
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(_test_pool, service.download_obs, record_id)
            return result
        except (ValueError, KeyError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            return JSONResponse(status_code=500, content={"detail": str(exc)})

    @app.post("/api/reflection/batch-start")
    async def batch_start(request: Request): return call(service.batch_start, await request.json())

    @app.post("/api/reflection/batch-pause")
    async def batch_pause(request: Request): return call(service.batch_pause, await request.json())

    @app.post("/api/reflection/batch-cancel")
    async def batch_cancel(request: Request): return call(service.batch_cancel, await request.json())

    @app.post("/api/reflection/batch-retry")
    async def batch_retry(request: Request): return call(service.batch_retry, await request.json())

    @app.post("/api/reflection/batch-rerun")
    async def batch_rerun(request: Request): return call(service.batch_rerun, await request.json())

    @app.get("/api/reflection/tasks-summary")
    def tasks_summary(): return service.all_tasks_summary()

    return service
