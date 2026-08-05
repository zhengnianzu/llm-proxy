"""
utils/logs_routes.py — 日志管理页面路由

提供:
  GET  /logs-admin                 — 日志管理页面
  GET  /api/logs-admin/list        — 列出活跃 base + 历史路径（含大小/格式/请求数）
  POST /api/logs-admin/add         — 新增历史路径
  POST /api/logs-admin/remove      — 移除历史路径（不删磁盘文件）

「都是到 env-key 这个层级」：历史路径应指向 env-key 层目录，
或包含 index.jsonl 叶子（含可变深度，如 new-api 的 details）。
统计/预览会自动合并这些路径。
"""

import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool

from utils.logs_config import (
    get_active_base,
    get_history_entries,
    add_history_path,
    remove_history_path,
    dir_size,
    human_size,
)
from utils.log_scan import detect_format, iter_index_dirs


def _require_ajax(request: Request):
    if request.headers.get("x-requested-with", "").lower() != "xmlhttprequest":
        return JSONResponse({"detail": "Forbidden"}, status_code=403)
    return None


def _is_admin(request: Request) -> bool:
    role = request.session.get("monitor_role", "admin")
    return role == "admin"


def _verify_and_persist(path: str) -> dict:
    """实测核对某 new-api 源并增量落库，返回对外 verify 结构（仅公开字段）。

    仅在「核对」入口（用户主动点核对 / 首屏首次回退）调用——绝不放到 list 热路径。
    verify_root 逐叶实测（内部 30s TTL 缓存）；save_verify 把逐叶 verified + 根级
    计数增量写进 log_dir.db，供之后首屏 get_last_verify 秒读。
    """
    from utils.newapi_index_db import verify_root
    verify = verify_root(path)
    try:
        import utils.logdir_store as lds
        from utils.logs_config import get_root_id as _grid
        from utils.log_scan import dir_key_for
        lds.save_verify(
            _grid(path),
            {k: v for k, v in verify.items() if not k.startswith("_")},
            leaf_dir_keys={
                p: dir_key_for(Path(path), Path(p))
                for p in verify.get("_leaf_map", {})
            },
            completed_paths=verify.get("_completed_paths"),
        )
    except Exception:
        pass
    # 只回公开字段：_completed_paths(set) / _leaf_map 不可 JSON 序列化
    return {k: v for k, v in verify.items() if not k.startswith("_")}


def _describe(path: str, active: bool, active_env_dir: str = "", name: str = "default") -> dict:
    exists = os.path.isdir(path)
    fmt = detect_format(path) if exists else "missing"
    leaf_count = 0
    built_count = 0
    synced = False
    # root_id：活跃 env_dir 固定为 default，其余按稳定哈希，供用户核对来源标识
    try:
        from utils.logs_config import get_root_id
        root_id = "default" if active else get_root_id(path, active_env_dir)
    except Exception:
        root_id = "default"
    if exists:
        # 已回填/总数（仅 new-api 源有 index.db 概念）：优先读持久化的 log_dir.db，
        # 不再每次全盘 stat；DB 空（从未同步）时回退磁盘探测，保证首次可用。
        if fmt == "newapi":
            # 计数口径统一为 log_dir.db（与叶子详情、backfill-status 同源）。
            # 未同步（从未 sync）时不回退磁盘探测——保持 leaf_count=0 且 synced=False，
            # 前端提示先「同步」，避免磁盘探测数与 DB 数并存导致口径不一致。
            try:
                import utils.logdir_store as lds
                from utils.logs_config import get_root_id as _grid
                # DB 主键统一用纯路径哈希（不传 active_env_dir），与 sync_leaves/_run 一致
                rid = _grid(path)
                synced = lds.has_any(rid)
                if synced:
                    summ = lds.count_summary(rid)
                    leaf_count = summ.get("total", 0)
                    built_count = summ.get("built", 0)
            except Exception:
                synced = False
        else:
            # native 等非 newapi 源：叶子计数需在网络盘上全量递归遍历（iter_index_dirs），
            # 慢，且会拖死 list 热路径（SFS 上单源可达 30s+）。首屏不扫盘，leaf_count 置 None
            # 表「未统计」，前端显示占位 + 「统计」按钮，按需调 /api/logs-admin/count-leaves。
            leaf_count = None

        # 核对结果：只读 log_dir.db 里最近一次落库的结果（get_last_verify），首屏绝不扫盘。
        # 只对 new-api 源。DB 有缓存 → 秒读；从未核对过但已同步 → 首屏自动核对一次并落库
        # （之后即走缓存）；未同步 → 保持空，前端提示先「同步」。实时核对改由「核对」按钮
        # 主动触发（/api/logs-admin/verify），把逐叶实测这类贵操作移出 list 热路径。
        verify = {}
        if fmt == "newapi":
            try:
                import utils.logdir_store as lds
                from utils.logs_config import get_root_id as _grid
                rid = _grid(path)
                cached = lds.get_last_verify(rid)
                if cached:
                    verify = cached
                elif synced:
                    verify = _verify_and_persist(path)
            except Exception:
                verify = {}
    else:
        verify = {}

    return {
        "path": path,
        "name": name or "default",
        "root_id": root_id,
        "active": active,
        "exists": exists,
        "format": fmt,
        "leaf_count": leaf_count,
        "built_count": built_count,
        "synced": synced,
        "verify": verify,
        "status": "活跃" if active else "历史",
    }


def register_logs_routes(
    app: FastAPI,
    templates: Jinja2Templates,
    active_env_dir: str = "",
    context_builder=None,
) -> None:

    def _ctx(request: Request) -> dict:
        if context_builder is not None:
            return context_builder(request, "logs_admin")
        return {
            "active_page": "logs_admin",
            "user_role": request.session.get("monitor_role", "admin"),
            "user_name": request.session.get("monitor_user", ""),
            "user_permissions": [
                p.strip()
                for p in (request.session.get("monitor_permissions") or "").split(",")
                if p.strip()
            ],
        }

    @app.get("/logs-admin")
    def logs_admin_page(request: Request):
        return templates.TemplateResponse(request, "logs_admin.html", context=_ctx(request))

    @app.get("/api/logs-admin/list")
    def logs_admin_list(request: Request, with_size: bool = False):
        denied = _require_ajax(request)
        if denied:
            return denied

        active_base = get_active_base()
        rows = []

        # 活跃：进程实际写入的 env_dir（名称固定 default）
        active_row = _describe(active_env_dir, active=True, name="default")
        active_row["base"] = active_base
        rows.append(active_row)

        # 历史（带名称）
        for e in get_history_entries():
            p = e["path"]
            # 跳过与活跃 env_dir 相同的路径
            if os.path.normpath(p) == os.path.normpath(active_env_dir):
                continue
            rows.append(_describe(p, active=False, active_env_dir=active_env_dir, name=e.get("name") or "default"))

        if with_size:
            for r in rows:
                if r["exists"]:
                    nbytes = dir_size(r["path"])
                    r["size_bytes"] = nbytes
                    r["size"] = human_size(nbytes)
                else:
                    r["size_bytes"] = 0
                    r["size"] = "-"

        return JSONResponse({
            "active_base": active_base,
            "active_env_dir": active_env_dir,
            "rows": rows,
        })

    @app.post("/api/logs-admin/add")
    async def logs_admin_add(request: Request):
        denied = _require_ajax(request)
        if denied:
            return denied
        if not _is_admin(request):
            return JSONResponse({"ok": False, "msg": "仅管理员可操作"}, status_code=403)

        body = await request.json()
        path = (body.get("path") or "").strip()
        name = (body.get("name") or "").strip()
        ok, msg = add_history_path(path, name)
        return JSONResponse({"ok": ok, "msg": msg})

    @app.post("/api/logs-admin/remove")
    async def logs_admin_remove(request: Request):
        denied = _require_ajax(request)
        if denied:
            return denied
        if not _is_admin(request):
            return JSONResponse({"ok": False, "msg": "仅管理员可操作"}, status_code=403)

        body = await request.json()
        path = (body.get("path") or "").strip()
        ok, msg = remove_history_path(path)
        if ok:
            # 顺带清掉该源在 log_dir.db 里的叶子记录（不删磁盘文件）
            try:
                import utils.logdir_store as lds
                from utils.logs_config import get_root_id
                lds.delete_root(get_root_id(path))
            except Exception:
                pass
        return JSONResponse({"ok": ok, "msg": msg})

    @app.post("/api/logs-admin/backfill")
    async def logs_admin_backfill(request: Request):
        """启动 new-api 叶子 index.db 后台构建（ingest+补 meta+聚合 sessions；历史预览/导出前置）。"""
        denied = _require_ajax(request)
        if denied:
            return denied
        if not _is_admin(request):
            return JSONResponse({"ok": False, "msg": "仅管理员可操作"}, status_code=403)

        body = await request.json()
        path = (body.get("path") or "").strip()
        workers = body.get("workers")
        force = bool(body.get("force", False))
        if not os.path.isdir(path):
            return JSONResponse({"ok": False, "msg": f"目录不存在: {path}"}, status_code=400)
        from utils.newapi_backfill import start_backfill
        st = start_backfill(path, workers=int(workers) if workers else None, force=force)
        return JSONResponse({"ok": True, "status": st})

    @app.post("/api/logs-admin/sync")
    async def logs_admin_sync(request: Request):
        """手动同步某 new-api 源的叶子清单到 log_dir.db（只扫盘写 DB，不触发回填）。

        新增节点写入 state=pending，已存在的更新其构建状态。用户可随后点「构建索引」补齐。
        """
        denied = _require_ajax(request)
        if denied:
            return denied
        if not _is_admin(request):
            return JSONResponse({"ok": False, "msg": "仅管理员可操作"}, status_code=403)

        body = await request.json()
        path = (body.get("path") or "").strip()
        if not os.path.isdir(path):
            return JSONResponse({"ok": False, "msg": f"目录不存在: {path}"}, status_code=400)
        if detect_format(path) != "newapi":
            return JSONResponse({"ok": False, "msg": "仅 new-api 源支持同步"}, status_code=400)
        from utils.newapi_backfill import sync_leaves
        res = sync_leaves(path)
        return JSONResponse({
            "ok": True,
            "msg": f"已同步 {res['total']} 个节点（新增 {res['added']}，更新 {res['updated']}，"
                   f"已建 {res['built']}）。构建完整性以「构建索引」为准。",
            "result": res,
        })

    @app.post("/api/logs-admin/verify")
    async def logs_admin_verify(request: Request):
        """主动核对某 new-api 源：逐叶实测「是否已完成跟上」并增量落库，返回最新 verify。

        前端「核对」按钮触发。实测扫盘（可能读网络盘）丢到线程池执行，避免阻塞事件循环。
        """
        denied = _require_ajax(request)
        if denied:
            return denied
        if not _is_admin(request):
            return JSONResponse({"ok": False, "msg": "仅管理员可操作"}, status_code=403)

        body = await request.json()
        path = (body.get("path") or "").strip()
        if not os.path.isdir(path):
            return JSONResponse({"ok": False, "msg": f"目录不存在: {path}"}, status_code=400)
        if detect_format(path) != "newapi":
            return JSONResponse({"ok": False, "msg": "仅 new-api 源支持核对"}, status_code=400)
        verify = await run_in_threadpool(_verify_and_persist, path)
        return JSONResponse({"ok": True, "verify": verify})

    @app.get("/api/logs-admin/count-leaves")
    async def logs_admin_count_leaves(request: Request, path: str = ""):
        """按需统计某源的叶子数（含 index.jsonl 的目录数）。

        native 等非 newapi 源的计数需全量递归遍历（网络盘上慢），故移出 list 热路径，
        由前端「统计」按钮触发。扫盘丢线程池，不阻塞事件循环。
        """
        denied = _require_ajax(request)
        if denied:
            return denied
        if not path or not os.path.isdir(path):
            return JSONResponse({"ok": False, "msg": f"目录不存在: {path}"}, status_code=400)

        def _count() -> int:
            return sum(1 for _ in iter_index_dirs(Path(path)))

        try:
            n = await run_in_threadpool(_count)
        except Exception as e:  # noqa: BLE001
            return JSONResponse({"ok": False, "msg": f"统计失败: {e}"}, status_code=500)
        return JSONResponse({"ok": True, "leaf_count": n})

    @app.get("/api/logs-admin/leaves")
    def logs_admin_leaves(request: Request, path: str = ""):
        """列出某 new-api 源在 log_dir.db 里登记的全部叶子（供「叶子详情」展开）。

        以 log_dir.db 为准（与列表页计数同源）：每叶返回标识、构建状态、会话数、错误。
        DB 未同步（从未 sync）时返回 synced=False，前端提示先「同步」。
        """
        denied = _require_ajax(request)
        if denied:
            return denied
        if not path or not os.path.isdir(path):
            return JSONResponse({"ok": False, "msg": f"目录不存在: {path}"}, status_code=400)
        try:
            import utils.logdir_store as lds
            from utils.logs_config import get_root_id
            rid = get_root_id(path)
            synced = lds.has_any(rid)
            summ = lds.count_summary(rid) if synced else {}
            leaves = []
            for r in (lds.bulk_get(rid) if synced else []):
                dir_key = r.get("dir_key", "")
                rp = r.get("root_path", "") or path
                full_path = os.path.join(rp, dir_key) if dir_key else rp
                leaves.append({
                    "dir_key": dir_key,
                    "full_path": full_path,
                    "state": r.get("state", "pending"),
                    "built": bool(r.get("built")),
                    "last_error": r.get("last_error", ""),
                })
        except Exception as e:  # noqa: BLE001
            return JSONResponse({"ok": False, "msg": f"读取失败: {e}"}, status_code=500)
        return JSONResponse({
            "ok": True,
            "synced": synced,
            "summary": summ,
            "leaves": leaves,
        })

    @app.get("/api/logs-admin/backfill-status")
    def logs_admin_backfill_status(request: Request, path: str = ""):
        denied = _require_ajax(request)
        if denied:
            return denied
        from utils.newapi_backfill import get_backfill_status
        st = dict(get_backfill_status(path))
        # 计数口径统一为 log_dir.db。非运行态时以 DB 为准，并清除磁盘探测(from_disk)
        # 借用的 skipped_leaves —— 它的语义是"磁盘已建总数"，不是"本次跳过数"，
        # 会让前端打出误导性的"跳过 N 个已构建"。
        try:
            import utils.logdir_store as lds
            from utils.logs_config import get_root_id
            rid = get_root_id(path)
            running = st.get("status") in ("running", "queued")
            if lds.has_any(rid):
                summ = lds.count_summary(rid)
                st["db_summary"] = summ
                if not running:
                    total = summ.get("total", 0)
                    built = summ.get("built", 0)
                    st["total_leaves"] = total
                    st["done_leaves"] = built
                    st["pending_leaves"] = summ.get("pending", 0)
                    st["error_leaves"] = summ.get("error", 0)
                    st.pop("skipped_leaves", None)   # 磁盘探测残留，语义不符，去掉
                    st.pop("from_disk", None)
                    # 全部已建 → ready；部分已建 → partial（前端显示"部分就绪 N/M"）
                    st["status"] = "ready" if (total and built >= total) else "partial"
            elif not running:
                # DB 从未同步：无权威计数，显式标记，前端提示先「同步」，
                # 不再显示 from_disk 的假"已就绪/跳过 N"。
                st = {"status": "unsynced"}
        except Exception:
            pass
        return JSONResponse(st)

    @app.get("/api/logs-admin/queue")
    def logs_admin_queue(request: Request):
        """全局回填队列快照：当前正在跑的源(含实时进度) + 排队中的源列表。

        前端 load 时读一次即可把"正在构建/排队第几位"直接更新到对应行，无需轮询。
        """
        denied = _require_ajax(request)
        if denied:
            return denied
        from utils.newapi_backfill import get_queue_snapshot
        return JSONResponse(get_queue_snapshot())

    @app.get("/api/logs-admin/build-log")
    def logs_admin_build_log(request: Request, path: str = "", limit: int = 400):
        """读某源的构建日志（backfill.log 中带该 root 的行）。供「构建日志」弹窗展示。

        path 为空则返回全局尾部。limit 上限兜底，避免一次性返回过多行。
        """
        denied = _require_ajax(request)
        if denied:
            return denied
        limit = max(1, min(int(limit or 400), 2000))
        from utils.newapi_backfill import read_backfill_log
        res = read_backfill_log(root_filter=path, limit=limit)
        return JSONResponse(res)
