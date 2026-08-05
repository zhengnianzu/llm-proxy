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
    get_root_id,
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


def _describe(src: dict, active: bool, active_env_dir: str = "") -> dict:
    """把 sources 表一行渲染成列表页行。

    src: logdir_store.list_sources 返回的 dict（含 root_id/root_path/name/
    format/templates）。DB 是「日志路径列表」的唯一数据源——
    这里不再读 YAML history。
    """
    path = (src.get("root_path") or "").strip()
    root_id = src.get("root_id") or "default"
    name = (src.get("name") or "").strip() or "default"
    templates = src.get("templates") or []
    exists = os.path.isdir(path)
    fmt = src.get("format") or (detect_format(path) if exists else "missing")
    leaf_count = 0
    built_count = 0
    synced = False
    if exists:
        # 节点数/已 index 数：统一以 log_dir.db 为准（count_summary），首屏不扫盘。
        #   · newapi：built = 有 index.db 的叶子数。
        #   · native：index.jsonl 即索引，同步时叶子一律标 built，故 built == total。
        # 未同步（从未 sync）时：newapi 保持 0 且 synced=False（前端提示先同步）；
        # native 置 leaf_count=None 表「未统计」，前端显示「同步」入口（点了才扫盘写 DB）。
        if fmt in ("newapi", "native"):
            try:
                import utils.logdir_store as lds
                # DB 主键统一用纯路径哈希（不传 active_env_dir），与 sync_leaves/_run 一致
                rid = get_root_id(path)
                synced = lds.has_any(rid)
                if synced:
                    summ = lds.count_summary(rid)
                    leaf_count = summ.get("total", 0)
                    built_count = summ.get("built", 0)
                elif fmt == "native":
                    # native 从未同步：不首屏扫盘，标未统计，前端提示先「同步」。
                    leaf_count = None
            except Exception:
                synced = False
        else:
            # 其它非 newapi/native 源：叶子计数需网络盘全量递归遍历（iter_index_dirs），慢，
            # 移出 list 热路径；leaf_count 置 None，前端显示占位 + 「统计」按钮按需触发。
            leaf_count = None

    return {
        "path": path,
        "name": name,
        "root_id": root_id,
        "active": active,
        "exists": exists,
        "format": fmt,
        "leaf_count": leaf_count,
        "built_count": built_count,
        "synced": synced,
        "templates": templates,
        "status": "活跃" if active else "历史",
    }


def register_logs_routes(app: FastAPI, templates: Jinja2Templates, active_env_dir: str = "") -> None:

    def _ctx(request: Request) -> dict:
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
        import utils.logdir_store as lds
        # 行来源 = sources 表（活跃行 root_id='default' 排第一），DB 为准。
        srcs = lds.list_sources(active_env_dir)
        rows = []
        for src in srcs:
            active = (src.get("root_id") == "default")
            row = _describe(src, active=active, active_env_dir=active_env_dir)
            if active:
                row["base"] = active_base
            rows.append(row)

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
        templates = body.get("templates")
        # 以 DB 为准新增源：
        import utils.logdir_store as lds
        # 校验目录存在
        if not path:
            return JSONResponse({"ok": False, "msg": "路径不能为空"}, status_code=400)
        if not os.path.isdir(path):
            return JSONResponse({"ok": False, "msg": f"目录不存在: {path}"}, status_code=400)
        rid = get_root_id(path)
        # 重复 / 嵌套校验：对 DB sources 做（列表以 DB 为权威）
        if lds.get_source(rid):
            return JSONResponse({"ok": False, "msg": "路径已存在"})
        from utils.logs_config import _is_ancestor
        norm = os.path.normpath(path)
        for src in lds.list_sources(active_env_dir):
            sp = (src.get("root_path") or "").strip()
            if not sp:
                continue
            snorm = os.path.normpath(sp)
            if snorm == norm:
                continue
            if _is_ancestor(snorm, norm):
                return JSONResponse(
                    {"ok": False, "msg": (f"新路径在已有源「{src.get('name') or 'default'}」"
                                          f"（{sp}）之下，会被其递归扫描重复收录。"
                                          f"请改登记不与它嵌套的路径，或先移除该源再登记外层。")},
                    status_code=400)
            if _is_ancestor(norm, snorm):
                return JSONResponse(
                    {"ok": False, "msg": (f"新路径是已有源「{src.get('name') or 'default'}」"
                                          f"（{sp}）的上层，会把它递归收录导致重复。"
                                          f"请改登记不与它嵌套的路径，或先移除该源再登记外层。")},
                    status_code=400)
        # DB 先行
        lds.upsert_source(rid, root_path=os.path.normpath(path),
                          name=name or "default", format=detect_format(path),
                          templates=templates or [])
        return JSONResponse({"ok": True, "msg": "已添加"})

    @app.post("/api/logs-admin/remove")
    async def logs_admin_remove(request: Request):
        denied = _require_ajax(request)
        if denied:
            return denied
        if not _is_admin(request):
            return JSONResponse({"ok": False, "msg": "仅管理员可操作"}, status_code=403)

        body = await request.json()
        path = (body.get("path") or "").strip()
        # 活跃目录（root_id='default'）由进程 env 决定，不可移除
        if get_root_id(path) == "default":
            return JSONResponse({"ok": False, "msg": "活跃目录不可移除"})
        # DB 为准：删 sources 行 + 叶子记录（不删磁盘文件）
        import utils.logdir_store as lds
        rid = get_root_id(path)
        src = lds.get_source(rid)
        if not src:
            return JSONResponse({"ok": False, "msg": "路径不存在"})
        lds.delete_root(rid)
        return JSONResponse({"ok": True, "msg": "已移除（未删除磁盘文件）"})

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
        fmt = detect_format(path)
        if fmt not in ("newapi", "native"):
            return JSONResponse({"ok": False, "msg": "仅 new-api / 本项目(native) 源支持同步"}, status_code=400)
        # 模板优先取请求体（用户即时改），否则取该源在 sources 表里登记的；空则 sync_leaves 内按 fmt 默认回退。
        templates = body.get("templates")
        if templates is None:
            import utils.logdir_store as lds
            src = lds.get_source(get_root_id(path))
            templates = (src.get("templates") or []) if src else []
        from utils.newapi_backfill import sync_leaves
        res = await run_in_threadpool(sync_leaves, path, templates)
        # native：index.jsonl 即索引，同步即「已 index」，无「构建」步骤；提示语区分口径。
        if fmt == "native":
            msg = (f"已同步 {res['total']} 个节点（新增 {res['added']}，更新 {res['updated']}）。"
                   f"本项目源的 index.jsonl 即索引，无需再构建。")
        else:
            msg = (f"已同步 {res['total']} 个节点（新增 {res['added']}，更新 {res['updated']}，"
                   f"已建 {res['built']}）。构建完整性以「构建索引」为准。")
        return JSONResponse({"ok": True, "msg": msg, "result": res})

    @app.post("/api/logs-admin/rename")
    async def logs_admin_rename(request: Request):
        """改数据源名字（DB 为准）。"""
        denied = _require_ajax(request)
        if denied:
            return denied
        if not _is_admin(request):
            return JSONResponse({"ok": False, "msg": "仅管理员可操作"}, status_code=403)

        body = await request.json()
        path = (body.get("path") or "").strip()
        name = (body.get("name") or "").strip() or "default"
        import utils.logdir_store as lds
        rid = get_root_id(path)
        ok = lds.set_source_name(rid, name)
        if not ok:
            return JSONResponse({"ok": False, "msg": "路径不存在"})
        return JSONResponse({"ok": True, "msg": "已改名"})

    @app.post("/api/logs-admin/set-templates")
    async def logs_admin_set_templates(request: Request):
        """改数据源层级模板（DB 为准）。可选即时 re-sync（resync=true）。"""
        denied = _require_ajax(request)
        if denied:
            return denied
        if not _is_admin(request):
            return JSONResponse({"ok": False, "msg": "仅管理员可操作"}, status_code=403)

        body = await request.json()
        path = (body.get("path") or "").strip()
        templates = body.get("templates")
        resync = bool(body.get("resync", False))
        import utils.logdir_store as lds
        rid = get_root_id(path)
        ok = lds.set_source_templates(rid, templates)
        if not ok:
            return JSONResponse({"ok": False, "msg": "路径不存在"})
        result = None
        msg = "已更新模板"
        if resync and os.path.isdir(path) and detect_format(path) in ("newapi", "native"):
            from utils.newapi_backfill import sync_leaves
            tpls = templates if templates else []
            result = await run_in_threadpool(sync_leaves, path, tpls)
            msg = (f"{msg}；已按新模板同步 {result['total']} 个节点"
                   f"（新增 {result['added']}，更新 {result['updated']}）。")
        return JSONResponse({"ok": True, "msg": msg, "result": result})

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
                    "sessions": r.get("sessions", 0) or 0,
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
            running = st.get("status") == "running"
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
        """当前正在回填的源快照（含实时进度）。

        已去掉全局串行队列——多个源可并发回填，queued 恒为空。
        保留此端点仅为兼容旧调用方（如导出页），前端列表页不再调用。
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
