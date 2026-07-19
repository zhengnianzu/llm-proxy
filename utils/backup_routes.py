"""
utils/backup_routes.py — 备份管理 Web 路由

管理所有 mtime 目录的 OBS 同步状态，支持手动触发同步和删除。
"""

import logging
import os
import signal
import shutil
import subprocess as _subprocess
import sys
import threading
import time
from pathlib import Path
from typing import List

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates

from utils.backup_store import (
    append_log,
    clear_logs,
    get_dir,
    get_live_syncing_dirs,
    get_logs,
    has_any_dirs,
    list_dirs,
    list_env_names,
    mark_backed_up,
    update_sync_pid,
    update_sync_status,
    upsert_dir,
)
from utils.obs_utils import load_obs_base, load_sync_config, get_sync_config_path, obsutil_ls

logger = logging.getLogger(__name__)

_sync_lock = threading.Lock()
_syncing_dirs: set = set()
_live_sync_procs: dict = {}


def _is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _capture_sync_logs(proc: _subprocess.Popen, dir_path: str):
    try:
        for raw_line in iter(proc.stdout.readline, b""):
            line = raw_line.decode("utf-8", errors="replace").rstrip()
            if not line:
                continue
            level = "error" if "[ERROR]" in line or "[WARNING]" in line else "info"
            try:
                append_log(dir_path, line, level=level)
            except Exception:
                pass
    except Exception:
        pass
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        exit_code = proc.wait()
        try:
            dir_info = get_dir(dir_path)
            if dir_info and dir_info.get("status") == "live_syncing":
                if exit_code == 0:
                    update_sync_status(dir_path, "done")
                    append_log(dir_path, "在线同步进程正常退出")
                else:
                    update_sync_status(dir_path, "error", error_msg=f"exit_code={exit_code}")
                    append_log(dir_path, f"在线同步进程异常退出: exit_code={exit_code}", level="error")
                update_sync_pid(dir_path, None)
        except Exception:
            pass
        _live_sync_procs.pop(dir_path, None)


def _stop_live_sync(dir_path: str, timeout: int = 120) -> dict:
    if dir_path in _live_sync_procs:
        proc = _live_sync_procs[dir_path]["proc"]
        if proc.poll() is None:
            try:
                os.kill(proc.pid, signal.SIGTERM)
            except OSError:
                pass
            try:
                proc.wait(timeout=timeout)
            except _subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
            _live_sync_procs.pop(dir_path, None)
            update_sync_status(dir_path, "done")
            update_sync_pid(dir_path, None)
            append_log(dir_path, "在线同步已停止")
            return {"ok": True}
        else:
            _live_sync_procs.pop(dir_path, None)

    dir_info = get_dir(dir_path)
    if not dir_info:
        return {"ok": False, "detail": "目录不存在"}

    pid = dir_info.get("sync_pid")
    if pid and _is_pid_alive(pid):
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
        for _ in range(timeout * 2):
            time.sleep(0.5)
            if not _is_pid_alive(pid):
                break
        if _is_pid_alive(pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
            time.sleep(0.2)
        update_sync_status(dir_path, "done")
        update_sync_pid(dir_path, None)
        append_log(dir_path, "在线同步已停止 (via PID)")
        return {"ok": True}

    if dir_info.get("status") == "live_syncing":
        update_sync_status(dir_path, "done")
        update_sync_pid(dir_path, None)
        append_log(dir_path, "在线同步进程已不存在，状态已更新")
    return {"ok": True}


def _cleanup_stale_live_syncs():
    try:
        stale_dirs = get_live_syncing_dirs()
        for d in stale_dirs:
            pid = d.get("sync_pid")
            dir_path = d["dir_path"]
            if pid and _is_pid_alive(pid):
                logger.info("Stopping stale live sync: dir=%s pid=%d", dir_path, pid)
                try:
                    os.kill(pid, signal.SIGTERM)
                except OSError:
                    pass
                for _ in range(20):
                    time.sleep(0.5)
                    if not _is_pid_alive(pid):
                        break
                if _is_pid_alive(pid):
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except OSError:
                        pass
            update_sync_status(dir_path, "done")
            update_sync_pid(dir_path, None)
            append_log(dir_path, "应用重启，在线同步已自动停止")
    except Exception as e:
        logger.warning("Failed to cleanup stale live syncs: %s", e)


def _require_ajax(request: Request):
    if request.headers.get("x-requested-with") != "XMLHttpRequest":
        return JSONResponse({"detail": "Not found"}, status_code=404)
    return None


def _scan_data_dirs(logs_all: Path, env_name: str = None, port_env_map: dict = None) -> List[dict]:
    """扫描 logs_all 下有数据的 mtime 目录。可选只扫描指定 env。"""
    result = []
    if not logs_all.is_dir():
        return result
    if env_name:
        target = logs_all / env_name
        env_dirs = [target] if target.is_dir() else []
    else:
        env_dirs = sorted(d for d in logs_all.iterdir()
                          if d.is_dir() and not d.name.startswith("logs_") and not d.name.startswith("."))
    port_map = port_env_map or {}
    for env_dir in env_dirs:
        if not env_dir.is_dir() or env_dir.name.startswith("logs_") or env_dir.name.startswith("."):
            continue
        for mtime_dir in sorted(env_dir.iterdir()):
            if not mtime_dir.is_dir():
                continue
            index_file = mtime_dir / "index.jsonl"
            has_data = index_file.is_file()
            if not has_data:
                try:
                    has_data = any(f.name.endswith("-req.json") for f in mtime_dir.iterdir())
                except OSError:
                    continue
            if not has_data:
                continue
            file_count = 0
            try:
                file_count = sum(1 for f in mtime_dir.iterdir() if f.is_file())
            except OSError:
                pass
            result.append({
                "dir_path": f"{env_dir.name}/{mtime_dir.name}",
                "env_name": env_dir.name,
                "mtime_tag": mtime_dir.name,
                "file_count": file_count,
                "port": port_map.get(env_dir.name, ""),
            })
    return result


def _scan_obs_dirs(obs_base: str, env_name: str = None) -> List[dict]:
    """扫描 OBS 上 {obs_base}/raw/ 下的 env/mtime 目录结构。"""
    result = []
    if not obs_base:
        return result
    raw_path = f"{obs_base.rstrip('/')}/raw/"
    env_items = obsutil_ls(raw_path, show_dirs=True)
    for env_item in env_items:
        if not env_item.get("is_dir"):
            continue
        ename = env_item["name"]
        if env_name and ename != env_name:
            continue
        env_obs_path = env_item["path"]
        mtime_items = obsutil_ls(env_obs_path, show_dirs=True)
        for mt_item in mtime_items:
            if not mt_item.get("is_dir"):
                continue
            mtag = mt_item["name"]
            result.append({
                "dir_path": f"{ename}/{mtag}",
                "env_name": ename,
                "mtime_tag": mtag,
                "obs_path": mt_item["path"],
            })
    return result


def _run_sync_for_dir(dir_path: str, logs_all: Path, obs_base: str, workers: int, upload_script: str):
    """在后台线程中同步单个目录到 OBS，直接上传整个文件夹，流式输出日志。"""
    import subprocess as sp
    from utils.obs_utils import DEFAULT_UPLOAD_SCRIPT

    logs_dir = str(logs_all / dir_path)
    env_name, mtime_tag = dir_path.split("/", 1)
    obs_dst = f"{obs_base.rstrip('/')}/raw/{env_name}/{mtime_tag}/"
    script = upload_script or DEFAULT_UPLOAD_SCRIPT
    if not Path(script).is_absolute():
        script = str((Path(__file__).resolve().parent.parent / script).resolve())

    try:
        # 本地目录不存在：已被清理，标记为 backed_up 跳过上传
        if not (logs_all / dir_path).is_dir():
            dir_info = get_dir(dir_path)
            if dir_info and dir_info.get("obs_path"):
                update_sync_status(dir_path, "backed_up", obs_path=dir_info["obs_path"])
                append_log(dir_path, "本地目录不存在，已标记为已备份（跳过上传）")
            else:
                append_log(dir_path, "本地目录不存在且无 OBS 路径，跳过", level="error")
            return

        # synced=1：之前已成功同步过，status 可能被后续操作误覆盖，修正并跳过
        dir_info = get_dir(dir_path)
        if dir_info and dir_info.get("synced") and dir_info.get("obs_path"):
            if dir_info.get("status") != "done":
                update_sync_status(dir_path, "done", obs_path=dir_info["obs_path"])
                append_log(dir_path, f"已有成功同步记录（synced=1），修正状态为 done，跳过重复上传")
            else:
                append_log(dir_path, "已有成功同步记录（synced=1），跳过重复上传")
            return

        update_sync_status(dir_path, "syncing", obs_path=obs_dst)
        append_log(dir_path, f"开始同步: {dir_path} -> {obs_dst}")

        cmd = [script, logs_dir, obs_dst]
        append_log(dir_path, f"执行: {' '.join(cmd)}")
        proc = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.STDOUT, text=True)
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                append_log(dir_path, line)
        proc.wait(timeout=600)

        if proc.returncode == 0:
            update_sync_status(dir_path, "done", obs_path=obs_dst)
            append_log(dir_path, "同步完成")
        else:
            append_log(dir_path, f"上传失败: exit_code={proc.returncode}", level="error")
            update_sync_status(dir_path, "error", error_msg=f"exit_code={proc.returncode}", obs_path=obs_dst)
    except Exception as e:
        update_sync_status(dir_path, "error", error_msg=str(e))
        append_log(dir_path, f"同步失败: {e}", level="error")
        logger.exception("backup sync failed for %s", dir_path)
    finally:
        _syncing_dirs.discard(dir_path)


def _run_sync_batch(dirs: List[str], logs_all: Path):
    """后台批量同步。"""
    cfg = load_sync_config()
    obs_base = cfg.get("obs_base", "") or load_obs_base()
    workers = cfg.get("workers", 4)
    upload_script = cfg.get("upload_script", "")

    if not obs_base:
        for d in dirs:
            update_sync_status(d, "error", error_msg="obs_base 未配置")
            append_log(d, "同步失败: obs_base 未配置", level="error")
            _syncing_dirs.discard(d)
        return

    for d in dirs:
        _run_sync_for_dir(d, logs_all, obs_base, workers, upload_script)


def register_backup_routes(app: FastAPI, logs_dir: str, port: str = "") -> None:
    logs_all = Path(logs_dir).parent.parent
    templates = Jinja2Templates(directory="templates")
    parts = Path(logs_dir).parts
    _active_dir_path = f"{parts[-2]}/{parts[-1]}" if len(parts) >= 2 else ""
    _active_env_name = parts[-2] if len(parts) >= 2 else ""
    _project_root = str(Path(__file__).resolve().parent.parent)

    _port_env_map: dict = {}
    logs_root = Path("logs")
    if logs_root.is_dir():
        for port_dir in logs_root.iterdir():
            if port_dir.is_dir() and port_dir.name.startswith("port"):
                p = port_dir.name[4:]
                for env_d in port_dir.iterdir():
                    if env_d.is_dir():
                        _port_env_map[env_d.name] = p
    if _active_env_name and port:
        _port_env_map[_active_env_name] = port

    def _ctx(request: Request) -> dict:
        return {
            "active_page": "backup",
            "user_role": request.session.get("monitor_role", "user"),
            "user_name": request.session.get("monitor_user", ""),
            "user_permissions": [
                p.strip()
                for p in (request.session.get("monitor_permissions") or "").split(",")
                if p.strip()
            ],
        }

    @app.get("/backup")
    def backup_page(request: Request):
        return templates.TemplateResponse(request, "backup.html", context=_ctx(request))

    @app.get("/api/backup/dirs")
    def backup_list_dirs(request: Request):
        denied = _require_ajax(request)
        if denied:
            return denied

        env_name = request.query_params.get("env_name", "")

        # 只有该 env 在 DB 中完全没有记录时，才做一次性初始扫描（首次建立 DB 或 DB 丢失）
        if env_name and not has_any_dirs(env_name):
            fs_dirs = _scan_data_dirs(logs_all, env_name=env_name, port_env_map=_port_env_map)
            for d in fs_dirs:
                upsert_dir(d["dir_path"], d["env_name"], d["mtime_tag"], d["file_count"],
                           port=d.get("port", ""), has_local=True)

        db_dirs = list_dirs(env_name=env_name if env_name else None)

        # 从 token_index 按 env 批量加载统计数据（内存缓存，无额外 IO）
        _tok_cache: dict = {}
        def _get_tok_stats(env_n: str, mtime_t: str) -> dict:
            if env_n not in _tok_cache:
                try:
                    from utils.token_index import refresh_token_index
                    idx = refresh_token_index(str(logs_all / env_n))
                    _tok_cache[env_n] = idx.get("dirs", {})
                except Exception:
                    _tok_cache[env_n] = {}
            di = _tok_cache[env_n].get(mtime_t, {})
            return {
                "req_total": di.get("entry_count", 0),
                "req_success": sum(v.get("s_count", 0) for v in di.get("models", {}).values()),
                "req_error": sum(v.get("e_count", 0) for v in di.get("models", {}).values()),
                "tok_in": sum(v.get("s_tok_in", 0) + v.get("e_tok_in", 0) for v in di.get("models", {}).values()),
                "tok_out": sum(v.get("s_tok_out", 0) for v in di.get("models", {}).values()),
            }

        merged = []
        for d in db_dirs:
            stats = _get_tok_stats(d["env_name"], d["mtime_tag"])
            entry = {
                "dir_path": d["dir_path"],
                "env_name": d["env_name"],
                "mtime_tag": d["mtime_tag"],
                "file_count": d["file_count"],
                "synced": d.get("synced", 0),
                "status": d.get("status", "pending"),
                "sync_time": d.get("sync_time", ""),
                "obs_path": d.get("obs_path", ""),
                "error_msg": d.get("error_msg", ""),
                "sync_pid": d.get("sync_pid"),
                "is_active": d["dir_path"] == _active_dir_path,
                "has_local": bool(d.get("has_local", 1)),
                "has_obs": bool(d.get("has_obs", 0)),
                "port": d.get("port", ""),
                **stats,
            }
            merged.append(entry)

        return JSONResponse({"dirs": merged})

    @app.post("/api/backup/sync")
    async def backup_sync(request: Request):
        denied = _require_ajax(request)
        if denied:
            return denied

        body = await request.json()
        dirs = body.get("dirs", [])
        if not dirs:
            return JSONResponse({"detail": "没有选择目录"}, status_code=400)

        new_dirs = []
        for d in dirs:
            if d in _syncing_dirs:
                continue
            _syncing_dirs.add(d)
            new_dirs.append(d)

        if not new_dirs:
            return JSONResponse({"detail": "所选目录正在同步中"}, status_code=409)

        t = threading.Thread(target=_run_sync_batch, args=(new_dirs, logs_all), daemon=True)
        t.start()

        return JSONResponse({"started": new_dirs})

    @app.get("/api/backup/config")
    def backup_config_get(request: Request):
        denied = _require_ajax(request)
        if denied:
            return denied
        cfg = load_sync_config()
        return JSONResponse({
            "obs_base": cfg.get("obs_base", "") or load_obs_base(),
            "workers": cfg.get("workers", 4),
            "interval": cfg.get("interval", 600),
            "upload_script": cfg.get("upload_script", ""),
            "upload_timeout": cfg.get("upload_timeout", 3600),
            "upload_jobs": cfg.get("upload_jobs", 8),
        })

    @app.put("/api/backup/config")
    async def backup_config_put(request: Request):
        denied = _require_ajax(request)
        if denied:
            return denied

        body = await request.json()
        cfg_path = get_sync_config_path()
        if not cfg_path:
            return JSONResponse({"detail": "sync_config 未配置"}, status_code=400)

        import yaml
        try:
            if cfg_path.is_file():
                with open(cfg_path, "r", encoding="utf-8") as f:
                    existing = yaml.safe_load(f) or {}
            else:
                existing = {}

            for key in ("obs_base", "workers", "interval", "upload_script", "upload_timeout", "upload_jobs"):
                if key in body:
                    existing[key] = body[key]

            with open(cfg_path, "w", encoding="utf-8") as f:
                yaml.dump(existing, f, default_flow_style=False, allow_unicode=True)

            return JSONResponse({"ok": True})
        except Exception as e:
            return JSONResponse({"detail": str(e)}, status_code=500)

    @app.get("/api/backup/logs/{dir_path:path}")
    def backup_logs(request: Request, dir_path: str):
        denied = _require_ajax(request)
        if denied:
            return denied
        logs = get_logs(dir_path)
        dir_info = get_dir(dir_path)
        return JSONResponse({
            "logs": logs,
            "status": dir_info.get("status", "pending") if dir_info else "pending",
        })

    @app.delete("/api/backup/dirs/{dir_path:path}")
    def backup_delete_dir(request: Request, dir_path: str):
        denied = _require_ajax(request)
        if denied:
            return denied

        dir_info = get_dir(dir_path)
        if not dir_info:
            return JSONResponse({"detail": "目录不存在"}, status_code=404)

        if dir_info.get("status") != "done":
            return JSONResponse({"detail": "只能删除已同步完成的目录"}, status_code=400)

        target = logs_all / dir_path
        if target.is_dir():
            try:
                shutil.rmtree(str(target))
                append_log(dir_path, f"本地目录已删除: {target}")
            except Exception as e:
                append_log(dir_path, f"删除失败: {e}", level="error")
                return JSONResponse({"detail": f"删除失败: {e}"}, status_code=500)

        mark_backed_up(dir_path)
        return JSONResponse({"ok": True})

    @app.get("/api/backup/obs-browse/{obs_path:path}")
    def backup_obs_browse(request: Request, obs_path: str):
        denied = _require_ajax(request)
        if denied:
            return denied
        if not obs_path.startswith("obs://"):
            obs_path = "obs://" + obs_path
        try:
            items = obsutil_ls(obs_path)
            return JSONResponse({"path": obs_path, "items": items})
        except Exception as e:
            return JSONResponse({"detail": str(e)}, status_code=500)

    @app.get("/api/backup/active-dir")
    def backup_active_dir(request: Request):
        denied = _require_ajax(request)
        if denied:
            return denied
        return JSONResponse({
            "active_dir": _active_dir_path,
            "current_env": _active_env_name,
            "port": port,
        })

    @app.get("/api/backup/env-list")
    def backup_env_list(request: Request):
        denied = _require_ajax(request)
        if denied:
            return denied
        env_set = set(list_env_names())
        if logs_all.is_dir():
            for d in logs_all.iterdir():
                if d.is_dir() and not d.name.startswith("logs_") and not d.name.startswith("."):
                    env_set.add(d.name)
        envs = []
        for name in sorted(env_set):
            envs.append({"name": name, "port": _port_env_map.get(name, "")})
        return JSONResponse({"envs": envs, "current_env": _active_env_name})

    @app.post("/api/backup/scan-obs")
    async def backup_scan_obs(request: Request):
        denied = _require_ajax(request)
        if denied:
            return denied
        body = await request.json()
        env_name = body.get("env_name", "")
        cfg = load_sync_config()
        obs_base = cfg.get("obs_base", "") or load_obs_base()
        if not obs_base:
            return JSONResponse({"detail": "obs_base 未配置"}, status_code=400)
        obs_dirs = _scan_obs_dirs(obs_base, env_name=env_name if env_name else None)
        for d in obs_dirs:
            upsert_dir(d["dir_path"], d["env_name"], d["mtime_tag"],
                       port=_port_env_map.get(d["env_name"], ""), has_obs=True)
            existing = get_dir(d["dir_path"])
            if existing and existing.get("status") == "pending":
                update_sync_status(d["dir_path"], "backed_up", obs_path=d["obs_path"])
            elif existing and not existing.get("obs_path"):
                update_sync_status(d["dir_path"], existing["status"], obs_path=d["obs_path"])
        return JSONResponse({"found": len(obs_dirs)})

    @app.post("/api/backup/live-sync/start")
    async def backup_live_sync_start(request: Request):
        denied = _require_ajax(request)
        if denied:
            return denied

        body = await request.json()
        dir_path = body.get("dir_path", "")

        if dir_path != _active_dir_path:
            return JSONResponse({"detail": "只能对活跃目录开启在线同步"}, status_code=400)

        if dir_path in _live_sync_procs:
            proc = _live_sync_procs[dir_path]["proc"]
            if proc.poll() is None:
                return JSONResponse({"detail": "在线同步已在运行"}, status_code=409)

        dir_info = get_dir(dir_path)
        if dir_info and dir_info.get("status") == "live_syncing":
            old_pid = dir_info.get("sync_pid")
            if old_pid and _is_pid_alive(old_pid):
                return JSONResponse({"detail": "在线同步已在运行"}, status_code=409)

        cfg = load_sync_config()
        obs_base = cfg.get("obs_base", "") or load_obs_base()
        if not obs_base:
            return JSONResponse({"detail": "obs_base 未配置"}, status_code=400)

        env_name, mtime_tag = dir_path.split("/", 1)
        obs_dst = f"{obs_base.rstrip('/')}/raw/{env_name}/{mtime_tag}/"
        logs_dir_full = str(logs_all / dir_path)
        interval = cfg.get("interval", 600)
        workers = cfg.get("workers", 4)
        upload_script = cfg.get("upload_script")

        cmd = [
            sys.executable, "-m", "utils.obs_sync",
            "--logs-dir", logs_dir_full,
            "--obs-dst", obs_dst,
            "--interval", str(interval),
            "--workers", str(workers),
        ]
        if upload_script:
            cmd.extend(["--upload-script", str(upload_script)])

        proc = _subprocess.Popen(
            cmd,
            cwd=_project_root,
            stdout=_subprocess.PIPE,
            stderr=_subprocess.STDOUT,
            start_new_session=True,
        )

        log_thread = threading.Thread(
            target=_capture_sync_logs,
            args=(proc, dir_path),
            daemon=True,
        )
        log_thread.start()

        _live_sync_procs[dir_path] = {"proc": proc, "log_thread": log_thread}

        upsert_dir(dir_path, env_name, mtime_tag)
        update_sync_status(dir_path, "live_syncing", obs_path=obs_dst)
        update_sync_pid(dir_path, proc.pid)
        append_log(dir_path, f"在线同步已启动: pid={proc.pid}, interval={interval}s")

        return JSONResponse({"ok": True, "pid": proc.pid})

    @app.post("/api/backup/live-sync/stop")
    async def backup_live_sync_stop(request: Request):
        denied = _require_ajax(request)
        if denied:
            return denied

        body = await request.json()
        dir_path = body.get("dir_path", "")

        result = _stop_live_sync(dir_path)
        if result["ok"]:
            return JSONResponse({"ok": True})
        return JSONResponse({"detail": result.get("detail", "停止失败")}, status_code=400)

    _cleanup_stale_live_syncs()

    # 启动时将当前活跃 mtime 目录注册进 DB（append-only，不触发文件系统全量扫描）
    if _active_dir_path:
        _env_n, _mtime_t = _active_dir_path.split("/", 1)
        _active_local = logs_all / _active_dir_path
        _fcount = 0
        if _active_local.is_dir():
            try:
                _fcount = sum(1 for f in _active_local.iterdir() if f.is_file())
            except OSError:
                pass
        upsert_dir(_active_dir_path, _env_n, _mtime_t, _fcount,
                   port=_port_env_map.get(_env_n, port), has_local=True)
