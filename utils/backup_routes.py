"""
utils/backup_routes.py — 备份管理 Web 路由

管理所有 mtime 目录的 OBS 同步状态，支持手动触发同步和删除。
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import signal
import shutil
import subprocess as _subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates

from utils.backup_store import (
    append_log,
    claim_next_job,
    clear_logs,
    clear_failed_files,
    count_unresolved_failed,
    enqueue_job,
    finish_job,
    get_dir,
    get_live_syncing_dirs,
    get_logs,
    get_raw_download_job,
    has_any_dirs,
    list_dirs,
    list_env_names,
    list_failed_files,
    list_queue,
    mark_backed_up,
    mark_file_resolved,
    record_failed_files,
    reset_running_on_startup,
    create_raw_download_job,
    update_raw_download_job,
    update_sync_pid,
    update_sync_status,
    upsert_dir,
)
from utils.obs_utils import (
    load_obs_base,
    load_sync_config,
    get_sync_config_path,
    download_obs_object,
    obsutil_ls,
    find_failed_report,
    parse_failed_report,
    reupload_file,
    resolve_obsutil_config,
)

logger = logging.getLogger(__name__)

_sync_lock = threading.Lock()
_syncing_dirs: set = set()
_live_sync_procs: dict = {}

# 手动上传串行队列：单 dispatcher 线程逐个消费 DB 里的 backup_queue
_queue_cond = threading.Condition()
_dispatcher_thread = None
_dispatcher_ctx: dict = {}   # {logs_all, resolver} —— 由 register_backup_routes 填充

# 自动备份守护线程的共享状态（供 /api/backup/auto-status 透明展示）
_auto_lock = threading.Lock()
_auto_state: dict = {
    "enabled": False,
    "running": False,        # 是否正在跑一轮上传
    "disk_percent": None,
    "threshold": None,
    "queue": [],             # 待上传的 dir_path
    "current": "",           # 正在上传的 dir_path
    "uploaded": [],          # 本轮已成功上传（转 pending_delete）的 dir_path
    "pending_delete_count": 0,
    "last_check_ts": 0,
    "message": "",
}
_auto_thread = None


def _bounded_env_int(name: str, default: int, low: int, high: int) -> int:
    try:
        return max(low, min(high, int(os.getenv(name, str(default)))))
    except (TypeError, ValueError):
        return default


_RAW_JOB_EXECUTOR = ThreadPoolExecutor(
    max_workers=_bounded_env_int("OBS_DOWNLOAD_JOB_WORKERS", 2, 1, 4),
    thread_name_prefix="obs-download-job",
)
_RAW_OBJECT_EXECUTOR = ThreadPoolExecutor(
    max_workers=_bounded_env_int("OBS_DOWNLOAD_WORKERS", 4, 1, 8),
    thread_name_prefix="obs-download-object",
)
_RAW_CATALOG_CACHE: Dict[str, Tuple[float, List[dict], str]] = {}
_RAW_CATALOG_LOCK = threading.Lock()


class _RawCatalogError(RuntimeError):
    pass


def _raw_download_root() -> Path:
    raw = os.getenv("OBS_DOWNLOAD_ROOT", "").strip()
    root = Path(raw) if raw else Path(__file__).resolve().parent.parent / "downloads"
    if not root.is_absolute():
        root = Path(__file__).resolve().parent.parent / root
    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _safe_relative_file(filename: str) -> str:
    """把索引中的文件名规范为安全的 OBS 相对 key。"""
    raw = str(filename or "").strip().replace("\\", "/")
    if not raw:
        return ""
    # native index 可能记录上传前的绝对本地路径；OBS 目录里对应的是 basename。
    if raw.startswith("/") or re.match(r"^[A-Za-z]:/", raw):
        raw = raw.rsplit("/", 1)[-1]
    parts = [part for part in raw.split("/") if part not in ("", ".")]
    if not parts or any(part == ".." for part in parts):
        return ""
    return "/".join(parts)


def _critical_files(filename: str) -> List[str]:
    """返回关键原始文件：native 为 req+res，new-api 为合并单文件。"""
    rel = _safe_relative_file(filename)
    if not rel:
        return []
    if rel.endswith("-req.json"):
        return [rel, rel[:-len("-req.json")] + "-res.json"]
    return [rel]


def _best_session_file(session: dict) -> str:
    traces = [
        trace for trace in (session.get("trace_list") or [])
        if isinstance(trace, dict) and (trace.get("filename") or trace.get("req_file"))
    ]
    if traces:
        successful = [trace for trace in traces if trace.get("success") is not False]
        candidates = successful or traces

        def _rank(trace: dict):
            try:
                msg_count = int(trace.get("msg_count") or 0)
            except (TypeError, ValueError):
                msg_count = 0
            return msg_count, str(trace.get("ts") or "")

        best = max(candidates, key=_rank)
        return str(best.get("filename") or best.get("req_file") or "")
    return str(session.get("latest_file") or "")


def _session_catalog_item(session: dict, position: int, obs_path: str) -> Optional[dict]:
    filename = _best_session_file(session)
    files = _critical_files(filename)
    if not files:
        return None
    key = str(
        session.get("session")
        or session.get("_key")
        or session.get("first_ts")
        or f"row:{position}"
    )
    digest = hashlib.sha256(
        f"{obs_path}\0{key}\0{filename}".encode("utf-8")
    ).hexdigest()[:24]
    models = session.get("models") or []
    if isinstance(models, str):
        models = [models] if models else []
    try:
        msg_count = int(
            session.get("total_messages", session.get("msg_count", 0)) or 0
        )
    except (TypeError, ValueError):
        msg_count = 0
    return {
        "session_id": digest,
        "q1": str(session.get("q1") or session.get("q1_preview") or "")[:300],
        "last_ts": str(
            session.get("end_time")
            or session.get("last_ts")
            or session.get("ts")
            or ""
        ),
        "models": [str(model) for model in models if model],
        "msg_count": msg_count,
        "files_count": len(files),
        "_files": files,
    }


def _parse_session_index(index_path: Path, obs_path: str) -> List[dict]:
    items: List[dict] = []
    with index_path.open("r", encoding="utf-8", errors="replace") as stream:
        for position, raw in enumerate(stream):
            try:
                obj = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(obj, dict) or obj.get("_meta"):
                continue
            item = _session_catalog_item(obj, position, obs_path)
            if item:
                items.append(item)
    items.sort(key=lambda item: item["last_ts"], reverse=True)
    return items


def _parse_raw_index(index_path: Path, obs_path: str) -> List[dict]:
    """基础降级：利用 index 元数据按 api_key + 会话线索聚合。"""
    groups: Dict[str, dict] = {}
    with index_path.open("r", encoding="utf-8", errors="replace") as stream:
        for position, raw in enumerate(stream):
            try:
                obj = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(obj, dict):
                continue
            filename = obj.get("req_file") or obj.get("filename")
            if not filename:
                continue
            # index 的 req_file 可能相对其父目录；备份叶子中对象名统一取 basename。
            filename = str(filename).replace("\\", "/").rsplit("/", 1)[-1]
            hint = obj.get("q1_hash") or obj.get("chain_key") or filename
            group_key = f"{obj.get('api_key', '')}\0{hint}"
            group = groups.setdefault(
                group_key,
                {
                    "q1": "",
                    "last_ts": "",
                    "models": set(),
                    "best": None,
                    "position": position,
                },
            )
            if obj.get("q1_preview") and not group["q1"]:
                group["q1"] = str(obj["q1_preview"])[:300]
            if obj.get("model"):
                group["models"].add(str(obj["model"]))
            ts = str(obj.get("ts") or "")
            if ts > group["last_ts"]:
                group["last_ts"] = ts
            try:
                msg_count = int(obj.get("msg_count") or 0)
            except (TypeError, ValueError):
                msg_count = 0
            rank = (1 if obj.get("success") is not False else 0, msg_count, ts)
            current = group["best"]
            if current is None or rank > current["rank"]:
                group["best"] = {
                    "filename": filename,
                    "msg_count": msg_count,
                    "rank": rank,
                }

    items: List[dict] = []
    for group_key, group in groups.items():
        best = group["best"]
        if not best:
            continue
        files = _critical_files(str(best["filename"]))
        if not files:
            continue
        digest = hashlib.sha256(
            f"{obs_path}\0{group_key}\0{best['filename']}".encode("utf-8")
        ).hexdigest()[:24]
        items.append({
            "session_id": digest,
            "q1": group["q1"],
            "last_ts": group["last_ts"],
            "models": sorted(group["models"]),
            "msg_count": best["msg_count"],
            "files_count": len(files),
            "_files": files,
        })
    items.sort(key=lambda item: item["last_ts"], reverse=True)
    return items


def _catalog_cache_file(obs_path: str, filename: str) -> Path:
    digest = hashlib.sha256(obs_path.encode("utf-8")).hexdigest()[:24]
    cache_dir = _raw_download_root() / ".catalog" / digest
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / filename


def _download_catalog_file(
    obs_path: str,
    filename: str,
    config_path: str,
) -> Tuple[Optional[Path], str]:
    target = _catalog_cache_file(obs_path, filename)
    temp_path = target.with_name(f"{target.name}.tmp-{uuid.uuid4().hex}")
    object_path = f"{obs_path.rstrip('/')}/{filename}"
    ok, message = download_obs_object(
        object_path,
        str(temp_path),
        config_path=config_path,
        timeout=_bounded_env_int("OBS_CATALOG_DOWNLOAD_TIMEOUT", 120, 10, 900),
    )
    if not ok:
        try:
            temp_path.unlink()
        except OSError:
            pass
        return None, message
    temp_path.replace(target)
    return target, ""


def _catalog_failure_message(*errors: str) -> str:
    """将 obsutil 错误归一成可操作、且不泄露底层请求信息的提示。"""
    combined = " ".join(error or "" for error in errors)
    auth_markers = (
        "InvalidAccessKeyId",
        "AccessDenied",
        "SignatureDoesNotMatch",
        "Status [403]",
        "status [403]",
        "invalid header field value",
    )
    if any(marker in combined for marker in auth_markers):
        return "OBS 配置或凭据无效，请检查该桶对应的 obsutil 配置"
    if "没有可识别的会话" in combined or "解析失败" in combined:
        return "OBS 索引中没有可识别的会话，请检查索引格式"
    not_found_markers = ("NoSuchKey", "Status [404]", "status [404]")
    if combined and all(
        any(marker in error for marker in not_found_markers)
        for error in errors
        if error
    ):
        return "OBS 中未找到可用的 session_index.jsonl 或 index.jsonl"
    return "OBS 索引读取失败，请检查网络和该桶的 obsutil 配置"


def _load_cached_raw_catalog(obs_path: str) -> Tuple[Optional[List[dict]], str]:
    """读取最后一次成功落盘的会话目录，不触发任何 OBS 请求。"""
    candidates = (
        ("session_index.jsonl", _parse_session_index),
        ("index.jsonl", _parse_raw_index),
    )
    for filename, parser in candidates:
        cache_file = _catalog_cache_file(obs_path, filename)
        if not cache_file.is_file():
            continue
        try:
            items = parser(cache_file, obs_path)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            logger.exception("load cached raw catalog failed: %s", cache_file)
            continue
        if items:
            return items, filename
    return None, ""


def _load_raw_catalog(obs_path: str) -> Tuple[List[dict], str, str]:
    ttl = _bounded_env_int("OBS_CATALOG_TTL", 300, 0, 3600)
    now = time.time()
    with _RAW_CATALOG_LOCK:
        cached = _RAW_CATALOG_CACHE.get(obs_path)
        if cached and now - cached[0] <= ttl:
            return cached[1], cached[2], ""

        try:
            config_path = resolve_obsutil_config(obs_path)
        except (ValueError, FileNotFoundError) as exc:
            raise _RawCatalogError(str(exc)) from exc

        session_path, session_error = _download_catalog_file(
            obs_path, "session_index.jsonl", config_path
        )
        items: Optional[List[dict]] = None
        source = ""
        if session_path:
            try:
                session_items = _parse_session_index(session_path, obs_path)
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                logger.exception("parse session_index.jsonl failed: %s", session_path)
                session_error = "session_index.jsonl 解析失败"
            else:
                if session_items:
                    items = session_items
                    source = "session_index.jsonl"
                else:
                    session_error = "session_index.jsonl 中没有可识别的会话"

        if items is None:
            raw_path, raw_error = _download_catalog_file(
                obs_path, "index.jsonl", config_path
            )
            if raw_path:
                try:
                    items = _parse_raw_index(raw_path, obs_path)
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    logger.exception("parse index.jsonl failed: %s", raw_path)
                    raw_error = "index.jsonl 解析失败"
                else:
                    source = "index.jsonl"

            if items is None:
                logger.warning(
                    "raw catalog download failed: session=%s raw=%s",
                    session_error[-200:],
                    raw_error[-200:],
                )
                failure = _catalog_failure_message(session_error, raw_error)
                cached_items, cached_source = _load_cached_raw_catalog(obs_path)
                if cached_items is not None:
                    warning = (
                        f"{failure}；当前展示上次成功缓存的会话列表，"
                        "修复配置后才能下载原始文件"
                    )
                    return cached_items, cached_source, warning
                raise _RawCatalogError(failure)

        _RAW_CATALOG_CACHE[obs_path] = (now, items, source)
        return items, source, ""


def _run_raw_download_job(job_id: str) -> None:
    job = get_raw_download_job(job_id)
    if not job:
        return
    update_raw_download_job(job_id, status="running", downloaded_files=0)
    downloaded = 0
    errors: List[str] = []
    try:
        obs_path = str(job["obs_path"]).rstrip("/") + "/"
        config_path = resolve_obsutil_config(obs_path)
        allowed_root = _raw_download_root()
        job_root = Path(job["output_path"]).resolve()
        try:
            job_root.relative_to(allowed_root)
        except ValueError as exc:
            raise RuntimeError("下载目标路径越界") from exc
        job_root.mkdir(parents=True, exist_ok=True)

        def _download_one(spec: dict) -> Tuple[bool, str]:
            rel_file = _safe_relative_file(spec.get("relative_path", ""))
            session_id = str(spec.get("session_id", ""))
            if not rel_file or not re.fullmatch(r"[0-9a-f]{24}", session_id):
                return False, "无效的下载项"
            local_path = (job_root / session_id / rel_file).resolve()
            try:
                local_path.relative_to(job_root)
            except ValueError:
                return False, "下载目标路径越界"
            object_path = f"{obs_path}{rel_file}"
            return download_obs_object(
                object_path,
                str(local_path),
                config_path=config_path,
                timeout=_bounded_env_int("OBS_OBJECT_DOWNLOAD_TIMEOUT", 300, 10, 3600),
            )

        futures = {
            _RAW_OBJECT_EXECUTOR.submit(_download_one, spec): spec
            for spec in (job.get("manifest") or [])
        }
        for future in as_completed(futures):
            spec = futures[future]
            try:
                ok, message = future.result()
            except Exception as exc:  # noqa: BLE001
                ok, message = False, str(exc)
            if ok:
                downloaded += 1
                update_raw_download_job(job_id, downloaded_files=downloaded)
            elif len(errors) < 20:
                errors.append(
                    f"{spec.get('relative_path', 'unknown')}: {str(message)[-200:]}"
                )

        total = int(job.get("total_files") or 0)
        if downloaded == total:
            status = "completed"
        else:
            status = "failed"
        update_raw_download_job(
            job_id,
            status=status,
            downloaded_files=downloaded,
            error_msg="\n".join(errors)[:4000],
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("raw OBS download failed: job=%s", job_id)
        update_raw_download_job(
            job_id,
            status="failed",
            downloaded_files=downloaded,
            error_msg=str(exc)[:1000],
        )


def _auto_set(**kw):
    with _auto_lock:
        _auto_state.update(kw)


def _auto_snapshot() -> dict:
    with _auto_lock:
        return dict(_auto_state)


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


def _human_size(nbytes: int) -> str:
    val = float(nbytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if val < 1024 or unit == "TB":
            return f"{int(val)}{unit}" if unit == "B" else f"{val:.1f}{unit}"
        val /= 1024
    return f"{val:.1f}TB"


def _disk_usage(path: Path) -> dict:
    """返回 path 所在挂载点的磁盘使用情况（df）。"""
    try:
        st = shutil.disk_usage(str(path))
        return {
            "total": st.total, "used": st.used, "free": st.free,
            "total_h": _human_size(st.total), "used_h": _human_size(st.used),
            "free_h": _human_size(st.free),
            "percent": round(st.used / st.total * 100, 1) if st.total else 0,
        }
    except OSError:
        return {}


def _du_size(path: Path) -> int:
    """递归统计目录字节数（du）。大目录可能较慢。"""
    total = 0
    try:
        for dirpath, _dn, filenames in os.walk(str(path)):
            for fn in filenames:
                try:
                    total += os.path.getsize(os.path.join(dirpath, fn))
                except OSError:
                    continue
    except OSError:
        pass
    return total


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


def _scan_root_leaves(root: str) -> List[dict]:
    """扫描一个非 logs_all 根（如 new-api 的 details）下含 index.jsonl 的叶子。

    env_name = 根的稳定 root_id（消除同 basename 冲突）；
    mtime_tag = 叶子相对根的路径（天/小时）；dir_path = env_name/mtime_tag。
    """
    result = []
    try:
        from utils.log_scan import iter_index_dirs, dir_key_for
        from utils.logs_config import get_root_id
    except Exception:
        return result
    rp = Path(root)
    if not rp.is_dir():
        return result
    env_name = get_root_id(root)
    for leaf in iter_index_dirs(rp):
        rel = dir_key_for(rp, leaf)
        try:
            file_count = sum(1 for f in leaf.iterdir() if f.is_file())
        except OSError:
            file_count = 0
        result.append({
            "dir_path": f"{env_name}/{rel}",
            "env_name": env_name,
            "mtime_tag": rel,
            "file_count": file_count,
            "port": "",
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


def _run_sync_for_dir(dir_path: str, logs_all: Path, obs_base: str, workers: int, upload_script: str,
                      resolver=None):
    """在后台线程中同步单个目录到 OBS，直接上传整个文件夹，流式输出日志。

    resolver: 可选 dir_path -> 真实磁盘 Path 的解析器（供 new-api 等非 logs_all 根）。
    """
    import subprocess as sp
    from utils.obs_utils import DEFAULT_UPLOAD_SCRIPT

    real_dir = resolver(dir_path) if resolver else (logs_all / dir_path)
    logs_dir = str(real_dir)
    env_name, mtime_tag = dir_path.split("/", 1)
    obs_dst = f"{obs_base.rstrip('/')}/raw/{env_name}/{mtime_tag}/"
    script = upload_script or DEFAULT_UPLOAD_SCRIPT
    if not Path(script).is_absolute():
        script = str((Path(__file__).resolve().parent.parent / script).resolve())

    try:
        # 本地目录不存在：已被清理，标记为 backed_up 跳过上传
        if not Path(real_dir).is_dir():
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
        task_id = ""
        output_dir = ""
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                append_log(dir_path, line)
                # obsutil 开始时打印 Task id / OutputDir，捕获用于定位失败报告
                if not task_id and line.startswith("Task id:"):
                    task_id = line.split(":", 1)[1].strip()
                elif not output_dir and line.startswith("OutputDir:"):
                    output_dir = line.split(":", 1)[1].strip()
        proc.wait(timeout=600)

        if proc.returncode == 0:
            update_sync_status(dir_path, "done", obs_path=obs_dst)
            clear_failed_files(dir_path)
            append_log(dir_path, "同步完成")
        else:
            # 尝试从 obsutil 失败报告解析出失败文件清单 → 标记 partial（部分失败）
            failed_items = []
            report = find_failed_report(task_id, output_dir) if task_id else None
            if report:
                failed_items = parse_failed_report(report)
            if failed_items:
                record_failed_files(dir_path, failed_items)
                msg = f"部分失败: {len(failed_items)} 个文件上传失败 (exit_code={proc.returncode})"
                append_log(dir_path, f"{msg}，可点击「补同步」重传", level="error")
                update_sync_status(dir_path, "partial", error_msg=msg, obs_path=obs_dst)
            else:
                append_log(dir_path, f"上传失败: exit_code={proc.returncode}", level="error")
                update_sync_status(dir_path, "error", error_msg=f"exit_code={proc.returncode}", obs_path=obs_dst)
    except Exception as e:
        update_sync_status(dir_path, "error", error_msg=str(e))
        append_log(dir_path, f"同步失败: {e}", level="error")
        logger.exception("backup sync failed for %s", dir_path)
    finally:
        _syncing_dirs.discard(dir_path)


def _run_resync_failed(dir_path: str):
    """补同步：逐个重传该目录 obsutil 上报的失败文件。"""
    try:
        items = list_failed_files(dir_path, only_unresolved=True)
        if not items:
            append_log(dir_path, "没有待补传的失败文件")
            # 无遗留失败 → 若之前是 partial，视为已完成
            dir_info = get_dir(dir_path)
            if dir_info and dir_info.get("status") == "partial":
                update_sync_status(dir_path, "done", obs_path=dir_info.get("obs_path", ""))
                append_log(dir_path, "无待补文件，状态修正为 done")
            return

        update_sync_status(dir_path, "syncing")
        append_log(dir_path, f"开始补同步: {len(items)} 个失败文件")
        ok_count = 0
        fail_count = 0
        for it in items:
            local = it["local_path"]
            obs = it["obs_path"]
            success, msg = reupload_file(local, obs)
            if success:
                mark_file_resolved(dir_path, obs)
                ok_count += 1
                append_log(dir_path, f"补传成功: {os.path.basename(local)}")
            else:
                fail_count += 1
                append_log(dir_path, f"补传失败: {os.path.basename(local)} — {msg}", level="error")

        remaining = count_unresolved_failed(dir_path)
        dir_info = get_dir(dir_path)
        obs_path = dir_info.get("obs_path", "") if dir_info else ""
        if remaining == 0:
            update_sync_status(dir_path, "done", obs_path=obs_path)
            clear_failed_files(dir_path)
            append_log(dir_path, f"补同步完成: {ok_count} 个成功，全部已备份")
        else:
            update_sync_status(dir_path, "partial",
                               error_msg=f"仍有 {remaining} 个文件未成功", obs_path=obs_path)
            append_log(dir_path, f"补同步部分完成: {ok_count} 成功 / {fail_count} 失败，剩余 {remaining}",
                       level="error")
    except Exception as e:
        update_sync_status(dir_path, "partial", error_msg=f"补同步异常: {e}")
        append_log(dir_path, f"补同步异常: {e}", level="error")
        logger.exception("resync failed for %s", dir_path)
    finally:
        _syncing_dirs.discard(dir_path)


def _queue_dispatch_loop():
    """单 dispatcher：从 backup_queue 逐个领取手动上传任务，串行执行。"""
    while True:
        job = claim_next_job()
        if not job:
            with _queue_cond:
                _queue_cond.wait(timeout=30)
            continue

        dir_path = job["dir_path"]
        job_type = job.get("job_type", "sync")
        _syncing_dirs.add(dir_path)
        try:
            if job_type == "resync":
                _run_resync_failed(dir_path)
            else:
                cfg = load_sync_config()
                obs_base = cfg.get("obs_base", "") or load_obs_base()
                workers = cfg.get("workers", 4)
                upload_script = cfg.get("upload_script", "")
                if not obs_base:
                    update_sync_status(dir_path, "error", error_msg="obs_base 未配置")
                    append_log(dir_path, "同步失败: obs_base 未配置", level="error")
                    _syncing_dirs.discard(dir_path)
                else:
                    logs_all = _dispatcher_ctx.get("logs_all")
                    resolver = _dispatcher_ctx.get("resolver")
                    _run_sync_for_dir(dir_path, logs_all, obs_base, workers,
                                      upload_script, resolver=resolver)
        except Exception:
            logger.exception("queue dispatch failed for %s", dir_path)
            _syncing_dirs.discard(dir_path)
        finally:
            finish_job(job["id"])


def _ensure_dispatcher():
    """懒启动单 dispatcher 线程（幂等）。"""
    global _dispatcher_thread
    with _queue_cond:
        if _dispatcher_thread is not None and _dispatcher_thread.is_alive():
            return
        _dispatcher_thread = threading.Thread(
            target=_queue_dispatch_loop, daemon=True, name="backup-queue-dispatcher"
        )
        _dispatcher_thread.start()


def _notify_dispatcher():
    with _queue_cond:
        _queue_cond.notify()


# ---------------------------------------------------------------------------
# 自动备份：磁盘达阈值时自动上传未备份目录，成功后转 pending_delete（等人工确认删除）
# ---------------------------------------------------------------------------

def _auto_cfg() -> dict:
    """从 sync_config(settings/obs_rl.yaml) 读自动备份配置。"""
    cfg = load_sync_config()
    return {
        "enabled": bool(cfg.get("auto_backup_enabled", False)),
        "threshold": float(cfg.get("auto_backup_disk_percent", 70) or 70),
        "interval": max(30, int(cfg.get("auto_backup_check_interval", 120) or 120)),
        "stop_percent": float(cfg.get("auto_backup_stop_percent", 0) or 0),
        "workers": int(cfg.get("workers", 4) or 4),
        "upload_script": cfg.get("upload_script", "") or "",
        "obs_base": cfg.get("obs_base", "") or load_obs_base(),
    }


def _run_auto_backup_round(logs_all: Path, active_env: str, active_path: str,
                           resolver, forced: bool = False) -> None:
    """跑一轮自动上传：把 active_env 下未备份目录最旧优先逐个上传，成功转 pending_delete。

    forced=True 时无视阈值直接跑（手动触发/立即备份）。
    """
    from utils.backup_store import list_backup_candidates, count_pending_delete

    cfg = _auto_cfg()
    if not cfg["obs_base"]:
        _auto_set(running=False, message="未配置 OBS(obs_base)，无法自动备份")
        return

    candidates = list_backup_candidates(active_env)
    queue = [d["dir_path"] for d in candidates]
    _auto_set(running=True, queue=list(queue), current="", uploaded=[],
              message=("手动触发，" if forced else "") + f"待上传 {len(queue)} 个目录")

    if not queue:
        _auto_set(running=False, current="", message="没有待备份的目录")
        return

    uploaded: List[str] = []
    for dir_path in queue:
        # 与手动同步互斥
        with _sync_lock:
            if dir_path in _syncing_dirs:
                continue
            _syncing_dirs.add(dir_path)
        _auto_set(current=dir_path,
                  queue=[d for d in queue if d not in uploaded and d != dir_path])
        try:
            _run_sync_for_dir(dir_path, logs_all, cfg["obs_base"], cfg["workers"],
                              cfg["upload_script"], resolver=resolver)
        except Exception as e:  # noqa: BLE001
            logger.exception("auto backup upload failed: %s", dir_path)
            append_log(dir_path, f"自动备份上传异常: {e}", level="error")
            continue

        # 上传结果以 DB 状态为准：done 才转 pending_delete（partial/error 保留原状态待人工处理）
        info = get_dir(dir_path)
        if info and info.get("status") == "done" and info.get("obs_path"):
            update_sync_status(dir_path, "pending_delete", obs_path=info["obs_path"])
            append_log(dir_path, "自动备份上传成功，转入待确认删除")
            uploaded.append(dir_path)
            _auto_set(uploaded=list(uploaded),
                      pending_delete_count=count_pending_delete(active_env))
        else:
            append_log(dir_path, "自动备份未成功（保留状态，需人工处理）", level="error")

        # 每传完一个重查磁盘：若配置了 stop_percent 且已降到其下则提前结束本轮
        disk = _disk_usage(Path(active_path))
        pct = disk.get("percent")
        _auto_set(disk_percent=pct)
        if not forced and cfg["stop_percent"] and pct is not None and pct <= cfg["stop_percent"]:
            _auto_set(message=f"磁盘已降至 {pct}% ≤ 停止阈值 {cfg['stop_percent']}%，本轮结束")
            break

    _auto_set(running=False, current="", queue=[],
              pending_delete_count=count_pending_delete(active_env),
              message=f"本轮完成：成功上传 {len(uploaded)} 个，待确认删除")


def _auto_backup_loop(logs_all: Path, active_env: str, active_path: str, resolver) -> None:
    """后台守护：定时查 df，达阈值则跑一轮自动上传。仅 leader worker 启动。"""
    from utils.backup_store import count_pending_delete

    logger.info("auto-backup daemon started (env=%s, path=%s)", active_env, active_path)
    while True:
        try:
            cfg = _auto_cfg()
            disk = _disk_usage(Path(active_path))
            pct = disk.get("percent")
            _auto_set(enabled=cfg["enabled"], threshold=cfg["threshold"],
                      disk_percent=pct, last_check_ts=time.time(),
                      pending_delete_count=count_pending_delete(active_env))

            if not cfg["enabled"]:
                _auto_set(message="自动备份未开启")
            elif pct is None:
                _auto_set(message="无法读取磁盘使用率")
            elif pct < cfg["threshold"]:
                _auto_set(message=f"磁盘 {pct}% < 阈值 {cfg['threshold']}%，无需自动备份")
            else:
                _auto_set(message=f"磁盘 {pct}% ≥ 阈值 {cfg['threshold']}%，开始自动上传")
                _run_auto_backup_round(logs_all, active_env, active_path, resolver)

            time.sleep(cfg["interval"])
        except Exception:  # noqa: BLE001 — 守护线程不能崩
            logger.exception("auto-backup loop error")
            time.sleep(60)


def register_backup_routes(
    app: FastAPI,
    logs_dir: str,
    port: str = "",
    context_builder: Optional[Callable[[Request, str], dict]] = None,
) -> None:
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

    # --- 多根支持：把配置的历史路径（含 new-api）也纳入备份 ---
    # env_name = 根目录的稳定 root_id（消除同 basename 冲突，如两个 new-api details 根）。
    # 对历史根，其 root_id 作为一个「env」，其 天/小时 叶子作为 mtime_tag。
    def _extra_roots() -> list:
        """配置的历史路径（排除活跃 logs_all 自身，那条已由 logs_all 扫描覆盖）。"""
        try:
            from utils.logs_config import get_history_paths
            return [p for p in get_history_paths() if Path(p).is_dir()]
        except Exception:
            return []

    def _root_env_name(root: str) -> str:
        from utils.logs_config import get_root_id
        return get_root_id(root, logs_dir)

    def _root_basename(root: str) -> str:
        """旧标识（basename），仅用于兼容存量 dir_path 的回退匹配。"""
        return os.path.basename(os.path.normpath(root))

    def _extra_root_for_env(env_name: str):
        for root in _extra_roots():
            if _root_env_name(root) == env_name or _root_basename(root) == env_name:
                return root
        return None

    def _resolve_real_dir(dir_path: str) -> Path:
        """把 dir_path（env_name/mtime_tag[/...]）解析为真实磁盘路径。

        优先按 logs_all/{dir_path}；不存在时在配置的历史根里按 root_id/<rel> 匹配，
        并对存量记录里遗留的旧 basename 前缀保留回退匹配。
        """
        cand = logs_all / dir_path
        if cand.is_dir():
            return cand
        env_name = dir_path.split("/", 1)[0]
        for root in _extra_roots():
            if _root_env_name(root) == env_name or _root_basename(root) == env_name:
                rel = dir_path.split("/", 1)[1] if "/" in dir_path else ""
                from utils.log_scan import resolve_leaf
                real = resolve_leaf(root, rel)
                if real.is_dir():
                    return real
        return cand  # 兜底：返回 logs_all 下的路径（可能不存在）

    # 一次性幂等迁移：存量记录旧 env 前缀（basename）-> root_id
    try:
        from utils.backup_store import migrate_env_prefix
        _mig = {}
        for root in _extra_roots():
            old = _root_basename(root)
            new = _root_env_name(root)
            if old != new:
                _mig[old] = new
        migrate_env_prefix(_mig)
    except Exception:
        logger.exception("backup env-prefix migration failed")

    # 启动手动上传串行队列 dispatcher：登记上下文 + 恢复中断项 + 起线程
    _dispatcher_ctx["logs_all"] = logs_all
    _dispatcher_ctx["resolver"] = _resolve_real_dir
    try:
        reset_running_on_startup()
    except Exception:
        logger.exception("reset backup_queue running-on-startup failed")
    _ensure_dispatcher()

    def _ctx(request: Request) -> dict:
        if context_builder is not None:
            return context_builder(request, "backup")
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
        _newapi_root = _extra_root_for_env(env_name) if env_name else None

        # 只有该 env 在 DB 中完全没有记录时，才做一次性初始扫描（首次建立 DB 或 DB 丢失）
        if env_name and not has_any_dirs(env_name):
            if _newapi_root:
                fs_dirs = _scan_root_leaves(_newapi_root)
            else:
                fs_dirs = _scan_data_dirs(logs_all, env_name=env_name, port_env_map=_port_env_map)
            for d in fs_dirs:
                upsert_dir(d["dir_path"], d["env_name"], d["mtime_tag"], d["file_count"],
                           port=d.get("port", ""), has_local=True)

        db_dirs = list_dirs(env_name=env_name if env_name else None)

        # root_id -> 展示名（配置的历史根用其 name；logs_all 下 env 目录名即展示名）
        from utils.logs_config import get_path_name
        _label_map = {}
        for root in _extra_roots():
            _label_map[_root_env_name(root)] = get_path_name(root)

        # 从 token_index 按 env 批量加载统计数据（内存缓存，无额外 IO）
        # new-api env 的 token_index 以真实根为 key（mtime_tag 为 天/小时 相对路径）。
        _tok_cache: dict = {}
        def _get_tok_stats(env_n: str, mtime_t: str) -> dict:
            if env_n not in _tok_cache:
                try:
                    from utils.token_index import refresh_token_index
                    _root = _extra_root_for_env(env_n) or str(logs_all / env_n)
                    idx = refresh_token_index(str(_root))
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
                "env_label": _label_map.get(d["env_name"], d["env_name"]),
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
                "failed_count": count_unresolved_failed(d["dir_path"]),
                **stats,
            }
            merged.append(entry)

        return JSONResponse({"dirs": merged})

    def _raw_catalog_for_dir(dir_path: str) -> Tuple[List[dict], str, str, str]:
        info = get_dir(dir_path)
        if not info:
            raise ValueError("备份记录不存在")
        obs_path = str(info.get("obs_path") or "").strip()
        if not obs_path.startswith("obs://"):
            raise ValueError("该备份没有可用的 OBS 路径")
        items, source, warning = _load_raw_catalog(obs_path)
        return items, source, obs_path, warning

    @app.get("/api/backup/raw-sessions")
    def backup_raw_sessions(request: Request):
        """分页返回某条 OBS 备份中的会话目录。"""
        denied = _require_ajax(request)
        if denied:
            return denied

        dir_path = request.query_params.get("dir_path", "").strip()
        if not dir_path:
            return JSONResponse({"detail": "缺少 dir_path"}, status_code=400)
        try:
            offset = max(0, int(request.query_params.get("offset", "0")))
            limit = max(1, min(200, int(request.query_params.get("limit", "50"))))
        except ValueError:
            return JSONResponse({"detail": "offset/limit 必须为整数"}, status_code=400)
        keyword = request.query_params.get("q", "").strip().casefold()

        try:
            catalog, source, _, warning = _raw_catalog_for_dir(dir_path)
        except (ValueError, _RawCatalogError) as exc:
            return JSONResponse({"detail": str(exc)}, status_code=400)
        if keyword:
            catalog = [
                item for item in catalog
                if keyword in item.get("q1", "").casefold()
                or any(keyword in model.casefold() for model in item.get("models", []))
            ]
        page = []
        for item in catalog[offset:offset + limit]:
            page.append({key: value for key, value in item.items() if key != "_files"})
        return JSONResponse({
            "items": page,
            "total": len(catalog),
            "offset": offset,
            "limit": limit,
            "source": source,
            "warning": warning,
            "download_available": not bool(warning),
        })

    @app.post("/api/backup/raw-download")
    async def backup_raw_download(request: Request):
        """创建选择性 OBS 单对象下载任务。"""
        denied = _require_ajax(request)
        if denied:
            return denied
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"detail": "请求 JSON 无效"}, status_code=400)

        dir_path = str(body.get("dir_path") or "").strip()
        raw_ids = body.get("session_ids", body.get("sessions", []))
        if not dir_path:
            return JSONResponse({"detail": "缺少 dir_path"}, status_code=400)
        if not isinstance(raw_ids, list) or not raw_ids:
            return JSONResponse({"detail": "请选择至少一个会话"}, status_code=400)

        max_sessions = _bounded_env_int("OBS_DOWNLOAD_MAX_SESSIONS", 500, 1, 5000)
        session_ids = list(dict.fromkeys(str(value) for value in raw_ids if value))
        if len(session_ids) > max_sessions:
            return JSONResponse(
                {"detail": f"单次最多选择 {max_sessions} 个会话"},
                status_code=400,
            )

        try:
            catalog, _, obs_path, warning = await asyncio.to_thread(
                _raw_catalog_for_dir, dir_path
            )
        except (ValueError, _RawCatalogError) as exc:
            return JSONResponse({"detail": str(exc)}, status_code=400)
        if warning:
            return JSONResponse({"detail": warning}, status_code=503)

        by_id = {item["session_id"]: item for item in catalog}
        missing = [session_id for session_id in session_ids if session_id not in by_id]
        if missing:
            return JSONResponse(
                {"detail": "所选会话已失效，请刷新列表后重试"},
                status_code=400,
            )

        manifest = []
        for session_id in session_ids:
            for relative_path in by_id[session_id]["_files"]:
                manifest.append({
                    "session_id": session_id,
                    "relative_path": relative_path,
                })
        if not manifest:
            return JSONResponse({"detail": "所选会话没有可下载文件"}, status_code=400)

        job_id = uuid.uuid4().hex
        output_path = _raw_download_root() / job_id
        try:
            create_raw_download_job(
                job_id=job_id,
                dir_path=dir_path,
                obs_path=obs_path,
                output_path=str(output_path),
                manifest=manifest,
            )
            _RAW_JOB_EXECUTOR.submit(_run_raw_download_job, job_id)
        except Exception as exc:  # noqa: BLE001
            logger.exception("create raw download job failed")
            return JSONResponse({"detail": f"创建下载任务失败: {exc}"}, status_code=500)
        return JSONResponse({"job_id": job_id})

    @app.get("/api/backup/raw-download/status/{job_id}")
    @app.get("/api/backup/raw-download/{job_id}")
    def backup_raw_download_status(request: Request, job_id: str):
        denied = _require_ajax(request)
        if denied:
            return denied
        if not re.fullmatch(r"[0-9a-f]{32}", job_id):
            return JSONResponse({"detail": "下载任务不存在"}, status_code=404)
        job = get_raw_download_job(job_id)
        if not job:
            return JSONResponse({"detail": "下载任务不存在"}, status_code=404)
        return JSONResponse({
            "status": job.get("status", "failed"),
            "total_files": int(job.get("total_files") or 0),
            "downloaded_files": int(job.get("downloaded_files") or 0),
            "output_path": job.get("output_path", ""),
            "error_msg": job.get("error_msg", ""),
        })

    @app.post("/api/backup/sync")
    async def backup_sync(request: Request):
        denied = _require_ajax(request)
        if denied:
            return denied

        body = await request.json()
        dirs = body.get("dirs", [])
        if not dirs:
            return JSONResponse({"detail": "没有选择目录"}, status_code=400)

        enqueued = []
        already = []
        for d in dirs:
            if enqueue_job(d, "sync"):
                enqueued.append(d)
            else:
                already.append(d)

        if not enqueued:
            return JSONResponse({"detail": "所选目录已在队列中", "already": already},
                                status_code=409)

        _notify_dispatcher()
        return JSONResponse({"enqueued": enqueued, "already": already})

    @app.post("/api/backup/resync-failed")
    async def backup_resync_failed(request: Request):
        denied = _require_ajax(request)
        if denied:
            return denied

        body = await request.json()
        dir_path = body.get("dir_path", "")
        if not dir_path:
            return JSONResponse({"detail": "缺少 dir_path"}, status_code=400)

        dir_info = get_dir(dir_path)
        if not dir_info:
            return JSONResponse({"detail": "目录不存在"}, status_code=404)

        failed = list_failed_files(dir_path, only_unresolved=True)
        if not failed:
            return JSONResponse({"detail": "没有待补传的失败文件，请重新同步"}, status_code=400)

        if not enqueue_job(dir_path, "resync"):
            return JSONResponse({"detail": "该目录已在队列中", "already": dir_path},
                                status_code=409)

        _notify_dispatcher()
        return JSONResponse({"enqueued": dir_path, "total": len(failed)})

    @app.get("/api/backup/queue-status")
    def backup_queue_status(request: Request):
        denied = _require_ajax(request)
        if denied:
            return denied
        jobs = list_queue()
        running = ""
        queued = []
        for j in jobs:
            if j.get("state") == "running":
                running = j["dir_path"]
            else:
                queued.append(j["dir_path"])
        return JSONResponse({
            "running": running,
            "queued": queued,
            "queue_len": len(queued),
            "jobs": [
                {
                    "dir_path": j["dir_path"],
                    "job_type": j.get("job_type", "sync"),
                    "state": j.get("state", "queued"),
                    "enqueued_at": j.get("enqueued_at", ""),
                }
                for j in jobs
            ],
        })

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

        target = _resolve_real_dir(dir_path)
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
        from utils.logs_config import get_path_name
        env_set = set(list_env_names())
        if logs_all.is_dir():
            for d in logs_all.iterdir():
                if d.is_dir() and not d.name.startswith("logs_") and not d.name.startswith("."):
                    env_set.add(d.name)
        # new-api 等配置的历史根：root_id 作为 env 标识（筛选键），配置名作展示 label
        newapi_envs = set()
        label_map = {}
        for root in _extra_roots():
            en = _root_env_name(root)
            env_set.add(en)
            newapi_envs.add(en)
            label_map[en] = get_path_name(root)
        envs = []
        for name in sorted(env_set):
            envs.append({
                "name": name,
                "label": label_map.get(name, name),
                "port": _port_env_map.get(name, ""),
                "newapi": name in newapi_envs,
            })
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
        logs_dir_full = str(_resolve_real_dir(dir_path))
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

    # du 结果缓存：{path: (size_bytes, computed_at)}；du 在后台线程算，避免大目录阻塞请求
    _du_cache: dict = {}
    _du_inflight: set = set()
    _du_lock = threading.Lock()
    _DU_TTL = 600  # 秒

    def _du_worker(path: str):
        try:
            sz = _du_size(Path(path))
            with _du_lock:
                _du_cache[path] = (sz, time.time())
        finally:
            with _du_lock:
                _du_inflight.discard(path)

    @app.get("/api/backup/storage")
    def backup_storage(request: Request):
        """返回各日志根所在磁盘用量（df，实时）+ 目录占用（du，后台计算带缓存）。

        with_size=true 时：命中缓存直接返回；否则后台起线程计算，本次返回 size_pending=true，
        前端轮询后续请求获取结果（不阻塞、不超时）。
        """
        denied = _require_ajax(request)
        if denied:
            return denied
        with_size = request.query_params.get("with_size", "").lower() in ("1", "true", "yes")

        # 参与备份的所有根：活跃 env 的父（logs_all）+ 各历史根
        roots = []
        active_root = str(logs_all)
        _active_path = str((logs_all / _active_env_name)) if _active_env_name else active_root
        # 活跃 env 目录尚未创建时，退回 logs_all（再退回项目根），保证 df 仍有值
        if not Path(_active_path).exists():
            _active_path = active_root if Path(active_root).exists() else _project_root
        roots.append({"name": _active_env_name or os.path.basename(active_root),
                      "path": _active_path,
                      "active": True})
        from utils.logs_config import get_path_name
        for r in _extra_roots():
            roots.append({"name": get_path_name(r), "path": r, "active": False})

        out = []
        for r in roots:
            p = Path(r["path"])
            info = {"name": r["name"], "path": r["path"], "active": r["active"],
                    "disk": _disk_usage(p)}
            if with_size and p.is_dir():
                with _du_lock:
                    cached = _du_cache.get(r["path"])
                    fresh = cached and (time.time() - cached[1] < _DU_TTL)
                    if fresh:
                        info["size_bytes"] = cached[0]
                        info["size_h"] = _human_size(cached[0])
                    else:
                        info["size_pending"] = True
                        if r["path"] not in _du_inflight:
                            _du_inflight.add(r["path"])
                            threading.Thread(target=_du_worker, args=(r["path"],),
                                             daemon=True, name="backup-du").start()
            out.append(info)

        return JSONResponse({"roots": out})

    # -----------------------------------------------------------------
    # 自动备份：透明状态 / 立即触发 / 待确认删除
    # -----------------------------------------------------------------
    def _auto_active_path() -> str:
        active_root = str(logs_all)
        p = str((logs_all / _active_env_name)) if _active_env_name else active_root
        if not Path(p).exists():
            p = active_root if Path(active_root).exists() else _project_root
        return p

    @app.get("/api/backup/auto-status")
    def backup_auto_status(request: Request):
        denied = _require_ajax(request)
        if denied:
            return denied
        snap = _auto_snapshot()
        # 补一份实时配置（阈值/开关可能刚改了 yaml）
        cfg = _auto_cfg()
        snap["enabled"] = cfg["enabled"]
        snap["threshold"] = cfg["threshold"]
        snap["obs_configured"] = bool(cfg["obs_base"])
        return JSONResponse(snap)

    @app.post("/api/backup/auto-trigger")
    def backup_auto_trigger(request: Request):
        """手动立即跑一轮自动上传（无视阈值，仍需 obs_base 已配置）。"""
        denied = _require_ajax(request)
        if denied:
            return denied
        with _auto_lock:
            if _auto_state.get("running"):
                return JSONResponse({"ok": False, "msg": "自动备份正在运行中"}, status_code=409)
        env = _active_env_name
        apath = _auto_active_path()
        t = threading.Thread(
            target=_run_auto_backup_round,
            args=(logs_all, env, apath, _resolve_real_dir),
            kwargs={"forced": True}, daemon=True, name="backup-auto-manual")
        t.start()
        return JSONResponse({"ok": True, "msg": "已开始自动上传"})

    @app.get("/api/backup/pending-delete")
    def backup_pending_delete_list(request: Request):
        denied = _require_ajax(request)
        if denied:
            return denied
        from utils.backup_store import list_pending_delete
        from utils.logs_config import get_path_name
        env = request.query_params.get("env_name", "") or None
        rows = list_pending_delete(env)
        _label_map = {_root_env_name(r): get_path_name(r) for r in _extra_roots()}
        for row in rows:
            row["env_label"] = _label_map.get(row.get("env_name"), row.get("env_name"))
        return JSONResponse({"dirs": rows})

    @app.post("/api/backup/confirm-delete")
    async def backup_confirm_delete(request: Request):
        """人工确认删除：仅对 pending_delete 且已上传(has_obs)的目录，删本地并标 backed_up。"""
        denied = _require_ajax(request)
        if denied:
            return denied
        body = await request.json()
        dirs = body.get("dirs", [])
        if not dirs:
            return JSONResponse({"detail": "没有选择目录"}, status_code=400)

        deleted, skipped = [], []
        for dir_path in dirs:
            info = get_dir(dir_path)
            # 安全校验：必须是待删状态 + 已上传 OBS，防止误删未备份数据
            if not info or info.get("status") != "pending_delete" \
                    or not info.get("has_obs") or not info.get("obs_path"):
                skipped.append(dir_path)
                append_log(dir_path, "确认删除被拒绝：非待删状态或未成功上传", level="error")
                continue
            target = _resolve_real_dir(dir_path)
            if target.is_dir():
                try:
                    shutil.rmtree(str(target))
                    append_log(dir_path, f"人工确认删除，本地目录已删除: {target}")
                except Exception as e:  # noqa: BLE001
                    append_log(dir_path, f"删除失败: {e}", level="error")
                    skipped.append(dir_path)
                    continue
            mark_backed_up(dir_path)
            deleted.append(dir_path)
        return JSONResponse({"ok": True, "deleted": deleted, "skipped": skipped})

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

    # 启动自动备份守护线程（仅 leader worker，避免多 worker 重复上传）
    global _auto_thread
    _is_leader = True
    try:
        from utils.leader_lock import is_leader
        _is_leader = is_leader()
    except Exception:
        _is_leader = True
    if _is_leader and _active_env_name and (_auto_thread is None or not _auto_thread.is_alive()):
        _auto_thread = threading.Thread(
            target=_auto_backup_loop,
            args=(logs_all, _active_env_name, _auto_active_path(), _resolve_real_dir),
            daemon=True, name="backup-auto-daemon")
        _auto_thread.start()
