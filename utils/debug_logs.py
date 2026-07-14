"""Debug 日志写入、缓存、路由。"""

import json
import logging
import os
import threading
import time
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse


def write_debug(logs_debug: str, ts: str, attempt: int, model: str, reason: str, body: str):
    """将失败尝试追加到 per-request .jsonl 文件。同一 request 的所有 retry 写同一个文件。"""
    try:
        os.makedirs(logs_debug, exist_ok=True)
        safe_model = model.replace("/", "_").replace(":", "_")
        filename = f"{ts}_{safe_model}.jsonl"
        path = os.path.join(logs_debug, filename)
        entry = {"attempt": attempt, "reason": reason, "body": body, "written_at": time.time()}
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        _append_debug_index(logs_debug, filename, ts, attempt, safe_model, reason, len(body.encode("utf-8")))
        return filename
    except Exception as ex:
        logging.warning(f"Failed to write debug log: {ex}")
        return None


def debug_filename(ts: str, model: str) -> str:
    safe_model = model.replace("/", "_").replace(":", "_")
    return f"{ts}_{safe_model}.jsonl"


def _append_debug_index(debug_dir: str, filename: str, ts: str,
                        attempt: int, model: str, reason: str, size: int):
    """每个 request 在 .log_index.jsonl 只占一条，后续 retry 更新 attempt_count。"""
    index_path = os.path.join(debug_dir, ".log_index.jsonl")
    try:
        parts = ts.rsplit("_", 1)
        ts_dt = parts[0] if len(parts) == 2 else ts
        seq = parts[1] if len(parts) == 2 else ""

        entries = []
        found = False
        if os.path.isfile(index_path):
            with open(index_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        e = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if e.get("filename") == filename:
                        e["attempt_count"] = e.get("attempt_count", 1) + 1
                        e["size"] = e.get("size", 0) + size
                        e["written_at"] = time.time()
                        e["last_reason"] = reason
                        found = True
                    entries.append(e)

        if not found:
            entries.append({
                "filename": filename, "ts": ts, "ts_dt": ts_dt, "seq": seq,
                "attempt": attempt, "attempt_count": 1,
                "model": model, "reason": reason,
                "size": size, "written_at": time.time(),
            })

        with open(index_path, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 缓存
# ---------------------------------------------------------------------------

_DEBUG_CACHE_LOCK = threading.Lock()
_debug_mem_cache: dict = {}
_DEBUG_MEM_TTL = 10


def _collect_debug_roots() -> list[Path]:
    logs_root = Path("logs")
    roots = []
    for port_dir in sorted(logs_root.glob("port*")):
        if not port_dir.is_dir():
            continue
        for env_dir in port_dir.iterdir():
            if not env_dir.is_dir():
                continue
            debug_dir = env_dir / "debug"
            if debug_dir.is_dir():
                roots.append(debug_dir)
    legacy = Path("logs", "debug")
    if legacy.is_dir():
        roots.append(legacy)
    return roots


def _rebuild_debug_cache(debug_root: Path) -> tuple:
    """v4: 每个 hour 独立存 items。返回 (hour_dirs_map, models)。"""
    cache_path = debug_root / ".log_cache.json"
    cache: dict = {"_version": 4, "updated_at": 0, "hour_dirs": {}, "models": []}
    if cache_path.is_file():
        try:
            raw = json.loads(cache_path.read_text(encoding="utf-8"))
            if raw.get("_version", 0) >= 4:
                cache = raw
            else:
                cache = {"_version": 4, "updated_at": 0, "hour_dirs": {}, "models": []}
        except (json.JSONDecodeError, OSError):
            pass

    hour_meta = cache.get("hour_dirs", {})
    hour_dirs = sorted(
        [d for d in debug_root.iterdir() if d.is_dir() and not d.name.startswith(".")],
        key=lambda d: d.name,
    )

    try:
        env_label = str(debug_root.relative_to(Path("logs")))
    except ValueError:
        env_label = str(debug_root)

    changed = False
    seen_hour_keys = set()
    for hdir in hour_dirs:
        hkey = hdir.name
        seen_hour_keys.add(hkey)
        idx_path = hdir / ".log_index.jsonl"

        if not idx_path.is_file():
            continue

        st = idx_path.stat()
        prev = hour_meta.get(hkey, {})
        if (prev.get("index_size") == st.st_size
                and prev.get("index_mtime") == st.st_mtime
                and prev.get("items")):
            continue

        entries = []
        try:
            with open(idx_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except OSError:
            continue

        entries.sort(key=lambda e: e.get("written_at", 0), reverse=True)

        items = []
        for e in entries:
            ts_dt = e.get("ts_dt", "")
            created_at = ""
            if ts_dt:
                try:
                    created_at = datetime.strptime(ts_dt, "%Y-%m-%d_%H-%M-%S").strftime("%Y-%m-%d %H:%M:%S")
                except ValueError:
                    created_at = ts_dt

            rel_name = f"{env_label}/{hkey}/{e['filename']}" if env_label else f"{hkey}/{e['filename']}"
            items.append({
                "filename": rel_name,
                "created_at": created_at,
                "attempt": e.get("attempt", 0),
                "attempt_count": e.get("attempt_count", 1),
                "model": e.get("model", ""),
                "reason": e.get("last_reason", "") or e.get("reason", ""),
                "size": e.get("size", 0),
                "env": f"{env_label}/{hkey}" if env_label else hkey,
            })

        hour_meta[hkey] = {
            "index_size": st.st_size,
            "index_mtime": st.st_mtime,
            "item_count": len(items),
            "items": items,
        }
        changed = True

    for old_key in list(hour_meta.keys()):
        if old_key not in seen_hour_keys:
            del hour_meta[old_key]
            changed = True

    all_models = set()
    for h in hour_meta.values():
        for item in h.get("items", []):
            m = item.get("model", "")
            if m:
                all_models.add(m)
    models = sorted(all_models)

    if changed:
        save_cache = {
            "_version": 4,
            "updated_at": time.time(),
            "hour_dirs": hour_meta,
            "models": models,
        }
        try:
            tmp = str(cache_path) + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(save_cache, f, ensure_ascii=False)
            os.replace(tmp, cache_path)
        except OSError:
            pass

    return hour_meta, models


def _load_debug_cache(env_filter: str = "", keyword: str = "", limit: int = 50, offset: int = 0) -> tuple:
    """返回 (paged_items, models, total)。"""
    global _debug_mem_cache

    now = time.time()
    cache_key = env_filter or "__all__"

    with _DEBUG_CACHE_LOCK:
        mem = _debug_mem_cache.get(cache_key)
        if mem and (now - mem["ts"]) < _DEBUG_MEM_TTL:
            items = mem["items"]
            models = mem["models"]
            if keyword:
                items = [i for i in items if keyword in i.get("filename", "").lower()
                         or keyword in i.get("model", "").lower()
                         or keyword in i.get("reason", "").lower()]
            return items[offset:offset + limit], models, len(items)

    all_items: list[dict] = []
    all_models: set = set()
    roots = _collect_debug_roots()

    if env_filter:
        target = Path("logs") / env_filter
        matched = False
        for root in roots:
            if target == root or target.parent == root:
                hour_map, models = _rebuild_debug_cache(root)
                all_models.update(models)
                if target != root:
                    hkey = target.name
                    all_items = hour_map.get(hkey, {}).get("items", [])
                else:
                    for h in hour_map.values():
                        all_items.extend(h.get("items", []))
                    all_items.sort(key=lambda i: i.get("created_at", ""), reverse=True)
                matched = True
                break
        if not matched:
            return [], [], 0
    else:
        for root in roots:
            hour_map, models = _rebuild_debug_cache(root)
            for h in hour_map.values():
                all_items.extend(h.get("items", []))
            all_models.update(models)
        all_items.sort(key=lambda i: i.get("created_at", ""), reverse=True)

    merged_models = sorted(all_models)

    with _DEBUG_CACHE_LOCK:
        _debug_mem_cache[cache_key] = {"ts": time.time(), "items": all_items, "models": merged_models}

    if keyword:
        all_items = [i for i in all_items if keyword in i.get("filename", "").lower()
                     or keyword in i.get("model", "").lower()
                     or keyword in i.get("reason", "").lower()]
    return all_items[offset:offset + limit], merged_models, len(all_items)


# ---------------------------------------------------------------------------
# 路由注册
# ---------------------------------------------------------------------------

def register_debug_routes(app: FastAPI, logs_debug: str, startup_date_tag: str):

    @app.get("/logs/debug/envs")
    def logs_debug_envs():
        """返回当前 port+env 的 debug 目录下所有 hour 目录列表。"""
        if not logs_debug:
            return JSONResponse([])
        debug_dir = Path(logs_debug).parent
        if not debug_dir.is_dir():
            return JSONResponse([])

        current_hour = startup_date_tag
        cache_meta = {}
        cache_path = debug_dir / ".log_cache.json"
        if cache_path.is_file():
            try:
                raw = json.loads(cache_path.read_text(encoding="utf-8"))
                cache_meta = raw.get("hour_dirs", {})
            except (json.JSONDecodeError, OSError):
                pass

        try:
            debug_rel = str(debug_dir.relative_to(Path("logs")))
        except ValueError:
            debug_rel = str(debug_dir)

        envs = []
        for hour_dir in sorted(debug_dir.iterdir(), reverse=True):
            if not hour_dir.is_dir() or hour_dir.name.startswith("."):
                continue
            hkey = hour_dir.name
            meta = cache_meta.get(hkey, {})
            count = meta.get("item_count", 0)
            if count == 0:
                idx = hour_dir / ".log_index.jsonl"
                if idx.is_file():
                    try:
                        count = sum(1 for line in idx.open() if line.strip())
                    except OSError:
                        pass
            try:
                mt = hour_dir.stat().st_mtime
            except OSError:
                mt = 0
            envs.append({
                "name": f"{debug_rel}/{hkey}",
                "label": hkey,
                "count": count,
                "mtime": datetime.fromtimestamp(mt).strftime("%m-%d %H:%M") if mt else "",
                "current": hkey == current_hour,
            })
        return JSONResponse(envs)

    @app.get("/logs/debug/list")
    def logs_debug_list(limit: int = 50, offset: int = 0, keyword: str = "", env: str = "", model: str = ""):
        safe_limit = max(1, min(limit, 1000))
        safe_offset = max(0, offset)
        keyword_lower = keyword.strip().lower()
        model_lower = model.strip().lower()
        items, models, total = _load_debug_cache(env_filter=env, keyword=keyword_lower, limit=safe_limit, offset=safe_offset)
        if model_lower:
            items = [i for i in items if model_lower in i.get("model", "").lower()]
        return JSONResponse({"items": items, "models": models, "total": total})

    @app.get("/logs/debug/file")
    def logs_debug_file(filename: str):
        if ".." in filename or not (filename.endswith(".txt") or filename.endswith(".jsonl")):
            return JSONResponse({"error": "invalid filename"}, status_code=400)
        path = Path("logs") / filename
        if not path.is_file():
            cur_path = Path(logs_debug) / Path(filename).name
            if cur_path.is_file():
                path = cur_path
            else:
                return JSONResponse({"error": "file not found"}, status_code=404)
        try:
            raw = path.read_text(encoding="utf-8")
        except Exception as ex:
            return JSONResponse({"error": f"read failed: {ex}"}, status_code=500)

        if filename.endswith(".jsonl"):
            parts = []
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    attempt = entry.get("attempt", "?")
                    reason = entry.get("reason", "")
                    parts.append(f"--- attempt {attempt} ({reason}) ---\n{entry.get('body', '')}")
                except json.JSONDecodeError:
                    parts.append(line)
            content = "\n\n".join(parts)
        else:
            content = raw

        return JSONResponse({
            "filename": filename,
            "content": content,
            "size": path.stat().st_size,
            "updated_at": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        })
