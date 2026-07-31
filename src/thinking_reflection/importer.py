from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

from . import db
from .extractor import extract_signatures


def _load_session_index(root: Path) -> list[str]:
    """从 session_analysis.json 读取 session 列表（folder name）。"""
    analysis_path = root / "session_analysis.json"
    if not analysis_path.is_file():
        raise ValueError(f"缺少 session_analysis.json: {root}")
    payload = json.loads(analysis_path.read_text(encoding="utf-8"))
    sessions = []
    items = payload.get("sessions") if isinstance(payload, dict) else payload
    # v1: sessions 为 list[dict]；v2: sessions 为 dict{"0":{...},...}（值仍是 dict）
    if isinstance(items, dict):
        items = list(items.values())
    if not isinstance(items, list):
        raise ValueError("session_analysis.json 格式错误")
    for entry in items:
        sid = entry.get("session", "") if isinstance(entry, dict) else ""
        if sid:
            sessions.append(sid)
    return sessions


def resolve_dataset_root(root: Path) -> Path:
    """穿透 obsutil 下载造成的单层同名嵌套，返回真正含数据的根目录。

    obsutil cp <obs>/xxx/ <local> -r 会把源目录名 xxx 本身也建到 local 下，
    形成 local/xxx/session_analysis.json 的双层结构。若 root 下没有
    session_analysis.json、且恰好只有一个非隐藏子目录含 session_analysis.json，
    则下钻到该子目录。否则原样返回 root。

    统计 / 导入 / 查看轨迹共用此逻辑，保证各入口对同一数据集解析出的
    root 一致（否则会出现统计只认出 1 个 session、查看轨迹找不到文件等错乱）。
    """
    if not root.is_dir():
        return root
    if (root / "session_analysis.json").is_file():
        return root
    subdirs = [d for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")]
    if len(subdirs) == 1 and (subdirs[0] / "session_analysis.json").is_file():
        return subdirs[0]
    return root


def _collect_files(root: Path) -> list[tuple[str, Path]]:
    """收集轨迹文件。优先 session_analysis.json 索引，否则 rglob 全扫。

    返回 [(session_id, file_path), ...]
    """
    root = resolve_dataset_root(root)
    excluded = {"session_analysis.json", "session_index.json", "failure_report.json", "manifest.json"}
    analysis_path = root / "session_analysis.json"
    if analysis_path.is_file():
        session_ids = _load_session_index(root)
        result = []
        for sid in session_ids:
            session_dir = root / sid
            if not session_dir.is_dir():
                continue
            for p in sorted(session_dir.glob("*.json")):
                if p.name not in excluded and not p.name.endswith("--thinking.json"):
                    result.append((sid, p))
        return result

    result = []
    for p in sorted(root.rglob("*.json")):
        if p.name in excluded or p.name.endswith("--thinking.json"):
            continue
        try:
            rel = p.relative_to(root)
        except ValueError:
            continue
        if len(rel.parts) < 2:
            continue
        sid = rel.parts[0]
        result.append((sid, p))
    return result


def validate_quality_dir(root: Path) -> None:
    if not root.is_dir():
        raise ValueError(f"质检目录不存在: {root}")
    if not _collect_files(root):
        raise ValueError("质检目录中没有 Session trajectory JSON")


def _preview_count_one(args: tuple[str, str]) -> tuple[str, int]:
    """子进程 worker：解析单个 trajectory，返回 (session_id, signature 数)。

    返回 -1 表示读取/解析失败或非 dict（供主进程计入 files_bad）。
    需为模块级函数才能被 ProcessPoolExecutor pickle。
    """
    sid, path = args
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return (sid, -1)
    if not isinstance(raw, dict):
        return (sid, -1)
    return (sid, sum(1 for _ in extract_signatures(raw)))


def preview_run(root: Path) -> dict[str, int]:
    """只读预统计：扫描 quality 目录，统计可解析出的 signature 任务数。

    不写 dataset_tasks、不落 detail 文件、不建 run。仅用于「统计」按钮，
    让用户在真正导入前看到规模。返回文件数 / signature 总数 / 涉及 session 数。

    大 key（数万文件）单进程逐个 json.loads 会很慢（~47s/46k 文件），
    这里用多进程并行解析，实测降到 ~2s。
    """
    files = _collect_files(root)
    if not files:
        raise ValueError("目录中没有可导入的轨迹文件")

    sessions: set[str] = set()
    signatures = 0
    files_ok = 0
    files_bad = 0
    workers = min(16, os.cpu_count() or 4)
    payload = [(sid, str(src)) for sid, src in files]
    # 小规模不值得起进程池，直接串行
    if len(payload) < 200 or workers <= 1:
        results = (_preview_count_one(a) for a in payload)
    else:
        from concurrent.futures import ProcessPoolExecutor

        ex = ProcessPoolExecutor(max_workers=workers)
        try:
            results = list(ex.map(_preview_count_one, payload, chunksize=200))
        finally:
            ex.shutdown()
    for _sid, cnt in results:
        if cnt < 0:
            files_bad += 1
            continue
        files_ok += 1
        if cnt:
            signatures += cnt
            sessions.add(_sid)
    return {
        "files": len(files),
        "files_ok": files_ok,
        "files_bad": files_bad,
        "signatures": signatures,
        "sessions": len(sessions),
    }


def import_run(db_path: Path, export_id: int, root: Path, max_retries: int, detail_dir: Path) -> dict[str, int]:
    # 与 _collect_files 一致地穿透嵌套，保证 relative / source_root 与 session_trajectory 查询口径相同
    root = resolve_dataset_root(root)
    files = _collect_files(root)
    if not files:
        raise ValueError("目录中没有可导入的轨迹文件")

    detail_dir.mkdir(parents=True, exist_ok=True)
    traj_inserted = traj_present = 0
    task_inserted = task_present = 0
    now = time.time()
    with db.connect(db_path) as conn:
        for sid, source in files:
            try:
                raw = json.loads(source.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"无法读取 trajectory: {source}") from exc
            if not isinstance(raw, dict):
                continue
            relative = source.relative_to(root).as_posix()
            trajectory_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{export_id}:{relative}"))
            cursor = conn.execute(
                "INSERT OR IGNORE INTO dataset_trajectories"
                "(export_id,trajectory_path,trajectory_id,session_id,source_root,created_at,updated_at) "
                "VALUES(?,?,?,?,?,?,?)",
                (export_id, relative, trajectory_id, sid, root.as_posix(), now, now),
            )
            if cursor.rowcount == 1:
                traj_inserted += 1
            else:
                traj_present += 1
            for item in extract_signatures(raw):
                task_uuid = str(uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"{export_id}:{sid}:{relative}:{item['block_path']}"))
                detail_file = detail_dir / f"{sid}_{task_uuid}.json"
                if not detail_file.exists():
                    detail_file.write_text(json.dumps({
                        "original_thinking": item["original_thinking"],
                        "signature": item["signature"],
                    }, ensure_ascii=False, indent=2), encoding="utf-8")
                cursor = conn.execute(
                    "INSERT OR IGNORE INTO dataset_tasks("
                    "uuid,export_id,session_id,trajectory_id,trajectory_path,block_path,"
                    "message_index,original_thinking,signature,signature_len,detail_path,"
                    "latest_status,retry_count,max_retries,created_at,updated_at)"
                    " VALUES(?,?,?,?,?,?,?,NULL,'',?,?, 'pending',0,?,?,?)",
                    (task_uuid, export_id, sid, trajectory_id, relative, item["block_path"],
                     item["message_index"], len(item["signature"]),
                     detail_file.as_posix(), max_retries, now, now))
                if cursor.rowcount == 1:
                    task_inserted += 1
                else:
                    task_present += 1
    return {
        "trajectories": traj_inserted + traj_present,
        "trajectories_inserted": traj_inserted,
        "trajectories_present": traj_present,
        "tasks": task_inserted + task_present,
        "tasks_inserted": task_inserted,
        "tasks_present": task_present,
    }
