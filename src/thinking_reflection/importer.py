from __future__ import annotations

import json
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
    if not isinstance(items, list):
        raise ValueError("session_analysis.json 格式错误")
    for entry in items:
        sid = entry.get("session", "") if isinstance(entry, dict) else ""
        if sid:
            sessions.append(sid)
    return sessions


def _collect_files(root: Path) -> list[tuple[str, Path]]:
    """收集轨迹文件。优先 session_analysis.json 索引，否则 rglob 全扫。

    返回 [(session_id, file_path), ...]
    """
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


def import_run(db_path: Path, run_id: str, root: Path, max_retries: int, detail_dir: Path) -> dict[str, int]:
    files = _collect_files(root)
    if not files:
        raise ValueError("目录中没有可导入的轨迹文件")

    detail_dir.mkdir(parents=True, exist_ok=True)
    trajectories = tasks = 0
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
            trajectory_id = str(uuid.uuid5(uuid.NAMESPACE_URL, relative))
            conn.execute("""INSERT OR IGNORE INTO run_trajectories
              (run_id,trajectory_id,session_id,trajectory_path,raw_json,source_root) VALUES(?,?,?,?,?,?)""",
              (run_id, trajectory_id, sid, relative, '', root.as_posix()))
            trajectories += 1
            for item in extract_signatures(raw):
                task_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{run_id}:{relative}:{item['block_path']}"))
                detail_file = detail_dir / f"{sid}_{task_uuid}.json"
                detail_file.write_text(json.dumps({
                    "original_thinking": item["original_thinking"],
                    "signature": item["signature"],
                }, ensure_ascii=False, indent=2), encoding="utf-8")
                cursor = conn.execute("""INSERT OR IGNORE INTO thinking_tasks(
                  uuid,run_id,trajectory_id,session_id,trajectory_path,block_path,message_index,
                  original_thinking,signature,signature_len,status,max_retries,detail_path,created_at,updated_at)
                  VALUES(?,?,?,?,?,?,?,NULL,'',?,'pending',?,?,?,?)""",
                  (task_uuid, run_id, trajectory_id, sid, relative, item["block_path"],
                   item["message_index"], len(item["signature"]),
                   max_retries, detail_file.as_posix(), now, now))
                tasks += cursor.rowcount
    return {"trajectories": trajectories, "tasks": tasks}
