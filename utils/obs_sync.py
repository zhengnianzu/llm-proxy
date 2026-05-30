"""
utils/obs_sync.py — 独立 raw 日志同步模块

将 logs_dir 中的三元组文件增量上传到 OBS，基于 index.jsonl 行号偏移。
供 ./app sync 调用，也可独立运行: python -m utils.obs_sync --help
"""

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_shutdown_requested = False


def _handle_signal(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True


def _interruptible_sleep(seconds: int):
    for _ in range(seconds):
        if _shutdown_requested:
            break
        time.sleep(1)


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

def _run_upload_cmd(
    local: str,
    dst: str,
    upload_script: Optional[str] = None,
    timeout: int = 120,
) -> Tuple[bool, str]:
    project_root = Path(__file__).resolve().parent.parent
    if upload_script is None:
        upload_script = str(project_root / "tools" / "obs_upload.sh")
    else:
        upload_script = str((project_root / upload_script).resolve())

    cmd = [upload_script, local, dst]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode == 0:
            return True, result.stdout.strip()
        return False, (result.stderr or result.stdout).strip()
    except Exception as e:
        return False, str(e)


def _upload_file(
    local_path: Path,
    obs_dst: str,
    upload_script: Optional[str],
) -> Tuple[str, bool, str]:
    dst = obs_dst.rstrip("/") + "/" + local_path.name
    ok, msg = _run_upload_cmd(str(local_path), dst, upload_script)
    return local_path.name, ok, msg


# ---------------------------------------------------------------------------
# Index reading
# ---------------------------------------------------------------------------

def _read_new_entries(
    logs_dir: str,
    line_offset: int,
) -> Tuple[List[dict], int]:
    index_path = Path(logs_dir) / "index.jsonl"
    if not index_path.exists():
        return [], line_offset

    entries: List[dict] = []
    total = 0
    with open(index_path, "r", encoding="utf-8") as f:
        for i, raw in enumerate(f):
            total += 1
            if i < line_offset:
                continue
            line = raw.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not entry.get("ts"):
                continue
            entries.append(entry)
    return entries, total


def _collect_files(logs_dir: str, entries: List[dict]) -> Dict[str, List[Path]]:
    base = Path(logs_dir)
    result: Dict[str, List[Path]] = {}
    for entry in entries:
        ts = entry["ts"]
        files: List[Path] = []
        for suffix in ("-req.json", "-headers.json", "-res.json"):
            p = base / f"{ts}{suffix}"
            if p.is_file():
                files.append(p)
        if files:
            result[ts] = files
    return result


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

def _state_path(logs_dir: str) -> Path:
    return Path(logs_dir) / ".sync_state.json"


def _load_state(logs_dir: str) -> dict:
    sp = _state_path(logs_dir)
    if sp.exists():
        try:
            with open(sp, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"line_offset": 0}


def _save_state(logs_dir: str, state: dict):
    sp = _state_path(logs_dir)
    state["last_sync_at"] = datetime.now().isoformat(timespec="seconds")
    with open(sp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Sync cycle
# ---------------------------------------------------------------------------

def _run_sync_cycle(
    logs_dir: str,
    obs_dst: str,
    workers: int,
    upload_script: Optional[str],
    logger: logging.Logger,
) -> None:
    state = _load_state(logs_dir)
    offset = state.get("line_offset", 0)

    entries, total_lines = _read_new_entries(logs_dir, offset)
    if not entries:
        logger.debug("no new entries (offset=%d, total=%d)", offset, total_lines)
        state["line_offset"] = total_lines if total_lines else offset
        _save_state(logs_dir, state)
        return

    logger.info("found %d new entries (offset %d -> %d)", len(entries), offset, total_lines)

    files_by_ts = _collect_files(logs_dir, entries)
    all_files: List[Path] = []
    for ts_files in files_by_ts.values():
        all_files.extend(ts_files)

    if not all_files:
        logger.info("no files to upload")
        state["line_offset"] = total_lines
        _save_state(logs_dir, state)
        return

    logger.info("uploading %d files to %s", len(all_files), obs_dst)
    ok_count = 0
    fail_count = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_upload_file, fp, obs_dst, upload_script): fp
            for fp in all_files
        }
        for future in as_completed(futures):
            if _shutdown_requested:
                break
            name, ok, msg = future.result()
            if ok:
                ok_count += 1
            else:
                fail_count += 1
                logger.error("upload FAIL %s: %s", name, msg)

    logger.info("upload summary: %d ok, %d failed", ok_count, fail_count)

    # 即使有失败也推进 offset，避免下次重复处理已成功的条目
    # 失败的文件在下次 app.py 重启时会重新写入 index.jsonl
    state["line_offset"] = total_lines
    _save_state(logs_dir, state)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def sync_loop(
    logs_dir: str,
    obs_dst: str,
    interval: int = 600,
    workers: int = 4,
    upload_script: Optional[str] = None,
    once: bool = False,
    logger: Optional[logging.Logger] = None,
):
    if logger is None:
        logger = _setup_logging(logs_dir)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    logger.info(
        "obs_sync starting. logs_dir=%s obs_dst=%s interval=%ds workers=%d",
        logs_dir, obs_dst, interval, workers,
    )

    while not _shutdown_requested:
        try:
            _run_sync_cycle(logs_dir, obs_dst, workers, upload_script, logger)
        except Exception:
            logger.exception("sync cycle failed")

        if once:
            break
        _interruptible_sleep(interval)

    logger.info("obs_sync exiting")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logging(logs_dir: str) -> logging.Logger:
    logger = logging.getLogger("obs_sync")
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    return logger


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Incremental raw log sync to OBS"
    )
    parser.add_argument("--logs-dir", required=True, help="Log directory to sync")
    parser.add_argument("--obs-dst", required=True, help="OBS destination (obs://bucket/path/)")
    parser.add_argument("--interval", type=int, default=600, help="Sync interval in seconds (default 600)")
    parser.add_argument("--workers", type=int, default=4, help="Upload concurrency (default 4)")
    parser.add_argument("--upload-script", default=None, help="Custom upload script")
    parser.add_argument("--once", action="store_true", help="Run single cycle then exit")
    return parser.parse_args()


def main():
    args = parse_args()
    sync_loop(
        logs_dir=args.logs_dir,
        obs_dst=args.obs_dst,
        interval=args.interval,
        workers=args.workers,
        upload_script=args.upload_script,
        once=args.once,
    )


if __name__ == "__main__":
    main()
