from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path

from utils.obs_utils import load_obs_base, load_sync_config, OBSUTIL_BIN, DEFAULT_UPLOAD_SCRIPT

from . import db


def upload_run_to_obs(db_path: Path, run_id: str) -> dict:
    run = db.get_run(db_path, run_id)
    if not run:
        raise ValueError("Run 不存在")
    export_root = run.get("export_root", "")
    local_dir = str(Path(export_root) / run_id)
    if not Path(local_dir).is_dir():
        raise ValueError(f"导出目录不存在: {local_dir}")
    obs_base = load_obs_base()
    if not obs_base:
        raise ValueError("OBS 未配置")
    sync_cfg = load_sync_config()
    upload_script = sync_cfg.get("upload_script")
    if upload_script:
        p = Path(upload_script)
        if not p.is_absolute():
            upload_script = str((Path(__file__).resolve().parent.parent.parent / p).resolve())
    else:
        upload_script = DEFAULT_UPLOAD_SCRIPT
    dst = f"{obs_base.rstrip('/')}/reflection/{run.get('source_key', 'unknown')}/{run_id}/"

    def _bg():
        _log = lambda msg, level="info": db.append_run_log(db_path, run_id, msg, level)
        _log(f"OBS 上传开始: {local_dir} -> {dst}")
        cmd = [upload_script, local_dir, dst]
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            for line in proc.stdout:
                line = line.rstrip("\n\r")
                if line:
                    _log(line)
            proc.wait(timeout=600)
            if proc.returncode == 0:
                _log("OBS 上传完成")
                with db.connect(db_path) as conn:
                    conn.execute(
                        "UPDATE reflection_runs SET obs_root=?,updated_at=? WHERE run_id=?",
                        (dst, time.time(), run_id),
                    )
            else:
                err = f"obsutil 退出码: {proc.returncode}"
                _log(f"OBS 上传失败: {err}", "error")
                with db.connect(db_path) as conn:
                    conn.execute(
                        "UPDATE reflection_runs SET obs_root=?,updated_at=? WHERE run_id=?",
                        (f"error:{err[:200]}", time.time(), run_id),
                    )
        except Exception as e:
            msg = str(e)
            _log(f"OBS 上传异常: {msg}", "error")
            with db.connect(db_path) as conn:
                conn.execute(
                    "UPDATE reflection_runs SET obs_root=?,updated_at=? WHERE run_id=?",
                    (f"error:{msg[:200]}", time.time(), run_id),
                )

    threading.Thread(target=_bg, daemon=True, name=f"obs-upload-{run_id}").start()
    return {"status": "uploading", "dst": dst}
