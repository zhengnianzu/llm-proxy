"""
utils/obs_utils.py — OBS 路径管理与工具函数

集中管理：
- obsutil 二进制路径
- obs_base 配置读取（sync_config → settings/obs_base.yaml fallback）
- 上传命令（_run_upload_cmd）
- 目录/文件列表（obsutil ls）
"""

import os
import subprocess
from pathlib import Path
from subprocess import PIPE, STDOUT, Popen
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OBSUTIL_BIN = str(PROJECT_ROOT / "tools" / "obsutil" / "obsutil")
DEFAULT_UPLOAD_SCRIPT = str(PROJECT_ROOT / "tools" / "obs_upload.sh")
DEFAULT_DOWNLOAD_SCRIPT = str(PROJECT_ROOT / "tools" / "obs_download.sh")


# ---------------------------------------------------------------------------
# obs_base 配置读取
# ---------------------------------------------------------------------------

def load_obs_base() -> str:
    """读取 obs_base。优先 .cli_state.yaml -> sync_config，fallback settings/obs_base.yaml。"""
    import yaml

    candidates = []

    cli_state_path = PROJECT_ROOT / ".cli_state.yaml"
    if cli_state_path.is_file():
        try:
            with open(cli_state_path, "r", encoding="utf-8") as f:
                state = yaml.safe_load(f) or {}
            sync_cfg = state.get("sync_config", "")
            if sync_cfg:
                p = Path(sync_cfg)
                if not p.is_absolute():
                    p = PROJECT_ROOT / p
                candidates.append(p)
        except Exception:
            pass

    candidates.append(PROJECT_ROOT / "settings" / "obs_base.yaml")

    for cfg_path in candidates:
        if not cfg_path.is_file():
            continue
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            val = cfg.get("obs_base", "").strip().rstrip("/")
            if val:
                return val
        except Exception:
            continue
    return ""


def load_sync_config() -> dict:
    """读取 sync_config 配置文件（obs_base, interval, workers, upload_script）。"""
    import yaml

    cli_state_path = PROJECT_ROOT / ".cli_state.yaml"
    if not cli_state_path.is_file():
        return {}
    try:
        with open(cli_state_path, "r", encoding="utf-8") as f:
            state = yaml.safe_load(f) or {}
        sync_cfg = state.get("sync_config", "")
        if not sync_cfg:
            return {}
        p = Path(sync_cfg)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if not p.is_file():
            return {}
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def get_sync_config_path() -> Optional[Path]:
    """返回 sync_config YAML 文件路径（如果配置了）。"""
    import yaml

    cli_state_path = PROJECT_ROOT / ".cli_state.yaml"
    if not cli_state_path.is_file():
        return None
    try:
        with open(cli_state_path, "r", encoding="utf-8") as f:
            state = yaml.safe_load(f) or {}
        sync_cfg = state.get("sync_config", "")
        if not sync_cfg:
            return None
        p = Path(sync_cfg)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return p if p.is_file() else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 上传
# ---------------------------------------------------------------------------

def run_upload_cmd(
    local: str,
    dst: str,
    upload_script: Optional[str] = None,
    timeout: Optional[int] = None,
    jobs: Optional[int] = None,
    log_cb=None,
) -> Tuple[bool, str]:
    """流式执行上传脚本（Popen 模式），实时打印进度。timeout/jobs 默认从 sync_config 读取。
    log_cb: 可选回调 fn(str)，每 10 秒汇报一次上传进度（避免 progress_log 过于庞大）。
    """
    import time as _time
    cfg = load_sync_config()
    if timeout is None:
        timeout = int(cfg.get("upload_timeout", 3600))
    if jobs is None:
        jobs = int(cfg.get("upload_jobs", 8))

    if upload_script is None:
        upload_script = DEFAULT_UPLOAD_SCRIPT
    else:
        p = Path(upload_script)
        if not p.is_absolute():
            upload_script = str((PROJECT_ROOT / p).resolve())

    cmd = [upload_script, local, dst, str(jobs)]
    try:
        proc = Popen(cmd, stdout=PIPE, stderr=STDOUT, text=True)
        lines: List[str] = []
        line_count = 0
        last_log_ts = _time.monotonic()
        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            lines.append(line)
            line_count += 1
            print(line, flush=True)
            if log_cb:
                now = _time.monotonic()
                if now - last_log_ts >= 10:
                    try:
                        log_cb(f"上传中... 已处理 {line_count} 行，最新: {line[-120:]}")
                    except Exception:
                        pass
                    last_log_ts = now
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            return False, f"Command timed out after {timeout}s"
        if proc.returncode == 0:
            return True, lines[-1] if lines else ""
        return False, "\n".join(lines[-5:]) if lines else "upload failed"
    except Exception as e:
        return False, str(e)


# ---------------------------------------------------------------------------
# 下载
# ---------------------------------------------------------------------------

def run_download_cmd(
    obs_path: str,
    local: str,
    download_script: Optional[str] = None,
    timeout: int = 300,
) -> Tuple[bool, str]:
    """执行下载脚本，将 OBS 文件/目录下载到本地。"""
    if download_script is None:
        download_script = DEFAULT_DOWNLOAD_SCRIPT
    else:
        p = Path(download_script)
        if not p.is_absolute():
            download_script = str((PROJECT_ROOT / p).resolve())

    cmd = [download_script, obs_path, local]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode == 0:
            return True, result.stdout.strip()
        return False, (result.stderr or result.stdout).strip()
    except Exception as e:
        return False, str(e)


# ---------------------------------------------------------------------------
# 目录/文件列表
# ---------------------------------------------------------------------------

def obsutil_ls(path: str, show_dirs: bool = False, limit: int = 500) -> List[dict]:
    """执行 obsutil ls，返回解析后的文件/目录列表。"""
    if not os.path.isfile(OBSUTIL_BIN):
        return []
    if not path.endswith("/"):
        path += "/"
    args = [OBSUTIL_BIN, "ls", path, f"-limit={limit}"]
    if show_dirs:
        args.append("-d")
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return []
    except Exception:
        return []

    items = []
    for line in result.stdout.strip().splitlines():
        line = line.strip()
        if not line.startswith("obs://"):
            continue
        parts = line.split()
        full_path = parts[0] if parts else line
        name = full_path.replace(path, "").strip("/")
        if not name:
            continue
        is_dir = full_path.endswith("/")
        if is_dir:
            name = name.rstrip("/")
        size = None
        if len(parts) > 1:
            for p in parts[1:]:
                if any(p.endswith(u) for u in ("B", "KB", "MB", "GB")):
                    size = p
                    break
        items.append({"name": name, "path": full_path, "is_dir": is_dir, "size": size})
    return items
