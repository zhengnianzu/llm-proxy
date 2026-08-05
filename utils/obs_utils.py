"""
utils/obs_utils.py — OBS 路径管理与工具函数

集中管理：
- obsutil 二进制路径
- obs_base 配置读取（唯一来源：sync_config 指向的 yaml，如 settings/obs_rl.yaml）
- 上传命令（_run_upload_cmd）
- 目录/文件列表（obsutil ls）
"""

import glob
import json
import os
import re
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from subprocess import PIPE, STDOUT, Popen
from typing import Dict, Iterator, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OBSUTIL_BIN = str(PROJECT_ROOT / "tools" / "obsutil" / "obsutil")
DEFAULT_UPLOAD_SCRIPT = str(PROJECT_ROOT / "tools" / "obs_upload.sh")
DEFAULT_DOWNLOAD_SCRIPT = str(PROJECT_ROOT / "tools" / "obs_download.sh")
DEFAULT_OBSUTIL_CONFIG_DIR = "/mnt/tanpeng/conf"
DEFAULT_OBSUTIL_RUNTIME_CONFIG_DIR = PROJECT_ROOT / "runtime" / "obsutil-config"

_OBS_BUCKET_RE = re.compile(
    r"^obs://(?P<bucket>[A-Za-z0-9](?:[A-Za-z0-9.-]{0,61}[A-Za-z0-9])?)(?:/|$)"
)

# sync_config 未初始化时的默认配置文件（.cli_state.yaml 里没有 sync_config 时回退）
DEFAULT_SYNC_CONFIG = PROJECT_ROOT / "settings" / "obs_base.yaml"


# ---------------------------------------------------------------------------
# obs_base 配置读取
# ---------------------------------------------------------------------------

def get_sync_config_path() -> Optional[Path]:
    """返回 sync_config YAML 文件路径。

    优先用 .cli_state.yaml 的 sync_config 指向的文件；未配置或不存在时，
    回退到默认的 settings/obs_base.yaml（若存在）。
    """
    import yaml

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
                if p.is_file():
                    return p
        except Exception:
            pass
    # 回退默认配置
    return DEFAULT_SYNC_CONFIG if DEFAULT_SYNC_CONFIG.is_file() else None


def load_sync_config() -> dict:
    """读取 sync_config 配置文件（obs_base, interval, workers, upload_script）。

    未初始化时回退到默认 settings/obs_base.yaml。
    """
    import yaml

    p = get_sync_config_path()
    if p is None:
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def load_obs_base() -> str:
    """读取 obs_base。来源：sync_config 指向的配置文件（如 settings/obs_rl.yaml）；
    未初始化时回退默认 settings/obs_base.yaml。"""
    return load_sync_config().get("obs_base", "").strip().rstrip("/")


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
            if log_cb:
                try:
                    log_cb(f"上传超时（{timeout}s）")
                except Exception:
                    pass
            return False, f"Command timed out after {timeout}s"
        if proc.returncode == 0:
            last = lines[-1] if lines else ""
            if log_cb and last:
                try:
                    log_cb(f"obsutil: {last}")
                except Exception:
                    pass
            return True, last
        tail = "\n".join(lines[-5:]) if lines else "upload failed"
        if log_cb and tail:
            try:
                log_cb(f"obsutil 错误: {tail[-200:]}")
            except Exception:
                pass
        return False, tail
    except Exception as e:
        return False, str(e)


# ---------------------------------------------------------------------------
# 下载
# ---------------------------------------------------------------------------

def get_obs_bucket(obs_path: str) -> str:
    """从 ``obs://bucket/key`` 中提取并校验桶名。"""
    match = _OBS_BUCKET_RE.match((obs_path or "").strip())
    if not match:
        raise ValueError("无效的 OBS 路径")
    bucket = match.group("bucket")
    if ".." in bucket:
        raise ValueError("无效的 OBS 桶名")
    return bucket


def _obsutil_config_mapping() -> Dict[str, str]:
    """读取可信配置中的 bucket -> obsutil config 映射。

    支持 sync_config 的 ``obsutil_configs``，以及环境变量
    ``OBSUTIL_CONFIG_MAP``（JSON 对象）。环境变量优先，便于测试部署覆盖。
    """
    mapping: Dict[str, str] = {}
    cfg_mapping = load_sync_config().get("obsutil_configs", {})
    if isinstance(cfg_mapping, dict):
        mapping.update({str(k): str(v) for k, v in cfg_mapping.items() if k and v})
    raw = os.getenv("OBSUTIL_CONFIG_MAP", "").strip()
    if raw:
        try:
            env_mapping = json.loads(raw)
            if isinstance(env_mapping, dict):
                mapping.update(
                    {str(k): str(v) for k, v in env_mapping.items() if k and v}
                )
        except (json.JSONDecodeError, TypeError):
            pass
    return mapping


def resolve_obsutil_config(obs_path: str, require_exists: bool = True) -> str:
    """按 OBS 桶解析 obsutil 配置文件。

    默认约定为 ``/mnt/tanpeng/conf/<bucket>``。可通过 sync_config 的
    ``obsutil_config_dir`` / ``obsutil_configs``，或环境变量
    ``OBSUTIL_CONFIG_DIR`` / ``OBSUTIL_CONFIG_MAP`` 覆盖。客户端不能传入
    配置路径，避免借下载接口读取任意凭据。
    """
    bucket = get_obs_bucket(obs_path)
    cfg = load_sync_config()
    config_dir = Path(
        os.getenv(
            "OBSUTIL_CONFIG_DIR",
            str(cfg.get("obsutil_config_dir") or DEFAULT_OBSUTIL_CONFIG_DIR),
        )
    ).expanduser()

    config_dir = config_dir.resolve()
    configured = _obsutil_config_mapping().get(bucket, bucket)
    candidate = Path(configured).expanduser()
    if not candidate.is_absolute():
        candidate = config_dir / candidate
    candidate = candidate.resolve()

    # 相对映射和默认“文件名即桶名”必须始终落在可信配置目录中。
    # 仅服务端显式配置的绝对路径可以位于其它目录。
    if not Path(configured).expanduser().is_absolute():
        try:
            candidate.relative_to(config_dir)
        except ValueError as exc:
            raise ValueError(f"OBS 桶 {bucket} 的配置路径越界") from exc
    if require_exists and not candidate.is_file():
        raise FileNotFoundError(f"OBS 桶 {bucket} 未配置")
    return str(candidate)


def _obsutil_runtime_config_dir() -> Path:
    """返回 obsutil 私有运行时配置目录，不复用共享凭据文件。"""
    configured = os.getenv("OBSUTIL_RUNTIME_CONFIG_DIR", "").strip()
    requested_dir = (
        Path(configured).expanduser()
        if configured
        else DEFAULT_OBSUTIL_RUNTIME_CONFIG_DIR
    )
    if not requested_dir.is_absolute():
        requested_dir = PROJECT_ROOT / requested_dir
    if requested_dir.exists() and requested_dir.is_symlink():
        raise ValueError("obsutil 私有配置目录不能是符号链接")
    requested_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir = requested_dir.resolve()
    if not runtime_dir.is_dir():
        raise ValueError("obsutil 私有配置目录无效")
    if os.name != "nt":
        runtime_dir.chmod(0o700)
    return runtime_dir


@contextmanager
def isolated_obsutil_config(config_path: str) -> Iterator[str]:
    """为一次 obsutil 命令创建私有配置副本，并在命令结束后删除。

    obsutil 可能加密并回写 ``-config`` 指向的文件，因此绝不能把
    ``/mnt/tanpeng/conf`` 中的共享原件直接传给子进程。
    """
    source = Path(config_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"obsutil 配置不存在: {source}")

    runtime_dir = _obsutil_runtime_config_dir()
    prefix = re.sub(r"[^A-Za-z0-9_.-]", "_", source.name)[:64] or "obsutil"
    fd, temp_name = tempfile.mkstemp(
        prefix=f"{prefix}.",
        suffix=".conf",
        dir=str(runtime_dir),
    )
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        if os.name != "nt":
            temp_path.chmod(0o600)
        shutil.copyfile(source, temp_path)
        if os.name != "nt":
            temp_path.chmod(0o600)
        yield str(temp_path)
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass


def download_obs_object(
    obs_path: str,
    local_path: str,
    config_path: Optional[str] = None,
    timeout: int = 300,
) -> Tuple[bool, str]:
    """下载单个 OBS 对象，不使用 ``-r``。

    ``config_path`` 仅供服务端可信调用；缺省时按桶自动选择配置。
    """
    get_obs_bucket(obs_path)
    object_key = obs_path.split("/", 3)[-1] if "/" in obs_path[6:] else ""
    if not object_key or obs_path.endswith("/"):
        return False, "仅支持下载单个 OBS 对象"
    if not os.path.isfile(OBSUTIL_BIN):
        return False, "obsutil not found"

    try:
        resolved_config = config_path or resolve_obsutil_config(obs_path)
        config_file = Path(resolved_config).expanduser().resolve()
        if not config_file.is_file():
            return False, f"OBS 桶 {get_obs_bucket(obs_path)} 未配置"

        target = Path(local_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with isolated_obsutil_config(str(config_file)) as runtime_config:
            cmd = [
                OBSUTIL_BIN,
                "cp",
                obs_path,
                str(target),
                "-f",
                f"-config={runtime_config}",
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout
            )
        if result.returncode == 0:
            return True, (result.stdout or "").strip()[-500:]
        return False, ((result.stderr or result.stdout) or "download failed").strip()[-500:]
    except subprocess.TimeoutExpired:
        return False, f"下载超时（{timeout}s）"
    except Exception as exc:
        return False, str(exc)

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

def obsutil_ls(
    path: str,
    show_dirs: bool = False,
    limit: int = 1000,
    config_path: Optional[str] = None,
) -> List[dict]:
    """执行 obsutil ls -d，只列出当前一层的子目录和文件（不递归）。

    obsutil ls -d 输出分两段：
        Folder list:
        obs://.../subdir/
        ...
        Object list:
        key   LastModified   Size   StorageClass   ETag
        obs://.../file.json
                          2026-...   472B   standard   "..."
    文件的元数据（大小等）在文件 key 的下一行，需要跨行解析。
    参数 show_dirs 保留兼容，实际始终 -d。
    """
    if not os.path.isfile(OBSUTIL_BIN):
        return []
    if not path.endswith("/"):
        path += "/"
    args = [OBSUTIL_BIN, "ls", path, "-d", f"-limit={limit}"]
    cfg_path = None
    if config_path:
        cfg_path = Path(config_path).expanduser().resolve()
        if not cfg_path.is_file():
            return []
    try:
        if cfg_path:
            with isolated_obsutil_config(str(cfg_path)) as runtime_config:
                result = subprocess.run(
                    [*args, f"-config={runtime_config}"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
        else:
            result = subprocess.run(
                args, capture_output=True, text=True, timeout=30
            )
        if result.returncode != 0:
            return []
    except Exception:
        return []

    dirs: List[dict] = []
    files: List[dict] = []
    section = None  # "folder" | "object"
    lines = result.stdout.splitlines()
    i = 0
    while i < len(lines):
        raw = lines[i]
        line = raw.strip()
        i += 1

        low = line.lower()
        if low.startswith("folder list"):
            section = "folder"
            continue
        if low.startswith("object list"):
            section = "object"
            continue
        if not line:
            continue

        if section == "folder":
            if not line.startswith("obs://"):
                continue
            full_path = line
            name = full_path.replace(path, "", 1).strip("/")
            if not name:  # 当前目录自身，跳过
                continue
            dirs.append({"name": name, "path": full_path, "is_dir": True, "size": None})

        elif section == "object":
            if line.startswith("key") and "LastModified" in raw:
                continue  # 表头
            if not line.startswith("obs://"):
                continue
            full_path = line
            name = full_path.replace(path, "", 1).strip("/")
            if not name:  # 目录占位对象（path 自身），跳过
                continue
            # 元数据在下一行：<空格>LastModified  Size  StorageClass  ETag
            size = None
            if i < len(lines):
                meta = lines[i].strip()
                if meta and not meta.startswith("obs://"):
                    for tok in meta.split():
                        if any(tok.endswith(u) for u in ("B", "KB", "MB", "GB", "TB")):
                            size = tok
                            break
                    i += 1  # 消费元数据行
            files.append({"name": name, "path": full_path, "is_dir": False, "size": size})

    return dirs + files


# ---------------------------------------------------------------------------
# 失败报告解析 + 单文件补传（"补同步"）
# ---------------------------------------------------------------------------

DEFAULT_OBSUTIL_OUTPUT_DIR = str(Path.home() / ".obsutil_output")

# 匹配失败报告数据行的 status/error message，用于记录失败原因
_STATUS_RE = re.compile(r"status \[(\d+)\]")
_ERRMSG_RE = re.compile(r"error message \[([^\]]*)\]")


def find_failed_report(task_id: str, output_dir: str = "") -> Optional[str]:
    """在 output_dir 下查找 obsutil 某次 cp 的失败报告，返回最新匹配文件路径。

    obsutil 命名：cp_failed_report_<timestamp>_<task_id>.txt
    """
    if not task_id:
        return None
    base = output_dir or DEFAULT_OBSUTIL_OUTPUT_DIR
    pattern = os.path.join(base, f"cp_failed_report_*_{task_id}.txt")
    matches = glob.glob(pattern)
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def parse_failed_report(report_path: str) -> List[Tuple[str, str, str]]:
    """解析 obsutil 失败报告，返回 [(local_path, obs_path, error), ...]。

    数据行形如：
        <ts> <size>, <local_src> --> <obs_dst>, cost [N], status [500], error code [..], error message [..], request id [..]
    表头行是 `[file size, src --> dst, ...]`（含 `-->` 但被方括号包裹），需排除。
    """
    result: List[Tuple[str, str, str]] = []
    try:
        with open(report_path, "r", encoding="utf-8", errors="replace") as f:
            for raw in f:
                line = raw.strip()
                if " --> " not in line:
                    continue
                # 排除表头：形如 "... [file size, src --> dst, cost(ms), ...]"
                if "[file size" in line or "src --> dst" in line:
                    continue
                # 切出 src：--> 左侧，取最后一个 ", " 之后（前面是时间戳+大小）
                left, _, right = line.partition(" --> ")
                # left 末尾是 "<ts> <size>, <local_src>"，local 从第一个 ", " 后开始
                if ", " in left:
                    local_path = left.split(", ", 1)[1].strip()
                else:
                    local_path = left.strip()
                # 切出 dst：--> 右侧到 ", cost [" 之前
                obs_path = right.split(", cost [", 1)[0].strip()
                if not obs_path:
                    obs_path = right.strip()
                # 错误信息
                status_m = _STATUS_RE.search(line)
                errmsg_m = _ERRMSG_RE.search(line)
                parts = []
                if status_m:
                    parts.append(f"status {status_m.group(1)}")
                if errmsg_m:
                    parts.append(errmsg_m.group(1))
                error = " - ".join(parts) if parts else "upload failed"
                if local_path and obs_path.startswith("obs://"):
                    result.append((local_path, obs_path, error))
    except OSError:
        return []
    return result


def reupload_file(local_path: str, obs_path: str, timeout: int = 120) -> Tuple[bool, str]:
    """单文件补传：obsutil cp <local> <obs> -f（非 -r，幂等覆盖）。"""
    if not os.path.isfile(OBSUTIL_BIN):
        return False, "obsutil not found"
    if not os.path.isfile(local_path):
        return False, f"本地文件不存在: {local_path}"
    cmd = [OBSUTIL_BIN, "cp", local_path, obs_path, "-f"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode == 0:
            return True, (result.stdout or "").strip()[-200:]
        return False, ((result.stderr or result.stdout) or "").strip()[-200:]
    except subprocess.TimeoutExpired:
        return False, f"补传超时（{timeout}s）"
    except Exception as e:
        return False, str(e)
