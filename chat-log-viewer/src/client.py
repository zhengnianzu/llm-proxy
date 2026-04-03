"""
client.py — OBS 同步下载客户端

功能：
  1. 从 OBS 目录下载 index.jsonl 文件
  2. 解析 index.jsonl 的新增行，下载对应的 session folder
  3. 支持增量下载：通过本地状态（.client_state.json / .last_sync_ts）记录已处理的 index.jsonl 行偏移
  4. 支持定时任务：通过 --interval 参数定时轮询执行同步下载
  5. 支持 YAML 配置文件

用法：
    # 初次全量（从 index.jsonl 第 1 行开始处理）
    python client.py --obs-path obs://bucket/sessions/ --output ./local_sessions

    # 增量下载（从 base-output 的 index.jsonl 最后一行开始继续）
    python client.py --obs-path obs://bucket/sessions/ --output ./local_sessions --base-output ./local_sessions

    # 定时增量同步（每隔 3600 秒）
    python client.py --obs-path obs://bucket/sessions/ --output ./local_sessions --base-output ./local_sessions --interval 3600

    # 使用配置文件
    python client.py --config configs/client.yaml

配置文件示例 (configs/client.yaml):
    obs_path: obs://bucket/sessions/
    output: ./local_sessions
    base_output: ./local_sessions      # 可选：存在时用其 index.jsonl 初始化增量状态（从最后一行继续）
    download_script: ./obs_download.sh # 可选，默认 obs_download.sh
    workers: 4                         # 可选，默认 4
    interval: 3600                     # 可选，指定后启用定时任务

配置优先级：命令行参数 > 配置文件 > 默认值
"""

import argparse
import json
import logging
import re
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_DOWNLOAD_SCRIPT = "obs_download.sh"
DEFAULT_WORKERS = 4
DEFAULT_CONFIG = "configs/client.yaml"
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging() -> logging.Logger:
    logger = logging.getLogger("obs_client")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(handler)
    return logger


logger = setup_logging()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_yaml_config(config_path: Path) -> dict:
    if not config_path.exists():
        return {}
    if not _YAML_AVAILABLE:
        logger.warning("PyYAML not installed, cannot read config file. Install with: pip install pyyaml")
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_config(args: argparse.Namespace) -> dict:
    """合并配置文件与命令行参数，命令行优先。"""
    # 先从配置文件加载
    config_path = Path(args.config) if args.config else Path(__file__).parent / DEFAULT_CONFIG
    cfg = load_yaml_config(config_path)
    if args.config and not config_path.exists():
        sys.exit(f"[error] Config file not found: {args.config}")

    # 命令行参数覆盖配置文件（仅当命令行有显式指定时）
    if args.obs_path:
        cfg["obs_path"] = args.obs_path
    if args.output:
        cfg["output"] = args.output
    if args.base_output:
        cfg["base_output"] = args.base_output
    if args.download_script != DEFAULT_DOWNLOAD_SCRIPT:
        cfg["download_script"] = args.download_script
    if args.workers != DEFAULT_WORKERS:
        cfg["workers"] = args.workers
    if args.interval is not None:
        cfg["interval"] = args.interval

    # 填充默认值
    cfg.setdefault("download_script", DEFAULT_DOWNLOAD_SCRIPT)
    cfg.setdefault("workers", DEFAULT_WORKERS)
    cfg.setdefault("interval", None)
    cfg.setdefault("base_output", None)

    # 校验必填项
    if not cfg.get("obs_path"):
        sys.exit("[error] obs_path is required (--obs-path or config: obs_path)")
    if not cfg.get("output"):
        sys.exit("[error] output is required (--output or config: output)")

    # 规范化 obs_path
    obs_path = cfg["obs_path"]
    if not obs_path.startswith("obs://"):
        sys.exit(f"[error] obs_path must start with obs://: {obs_path}")
    if not obs_path.endswith("/"):
        cfg["obs_path"] = obs_path + "/"

    return cfg


# ---------------------------------------------------------------------------
# OBS download wrapper
# ---------------------------------------------------------------------------

def download_file(
    obs_path: str,
    local_path: Path,
    download_script: str,
    timeout: int = 300,
) -> Tuple[bool, str]:
    """使用 download_script 下载单个文件或目录。"""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [download_script, obs_path, str(local_path)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode == 0:
            return True, result.stdout.strip()
        return False, (result.stderr or result.stdout).strip()
    except FileNotFoundError:
        return False, f"download_script not found: {download_script}"
    except subprocess.TimeoutExpired:
        return False, f"download timed out (>{timeout}s)"
    except Exception as e:
        return False, str(e)


# ---------------------------------------------------------------------------
# Index.jsonl parsing
# ---------------------------------------------------------------------------

def download_index_file(obs_path: str, filename: str, download_script: str, output: Path) -> Optional[Path]:
    """下载 OBS 上的 index 文件（index.jsonl 或 index.json）到 output 目录，失败返回 None。"""
    obs_file = obs_path.rstrip("/") + "/" + filename

    logger.info("Downloading %s from %s", filename, obs_file)
    ok, msg = download_file(obs_file, output, download_script, timeout=60)

    local_path = output / filename
    if not ok:
        logger.warning("Failed to download %s: %s", filename, msg)
        return None

    if not local_path.exists() or local_path.stat().st_size == 0:
        logger.warning("Downloaded %s is empty or missing", filename)
        return None

    logger.info("Downloaded %s successfully", filename)
    return local_path


def parse_index_jsonl(index_path: Path, line_offset: int = 0) -> Tuple[List[dict], int]:
    """
    解析 index.jsonl，从 line_offset 行开始读取新增行。
    去重保留每个 folder 最后一条（最新状态）。

    Returns:
        (entries, total_lines)
    """
    seen: dict = {}
    total_lines = 0
    with open(index_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            total_lines = line_no
            if line_no <= line_offset:
                continue
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if "folder" not in entry:
                    logger.warning("Line %d missing 'folder' field, skipping", line_no)
                    continue
                seen[entry["folder"]] = entry
            except json.JSONDecodeError as e:
                logger.warning("Line %d invalid JSON: %s", line_no, e)

    entries = list(seen.values())
    logger.info("Parsed %d unique entries from index.jsonl (offset=%d, total_lines=%d)",
                len(entries), line_offset, total_lines)
    return entries, total_lines


def parse_index_json(index_path: Path) -> List[dict]:
    """解析 index.json（备用方案），返回所有条目。"""
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            entries = json.load(f)
        if not isinstance(entries, list):
            logger.error("index.json is not a list")
            return []
        logger.info("Parsed %d entries from index.json", len(entries))
        return entries
    except json.JSONDecodeError as e:
        logger.error("Failed to parse index.json: %s", e)
        return []
    except Exception as e:
        logger.error("Error reading index.json: %s", e)
        return []


# ---------------------------------------------------------------------------
# Incremental filtering
# ---------------------------------------------------------------------------

_FOLDER_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})")


def extract_date_from_folder(folder_name: str) -> Optional[datetime]:
    """从文件夹名称提取日期时间，格式 2026-03-25_15-42-10_366（末尾毫秒可选）。"""
    m = _FOLDER_TS_RE.match(folder_name)
    if not m:
        return None
    try:
        return datetime.strptime(f"{m.group(1)}_{m.group(2)}", "%Y-%m-%d_%H-%M-%S")
    except ValueError:
        return None


def extract_date_from_latest_file(latest_file: str) -> Optional[datetime]:
    """从 latest_file 字段提取日期时间，格式 2026-03-25_15-42-10_366.json。"""
    name = latest_file.rsplit(".", 1)[0]  # 去掉扩展名
    return extract_date_from_folder(name)


def get_latest_date_from_local_index(base_output: Path) -> Optional[datetime]:
    """从本地 index.jsonl 的最后一行提取 latest_file 日期。"""
    index_jsonl = base_output / "index.jsonl"

    if not index_jsonl.exists():
        logger.info("No local index.jsonl found in base_output")
        return None

    try:
        # 读取最后一行
        last_line = None
        with open(index_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    last_line = line

        if not last_line:
            logger.info("index.jsonl is empty")
            return None

        entry = json.loads(last_line)
        latest_file = entry.get("latest_file")
        if not latest_file:
            logger.warning("Last entry missing latest_file field")
            return None

        latest_date = extract_date_from_latest_file(latest_file)
        if latest_date:
            logger.info("Latest date from local index.jsonl: %s", latest_date.isoformat())
        else:
            logger.warning("Could not extract date from latest_file: %s", latest_file)

        return latest_date
    except Exception as e:
        logger.warning("Failed to read local index.jsonl: %s", e)
        return None


def filter_entries_by_date(entries: List[dict], cutoff_date: Optional[datetime]) -> List[dict]:
    """
    过滤出日期晚于 cutoff_date 的条目。
    优先从 folder 名称提取日期，如果失败则尝试从 latest_file 提取。
    """
    if cutoff_date is None:
        return entries

    filtered = []
    for e in entries:
        folder_date = extract_date_from_folder(e.get("folder", ""))
        if not folder_date and "latest_file" in e:
            folder_date = extract_date_from_latest_file(e["latest_file"])
        if folder_date and folder_date > cutoff_date:
            filtered.append(e)

    logger.info("Filtered %d/%d entries (after %s)",
                len(filtered), len(entries), cutoff_date.isoformat())
    return filtered


_STATE_FILE = ".client_state.json"


def read_sync_state(output: Path) -> dict:
    """读取上次同步状态（行偏移 + 最新日期）。"""
    state_file = output / _STATE_FILE
    if not state_file.exists():
        return {"index_line_offset": 0, "latest_date": None}
    try:
        with open(state_file, "r", encoding="utf-8") as f:
            state = json.load(f)
        offset = state.get("index_line_offset", 0)
        latest_date_str = state.get("latest_date")
        latest_date = datetime.fromisoformat(latest_date_str) if latest_date_str else None
        logger.info("Loaded sync state: offset=%d, latest_date=%s", offset, latest_date_str or "None")
        return {"index_line_offset": offset, "latest_date": latest_date}
    except Exception as e:
        logger.warning("Failed to read %s: %s", state_file, e)
        return {"index_line_offset": 0, "latest_date": None}


def write_sync_state(output: Path, offset: int, latest_date: Optional[datetime]) -> None:
    """保存同步状态（行偏移 + 最新日期）。"""
    state_file = output / _STATE_FILE
    state = {
        "index_line_offset": offset,
        "latest_date": latest_date.isoformat() if latest_date else None,
    }
    try:
        tmp = output / (_STATE_FILE + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        tmp.replace(state_file)
        logger.info("Saved sync state: offset=%d, latest_date=%s", offset, state["latest_date"] or "None")
    except Exception as e:
        logger.warning("Failed to write %s: %s", state_file, e)


# ---------------------------------------------------------------------------
# Download folders
# ---------------------------------------------------------------------------

def download_folder(
    obs_base: str,
    folder_name: str,
    output_dir: Path,
    download_script: str,
) -> Tuple[str, bool, str]:
    obs_folder = obs_base.rstrip("/") + "/" + folder_name + "/"
    ok, msg = download_file(obs_folder, output_dir, download_script, timeout=300)
    return folder_name, ok, msg


def download_folders_parallel(
    obs_base: str,
    folders: List[str],
    output_dir: Path,
    download_script: str,
    workers: int,
) -> Tuple[int, int]:
    """并发下载多个文件夹，返回 (success_count, fail_count)。"""
    if not folders:
        logger.info("No folders to download")
        return 0, 0

    logger.info("Starting download of %d folders (workers=%d)", len(folders), workers)
    success_count = 0
    fail_count = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(download_folder, obs_base, folder, output_dir, download_script): folder
            for folder in folders
        }
        for future in as_completed(futures):
            folder_name, ok, msg = future.result()
            if ok:
                logger.info("✓ %s", folder_name)
                success_count += 1
            else:
                logger.error("✗ %s — %s", folder_name, msg)
                fail_count += 1

    logger.info("Download summary: %d success, %d failed", success_count, fail_count)
    return success_count, fail_count


# ---------------------------------------------------------------------------
# One sync cycle
# ---------------------------------------------------------------------------

def sync_once(
    obs_path: str,
    output: Path,
    base_output: Optional[Path],
    download_script: str,
    workers: int,
) -> None:
    """
    执行一次完整的同步。

    增量策略（基于行偏移 + 时间戳）：
    1. 读取 output/.client_state.json 获取上次的 index_line_offset 和 latest_date
    2. 下载 OBS 上的 index.jsonl，从 line_offset 开始读取新增行
    3. 对新增条目，按 latest_file 日期过滤（晚于 latest_date）
    4. 下载过滤后的 folders
    5. 更新 state：新的 line_offset 和本次下载的最新日期
    """
    output.mkdir(parents=True, exist_ok=True)

    # Step 1: 读取上次同步状态
    state = read_sync_state(output)
    line_offset = state["index_line_offset"]
    cutoff_date = state["latest_date"]

    # fallback: 如果 state 为空且指定了 base_output，从 base_output 初始化
    if line_offset == 0 and cutoff_date is None and base_output:
        cutoff_date = get_latest_date_from_local_index(base_output)
        if cutoff_date:
            logger.info("Initialized cutoff_date from base_output: %s", cutoff_date.isoformat())

    if cutoff_date:
        logger.info("Incremental mode: line_offset=%d, cutoff_date=%s", line_offset, cutoff_date.isoformat())
    else:
        logger.info("Full download mode: line_offset=%d (no cutoff date)", line_offset)

    # Step 2: 下载 OBS 上的最新 index.jsonl
    index_path = download_index_file(obs_path, "index.jsonl", download_script, output)
    new_entries = []
    total_lines = line_offset

    if index_path:
        try:
            new_entries, total_lines = parse_index_jsonl(index_path, line_offset)
            logger.info("Downloaded and parsed index.jsonl: %d new entries", len(new_entries))
        except Exception as e:
            logger.warning("Failed to parse index.jsonl: %s", e)

    if not new_entries:
        logger.info("No new entries to download (offset=%d, total_lines=%d)", line_offset, total_lines)
        # 即使没有新条目，也更新 line_offset
        write_sync_state(output, total_lines, cutoff_date)
        return

    # Step 3: 按日期过滤（只下载 latest_file 日期晚于 cutoff_date 的）
    if cutoff_date:
        filtered_entries = filter_entries_by_date(new_entries, cutoff_date)
        if not filtered_entries:
            logger.info("No new entries after date filtering (all up-to-date)")
            write_sync_state(output, total_lines, cutoff_date)
            return
        folders_to_download = [e["folder"] for e in filtered_entries]
    else:
        folders_to_download = [e["folder"] for e in new_entries]
        logger.info("Full download mode: %d folders", len(folders_to_download))

    # Step 4: 并发下载 folders
    success, failed = download_folders_parallel(
        obs_path, folders_to_download, output, download_script, workers
    )

    # Step 5: 保存新 index 到本地（合并所有条目）
    # 需要读取完整的 index.jsonl 来生成 index.json
    all_entries_dict = {}
    if index_path and index_path.exists():
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        if "folder" in entry:
                            all_entries_dict[entry["folder"]] = entry
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            logger.warning("Failed to read full index.jsonl for index.json generation: %s", e)

    with open(output / "index.json", "w", encoding="utf-8") as f:
        json.dump(list(all_entries_dict.values()), f, ensure_ascii=False, indent=2)
    logger.info("Saved index.json to %s", output / "index.json")

    # Step 6: 更新同步状态
    if success > 0:
        downloaded_entries = [e for e in new_entries if e.get("folder") in set(folders_to_download)]
        logger.debug("Downloaded entries count: %d", len(downloaded_entries))
        max_date = cutoff_date  # 保留旧的 cutoff_date 作为基准
        for e in downloaded_entries:
            dt = None
            if "latest_file" in e:
                dt = extract_date_from_latest_file(e["latest_file"])
            elif "folder" in e:
                dt = extract_date_from_folder(e["folder"])
            logger.debug("Entry %s -> date %s", e.get("folder"), dt)
            if dt and (max_date is None or dt > max_date):
                max_date = dt
        logger.info("Max date from downloaded entries: %s", max_date)
        if max_date:
            write_sync_state(output, total_lines, max_date)
        else:
            logger.warning("Could not extract max_date from downloaded entries; updating offset only")
            write_sync_state(output, total_lines, cutoff_date)
    else:
        logger.info("No successful downloads in this cycle; updating offset only")
        write_sync_state(output, total_lines, cutoff_date)

    logger.info("Sync completed: %d downloaded, %d failed", success, failed)


# ---------------------------------------------------------------------------
# Daemon loop
# ---------------------------------------------------------------------------

_shutdown_requested = False


def _handle_signal(signum, frame) -> None:
    global _shutdown_requested
    _shutdown_requested = True
    print("\n[obs_client] Shutdown requested, finishing current cycle...")


def _interruptible_sleep(seconds: int) -> None:
    for _ in range(seconds):
        if _shutdown_requested:
            break
        time.sleep(1)


def run_daemon(cfg: dict) -> None:
    obs_path: str = cfg["obs_path"]
    output: Path = Path(cfg["output"]).resolve()
    base_output: Optional[Path] = Path(cfg["base_output"]).resolve() if cfg.get("base_output") else None
    download_script: str = str(Path(cfg["download_script"]).resolve())
    workers: int = int(cfg["workers"])
    interval: Optional[int] = int(cfg["interval"]) if cfg.get("interval") else None

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    logger.info("OBS sync download starting")
    logger.info("  obs_path:        %s", obs_path)
    logger.info("  output:          %s", output)
    logger.info("  base_output:     %s", base_output or "None")
    logger.info("  download_script: %s", download_script)
    logger.info("  workers:         %d", workers)
    if interval:
        logger.info("  mode:            daemon (interval=%ds)", interval)
    else:
        logger.info("  mode:            one-shot")

    while not _shutdown_requested:
        logger.info("--- Sync cycle start ---")
        try:
            sync_once(obs_path, output, base_output, download_script, workers)
        except KeyboardInterrupt:
            logger.info("Interrupted by user (Ctrl+C)")
            break
        except Exception:
            logger.exception("Unexpected error during sync cycle")

        if not interval or _shutdown_requested:
            break

        logger.info("Sleeping %ds until next sync", interval)
        _interruptible_sleep(interval)

    logger.info("Exiting")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OBS sync download client — download sessions from OBS based on index.jsonl"
    )
    parser.add_argument(
        "--config", "-c", default=None,
        help=f"YAML config file path (default: {DEFAULT_CONFIG} next to script)",
    )
    parser.add_argument(
        "--obs-path", default=None,
        help="OBS source path (e.g., obs://bucket/sessions/)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Local output directory",
    )
    parser.add_argument(
        "--base-output", default=None,
        help="Local base directory for incremental download (compare latest folder date)",
    )
    parser.add_argument(
        "--download-script", default=DEFAULT_DOWNLOAD_SCRIPT,
        help=f"Download script path (default: {DEFAULT_DOWNLOAD_SCRIPT})",
    )
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        help=f"Concurrent download workers (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--interval", type=int, default=None,
        help="Sync interval in seconds; omit to run once",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = build_config(args)

    download_script = Path(cfg["download_script"]).resolve()
    if not download_script.exists():
        logger.error("Download script not found: %s", download_script)
        sys.exit(1)
    cfg["download_script"] = str(download_script)

    run_daemon(cfg)


if __name__ == "__main__":
    main()
