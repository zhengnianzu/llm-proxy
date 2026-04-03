"""
sync_sessions.py — 实时增量同步守护进程

三种运行模式，由参数组合决定:

  1. raw 模式（--src + obs_raw，无 --session-dir）:
     按 src/index.jsonl 增量读取原始三元组文件，整批上传到 OBS（obs_raw）。
     upload_erase=true 时，上传成功的文件从 src 中删除。

  2. export 模式（--src + --session-dir + obs_session）:
     先从 src 增量 export 出 session 文件到 session-dir，再上传到 OBS（obs_session）。
     upload_erase=true 时，上传 session folder 成功后删除 src 中对应的原始三元组文件。
     可同时指定 obs_raw 进行双备份（先 raw 上传，再 export 上传，各自独立 erase）。

  3. upload-only 模式（--session-dir，无 --src）:
     session-dir 由外部写入，内含 index.jsonl。
     读增量行找出变化 folder，上传到 OBS（obs_session）。
     upload_erase=true 时，上传成功后删除本地 session folder。

用法:
    # raw 模式
    python sync_sessions.py --src /path/to/logs_anthropic \\
                            --obs-raw obs://bucket/raw/

    # export 模式（双备份）
    python sync_sessions.py --src /path/to/logs_anthropic --session-dir /path/to/out \\
                            --obs-raw obs://bucket/raw/ --obs-session obs://bucket/sessions/

    # upload-only 模式
    python sync_sessions.py --session-dir /path/to/sessions \\
                            --obs-session obs://bucket/sessions/

    # 单次运行（测试/cron）
    python sync_sessions.py --config sync_config.yaml --once

配置字段说明:
    src:              logs_anthropic 目录（有则启用 raw/export 模式）
    session_dir:      session 目录（export 时自动创建，upload-only 时须已存在）
    obs_raw:          原始三元组上传目标（raw/export 模式均可指定）
    obs_session:      session 上传目标（export/upload-only 模式使用）
    upload_erase:     上传成功后是否清除本地数据（默认 false）
    upload_script:    自定义上传脚本，签名: <script> <local_path> <obs_path>
    interval_seconds: 同步间隔（默认 3600）
    upload_workers:   上传并发数（默认 4）

session_dir/index.jsonl 格式（每行一条）:
    {"folder": "2026-03-25_15-42-10_366", "q1": "...",
     "latest_file": "2026-03-25_15-42-10_366.json", "msg_count": 5, "model": "..."}
"""

import argparse
import json
import logging
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

from src.utils.message_utils import (
    count_user_messages,
    extract_messages,
    get_first_user_text,
    load_json,
    parse_response,
)
from src.utils.triplet_collector import collect_new_triplets, read_session_index_jsonl


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STATE_FILE = ".sync_state.json"
LOG_FILE = "sync_sessions.log"
DEFAULT_CONFIG = "sync_config.yaml"

DEFAULTS = {
    "interval_seconds": 3600,
    "upload_workers": 4,
    "upload_erase": False,
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        return {}
    suffix = config_path.suffix.lower()
    with open(config_path, "r", encoding="utf-8") as f:
        if suffix in (".yaml", ".yml"):
            if not _YAML_AVAILABLE:
                print("[warn] PyYAML not installed, cannot read .yaml config. Install with: pip install pyyaml")
                return {}
            return yaml.safe_load(f) or {}
        elif suffix == ".json":
            return json.load(f)
        else:
            print(f"[warn] Unknown config format: {suffix}, skipping")
            return {}


def build_config(args: argparse.Namespace) -> dict:
    config = dict(DEFAULTS)

    config_path = Path(args.config) if args.config else Path(__file__).parent / DEFAULT_CONFIG
    file_cfg = load_config(config_path)
    if file_cfg:
        config.update(file_cfg)
    elif args.config:
        sys.exit(f"[error] Config file not found: {args.config}")

    if args.src:
        config["src"] = args.src
    if getattr(args, "session_dir", None):
        config["session_dir"] = args.session_dir
    if getattr(args, "upload_script", None):
        config["upload_script"] = args.upload_script
    if getattr(args, "obs_raw", None):
        config["obs_raw"] = args.obs_raw
    if getattr(args, "obs_session", None):
        config["obs_session"] = args.obs_session
    if getattr(args, "upload_erase", None) is not None:
        config["upload_erase"] = args.upload_erase
    if args.interval is not None:
        config["interval_seconds"] = args.interval
    if args.workers is not None:
        config["upload_workers"] = args.workers

    # 规范化 obs_raw / obs_session
    def _normalize_obs(key: str) -> None:
        val = config.get(key)
        if not val:
            return
        if not val.startswith("obs://"):
            sys.exit(f"[error] {key} must start with obs://: {val}")
        if not val.endswith("/"):
            config[key] = val + "/"

    _normalize_obs("obs_raw")
    _normalize_obs("obs_session")

    has_src = bool(config.get("src"))
    has_session_dir = bool(config.get("session_dir"))
    has_obs_raw = bool(config.get("obs_raw"))
    has_obs_session = bool(config.get("obs_session"))

    # 至少需要一个 OBS 目标
    if not has_obs_raw and not has_obs_session:
        sys.exit(
            "[error] Must specify at least one OBS target:\n"
            "  --obs-raw   (for raw/export mode)\n"
            "  --obs-session (for export/upload-only mode)"
        )

    if has_src:
        # 有 src：raw 模式 或 export 模式（或两者同时）
        src = Path(config["src"]).resolve()
        if not src.is_dir():
            sys.exit(f"[error] src directory not found: {src}")
        config["src"] = str(src)

        if has_session_dir:
            # export 模式（含 raw 双备份）
            if not has_obs_session:
                sys.exit("[error] export mode requires obs_session")
            sd = Path(config["session_dir"]).resolve()
            sd.mkdir(parents=True, exist_ok=True)
            config["session_dir"] = str(sd)
            config["mode"] = "export"
        else:
            # raw-only 模式
            if not has_obs_raw:
                sys.exit("[error] raw mode requires obs_raw (no --session-dir given)")
            config["mode"] = "raw"
    else:
        # 无 src：upload-only 模式
        if not has_session_dir:
            sys.exit(
                "[error] Must specify --session-dir for upload-only mode (no --src given)"
            )
        if not has_obs_session:
            sys.exit("[error] upload-only mode requires obs_session")
        sd = Path(config["session_dir"]).resolve()
        if not sd.is_dir():
            sys.exit(f"[error] session_dir not found: {sd}")
        config["session_dir"] = str(sd)
        config["mode"] = "upload-only"

    # upload_erase 规范为 bool
    config["upload_erase"] = bool(config.get("upload_erase", False))

    # upload_script 校验
    if config.get("upload_script"):
        script = Path(config["upload_script"]).resolve()
        if not script.is_file():
            sys.exit(f"[error] upload_script not found: {script}")
        config["upload_script"] = str(script)

    return config


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(out_dir: Path) -> logging.Logger:
    logger = logging.getLogger("sync_sessions")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = RotatingFileHandler(
        out_dir / LOG_FILE,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

def load_state(out_dir: Path) -> dict:
    state_file = out_dir / STATE_FILE
    if not state_file.exists():
        return {}
    try:
        with open(state_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(out_dir: Path, cutoff: Optional[str], index_line_offset: int = 0, mode: str = "export") -> None:
    # 用 mode 前缀区分两种模式的 offset，避免共用 state_dir 时互相覆盖
    offset_key = "src_index_line_offset" if mode == "export" else "session_index_line_offset"
    state = {
        "cutoff": cutoff,
        offset_key: index_line_offset,
        "last_sync_utc": datetime.now(timezone.utc).isoformat(),
    }
    tmp = out_dir / (STATE_FILE + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False)
    tmp.replace(out_dir / STATE_FILE)


# ---------------------------------------------------------------------------
# Incremental export
# ---------------------------------------------------------------------------

def run_export(
    src: Path,
    out: Path,
    cutoff: Optional[str],
    index_line_offset: int,
    logger: logging.Logger,
) -> Tuple[int, int, List[str], Optional[str], int]:
    """
    增量导出：只处理 prefix > cutoff 的三元组，原地更新 out/。
    同时将变化的 session 条目追加到 out/index.jsonl（供 upload-only 模式消费）。

    Returns:
        (exported_files, sessions_updated, changed_folders, new_cutoff, new_index_line_offset)
    """
    # Load existing index as base
    base_index: List[dict] = []
    idx_path = out / "index.json"

    if idx_path.exists():
        try:
            with open(idx_path, "r", encoding="utf-8") as f:
                base_index = json.load(f)
        except Exception as e:
            logger.warning("Failed to read existing index.json: %s", e)

    # Collect new triplets
    new_triplets, new_prefixes, total_lines = collect_new_triplets(
        src, cutoff, index_line_offset
    )
    new_index_line_offset = total_lines if total_lines is not None else index_line_offset

    if not new_prefixes:
        logger.info("No new triplets — nothing to do")
        return 0, 0, [], cutoff, new_index_line_offset, {}

    logger.info("New triplets: %d (index_line_offset: %d→%d)",
                len(new_prefixes), index_line_offset, new_index_line_offset)

    # Build sessions
    sessions: List[dict] = []
    latest_session_by_q1: Dict[str, dict] = {}

    for entry in base_index:
        q1 = entry.get("q1", "")
        if not q1:
            continue
        session = {
            "folder_prefix": entry["folder"],
            "q1": q1,
            "items": [],
            "from_base": True,
        }
        sessions.append(session)
        latest_session_by_q1[q1] = session

    skipped = 0
    for prefix in new_prefixes:
        tri = new_triplets[prefix]
        if "req" not in tri:
            skipped += 1
            continue
        try:
            req_data = load_json(tri["req"])
        except Exception as e:
            logger.warning("Failed to read req %s: %s", prefix, e)
            skipped += 1
            continue

        messages = extract_messages(req_data)
        if not messages:
            skipped += 1
            continue

        q1 = get_first_user_text(messages)
        user_count = count_user_messages(messages)

        if user_count <= 1:
            session = {"folder_prefix": prefix, "q1": q1, "items": [(prefix, tri)], "from_base": False}
            sessions.append(session)
            latest_session_by_q1[q1] = session
        else:
            session = latest_session_by_q1.get(q1)
            if session is None:
                session = {"folder_prefix": prefix, "q1": q1, "items": [], "from_base": False}
                sessions.append(session)
                latest_session_by_q1[q1] = session
            session["items"].append((prefix, tri))

    logger.debug("Sessions total: %d, skipped triplets: %d", len(sessions), skipped)

    # Export active sessions
    active_sessions = [s for s in sessions if s.get("items")]
    all_items = [
        (s, prefix, tri)
        for s in active_sessions
        for prefix, tri in sorted(s["items"], key=lambda x: x[0])
    ]

    for s in active_sessions:
        (out / s["folder_prefix"]).mkdir(parents=True, exist_ok=True)

    def _export_one(task):
        session, prefix, tri = task
        try:
            req_data = load_json(tri["req"])
        except Exception as e:
            logger.warning("Failed to read req %s: %s", prefix, e)
            return session["folder_prefix"], f"{prefix}.json", None

        merged = dict(req_data)
        if "headers" in tri:
            try:
                merged["header"] = load_json(tri["headers"])
            except Exception as e:
                logger.warning("Failed to read headers %s: %s", prefix, e)
                merged["header"] = {}
        else:
            merged["header"] = {}

        if "res" in tri:
            try:
                merged["response"] = parse_response(load_json(tri["res"]))
            except Exception as e:
                logger.warning("Failed to parse res %s: %s", prefix, e)
                merged["response"] = {}
        else:
            merged["response"] = {}

        out_file = out / session["folder_prefix"] / f"{prefix}.json"
        with open(out_file, "w", encoding="utf-8") as fh:
            json.dump(merged, fh, ensure_ascii=False, indent=2)
        return session["folder_prefix"], f"{prefix}.json", merged

    results: Dict[str, List] = {s["folder_prefix"]: [] for s in active_sessions}
    exported_files = 0
    changed_folders_set: set = set()
    # folder_prefix -> 本次导出涉及的 triplet prefix 列表（用于 erase_src_triplets）
    folder_to_prefixes: Dict[str, List[str]] = {s["folder_prefix"]: [] for s in active_sessions}
    for s in active_sessions:
        for prefix, _tri in s["items"]:
            folder_to_prefixes[s["folder_prefix"]].append(prefix)

    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(_export_one, task): task for task in all_items}
        for future in as_completed(futures):
            folder_prefix, filename, merged = future.result()
            if merged is not None:
                results[folder_prefix].append((filename, merged))
                changed_folders_set.add(folder_prefix)
                exported_files += 1

    # Update index
    index_by_folder: Dict[str, dict] = {e["folder"]: e for e in base_index}

    for session in active_sessions:
        best_file: Optional[str] = None
        best_msg_count = -1
        best_model = ""

        existing = index_by_folder.get(session["folder_prefix"])
        if existing:
            best_file = existing.get("latest_file")
            best_msg_count = existing.get("msg_count", -1)
            best_model = existing.get("model", "")

        for filename, merged in results[session["folder_prefix"]]:
            messages = extract_messages(merged) or []
            has_response = bool((merged.get("response") or {}).get("content"))
            msg_count = len(messages) + (1 if has_response else 0)
            if msg_count > best_msg_count:
                best_msg_count = msg_count
                best_file = filename
                best_model = merged.get("model", "")

        if best_file:
            index_by_folder[session["folder_prefix"]] = {
                "folder": session["folder_prefix"],
                "q1": session["q1"],
                "latest_file": best_file,
                "msg_count": best_msg_count,
                "model": best_model,
            }

    with open(idx_path, "w", encoding="utf-8") as fh:
        json.dump(list(index_by_folder.values()), fh, ensure_ascii=False, indent=2)

    # 追加写 index.jsonl：将本轮变化的 session 条目逐行写入
    # 供外部（或 upload-only 模式）通过增量行检测变化
    if changed_folders_set:
        jsonl_path = out / "index.jsonl"
        with open(jsonl_path, "a", encoding="utf-8") as fh:
            for folder in sorted(changed_folders_set):
                entry = index_by_folder.get(folder)
                if entry:
                    fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    new_cutoff = max(new_prefixes) if new_prefixes else cutoff
    changed_folders = sorted(changed_folders_set)

    return exported_files, len(changed_folders), changed_folders, new_cutoff, new_index_line_offset, folder_to_prefixes


# ---------------------------------------------------------------------------
# Upload-only mode (session_dir with index.jsonl)
# ---------------------------------------------------------------------------

def run_upload_only(
    session_dir: Path,
    index_line_offset: int,
    logger: logging.Logger,
) -> Tuple[List[str], int]:
    """
    读取 session_dir/index.jsonl 的增量行，返回需要上传的 folder 列表。

    index.jsonl 由外部写入，每次 session 新增或更新时追加一行。
    同一个 folder 可能出现多次（多次更新），去重后返回。

    Returns:
        (changed_folders, new_index_line_offset)
    """
    index_path = session_dir / "index.jsonl"
    if not index_path.exists():
        logger.warning("session_dir/index.jsonl not found: %s", index_path)
        return [], index_line_offset

    new_entries, total_lines = read_session_index_jsonl(session_dir, index_line_offset)
    if not new_entries:
        logger.info("No new entries in index.jsonl (offset=%d, total=%d)",
                    index_line_offset, total_lines)
        return [], total_lines

    # 去重，保留每个 folder 最后一次出现的条目（最新状态）
    seen: Dict[str, dict] = {}
    for entry in new_entries:
        seen[entry["folder"]] = entry

    changed_folders = sorted(seen)
    logger.info("New index.jsonl entries: %d lines → %d unique folders",
                len(new_entries), len(changed_folders))
    return changed_folders, total_lines


# ---------------------------------------------------------------------------
# OBS upload
# ---------------------------------------------------------------------------

def _run_upload_cmd(
    upload_script: Optional[str],
    local: str,
    dst: str,
    timeout: int,
) -> Tuple[bool, str]:
    """执行上传命令，返回 (success, message)。"""
    if upload_script:
        cmd = [upload_script, local, dst]
        not_found_msg = f"upload_script not found: {upload_script}"
    else:
        cmd = ["obs_utils", "cp", local, dst]
        not_found_msg = "obs_utils not found in PATH"
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode == 0:
            return True, result.stdout.strip()
        return False, (result.stderr or result.stdout).strip()
    except FileNotFoundError:
        return False, not_found_msg
    except subprocess.TimeoutExpired:
        return False, f"upload timed out (>{timeout}s)"
    except Exception as e:
        return False, str(e)


def upload_folder(
    local_folder: Path,
    obs_dst: str,
    folder_name: str,
    upload_script: Optional[str] = None,
) -> Tuple[str, bool, str]:
    dst = obs_dst
    ok, msg = _run_upload_cmd(upload_script, str(local_folder) + "/", dst, timeout=120)
    return folder_name, ok, msg


def upload_index(
    out_dir: Path,
    obs_dst: str,
    upload_script: Optional[str] = None,
) -> Tuple[bool, str]:
    dst = obs_dst.rstrip("/") + "/index.json"
    return _run_upload_cmd(upload_script, str(out_dir / "index.json"), dst, timeout=60)


def upload_index_jsonl(
    out_dir: Path,
    obs_dst: str,
    upload_script: Optional[str] = None,
) -> Tuple[bool, str]:
    dst = obs_dst.rstrip("/") + "/index.jsonl"
    return _run_upload_cmd(upload_script, str(out_dir / "index.jsonl"), dst, timeout=60)


def run_uploads(
    out_dir: Path,
    obs_dst: str,
    changed_folders: List[str],
    workers: int,
    logger: logging.Logger,
    upload_script: Optional[str] = None,
) -> List[str]:
    """上传 changed_folders，返回上传成功的 folder 列表。"""
    ok_folders: List[str] = []
    fail_count = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(upload_folder, out_dir / folder, obs_dst, folder, upload_script): folder
            for folder in changed_folders
        }
        for future in as_completed(futures):
            folder_name, ok, msg = future.result()
            if ok:
                logger.info("upload OK  folder=%s", folder_name)
                ok_folders.append(folder_name)
            else:
                logger.error("upload FAIL folder=%s err=%s", folder_name, msg)
                fail_count += 1

    logger.info("upload summary: %d ok, %d failed", len(ok_folders), fail_count)

    ok, msg = upload_index(out_dir, obs_dst, upload_script)
    if ok:
        logger.info("index.json upload OK")
    else:
        logger.error("index.json upload FAIL: %s", msg)

    ok, msg = upload_index_jsonl(out_dir, obs_dst, upload_script)
    if ok:
        logger.info("index.jsonl upload OK")
    else:
        logger.error("index.jsonl upload FAIL: %s", msg)

    return ok_folders


# ---------------------------------------------------------------------------
# Erase helpers
# ---------------------------------------------------------------------------

def erase_src_triplets(
    src: Path,
    prefixes: List[str],
    logger: logging.Logger,
) -> None:
    """删除 src 中指定 prefix 对应的原始三元组文件（req/headers/res）。"""
    from src.utils.triplet_collector import TIMESTAMP_RE  # noqa: F401 — already imported via collect_new_triplets

    deleted = 0
    for prefix in prefixes:
        for suffix in ("-req.json", "-headers.json", "-res.json"):
            candidates = list(src.rglob(f"{prefix}{suffix}"))
            for p in candidates:
                try:
                    p.unlink()
                    deleted += 1
                except Exception as e:
                    logger.warning("erase_src: failed to delete %s: %s", p, e)

    logger.info("erase_src: deleted %d raw files for %d prefixes", deleted, len(prefixes))


def erase_session_folders(
    session_dir: Path,
    folders: List[str],
    logger: logging.Logger,
) -> None:
    """删除 session_dir 中已上传的 folder 子目录。"""
    import shutil
    deleted = 0
    for folder in folders:
        fp = session_dir / folder
        if fp.is_dir():
            try:
                shutil.rmtree(fp)
                deleted += 1
            except Exception as e:
                logger.warning("erase_session: failed to delete %s: %s", fp, e)
    logger.info("erase_session: deleted %d folders", deleted)


# ---------------------------------------------------------------------------
# Raw mode
# ---------------------------------------------------------------------------

def run_raw(
    src: Path,
    obs_raw: str,
    index_line_offset: int,
    workers: int,
    upload_erase: bool,
    logger: logging.Logger,
    upload_script: Optional[str] = None,
) -> Tuple[int, int]:
    """
    raw 模式：将 src/index.jsonl 中新增的三元组文件整批上传到 obs_raw。
    每个 prefix 的文件上传到 obs_raw/<prefix>/ 下。
    upload_erase=True 时，上传成功的文件从 src 删除。

    Returns:
        (uploaded_count, new_index_line_offset)
    """
    from src.utils.triplet_collector import collect_triplets_from_index

    index_path = src / "index.jsonl"
    if not index_path.exists():
        logger.warning("raw mode: src/index.jsonl not found, falling back to scan")
        from src.utils.triplet_collector import collect_triplets_by_scan
        triplets = collect_triplets_by_scan(src)
        total_lines = index_line_offset
    else:
        triplets, total_lines = collect_triplets_from_index(src, index_line_offset)

    if not triplets:
        logger.info("raw mode: no new triplets (offset=%d)", index_line_offset)
        return 0, total_lines if total_lines else index_line_offset

    logger.info("raw mode: %d new triplets to upload", len(triplets))

    def _upload_triplet(prefix: str, tri: dict) -> Tuple[str, bool, List[Path]]:
        dst_prefix = obs_raw.rstrip("/") + "/" + prefix + "/"
        uploaded_paths: List[Path] = []
        all_ok = True
        for kind, file_path in tri.items():
            dst = dst_prefix + file_path.name
            ok, msg = _run_upload_cmd(upload_script, str(file_path), dst, timeout=120)
            if ok:
                uploaded_paths.append(file_path)
            else:
                logger.error("raw upload FAIL prefix=%s file=%s err=%s", prefix, file_path.name, msg)
                all_ok = False
        return prefix, all_ok, uploaded_paths

    ok_count = 0
    ok_prefixes: List[str] = []
    uploaded_files: List[Path] = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_upload_triplet, prefix, tri): prefix
            for prefix, tri in triplets.items()
        }
        for future in as_completed(futures):
            prefix, all_ok, paths = future.result()
            # 无论是否全部成功，已上传的文件都记录下来（供 erase 使用）
            uploaded_files.extend(paths)
            if all_ok:
                ok_count += 1
                ok_prefixes.append(prefix)
            else:
                logger.error("raw upload incomplete for prefix=%s", prefix)

    logger.info("raw upload summary: %d/%d triplets fully OK", ok_count, len(triplets))

    if upload_erase and uploaded_files:
        logger.info("upload_erase: removing %d uploaded files from src", len(uploaded_files))
        deleted = 0
        for p in uploaded_files:
            try:
                p.unlink()
                deleted += 1
            except Exception as e:
                logger.warning("erase raw: failed to delete %s: %s", p, e)
        logger.info("upload_erase: deleted %d files", deleted)

    return ok_count, total_lines if total_lines else index_line_offset


# ---------------------------------------------------------------------------
# Daemon loop
# ---------------------------------------------------------------------------

_shutdown_requested = False


def _handle_signal(signum, frame) -> None:
    global _shutdown_requested
    _shutdown_requested = True
    print("\n[sync_sessions] Shutdown requested, finishing current cycle...")


def _interruptible_sleep(seconds: int) -> None:
    for _ in range(seconds):
        if _shutdown_requested:
            break
        time.sleep(1)


def run_daemon(config: dict, logger: logging.Logger, once: bool = False) -> None:
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    interval = int(config["interval_seconds"])
    workers = int(config["upload_workers"])
    mode = config["mode"]
    upload_script = config.get("upload_script") or None
    upload_erase = config["upload_erase"]
    obs_raw = config.get("obs_raw")
    obs_session = config.get("obs_session")

    # state 目录：有 session_dir 用 session_dir，否则用 src
    if config.get("session_dir"):
        state_dir = Path(config["session_dir"])
    else:
        state_dir = Path(config["src"])

    if config.get("src"):
        src = Path(config["src"])
    if config.get("session_dir"):
        session_dir = Path(config["session_dir"])

    while not _shutdown_requested:
        state = load_state(state_dir)

        if mode == "raw":
            # ── raw 模式 ──────────────────────────────────────────────
            raw_offset = state.get("src_index_line_offset", 0)
            logger.info("Sync start [raw]. src_index_line_offset=%d upload_erase=%s",
                        raw_offset, upload_erase)
            try:
                ok_count, new_offset = run_raw(
                    src, obs_raw, raw_offset, workers, upload_erase, logger, upload_script
                )
            except Exception:
                logger.exception("run_raw failed — skipping this cycle")
                if once:
                    break
                _interruptible_sleep(interval)
                continue
            save_state(state_dir, state.get("cutoff"), new_offset, mode="export")

        elif mode == "upload-only":
            # ── upload-only 模式 ──────────────────────────────────────
            sess_offset = state.get("session_index_line_offset", 0)
            logger.info("Sync start [upload-only]. session_index_line_offset=%d upload_erase=%s",
                        sess_offset, upload_erase)
            try:
                changed, new_offset = run_upload_only(session_dir, sess_offset, logger)
            except Exception:
                logger.exception("run_upload_only failed — skipping this cycle")
                if once:
                    break
                _interruptible_sleep(interval)
                continue

            logger.info("Sync done. changed_folders=%d", len(changed))
            if changed:
                logger.info("Uploading %d folders to OBS (obs_session)...", len(changed))
                ok_folders = run_uploads(session_dir, obs_session, changed, workers, logger, upload_script)
                if upload_erase and ok_folders:
                    logger.info("upload_erase: removing %d uploaded session folders", len(ok_folders))
                    erase_session_folders(session_dir, ok_folders, logger)
            else:
                logger.info("No changes — skipping upload")

            save_state(state_dir, state.get("cutoff"), new_offset, mode="upload-only")

        else:
            # ── export 模式 ───────────────────────────────────────────
            cutoff = state.get("cutoff")
            src_offset = state.get("src_index_line_offset", 0)
            logger.info("Sync start [export]. cutoff=%s src_index_line_offset=%d upload_erase=%s",
                        cutoff or "none", src_offset, upload_erase)

            # 1) 可选 raw 上传（先备份原始数据）
            if obs_raw:
                logger.info("export+raw: uploading raw triplets to obs_raw first")
                try:
                    # export 模式下 raw 只做备份，不删本地文件（export 步骤还需要这些文件）
                    # erase 统一在 session 上传成功后由 erase_src_triplets 完成
                    raw_ok, new_src_offset = run_raw(
                        src, obs_raw, src_offset, workers, False, logger, upload_script
                    )
                    # raw 上传后更新 offset，export 用同一 offset
                    src_offset = new_src_offset
                    save_state(state_dir, cutoff, src_offset, mode="export")
                except Exception:
                    logger.exception("run_raw (in export mode) failed — continuing with export")

            # 2) export + session 上传
            try:
                exported, updated, changed, new_cutoff, new_src_offset, folder_to_prefixes = run_export(
                    src, session_dir, cutoff, src_offset, logger
                )
            except Exception:
                logger.exception("run_export failed — skipping this cycle")
                if once:
                    break
                _interruptible_sleep(interval)
                continue

            logger.info("Export done. new_files=%d sessions_updated=%d", exported, updated)

            if changed:
                logger.info("Uploading %d changed folders to OBS (obs_session)...", len(changed))
                ok_folders = run_uploads(session_dir, obs_session, changed, workers, logger, upload_script)
                if upload_erase and ok_folders:
                    # session folder 已上传到 obs_session，可以删除
                    logger.info("upload_erase: removing %d uploaded session folders", len(ok_folders))
                    erase_session_folders(session_dir, ok_folders, logger)

                    # session 上传成功后，删除 src 中对应的原始三元组文件
                    # 注意：ok_folders 是 session folder 名，不一定等于 triplet prefix。
                    # 需要映射到本轮实际导出的 triplet prefixes 才能准确删除。
                    prefixes_to_erase: List[str] = []
                    for folder in ok_folders:
                        prefixes_to_erase.extend(folder_to_prefixes.get(folder, []))

                    logger.info(
                        "upload_erase: removing src triplets for %d uploaded folders (prefixes=%d)",
                        len(ok_folders), len(prefixes_to_erase),
                    )
                    if prefixes_to_erase:
                        erase_src_triplets(src, prefixes_to_erase, logger)
                    else:
                        logger.info("upload_erase: no src prefixes to erase for this cycle")
            else:
                logger.info("No changes — skipping session upload")

            save_state(state_dir, new_cutoff, new_src_offset, mode="export")

        if once or _shutdown_requested:
            break

        logger.info("Sleeping %ds until next sync", interval)
        _interruptible_sleep(interval)

    logger.info("Daemon exiting cleanly")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Incremental session sync daemon — exports logs_anthropic and uploads to OBS"
    )
    parser.add_argument("--config", "-c", default=None,
                        help=f"Config file path (default: {DEFAULT_CONFIG} next to script)")
    parser.add_argument("--src", "-s", default=None,
                        help="logs_anthropic source directory. "
                             "Presence selects raw/export mode; absence selects upload-only mode.")
    parser.add_argument("--session-dir", "-o", default=None, dest="session_dir",
                        help="Session directory: output for export mode, or existing dir for upload-only mode. "
                             "Not required for raw-only mode.")
    parser.add_argument("--obs-raw", default=None, dest="obs_raw",
                        help="OBS target for raw triplet files (e.g. obs://bucket/raw/). "
                             "Enables raw backup when --src is given.")
    parser.add_argument("--obs-session", default=None, dest="obs_session",
                        help="OBS target for session folders (e.g. obs://bucket/sessions/). "
                             "Required for export/upload-only mode.")
    parser.add_argument("--upload-erase", action="store_true", default=None, dest="upload_erase",
                        help="Delete local data after successful upload.")
    parser.add_argument("--upload-script", default=None, dest="upload_script",
                        help="Upload script path (e.g. ./obs_upload.sh). "
                             "Signature: <script> <local_path> <obs_path>. "
                             "Falls back to obs_utils if not specified.")
    parser.add_argument("--interval", type=int, default=None,
                        help="Sync interval in seconds (default 3600)")
    parser.add_argument("--workers", type=int, default=None,
                        help="OBS upload concurrency (default 4)")
    parser.add_argument("--once", action="store_true",
                        help="Run a single sync cycle then exit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = build_config(args)

    # logging 目录：优先 session_dir，其次 src
    log_dir = Path(config.get("session_dir") or config["src"])
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(log_dir)
    logger.info(
        "sync_sessions starting. mode=%s upload_erase=%s interval=%ds workers=%d",
        config["mode"], config["upload_erase"],
        config["interval_seconds"], config["upload_workers"],
    )
    if config.get("obs_raw"):
        logger.info("obs_raw=%s", config["obs_raw"])
    if config.get("obs_session"):
        logger.info("obs_session=%s", config["obs_session"])
    if config.get("src"):
        logger.info("src=%s", config["src"])
    if config.get("session_dir"):
        logger.info("session_dir=%s", config["session_dir"])

    run_daemon(config, logger, once=args.once)


if __name__ == "__main__":
    main()
