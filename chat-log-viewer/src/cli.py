#!/usr/bin/env python3
"""
cli.py — Chat Log Viewer 统一管理工具

用法:
    python -m src.cli <service> <action> [options]
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

BASE_DIR = Path(__file__).resolve().parent.parent
STATE_FILE = BASE_DIR / ".cli_state.yaml"

SERVICES = {
    "server": {
        "script": "src.server",
        "default_cfg": BASE_DIR / "configs" / "server.yaml",
        "default_pid": BASE_DIR / "logs" / "server.pid",
        "default_log": BASE_DIR / "logs" / "server.log",
        "description": "Chat Log Viewer (src/server.py)",
    },
    "sync": {
        "script": "src.sync_sessions",
        "default_cfg": BASE_DIR / "configs" / "sync_config.yaml",
        "default_pid": BASE_DIR / "logs" / "sync.pid",
        "default_log": BASE_DIR / "logs" / "sync.log",
        "description": "增量同步守护进程 (src/sync_sessions.py)",
    },
    "client": {
        "script": "src.client",
        "default_cfg": BASE_DIR / "configs" / "client.yaml",
        "default_pid": BASE_DIR / "logs" / "client.pid",
        "default_log": BASE_DIR / "logs" / "client.log",
        "description": "OBS 下载客户端 (src/client.py)",
    },
}


def _now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _relpath(path: Path) -> str:
    try:
        return os.path.relpath(path.resolve(), BASE_DIR)
    except ValueError:
        return str(path.resolve())


def _display_path(path: Path | str) -> str:
    path_obj = Path(path)
    try:
        return os.path.relpath(path_obj.resolve(), BASE_DIR)
    except Exception:
        return str(path_obj)


def _slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", text.strip())
    slug = slug.replace("/", "-").replace("\\", "-").replace(".", "-")
    slug = re.sub(r"-+", "-", slug).strip("-_")
    return slug or "default"


def _service_state(state: dict, service: str) -> dict:
    services = state.setdefault("services", {})
    svc = services.setdefault(service, {})
    svc.setdefault("default_instance", None)
    svc.setdefault("instances", {})
    return svc


def _upgrade_legacy_state(state: dict) -> dict:
    if "services" not in state:
        upgraded = {"services": {}}
        for service in SERVICES:
            legacy = state.get(service)
            if isinstance(legacy, dict):
                svc = upgraded["services"].setdefault(service, {"default_instance": None, "instances": {}})
                cfg = legacy.get("config")
                if cfg:
                    cfg_path = Path(cfg)
                    try:
                        cfg_key = _relpath(cfg_path)
                    except Exception:
                        cfg_key = str(cfg)
                    svc["default_instance"] = cfg_key
                    svc["instances"][cfg_key] = {
                        "config": str(Path(cfg).resolve()),
                        "name": _slugify(cfg_key),
                        "saved_at": legacy.get("updated_at"),
                    }
        return upgraded

    for service in SERVICES:
        _service_state(state, service)
    return state


def _load_state() -> dict:
    if not STATE_FILE.exists():
        return _upgrade_legacy_state({})
    try:
        with open(STATE_FILE, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except Exception:
        raw = {}
    return _upgrade_legacy_state(raw if isinstance(raw, dict) else {})


def _save_state(state: dict):
    state["updated_at"] = _now_text()
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        yaml.safe_dump(state, f, sort_keys=True, allow_unicode=True)


def _load_config(config_path: Path) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_config_path(config_arg: Optional[str], service: str) -> Path:
    if config_arg:
        path = Path(config_arg)
        if not path.is_absolute():
            path = BASE_DIR / path
        return path.resolve()
    return SERVICES[service]["default_cfg"].resolve()


def _config_key(config_path: Path) -> str:
    return _relpath(config_path)


def _instance_slug(instance: dict, config_key: str) -> str:
    name = (instance or {}).get("name")
    if name:
        return _slugify(str(name))
    return _slugify(config_key)


def _default_dynamic_token(service: str, cfg: dict) -> str:
    if service == "server":
        return f"port{cfg.get('port', 8080)}"
    if service == "sync":
        mode = cfg.get("mode")
        if not mode:
            has_src = bool(cfg.get("src"))
            has_session_dir = bool(cfg.get("session_dir"))
            if has_src and has_session_dir:
                mode = "export"
            elif has_src:
                mode = "raw"
            else:
                mode = "upload-only"
        interval = cfg.get("interval_seconds", 3600)
        return f"{mode}-interval{interval}"
    interval = cfg.get("interval")
    return f"interval{interval}" if interval else "once"


def _resolve_log_file(service: str, svc: dict, cfg: dict, instance_slug: str) -> Path:
    default_log = svc["default_log"]
    dynamic_token = _default_dynamic_token(service, cfg)
    dynamic_default = default_log.with_name(
        f"{default_log.stem}-{instance_slug}-{dynamic_token}{default_log.suffix}"
    )

    configured = cfg.get("log_file")
    if not configured:
        return dynamic_default

    configured_path = Path(configured)
    configured_text = str(configured_path)
    default_text = str(default_log)
    default_rel_text = str(default_log.relative_to(BASE_DIR))

    if configured_text in (default_text, default_rel_text):
        return dynamic_default

    configured_abs = configured_path if configured_path.is_absolute() else (BASE_DIR / configured_path)
    if configured_abs.suffix:
        return configured_abs.resolve()
    return (configured_abs / dynamic_default.name).resolve()


def _resolve_pid_file(service: str, svc: dict, cfg: dict, instance_slug: str) -> Path:
    default_pid = svc["default_pid"]
    dynamic_token = _default_dynamic_token(service, cfg)
    return default_pid.with_name(
        f"{default_pid.stem}-{instance_slug}-{dynamic_token}{default_pid.suffix}"
    )


def _read_pid(pid_file: Path) -> Optional[int]:
    if pid_file.exists():
        try:
            return int(pid_file.read_text(encoding="utf-8").strip())
        except (ValueError, OSError):
            pass
    return None


def _write_pid(pid_file: Path, pid: int):
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(f"{pid}\n", encoding="utf-8")


def _remove_pid(pid_file: Path):
    pid_file.unlink(missing_ok=True)


def _is_running(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


def _tail_lines(path: Path, n: int) -> List[str]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.readlines()[-n:]


def _build_server_cmd(cfg: dict) -> List[str]:
    cmd = [sys.executable, "-m", "src.server"]
    mode = cfg.get("mode", "scan")
    if mode == "scan":
        for d in cfg.get("dir", []):
            cmd += ["--dir", d]
        for d in cfg.get("dirs", []):
            cmd += ["--dirs", d]
    elif mode == "session":
        for d in cfg.get("session_dir", []):
            cmd += ["--session-dir", d]
        for d in cfg.get("session_dirs", []):
            cmd += ["--session-dirs", d]
    else:
        sys.exit(f"[error] server 未知 mode: {mode}，应为 scan 或 session")
    cmd += ["--host", str(cfg.get("host", "0.0.0.0"))]
    cmd += ["--port", str(cfg.get("port", 8080))]
    return cmd


def _build_sync_cmd(cfg: dict, once: bool = False) -> List[str]:
    cmd = [sys.executable, "-m", "src.sync_sessions"]
    if cfg.get("src"):
        cmd += ["--src", cfg["src"]]
    if cfg.get("session_dir"):
        cmd += ["--session-dir", cfg["session_dir"]]
    if cfg.get("obs_raw"):
        cmd += ["--obs-raw", cfg["obs_raw"]]
    if cfg.get("obs_session"):
        cmd += ["--obs-session", cfg["obs_session"]]
    if cfg.get("upload_erase"):
        cmd += ["--upload-erase"]
    if cfg.get("upload_script"):
        cmd += ["--upload-script", cfg["upload_script"]]
    if cfg.get("interval_seconds"):
        cmd += ["--interval", str(cfg["interval_seconds"])]
    if cfg.get("upload_workers"):
        cmd += ["--workers", str(cfg["upload_workers"])]
    if once:
        cmd += ["--once"]
    return cmd


def _build_client_cmd(cfg: dict) -> List[str]:
    cmd = [sys.executable, "-m", "src.client"]
    if cfg.get("mode"):
        cmd += ["--mode", cfg["mode"]]
    if cfg.get("obs_path"):
        cmd += ["--obs-path", cfg["obs_path"]]
    if cfg.get("output"):
        cmd += ["--output", cfg["output"]]
    if cfg.get("base_output"):
        cmd += ["--base-output", cfg["base_output"]]
    if cfg.get("download_script"):
        cmd += ["--download-script", cfg["download_script"]]
    if cfg.get("workers"):
        cmd += ["--workers", str(cfg["workers"])]
    if cfg.get("interval"):
        cmd += ["--interval", str(cfg["interval"])]
    return cmd


_CMD_BUILDERS = {
    "server": _build_server_cmd,
    "sync": lambda cfg, **kw: _build_sync_cmd(cfg, once=kw.get("once", False)),
    "client": _build_client_cmd,
}


def _resolve_instance_by_name(service: str, name: str, state: dict) -> Tuple[str, dict]:
    svc_state = _service_state(state, service)
    for config_key, instance in (svc_state.get("instances") or {}).items():
        if instance.get("name") == name:
            return config_key, instance
    sys.exit(f"[error] {service} 实例不存在: {name}")


def _resolve_target_instance(service: str, args, state: dict, require_existing: bool = False) -> Tuple[str, dict, Path]:
    svc_state = _service_state(state, service)
    instances = svc_state.setdefault("instances", {})

    if getattr(args, "name", None):
        config_key, instance = _resolve_instance_by_name(service, args.name, state)
        config = instance.get("config")
        if not config:
            sys.exit(f"[error] {service} 实例 {args.name} 缺少配置路径")
        return config_key, instance, Path(config)

    config_arg = getattr(args, "config", None)
    if config_arg:
        config_path = _resolve_config_path(config_arg, service)
        config_key = _config_key(config_path)
        instance = instances.get(config_key)
        if require_existing and instance is None and not config_path.exists():
            sys.exit(f"[error] 配置文件不存在: {config_path}")
        if instance is None:
            instance = {}
        return config_key, instance, config_path

    default_key = svc_state.get("default_instance")
    if default_key:
        instance = instances.get(default_key)
        if instance and instance.get("config"):
            return default_key, instance, Path(instance["config"])

    if require_existing:
        sys.exit(f"[error] {service} 未设置默认实例，请传 --config 或 --name")

    config_path = SERVICES[service]["default_cfg"].resolve()
    return _config_key(config_path), instances.get(_config_key(config_path), {}), config_path


def _instance_runtime(service: str, config_path: Path, instance: dict) -> Tuple[dict, str, Path, Path]:
    if not config_path.exists():
        sys.exit(f"[error] 配置文件不存在: {config_path}")
    cfg = _load_config(config_path)
    instance_slug = _instance_slug(instance, _config_key(config_path))
    svc = SERVICES[service]
    log_file = _resolve_log_file(service, svc, cfg, instance_slug)
    pid_file = _resolve_pid_file(service, svc, cfg, instance_slug)

    recorded_pid = instance.get("pid_file")
    if recorded_pid:
        pid_file = (BASE_DIR / recorded_pid).resolve()

    recorded_log = instance.get("log_file")
    if recorded_log:
        log_file = (BASE_DIR / recorded_log).resolve()

    return cfg, instance_slug, pid_file, log_file


def _instance_record(
    service: str,
    state: dict,
    config_key: str,
    config_path: Path,
    instance_name: Optional[str] = None,
) -> dict:
    svc_state = _service_state(state, service)
    instances = svc_state.setdefault("instances", {})
    instance = instances.setdefault(config_key, {})
    instance.setdefault("config", str(config_path.resolve()))
    if instance_name:
        instance["name"] = instance_name
    elif not instance.get("name"):
        instance["name"] = _slugify(config_key)
    return instance


def _conflicting_output(service: str, config_key: str, cfg: dict, state: dict) -> Optional[str]:
    svc_state = _service_state(state, service)
    target = None
    if service == "sync":
        target = cfg.get("session_dir") or cfg.get("src")
    elif service == "client":
        target = cfg.get("output")
    if not target:
        return None

    target_path = str(Path(target).expanduser().resolve())
    for other_key, instance in (svc_state.get("instances") or {}).items():
        if other_key == config_key:
            continue
        other_cfg_path = instance.get("config")
        pid_file = instance.get("pid_file")
        pid = _read_pid(BASE_DIR / pid_file) if pid_file else instance.get("pid")
        if not other_cfg_path or not _is_running(pid):
            continue
        try:
            other_cfg = _load_config(Path(other_cfg_path))
        except Exception:
            continue
        other_target = None
        if service == "sync":
            other_target = other_cfg.get("session_dir") or other_cfg.get("src")
        elif service == "client":
            other_target = other_cfg.get("output")
        if not other_target:
            continue
        if str(Path(other_target).expanduser().resolve()) == target_path:
            label = instance.get("name") or other_key
            return f"{label} -> {target_path}"
    return None


def cmd_start(service: str, args):
    state = _load_state()
    config_key, _existing_instance, config_path = _resolve_target_instance(service, args, state, require_existing=False)
    instance = _instance_record(service, state, config_key, config_path, getattr(args, "name", None))
    cfg, instance_slug, pid_file, log_file = _instance_runtime(service, config_path, instance)

    conflict = _conflicting_output(service, config_key, cfg, state)
    if conflict:
        sys.exit(f"[error] {service} 输出目录已被运行中的实例占用: {conflict}")

    pid = _read_pid(pid_file)
    if pid and _is_running(pid):
        print(f"[info] {service}/{instance['name']} 已在运行 (PID {pid})")
        print(f"[info] log: {_display_path(log_file)}")
        return

    extra = {}
    if service == "sync" and getattr(args, "once", False):
        extra["once"] = True

    cmd = _CMD_BUILDERS[service](cfg, **extra)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "ab") as log_fd:
        proc = subprocess.Popen(
            cmd,
            stdout=log_fd,
            stderr=log_fd,
            cwd=str(BASE_DIR),
            start_new_session=True,
        )

    _write_pid(pid_file, proc.pid)
    time.sleep(0.5)
    if not _is_running(proc.pid):
        sys.exit(f"[error] {service}/{instance['name']} 启动失败，请检查日志: {log_file}")

    instance.update({
        "config": str(config_path.resolve()),
        "name": instance["name"],
        "pid": proc.pid,
        "pid_file": _relpath(pid_file),
        "log_file": _relpath(log_file),
        "instance_slug": instance_slug,
        "started_at": _now_text(),
        "stopped_at": None,
    })
    if service == "server":
        instance["host"] = str(cfg.get("host", "0.0.0.0"))
        instance["port"] = int(cfg.get("port", 8080))
    _service_state(state, service)["default_instance"] = config_key
    _save_state(state)

    addr = ""
    if service == "server":
        addr = f" http://{cfg.get('host', '0.0.0.0')}:{cfg.get('port', 8080)}"
    print(f"[info] {service}/{instance['name']} 已启动 (PID {proc.pid}){addr}")
    print(f"[info] config: {_display_path(config_path)}")
    print(f"[info] log: {_display_path(log_file)}")


def cmd_stop(service: str, args):
    state = _load_state()
    config_key, instance, config_path = _resolve_target_instance(service, args, state, require_existing=True)
    instance = _instance_record(service, state, config_key, config_path, getattr(args, "name", None))
    cfg, _instance_slug_text, pid_file, _log_file = _instance_runtime(service, config_path, instance)

    pid = _read_pid(pid_file) or instance.get("pid")
    if pid is None:
        print(f"[info] {service}/{instance.get('name') or config_key} 未运行（无 PID 文件）")
        return
    if not _is_running(pid):
        print(f"[info] PID {pid} 进程不存在，清理 PID 文件")
        _remove_pid(pid_file)
        instance["pid"] = None
        instance["stopped_at"] = _now_text()
        _save_state(state)
        return

    os.kill(pid, signal.SIGTERM)
    for _ in range(20):
        time.sleep(0.5)
        if not _is_running(pid):
            break
    else:
        os.kill(pid, signal.SIGKILL)
        print(f"[warn] 进程 {pid} 未响应 SIGTERM，已强制 SIGKILL")
    _remove_pid(pid_file)
    instance["pid"] = None
    instance["stopped_at"] = _now_text()
    if service == "server":
        instance["host"] = str(cfg.get("host", "0.0.0.0"))
        instance["port"] = int(cfg.get("port", 8080))
    _save_state(state)
    print(f"[info] {service}/{instance.get('name') or config_key} 已停止 (PID {pid})")


def cmd_restart(service: str, args):
    cmd_stop(service, args)
    time.sleep(1)
    cmd_start(service, args)


def cmd_status(service: str, args):
    state = _load_state()
    config_key, instance, config_path = _resolve_target_instance(service, args, state, require_existing=True)
    instance = _instance_record(service, state, config_key, config_path, getattr(args, "name", None))
    cfg, _instance_slug_text, pid_file, log_file = _instance_runtime(service, config_path, instance)

    pid = _read_pid(pid_file) or instance.get("pid")
    name = instance.get("name") or config_key
    if pid is None:
        print(f"[status] {service}/{name}: stopped（无 PID 文件）")
        return
    if _is_running(pid):
        extra = ""
        if service == "server":
            extra = f" host={cfg.get('host', '0.0.0.0')} port={cfg.get('port', 8080)}"
        print(f"[status] {service}/{name}: running (PID {pid}){extra} log={_display_path(log_file)}")
    else:
        print(f"[status] {service}/{name}: stopped（PID {pid} 不存在，清理 PID 文件）")
        _remove_pid(pid_file)
        instance["pid"] = None
        instance["stopped_at"] = _now_text()
        _save_state(state)


def cmd_logs(service: str, args):
    state = _load_state()
    config_key, instance, config_path = _resolve_target_instance(service, args, state, require_existing=True)
    instance = _instance_record(service, state, config_key, config_path, getattr(args, "name", None))
    _cfg, _instance_slug_text, _pid_file, log_file = _instance_runtime(service, config_path, instance)
    if not log_file.exists():
        print(f"[info] 日志文件不存在: {_display_path(log_file)}")
        return
    if getattr(args, "follow", False):
        try:
            subprocess.run(["tail", "-n", str(args.lines), "-f", str(log_file)], check=False)
        except KeyboardInterrupt:
            pass
        return
    sys.stdout.writelines(_tail_lines(log_file, getattr(args, "lines", 50)))


def cmd_config(service: str, args):
    state = _load_state()
    svc_state = _service_state(state, service)

    if getattr(args, "clear", False):
        if getattr(args, "name", None):
            config_key, _instance = _resolve_instance_by_name(service, args.name, state)
            svc_state["instances"].pop(config_key, None)
            if svc_state.get("default_instance") == config_key:
                svc_state["default_instance"] = None
            _save_state(state)
            print(f"[config] {service}: 已移除实例 {args.name}")
            return
        svc_state["default_instance"] = None
        _save_state(state)
        print(f"[config] {service}: 已清除默认实例")
        return

    path_arg = getattr(args, "path", None)
    if path_arg:
        config_path = _resolve_config_path(path_arg, service)
        if not config_path.exists():
            sys.exit(f"[error] 配置文件不存在: {config_path}")
        config_key = _config_key(config_path)
        instance = _instance_record(service, state, config_key, config_path, getattr(args, "name", None))
        recorded_pid_file = instance.get("pid_file")
        recorded_pid = _read_pid(BASE_DIR / recorded_pid_file) if recorded_pid_file else instance.get("pid")
        if not _is_running(recorded_pid):
            instance.pop("pid_file", None)
            instance.pop("log_file", None)
            instance.pop("instance_slug", None)
            instance["pid"] = None
        instance["saved_at"] = _now_text()
        svc_state["default_instance"] = config_key
        _save_state(state)
        print(f"[config] {service}: 默认实例 -> {instance['name']}")
        print(f"[config] {service}: 配置文件 -> {_display_path(config_path)}")
        return

    default_key = svc_state.get("default_instance")
    print(f"[config] {service}: 默认实例 {default_key or '(无)'}")
    for config_key, instance in sorted((svc_state.get("instances") or {}).items()):
        marker = "*" if config_key == default_key else " "
        config_text = instance.get("config") or config_key
        print(f"{marker} name={instance.get('name') or '-'} config={_display_path(config_text)}")


def cmd_list(service_name: Optional[str], args):
    state = _load_state()
    if service_name and service_name in SERVICES:
        services = [service_name]
    elif getattr(args, "service_filter", None):
        services = [args.service_filter]
    else:
        services = list(SERVICES.keys())
    printed = False
    for service in services:
        svc_state = _service_state(state, service)
        default_key = svc_state.get("default_instance")
        for config_key, instance in sorted((svc_state.get("instances") or {}).items()):
            config = instance.get("config")
            if not config:
                continue
            try:
                cfg, _instance_slug_text, pid_file, log_file = _instance_runtime(service, Path(config), instance)
            except SystemExit:
                pid_file = BASE_DIR / instance.get("pid_file", "")
                log_file = BASE_DIR / instance.get("log_file", "")
                cfg = {}
            pid = _read_pid(pid_file) or instance.get("pid")
            running = _is_running(pid)
            if getattr(args, "running", False) and not running:
                continue
            marker = "*" if config_key == default_key else " "
            name = instance.get("name") or config_key
            extras = []
            if service == "server":
                extras.append(f"port={cfg.get('port', instance.get('port', '-'))}")
            print(
                f"{marker} {service}/{name}: {'running' if running else 'stopped'} "
                f"pid={pid or '-'} {' '.join(extras)} config={_display_path(config)} log={_display_path(log_file)}"
            )
            printed = True
    if not printed:
        print("[list] 无匹配实例")


def _add_common_instance_args(parser, include_config: bool = True):
    if include_config:
        parser.add_argument("--config", "-c", default=None, help="配置文件路径")
    parser.add_argument("--name", default=None, help="实例名（用于快速定位）")


def _add_service_subparser(sub, service: str, svc: dict):
    svc_p = sub.add_parser(service, help=svc["description"])
    svc_sub = svc_p.add_subparsers(dest="action", required=True)

    p = svc_sub.add_parser("start", help=f"启动 {service}")
    _add_common_instance_args(p)
    if service == "sync":
        p.add_argument("--once", action="store_true", help="单次运行后退出（测试/cron 模式）")

    p = svc_sub.add_parser("stop", help=f"停止 {service}")
    _add_common_instance_args(p)

    p = svc_sub.add_parser("restart", help=f"重启 {service}")
    _add_common_instance_args(p)
    if service == "sync":
        p.add_argument("--once", action="store_true", help="单次运行后退出")

    p = svc_sub.add_parser("status", help=f"查看 {service} 运行状态")
    _add_common_instance_args(p)

    p = svc_sub.add_parser("logs", help="查看最近日志")
    p.add_argument("--lines", "-n", type=int, default=50, help="显示最后 N 行 (默认 50)")
    p.add_argument("--follow", "-f", action="store_true", help="持续跟随日志输出")
    _add_common_instance_args(p)

    p = svc_sub.add_parser("config", help=f"设置或查看 {service} 默认实例")
    p.add_argument("path", nargs="?", help="要保存的配置文件路径")
    p.add_argument("--name", default=None, help="为实例设置别名")
    p.add_argument("--clear", action="store_true", help="清除默认实例；配合 --name 时删除该实例记录")

    p = svc_sub.add_parser("list", help=f"列出 {service} 的所有实例")
    p.add_argument("--running", action="store_true", help="仅显示运行中的实例")


def main():
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description="Chat Log Viewer 统一管理工具",
    )
    sub = parser.add_subparsers(dest="service", required=True)

    for svc_name, svc_info in SERVICES.items():
        _add_service_subparser(sub, svc_name, svc_info)

    p = sub.add_parser("list", help="列出所有服务实例")
    p.add_argument("--service", dest="service_filter", choices=list(SERVICES.keys()), help="仅列出指定服务")
    p.add_argument("--running", action="store_true", help="仅显示运行中的实例")
    p.set_defaults(action="list")

    args = parser.parse_args()

    if args.service == "list":
        cmd_list(None, args)
        return

    dispatch = {
        "start": cmd_start,
        "stop": cmd_stop,
        "restart": cmd_restart,
        "status": cmd_status,
        "logs": cmd_logs,
        "config": cmd_config,
        "list": cmd_list,
    }
    dispatch[args.action](args.service, args)


if __name__ == "__main__":
    main()
