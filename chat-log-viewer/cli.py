#!/usr/bin/env python3
"""
cli.py — Chat Log Viewer 统一管理工具

用法:
    python cli.py <service> <action> [options]

支持的服务:
    server   — 管理 server.py (日志浏览服务)
    sync     — 管理 sync_sessions.py (增量同步守护进程)

示例:
    python cli.py server start [--config configs/server.yaml]
    python cli.py server stop
    python cli.py server restart [--config configs/server.yaml]
    python cli.py server status
    python cli.py server logs [--lines 50]

    python cli.py sync start [--config configs/sync_config.yaml] [--once]
    python cli.py sync stop
    python cli.py sync restart [--config configs/sync_config.yaml]
    python cli.py sync status
    python cli.py sync logs [--lines 50]
"""

import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

import yaml

# ---------------------------------------------------------------------------
# 服务配置表
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent

# 每个服务的元信息
SERVICES = {
    "server": {
        "script":      "server.py",
        "default_cfg": BASE_DIR / "configs" / "server.yaml",
        "pid_file":    BASE_DIR / "logs" / "server.pid",
        "default_log": BASE_DIR / "logs" / "server.log",
        "description": "Chat Log Viewer (server.py)",
    },
    "sync": {
        "script":      "sync_sessions.py",
        "default_cfg": BASE_DIR / "configs" / "sync_config.yaml",
        "pid_file":    BASE_DIR / "logs" / "sync.pid",
        "default_log": BASE_DIR / "logs" / "sync.log",
        "description": "增量同步守护进程 (sync_sessions.py)",
    },
}


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def _setup_logging(log_file: Path, log_level: str = "INFO"):
    log_file.parent.mkdir(parents=True, exist_ok=True)
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )


def _resolve_log_file(service: str, svc: dict, cfg: dict) -> Path:
    default_log = svc["default_log"]
    if service == "server":
        port = cfg.get("port", 8080)
        default_log = default_log.with_name(
            f"{default_log.stem}-port{port}{default_log.suffix}"
        )
    elif service == "sync":
        interval = cfg.get("interval_seconds", 3600)
        default_log = default_log.with_name(
            f"{default_log.stem}-interval{interval}{default_log.suffix}"
        )

    configured = cfg.get("log_file")
    if not configured:
        return default_log

    configured_path = Path(configured)
    configured_text = str(configured_path)
    default_text = str(svc["default_log"])
    default_rel_text = str(svc["default_log"].relative_to(BASE_DIR))

    # 如果配置里只是沿用了通用默认值 logs/server.log / logs/sync.log，
    # 则仍然使用按端口/间隔区分后的动态日志文件名。
    if configured_text in (default_text, default_rel_text):
        return default_log

    return BASE_DIR / configured_path


def _resolve_pid_file(service: str, svc: dict, cfg: dict) -> Path:
    default_pid = svc["pid_file"]
    if service == "server":
        port = cfg.get("port", 8080)
        return default_pid.with_name(f"{default_pid.stem}-port{port}{default_pid.suffix}")
    return default_pid


def _read_pid(pid_file: Path) -> Optional[int]:
    if pid_file.exists():
        try:
            return int(pid_file.read_text().strip())
        except (ValueError, OSError):
            pass
    return None


def _write_pid(pid_file: Path, pid: int):
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(pid))


def _remove_pid(pid_file: Path):
    pid_file.unlink(missing_ok=True)


def _is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


# ---------------------------------------------------------------------------
# 构建启动命令
# ---------------------------------------------------------------------------

def _build_server_cmd(cfg: dict) -> List[str]:
    cmd = [sys.executable, str(BASE_DIR / "server.py")]
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
    cmd = [sys.executable, str(BASE_DIR / "sync_sessions.py")]
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
        cmd += ["--upload-workers", str(cfg["upload_workers"])]
    if once:
        cmd += ["--once"]
    return cmd


_CMD_BUILDERS = {
    "server": _build_server_cmd,
    "sync":   lambda cfg, **kw: _build_sync_cmd(cfg, once=kw.get("once", False)),
}


# ---------------------------------------------------------------------------
# 通用 start / stop / restart / status / logs
# ---------------------------------------------------------------------------

def cmd_start(service: str, args):
    svc = SERVICES[service]
    config_path = Path(args.config)
    if not config_path.exists():
        sys.exit(f"[error] 配置文件不存在: {config_path}")

    cfg = _load_config(config_path)
    log_file = _resolve_log_file(service, svc, cfg)
    pid_file = _resolve_pid_file(service, svc, cfg)
    _setup_logging(log_file, cfg.get("log_level", "INFO"))
    logger = logging.getLogger("cli")

    pid = _read_pid(pid_file)
    if pid and _is_running(pid):
        logger.info(f"{service} 已在运行 (PID {pid})")
        return

    extra = {}
    if service == "sync" and getattr(args, "once", False):
        extra["once"] = True

    cmd = _CMD_BUILDERS[service](cfg, **extra)
    log_fd = open(log_file, "a")
    proc = subprocess.Popen(
        cmd,
        stdout=log_fd,
        stderr=log_fd,
        cwd=str(BASE_DIR),
        start_new_session=True,
    )
    _write_pid(pid_file, proc.pid)
    port_info = f":{cfg.get('port',8080)}" if service == "server" else ""
    logger.info(f"{service} 已启动 (PID {proc.pid}){port_info}，日志: {log_file}")
    if service == "server":
        logger.info(f"访问地址: http://{cfg.get('host','0.0.0.0')}:{cfg.get('port',8080)}")


def cmd_stop(service: str, _args):
    svc = SERVICES[service]
    config_path = Path(getattr(_args, "config", svc["default_cfg"]))
    cfg = _load_config(config_path) if config_path.exists() else {}
    pid_file = _resolve_pid_file(service, svc, cfg)
    pid = _read_pid(pid_file)
    if pid is None:
        print(f"[info] {service} 未运行（无 PID 文件）")
        return
    if not _is_running(pid):
        print(f"[info] PID {pid} 进程不存在，清理 PID 文件")
        _remove_pid(pid_file)
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
    print(f"[info] {service} 已停止 (PID {pid})")


def cmd_restart(service: str, args):
    cmd_stop(service, args)
    time.sleep(1)
    cmd_start(service, args)


def cmd_status(service: str, _args):
    svc = SERVICES[service]
    config_path = Path(getattr(_args, "config", svc["default_cfg"]))
    cfg = _load_config(config_path) if config_path.exists() else {}
    pid_file = _resolve_pid_file(service, svc, cfg)
    pid = _read_pid(pid_file)
    if pid is None:
        print(f"[status] {service}: stopped（无 PID 文件）")
        return
    if _is_running(pid):
        print(f"[status] {service}: running (PID {pid})")
    else:
        print(f"[status] {service}: stopped（PID {pid} 不存在，清理 PID 文件）")
        _remove_pid(pid_file)


def cmd_logs(service: str, args):
    svc = SERVICES[service]
    log_file = svc["default_log"]
    config_path = Path(args.config) if getattr(args, "config", None) else svc["default_cfg"]
    if config_path.exists():
        cfg = _load_config(config_path)
        log_file = _resolve_log_file(service, svc, cfg)
    if not log_file.exists():
        print(f"[info] 日志文件不存在: {log_file}")
        return
    lines = getattr(args, "lines", 50)
    result = subprocess.run(
        ["tail", "-n", str(lines), str(log_file)],
        capture_output=True, text=True,
    )
    print(result.stdout, end="")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def _add_service_subparser(sub, service: str, svc: dict):
    """为指定服务注册 start/stop/restart/status/logs 子命令。"""
    svc_p = sub.add_parser(service, help=svc["description"])
    svc_sub = svc_p.add_subparsers(dest="action", required=True)

    # start
    p = svc_sub.add_parser("start", help=f"启动 {service}")
    p.add_argument("--config", "-c", default=str(svc["default_cfg"]),
                   help=f"配置文件路径 (默认: {svc['default_cfg'].relative_to(BASE_DIR)})")
    if service == "sync":
        p.add_argument("--once", action="store_true", help="单次运行后退出（测试/cron 模式）")

    # stop
    p = svc_sub.add_parser("stop", help=f"停止 {service}")
    p.add_argument("--config", "-c", default=str(svc["default_cfg"]),
                   help=f"配置文件路径 (默认: {svc['default_cfg'].relative_to(BASE_DIR)})")

    # restart
    p = svc_sub.add_parser("restart", help=f"重启 {service}")
    p.add_argument("--config", "-c", default=str(svc["default_cfg"]),
                   help=f"配置文件路径 (默认: {svc['default_cfg'].relative_to(BASE_DIR)})")
    if service == "sync":
        p.add_argument("--once", action="store_true", help="单次运行后退出")

    # status
    p = svc_sub.add_parser("status", help=f"查看 {service} 运行状态")
    p.add_argument("--config", "-c", default=str(svc["default_cfg"]),
                   help=f"配置文件路径 (默认: {svc['default_cfg'].relative_to(BASE_DIR)})")

    # logs
    p = svc_sub.add_parser("logs", help="查看最近日志")
    p.add_argument("--lines", "-n", type=int, default=50, help="显示最后 N 行 (默认 50)")
    p.add_argument("--config", "-c", default=str(svc["default_cfg"]),
                   help="配置文件路径（用于读取 log_file 路径）")


def main():
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description="Chat Log Viewer 统一管理工具",
    )
    sub = parser.add_subparsers(dest="service", required=True)

    for svc_name, svc_info in SERVICES.items():
        _add_service_subparser(sub, svc_name, svc_info)

    args = parser.parse_args()

    dispatch = {
        "start":   cmd_start,
        "stop":    cmd_stop,
        "restart": cmd_restart,
        "status":  cmd_status,
        "logs":    cmd_logs,
    }
    handler = dispatch.get(args.action)
    if handler:
        handler(args.service, args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
