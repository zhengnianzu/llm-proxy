#!/usr/bin/env python3
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
import re

BASE_DIR = Path(__file__).resolve().parent
APP_FILE = BASE_DIR / "app.py"
LOG_DIR = BASE_DIR / "logs"
STATE_FILE = BASE_DIR / ".cli_state.yaml"
DEFAULT_ENV = ".env"

try:
    import yaml  # type: ignore
except Exception:
    yaml = None


def eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


def load_state() -> dict:
    if not STATE_FILE.exists():
        return {"source_env": DEFAULT_ENV, "services": {}}
    if yaml is not None:
        try:
            with STATE_FILE.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            if isinstance(data, dict):
                data.setdefault("source_env", DEFAULT_ENV)
                data.setdefault("services", {})
                return data
        except Exception:
            pass

    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data.setdefault("source_env", DEFAULT_ENV)
            data.setdefault("services", {})
            return data
    except Exception:
        pass

    state = {}
    for line in STATE_FILE.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#") or ":" not in raw:
            continue
        key, value = raw.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value == "":
            state[key] = ""
            continue
        if value in ("null", "~"):
            state[key] = None
            continue
        if value in ("true", "false"):
            state[key] = value == "true"
            continue
        if value.startswith('"') or value.startswith("[") or value.startswith("{"):
            try:
                state[key] = json.loads(value)
                continue
            except Exception:
                pass
        if value.isdigit():
            state[key] = int(value)
            continue
        state[key] = value
    state.setdefault("source_env", DEFAULT_ENV)
    state.setdefault("services", {})
    return state


def save_state(state: dict) -> None:
    if yaml is not None:
        with STATE_FILE.open("w", encoding="utf-8") as f:
            yaml.safe_dump(state, f, sort_keys=True, allow_unicode=True)
        return

    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_env_file(env_path: Path) -> dict:
    values = {}
    if not env_path.exists():
        raise FileNotFoundError(f"env file not found: {env_path}")
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        line = line.split(" #", 1)[0].split("\t#", 1)[0]
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or not (key[0].isalpha() or key[0] == "_"):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        values[key] = value
    return values


def get_api_key_suffix(env_values: dict) -> str:
    raw = (env_values.get("UPSTREAM_API_KEY") or "").strip()
    if not raw:
        return ""
    first = ""
    for part in raw.split(","):
        part = part.strip()
        if part:
            first = part
            break
    if not first:
        return ""
    return first[-4:]


def resolve_env_path(source_env: Optional[str]) -> Path:
    env_name = source_env or DEFAULT_ENV
    env_path = Path(env_name)
    if not env_path.is_absolute():
        env_path = BASE_DIR / env_name
    return env_path.resolve()


def state_runtime(state: dict) -> tuple[Path, dict, str, int, Path, Path]:
    source_env = state.get("source_env") or DEFAULT_ENV
    env_path = resolve_env_path(source_env)
    env_values = parse_env_file(env_path)
    host = env_values.get("PROXY_HOST", "127.0.0.1")
    port = int(env_values.get("PROXY_PORT", "4000"))
    pid_file = LOG_DIR / f"app-port{port}.pid"
    log_file = LOG_DIR / f"app-port{port}.log"
    return env_path, env_values, host, port, pid_file, log_file


def get_service_key(env_path: Path) -> str:
    try:
        return os.path.relpath(env_path, BASE_DIR)
    except ValueError:
        return str(env_path)


def get_service_slug(service_key: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", service_key)
    slug = slug.replace("/", "-").replace("\\", "-").replace(".", "-")
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug or "default"


def get_selected_env(args: argparse.Namespace, state: dict) -> str:
    return args.env_file or state.get("source_env") or DEFAULT_ENV


def is_pid_running(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def read_pid(path: Path) -> Optional[int]:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def tail_lines(path: Path, n: int) -> list[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    return lines[-n:]


def cmd_config(args: argparse.Namespace) -> int:
    state = load_state()
    if args.env_file:
        env_path = resolve_env_path(args.env_file)
        if not env_path.exists():
            eprint(f"[app] env file not found: {env_path}")
            return 1
        state["source_env"] = get_service_key(env_path)
        state["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        save_state(state)
        print(f"[app] source_env -> {state['source_env']}")
        print(f"[app] state saved -> {STATE_FILE}")
        return 0

    env_path = resolve_env_path(state.get("source_env"))
    print(f"[app] source_env: {state.get('source_env', DEFAULT_ENV)}")
    print(f"[app] resolved_env: {env_path}")
    services = state.get("services") or {}
    if services:
        print("[app] services:")
        for key, svc in services.items():
            port = svc.get("port", "-")
            pid = svc.get("pid", "-")
            print(f"  - {key}: port={port} pid={pid}")
    print(f"[app] state_file: {STATE_FILE}")
    return 0


def cmd_start(args: argparse.Namespace) -> int:
    state = load_state()
    state["source_env"] = get_selected_env(args, state)
    env_path, env_values, host, port, pid_file, log_file = state_runtime(state)
    api_key_suffix = get_api_key_suffix(env_values)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    service_key = get_service_key(env_path)
    service_slug = get_service_slug(service_key)
    services = state.setdefault("services", {})
    service = services.setdefault(service_key, {})
    pid_file = LOG_DIR / f"app-{service_slug}-port{port}.pid"
    log_file = LOG_DIR / f"app-{service_slug}-port{port}.log"

    pid = read_pid(pid_file)
    if is_pid_running(pid):
        print(f"[app] already running: pid={pid} host={host} port={port}")
        print(f"[app] log -> {log_file}")
        return 0

    child_env = os.environ.copy()
    child_env.update(env_values)
    child_env["ENV_FILE"] = str(env_path)
    child_env["LOG_TASK_TAG"] = service_slug

    with log_file.open("ab") as log_fp:
        proc = subprocess.Popen(
            [sys.executable, str(APP_FILE)],
            cwd=str(BASE_DIR),
            env=child_env,
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    pid_file.write_text(f"{proc.pid}\n", encoding="utf-8")
    time.sleep(1)
    if not is_pid_running(proc.pid):
        eprint(f"[app] failed to start, check log: {log_file}")
        return 1

    service.update({
        "env_path": service_key,
        "pid": proc.pid,
        "host": host,
        "port": port,
        "api_key_suffix": api_key_suffix,
        "pid_file": os.path.relpath(pid_file, BASE_DIR),
        "log_file": os.path.relpath(log_file, BASE_DIR),
        "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })
    state["source_env"] = service_key
    save_state(state)
    print(f"[app] started: pid={proc.pid} host={host} port={port}")
    print(f"[app] env -> {service_key}")
    print(f"[app] log -> {log_file}")

    if getattr(args, "sync", False):
        config = load_sync_config(state)
        if config.get("obs_base"):
            meta_path = LOG_DIR / f"app-meta-port{port}.json"
            print("[app] --sync: waiting for app meta file...")
            for _ in range(6):
                if meta_path.exists():
                    break
                time.sleep(10)
            if not meta_path.exists():
                eprint("[app] --sync: meta file not ready after 60s, skipping sync")
                return 0
            args.interval = None
            cmd_sync_stop(args)
            return cmd_sync(args)
        eprint("[app] --sync: not configured. Run: sync config <yaml_path>")

    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    state = load_state()
    state["source_env"] = get_selected_env(args, state)
    env_path = resolve_env_path(state["source_env"])
    service_key = get_service_key(env_path)
    service_slug = get_service_slug(service_key)
    service = (state.get("services") or {}).get(service_key, {})
    pid = service.get("pid")
    pid_file = service.get("pid_file")
    if not pid_file and service.get("port"):
        pid_file = os.path.relpath(LOG_DIR / f"app-{service_slug}-port{service.get('port')}.pid", BASE_DIR)
    if pid_file:
        pid_from_file = read_pid(BASE_DIR / pid_file)
        if pid_from_file:
            pid = pid_from_file

    if not is_pid_running(pid):
        print(f"[app] not running: env={service_key}")
        if pid_file:
            (BASE_DIR / pid_file).unlink(missing_ok=True)
        service["pid"] = None
        save_state(state)
        return 0

    print(f"[app] stopping pid={pid} env={service_key}")
    os.kill(pid, signal.SIGTERM)
    for _ in range(20):
        time.sleep(0.5)
        if not is_pid_running(pid):
            break

    if is_pid_running(pid):
        print(f"[app] force kill pid={pid}")
        os.kill(pid, signal.SIGKILL)
        time.sleep(0.2)

    if pid_file:
        (BASE_DIR / pid_file).unlink(missing_ok=True)
    service["pid"] = None
    service["stopped_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_state(state)
    print("[app] stopped")
    return 0


def cmd_restart(args: argparse.Namespace) -> int:
    stop_code = cmd_stop(args)
    if stop_code != 0:
        return stop_code
    return cmd_start(args)


def cmd_logs(args: argparse.Namespace) -> int:
    state = load_state()
    state["source_env"] = get_selected_env(args, state)
    env_path = resolve_env_path(state["source_env"])
    service_key = get_service_key(env_path)
    service_slug = get_service_slug(service_key)
    service = (state.get("services") or {}).get(service_key, {})
    log_file_rel = service.get("log_file")
    if log_file_rel:
        log_file = BASE_DIR / log_file_rel
    else:
        _, _, _, port, _, _ = state_runtime(state)
        log_file = LOG_DIR / f"app-{service_slug}-port{port}.log"

    if not log_file.exists():
        eprint(f"[app] log file not found: {log_file}")
        return 1

    if args.follow:
        try:
            subprocess.run(["tail", "-n", str(args.lines), "-f", str(log_file)], check=False)
        except KeyboardInterrupt:
            pass
        return 0

    sys.stdout.writelines(tail_lines(log_file, args.lines))
    return 0


def _print_services(state: dict) -> int:
    services = state.get("services") or {}
    if not services:
        print("[app] no recorded services")
        return 0
    for key, service in services.items():
        pid = service.get("pid")
        pid_file = service.get("pid_file")
        if pid_file:
            pid_from_file = read_pid(BASE_DIR / pid_file)
            if pid_from_file:
                pid = pid_from_file
        running = is_pid_running(pid)
        host = service.get("host", "-")
        port = service.get("port", "-")
        api_key_suffix = service.get("api_key_suffix", "")
        log_file = service.get("log_file", "-")
        marker = "*" if key == state.get("source_env") else " "
        suffix_text = f" key=***{api_key_suffix}" if api_key_suffix else ""
        print(f"{marker} {key}: {'running' if running else 'stopped'} pid={pid or '-'} host={host} port={port}{suffix_text} log={log_file}")
    return 0


def cmd_status(_args: argparse.Namespace) -> int:
    state = load_state()
    print(f"[app] source_env: {state.get('source_env', DEFAULT_ENV)}")
    return _print_services(state)


def cmd_list(_args: argparse.Namespace) -> int:
    state = load_state()
    return _print_services(state)


# ---------------------------------------------------------------------------
# Connect — connection test
# ---------------------------------------------------------------------------

def cmd_connect(args: argparse.Namespace) -> int:
    from utils.connection_test import run_test, print_result

    state = load_state()
    state["source_env"] = get_selected_env(args, state)
    env_path, env_values, host, port, _, _ = state_runtime(state)

    model = args.model
    method = args.method

    if args.noproxy:
        upstream_url = env_values.get("UPSTREAM_URL", "")
        if not upstream_url:
            eprint("[connect] UPSTREAM_URL not found in env file")
            return 1
        upstream_url = upstream_url.rstrip("/").removesuffix("/v1")
        raw_key = env_values.get("UPSTREAM_API_KEY", "").strip().strip('"')
        api_key = raw_key.split(",")[0].strip() if raw_key else ""
        print(f"[connect] direct upstream: {upstream_url}")
        # 上游统一用 openai 协议
        method = "openai"
        base_url = upstream_url
    else:
        base_url = f"http://{host}:{port}"
        # 从 env 文件读取 API_KEY（客户端 key），取第一个
        raw_api_key = env_values.get("API_KEY", "").strip().strip('"')
        api_key = raw_api_key.split(",")[0].strip() if raw_api_key else ""
        print(f"[connect] via proxy: {base_url}")

    print(f"[connect] method={method} model={model}")
    result = run_test(base_url, method, model, api_key, args.timeout)
    print_result(result)
    return 0 if result["ok"] else 1


# ---------------------------------------------------------------------------
# Sync config — yaml file path stored in .cli_state.yaml["sync_config"]
# ---------------------------------------------------------------------------

def _resolve_sync_config_path(state: dict) -> Optional[Path]:
    rel = state.get("sync_config")
    if not rel:
        return None
    p = Path(rel)
    if not p.is_absolute():
        p = BASE_DIR / p
    return p.resolve()


def load_sync_config(state: dict) -> dict:
    path = _resolve_sync_config_path(state)
    if not path or not path.exists():
        return {}
    if yaml is not None:
        try:
            with path.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_app_meta(port: int) -> dict:
    meta_path = LOG_DIR / f"app-meta-port{port}.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _get_sync_service(state: dict, service_key: str) -> dict:
    return state.setdefault("sync_services", {}).setdefault(service_key, {})


# ---------------------------------------------------------------------------
# cmd_sync_config — point to a yaml config file
# ---------------------------------------------------------------------------

def cmd_sync_config(args: argparse.Namespace) -> int:
    state = load_state()
    config_path = getattr(args, "config_file", None)

    if config_path:
        p = Path(config_path)
        if not p.is_absolute():
            p = BASE_DIR / p
        p = p.resolve()
        if not p.exists():
            eprint(f"[sync] config file not found: {p}")
            return 1
        rel = os.path.relpath(p, BASE_DIR)
        state["sync_config"] = rel
        state["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        save_state(state)
        print(f"[sync] sync_config -> {rel}")
        config = load_sync_config(state)
        for k, v in config.items():
            print(f"[sync]   {k}: {v}")
        return 0

    # show current
    cfg_path = _resolve_sync_config_path(state)
    if not cfg_path:
        print("[sync] not configured")
        print("[sync] run: sync config <yaml_path>")
        return 0
    print(f"[sync] config: {state.get('sync_config')}")
    print(f"[sync] resolved: {cfg_path}")
    config = load_sync_config(state)
    for k, v in config.items():
        print(f"[sync]   {k}: {v}")
    return 0


# ---------------------------------------------------------------------------
# cmd_sync — start sync daemon
# ---------------------------------------------------------------------------

def cmd_sync(args: argparse.Namespace) -> int:
    state = load_state()
    config = load_sync_config(state)
    obs_base = config.get("obs_base", "")
    if not obs_base:
        eprint("[sync] not configured. Run: sync config <yaml_path>")
        return 1

    state = load_state()
    state["source_env"] = get_selected_env(args, state)
    env_path, env_values, host, port, _, _ = state_runtime(state)
    service_key = get_service_key(env_path)
    service_slug = get_service_slug(service_key)

    meta = _read_app_meta(port)
    logs_dir = meta.get("logs_dir")
    if not logs_dir:
        eprint(f"[sync] app meta not found: logs/app-meta-port{port}.json")
        eprint("[sync] start the app first: ./app start")
        return 1

    # OBS 目标：obs_base + logs_dir 相对于 logs_all/ 的部分
    # logs_dir 形如 logs_all/env-xxx-key/26052814
    if logs_dir.startswith("logs_all/"):
        env_segment = logs_dir[len("logs_all/"):]
    else:
        env_segment = os.path.basename(logs_dir)
    obs_dst = obs_base.rstrip("/") + "/" + env_segment.strip("/") + "/"

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    pid_file = LOG_DIR / f"sync-{service_slug}.pid"
    log_file = LOG_DIR / f"sync-{service_slug}.log"

    pid = read_pid(pid_file)
    if is_pid_running(pid):
        print(f"[sync] already running: pid={pid} env={service_key}")
        print(f"[sync] log -> {log_file}")
        return 0

    interval = args.interval or config.get("interval", 600)
    workers = config.get("workers", 4)
    upload_script = config.get("upload_script")

    cmd = [
        sys.executable, "-m", "utils.obs_sync",
        "--logs-dir", str(logs_dir),
        "--obs-dst", str(obs_dst),
        "--interval", str(interval),
        "--workers", str(workers),
    ]
    if upload_script:
        cmd.extend(["--upload-script", str(upload_script)])

    with log_file.open("ab") as log_fp:
        proc = subprocess.Popen(
            cmd,
            cwd=str(BASE_DIR),
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    pid_file.write_text(f"{proc.pid}\n", encoding="utf-8")
    time.sleep(1)
    if not is_pid_running(proc.pid):
        eprint(f"[sync] failed to start, check log: {log_file}")
        return 1

    sync_svc = _get_sync_service(state, service_key)
    sync_svc.update({
        "pid": proc.pid,
        "pid_file": os.path.relpath(pid_file, BASE_DIR),
        "log_file": os.path.relpath(log_file, BASE_DIR),
        "logs_dir": logs_dir,
        "obs_dst": obs_dst,
        "interval": interval,
        "config": state.get("sync_config", ""),
        "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })
    save_state(state)
    print(f"[sync] started: pid={proc.pid} env={service_key}")
    print(f"[sync] {logs_dir} -> {obs_dst}")
    print(f"[sync] log -> {log_file}")
    return 0


# ---------------------------------------------------------------------------
# cmd_sync_stop — stop sync daemon
# ---------------------------------------------------------------------------

def cmd_sync_stop(args: argparse.Namespace) -> int:
    state = load_state()
    state["source_env"] = get_selected_env(args, state)
    env_path = resolve_env_path(state["source_env"])
    service_key = get_service_key(env_path)
    sync_svcs = state.get("sync_services") or {}
    sync_svc = sync_svcs.get(service_key, {})

    pid_file_rel = sync_svc.get("pid_file")
    pid = sync_svc.get("pid")
    if pid_file_rel:
        pid_from_file = read_pid(BASE_DIR / pid_file_rel)
        if pid_from_file:
            pid = pid_from_file

    if not is_pid_running(pid):
        print(f"[sync] not running: env={service_key}")
        if pid_file_rel:
            (BASE_DIR / pid_file_rel).unlink(missing_ok=True)
        sync_svc["pid"] = None
        save_state(state)
        return 0

    print(f"[sync] stopping pid={pid} env={service_key}")
    os.kill(pid, signal.SIGTERM)
    for _ in range(20):
        time.sleep(0.5)
        if not is_pid_running(pid):
            break

    if is_pid_running(pid):
        print(f"[sync] force kill pid={pid}")
        os.kill(pid, signal.SIGKILL)
        time.sleep(0.2)

    if pid_file_rel:
        (BASE_DIR / pid_file_rel).unlink(missing_ok=True)
    sync_svc["pid"] = None
    sync_svc["stopped_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_state(state)
    print("[sync] stopped")
    return 0


# ---------------------------------------------------------------------------
# cmd_sync_logs — view sync daemon logs
# ---------------------------------------------------------------------------

def cmd_sync_logs(args: argparse.Namespace) -> int:
    state = load_state()
    state["source_env"] = get_selected_env(args, state)
    env_path = resolve_env_path(state["source_env"])
    service_key = get_service_key(env_path)
    service_slug = get_service_slug(service_key)
    sync_svc = (state.get("sync_services") or {}).get(service_key, {})

    log_file_rel = sync_svc.get("log_file")
    if log_file_rel:
        log_file = BASE_DIR / log_file_rel
    else:
        log_file = LOG_DIR / f"sync-{service_slug}.log"

    if not log_file.exists():
        eprint(f"[sync] log file not found: {log_file}")
        return 1

    if args.follow:
        try:
            subprocess.run(["tail", "-n", str(args.lines), "-f", str(log_file)], check=False)
        except KeyboardInterrupt:
            pass
        return 0

    sys.stdout.writelines(tail_lines(log_file, args.lines))
    return 0


# ---------------------------------------------------------------------------
# cmd_sync_status / cmd_sync_list — show sync services
# ---------------------------------------------------------------------------

def _print_sync_services(state: dict) -> int:
    sync_svcs = state.get("sync_services") or {}
    if not sync_svcs:
        print("[sync] no recorded sync services")
        return 0
    for key, svc in sync_svcs.items():
        pid = svc.get("pid")
        pid_file_rel = svc.get("pid_file")
        if pid_file_rel:
            pid_from_file = read_pid(BASE_DIR / pid_file_rel)
            if pid_from_file:
                pid = pid_from_file
        running = is_pid_running(pid)
        interval = svc.get("interval", "-")
        config_file = svc.get("config", "-")
        marker = "*" if key == state.get("source_env") else " "
        print(f"{marker} {key}: {'running' if running else 'stopped'} pid={pid or '-'} interval={interval}s config={config_file}")
        print(f"    src={svc.get('logs_dir', '-')}")
        print(f"    dst={svc.get('obs_dst', '-')}")
    return 0


def cmd_sync_status(_args: argparse.Namespace) -> int:
    state = load_state()
    cfg_rel = state.get("sync_config")
    if cfg_rel:
        print(f"[sync] config: {cfg_rel}")
    else:
        print("[sync] config: not set")
    return _print_sync_services(state)


def cmd_sync_list(_args: argparse.Namespace) -> int:
    state = load_state()
    return _print_sync_services(state)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LLM proxy service CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_start = subparsers.add_parser("start", help="Start app.py using configured source_env")
    p_start.add_argument("--env", dest="env_file", help="Env file to use for this start")
    p_start.add_argument("--sync", action="store_true", help="Also start sync daemon (requires obs_base configured)")
    p_start.set_defaults(func=cmd_start)

    p_stop = subparsers.add_parser("stop", help="Stop the running app")
    p_stop.add_argument("--env", dest="env_file", help="Env file to stop")
    p_stop.set_defaults(func=cmd_stop)

    p_restart = subparsers.add_parser("restart", help="Restart the app")
    p_restart.add_argument("--env", dest="env_file", help="Env file to restart")
    p_restart.add_argument("--sync", action="store_true", help="Also start sync daemon after restart")
    p_restart.set_defaults(func=cmd_restart)

    p_logs = subparsers.add_parser("logs", help="Show log output")
    p_logs.add_argument("--env", dest="env_file", help="Env file whose log should be shown")
    p_logs.add_argument("-f", "--follow", action="store_true", help="Follow the log file")
    p_logs.add_argument("-n", "--lines", type=int, default=100, help="Number of lines to show")
    p_logs.set_defaults(func=cmd_logs)

    p_config = subparsers.add_parser("config", help="Show or update source_env")
    p_config.add_argument("env_file", nargs="?", help="Env file path, such as .env or .env.prod")
    p_config.set_defaults(func=cmd_config)

    p_status = subparsers.add_parser("status", help="Show current service status")
    p_status.set_defaults(func=cmd_status)

    p_list = subparsers.add_parser("list", help="List all recorded env services")
    p_list.set_defaults(func=cmd_list)

    p_connect = subparsers.add_parser("connect", help="Connection test (proxy or upstream)")
    p_connect.add_argument("--env", dest="env_file", help="Env file to use")
    p_connect.add_argument("--model", default="claude-sonnet-4-6", help="Model to test (default: claude-sonnet-4-6)")
    p_connect.add_argument("--method", default="anthropic", choices=["anthropic", "openai"], help="API protocol (default: anthropic)")
    p_connect.add_argument("--noproxy", action="store_true", help="Test upstream directly, bypassing proxy")
    p_connect.add_argument("--timeout", type=int, default=30, help="Request timeout in seconds")
    p_connect.set_defaults(func=cmd_connect)

    # --- sync subcommands ---
    p_sync = subparsers.add_parser("sync", help="Manage sync daemon (start/stop/logs/status)")
    sync_sub = p_sync.add_subparsers(dest="sync_action", required=True)

    ps_start = sync_sub.add_parser("start", help="Start sync daemon")
    ps_start.add_argument("--env", dest="env_file", help="Env file to sync")
    ps_start.add_argument("--interval", type=int, default=None, help="Override sync interval (seconds)")
    ps_start.set_defaults(func=cmd_sync)

    ps_stop = sync_sub.add_parser("stop", help="Stop sync daemon")
    ps_stop.add_argument("--env", dest="env_file", help="Env file whose sync to stop")
    ps_stop.set_defaults(func=cmd_sync_stop)

    ps_logs = sync_sub.add_parser("logs", help="Show sync daemon logs")
    ps_logs.add_argument("--env", dest="env_file", help="Env file whose sync logs to show")
    ps_logs.add_argument("-f", "--follow", action="store_true", help="Follow the log file")
    ps_logs.add_argument("-n", "--lines", type=int, default=100, help="Number of lines to show")
    ps_logs.set_defaults(func=cmd_sync_logs)

    ps_status = sync_sub.add_parser("status", help="Show sync status and config")
    ps_status.set_defaults(func=cmd_sync_status)

    ps_list = sync_sub.add_parser("list", help="List all recorded sync services")
    ps_list.set_defaults(func=cmd_sync_list)

    ps_config = sync_sub.add_parser("config", help="Set or show sync config file")
    ps_config.add_argument("config_file", nargs="?", help="YAML config file path, e.g. settings/obs_base.yaml")
    ps_config.set_defaults(func=cmd_sync_config)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
