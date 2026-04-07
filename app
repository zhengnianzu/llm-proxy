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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LLM proxy service CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_start = subparsers.add_parser("start", help="Start app.py using configured source_env")
    p_start.add_argument("--env", dest="env_file", help="Env file to use for this start")
    p_start.set_defaults(func=cmd_start)

    p_stop = subparsers.add_parser("stop", help="Stop the running app")
    p_stop.add_argument("--env", dest="env_file", help="Env file to stop")
    p_stop.set_defaults(func=cmd_stop)

    p_restart = subparsers.add_parser("restart", help="Restart the app")
    p_restart.add_argument("--env", dest="env_file", help="Env file to restart")
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

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
