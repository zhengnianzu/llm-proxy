#!/usr/bin/env bash
# server.sh — manage LLM_PROXY-PT service
# Usage: bash server.sh {start|stop|restart|status}

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP="$SCRIPT_DIR/app.py"
PID_FILE="$SCRIPT_DIR/logs/app.pid"
LOG_FILE="$SCRIPT_DIR/logs/app.log"

# Load PROXY_HOST / PROXY_PORT from .env if present
# Handles inline comments (KEY=value  # comment) and quoted values
if [[ -f "$SCRIPT_DIR/.env" ]]; then
    while IFS= read -r line || [[ -n "$line" ]]; do
        # skip blank lines and full-line comments
        [[ -z "$line" || "$line" =~ ^\s*# ]] && continue
        # strip inline comment (space/tab + #)
        line="${line%%[[:space:]]#*}"
        # must look like KEY=...
        [[ "$line" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]] || continue
        export "$line"
    done < "$SCRIPT_DIR/.env"
fi

PROXY_HOST="${PROXY_HOST:-127.0.0.1}"
PROXY_PORT="${PROXY_PORT:-4000}"

mkdir -p "$SCRIPT_DIR/logs"

# ---------------------------------------------------------------------------
is_running() {
    [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null
}

do_start() {
    if is_running; then
        echo "[server] Already running (PID $(cat "$PID_FILE"))"
        return 0
    fi
    echo "[server] Starting on ${PROXY_HOST}:${PROXY_PORT} ..."
    nohup python "$APP" >> "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    sleep 1
    if is_running; then
        echo "[server] Started (PID $(cat "$PID_FILE")), log → $LOG_FILE"
    else
        echo "[server] Failed to start — check $LOG_FILE"
        return 1
    fi
}

port_in_use() {
    ss -tlnp 2>/dev/null | grep -q ":${PROXY_PORT} " \
        || lsof -ti :"${PROXY_PORT}" > /dev/null 2>&1
}

wait_port_free() {
    for i in $(seq 1 20); do
        port_in_use || return 0
        sleep 0.5
    done
    echo "[server] Warning: port ${PROXY_PORT} still in use, force-killing occupant"
    lsof -ti :"${PROXY_PORT}" | xargs -r kill -9 2>/dev/null || true
    sleep 0.5
}

do_stop() {
    if ! is_running; then
        echo "[server] Not running"
        rm -f "$PID_FILE"
        # port may still be held by a stale process
        port_in_use && wait_port_free
        return 0
    fi
    PID=$(cat "$PID_FILE")
    echo "[server] Stopping PID $PID ..."
    kill "$PID"
    for i in $(seq 1 10); do
        sleep 0.5
        kill -0 "$PID" 2>/dev/null || break
    done
    if kill -0 "$PID" 2>/dev/null; then
        echo "[server] Force-killing PID $PID"
        kill -9 "$PID"
    fi
    rm -f "$PID_FILE"
    wait_port_free
    echo "[server] Stopped"
}

do_status() {
    if is_running; then
        echo "[server] Running (PID $(cat "$PID_FILE")) on ${PROXY_HOST}:${PROXY_PORT}"
    else
        echo "[server] Not running"
    fi
}

# ---------------------------------------------------------------------------
case "${1:-}" in
    start)   do_start ;;
    stop)    do_stop ;;
    restart) do_stop; do_start ;;
    status)  do_status ;;
    *)
        echo "Usage: bash server.sh {start|stop|restart|status}"
        exit 1
        ;;
esac
