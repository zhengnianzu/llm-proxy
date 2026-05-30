#!/usr/bin/env bash
# 用法: source env.sh
# 将 llm-proxy 的命令加入当前 shell 环境

_PROXY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

_run_in_proxy_dir() {
    local cmd=$1
    shift
    (cd "$_PROXY_DIR" && $cmd "$@")
}

app() { _run_in_proxy_dir python3 ./app "$@"; }
sync() { _run_in_proxy_dir python3 ./app sync "$@"; }

export -f app sync

echo "[llm-proxy] 已加载，可用命令："
echo "  app   — 管理代理服务 (start/stop/restart/logs/status/config)"
echo "  sync  — 管理同步服务 (start/stop/logs/status/config)"
echo ""
echo "  示例:"
echo "    sync config settings/obs_base.yaml"
echo "    app start --env .env.prod"
echo "    app start --env .env.prod --sync"
echo "    sync start --env .env.prod"
echo "    sync stop --env .env.prod"
echo "    sync logs --env .env.prod -f"
echo "    sync status"
