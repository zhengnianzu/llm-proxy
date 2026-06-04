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
sess() { _run_in_proxy_dir python3 ./app sess "$@"; }

export -f app sync sess

echo "[llm-proxy] 已加载，可用命令："
echo "  app  — 管理代理服务 (start/stop/restart/logs/status/config)"
echo "  sync — 管理同步服务 (start/stop/logs/status/config)"
echo "  sess — 管理 session 导出 (export/config/list/logs)"
echo ""
echo "  示例:"
echo "    sync config settings/obs_base.yaml"
echo "    app start --env .env.prod"
echo "    app start --env .env.prod --sync"
echo "    sync start --env .env.prod"
echo "    sync stop --env .env.prod"
echo "    sync logs --env .env.prod -f"
echo "    sync status"
echo "    sess list                               # 列出所有 mtime 目录及导出状态"
echo "    sess config 26060309                    # 切换到指定 mtime 目录"
echo "    sess export                             # 导出当前 mtime 的 session_index.jsonl"
echo "    sess export --sync                      # 导出并上传三元组到 OBS"
echo "    sess logs                               # 查看 export 运行日志"
