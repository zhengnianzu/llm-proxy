#!/usr/bin/env bash
# 用法: source env.sh
# 将 chat-log-viewer 的命令加入当前 shell 环境

_CHAT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 定义命令函数
server() {
    python3 -m src.cli server "$@"
}

sync() {
    python3 -m src.cli sync "$@"
}

client() {
    python3 -m src.client "$@"
}

cli() {
    python3 -m src.cli "$@"
}

# 导出函数，使其在子 shell 中可用
export -f server sync client cli

# 切换到项目目录执行（确保相对路径正确）
_run_in_chat_dir() {
    local cmd=$1
    shift
    (cd "$_CHAT_DIR" && $cmd "$@")
}

# 重新定义为在项目目录执行
server() { _run_in_chat_dir python3 -m src.cli server "$@"; }
sync() { _run_in_chat_dir python3 -m src.cli sync "$@"; }
client() { _run_in_chat_dir python3 -m src.cli client "$@"; }
cli() { _run_in_chat_dir python3 -m src.cli "$@"; }

export -f server sync client cli

# 为 tools/ 下的 .sh 脚本和 obsutil 添加执行权限
if [ -d "$_CHAT_DIR/tools" ]; then
    chmod +x "$_CHAT_DIR/tools"/*.sh 2>/dev/null || true
    chmod +x "$_CHAT_DIR/tools/obsutil" 2>/dev/null || true
fi

echo "[chat-log-viewer] 已加载，可用命令："
echo "  server  — 管理 server 服务 (start/stop/restart/status/logs)"
echo "  sync    — 管理 sync 服务   (start/stop/restart/status/logs)"
echo "  cli     — 统一管理入口"
echo "  client  — OBS 下载客户端"
