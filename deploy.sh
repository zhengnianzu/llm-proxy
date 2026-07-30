#!/usr/bin/env bash
# deploy.sh — 在一台新机器上初始化 llm-proxy 运行环境
#
# 做的事（全部幂等，可重复执行）：
#   1. 安装 uv（若缺失），并把 uv 安装目录写进 ~/.bashrc 的 PATH
#   2. 用 uv 安装 Python 3.12
#   3. 在项目目录下用 uv 创建 .venv（Python 3.12）
#   4. uv pip install -r requirements.txt 装依赖到 .venv
#
# 用法:
#   bash deploy.sh
# 装完后：
#   source ~/.bashrc          # 让 uv 进 PATH（或重开 shell）
#   bash server.sh start .env.xxx

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_VERSION="3.12"
UV_BIN_DIR="$HOME/.local/bin"          # uv 官方安装器默认落点
BASHRC="$HOME/.bashrc"

log() { echo "[deploy] $*"; }

# ---------------------------------------------------------------------------
# 1. 安装 uv
# ---------------------------------------------------------------------------
ensure_uv() {
    if command -v uv >/dev/null 2>&1; then
        log "uv 已存在: $(command -v uv) ($(uv --version))"
        return 0
    fi
    # 可能已装但不在当前 PATH
    if [[ -x "$UV_BIN_DIR/uv" ]]; then
        log "uv 已安装于 $UV_BIN_DIR/uv，但不在当前 PATH，临时加入"
        export PATH="$UV_BIN_DIR:$PATH"
        return 0
    fi
    log "未检测到 uv，开始安装 ..."
    if command -v curl >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
    elif command -v wget >/dev/null 2>&1; then
        wget -qO- https://astral.sh/uv/install.sh | sh
    else
        log "错误: 需要 curl 或 wget 才能安装 uv" >&2
        exit 1
    fi
    export PATH="$UV_BIN_DIR:$PATH"
    command -v uv >/dev/null 2>&1 || { log "错误: uv 安装后仍不可用" >&2; exit 1; }
    log "uv 安装完成: $(uv --version)"
}

# ---------------------------------------------------------------------------
# 2. 把 uv 安装目录写进 ~/.bashrc 的 PATH（幂等）
# ---------------------------------------------------------------------------
ensure_path_in_bashrc() {
    local marker="# >>> llm-proxy uv PATH >>>"
    if grep -qF "$marker" "$BASHRC" 2>/dev/null; then
        log "~/.bashrc 已包含 uv PATH 配置，跳过"
        return 0
    fi
    log "把 $UV_BIN_DIR 写入 $BASHRC 的 PATH"
    {
        echo ""
        echo "$marker"
        echo "export PATH=\"$UV_BIN_DIR:\$PATH\""
        echo "# <<< llm-proxy uv PATH <<<"
    } >> "$BASHRC"
}

# ---------------------------------------------------------------------------
# 2b. 配置阿里云 PyPI 镜像（pip 与 uv 各一套，幂等，不覆盖已有配置）
# ---------------------------------------------------------------------------
ALIYUN_INDEX="https://mirrors.aliyun.com/pypi/simple/"
ALIYUN_HOST="mirrors.aliyun.com"

ensure_mirror() {
    # pip：seed 出来的 .venv/bin/pip 手动装包时用
    local pip_conf="$HOME/.config/pip/pip.conf"
    if [[ -f "$pip_conf" ]] && grep -qF "$ALIYUN_HOST" "$pip_conf"; then
        log "pip 镜像已配置，跳过 ($pip_conf)"
    else
        log "写入 pip 阿里镜像 → $pip_conf"
        mkdir -p "$(dirname "$pip_conf")"
        cat > "$pip_conf" <<EOF
[global]
index-url = $ALIYUN_INDEX
trusted-host = $ALIYUN_HOST
EOF
    fi

    # uv：uv pip install 装 requirements.txt 时用（pip.conf 对 uv 无效）
    local uv_conf="$HOME/.config/uv/uv.toml"
    if [[ -f "$uv_conf" ]] && grep -qF "$ALIYUN_HOST" "$uv_conf"; then
        log "uv 镜像已配置，跳过 ($uv_conf)"
    else
        log "写入 uv 阿里镜像 → $uv_conf"
        mkdir -p "$(dirname "$uv_conf")"
        cat > "$uv_conf" <<EOF
[[index]]
url = "$ALIYUN_INDEX"
default = true
EOF
    fi
}

# ---------------------------------------------------------------------------
# 3. 安装 Python 3.12（由 uv 托管）
# ---------------------------------------------------------------------------
ensure_python() {
    log "确保 Python $PY_VERSION 可用（uv 托管）"
    uv python install "$PY_VERSION"
}

# ---------------------------------------------------------------------------
# 4. 创建 .venv 并装依赖
# ---------------------------------------------------------------------------
setup_venv() {
    cd "$SCRIPT_DIR"
    if [[ -d .venv ]]; then
        log ".venv 已存在，复用（如需重建先 rm -rf .venv）"
    else
        log "创建 .venv（Python $PY_VERSION）"
        uv venv --python "$PY_VERSION" --seed .venv
    fi
    log "安装依赖 requirements.txt 到 .venv"
    uv pip install --python "$SCRIPT_DIR/.venv/bin/python" -r requirements.txt
    log ".venv Python: $("$SCRIPT_DIR/.venv/bin/python" --version)"
}

# ---------------------------------------------------------------------------
main() {
    ensure_uv
    ensure_path_in_bashrc
    ensure_mirror
    ensure_python
    setup_venv
    echo ""
    log "✅ 部署完成"
    log "下一步:"
    log "  source ~/.bashrc                    # 让 uv 进 PATH（或重开 shell）"
    log "  bash server.sh start .env.xxx       # 启动服务"
}

main "$@"
