#!/bin/bash
# ============================================================
# 推理演示启动脚本
# 用法:
#   ./sh/play.sh              使用 play.yaml 中的默认配置运行推理
#   ./sh/play.sh --headless   覆盖为无头模式 (仅录视频)
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
WORKSPACE_ROOT="$SCRIPT_DIR/.."
cd "$WORKSPACE_ROOT" || exit

CONFIG_PATH="$SCRIPT_DIR/config/play.yaml"

if [ ! -f "$CONFIG_PATH" ]; then
    echo "[ERROR] Config file not found: $CONFIG_PATH"
    exit 1
fi

python3 "$SCRIPT_DIR/play_runner.py" "$@"