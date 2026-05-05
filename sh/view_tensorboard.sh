#!/bin/bash
# 方便直接通过命令行调用 python 的 tensorboard 启动器
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
cd "$SCRIPT_DIR/.." || exit

echo "[INFO] Starting TensorBoard..."
python "$SCRIPT_DIR/view_tensorboard.py"