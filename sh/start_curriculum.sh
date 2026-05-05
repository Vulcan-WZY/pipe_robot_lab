#!/bin/bash
# 方便直接通过命令行调用 python 的 runner
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
cd "$SCRIPT_DIR/.." || exit

echo "[INFO] Invoking python auto curriculum loop runner..."
python "$SCRIPT_DIR/auto_train_loop.py"