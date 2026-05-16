#!/bin/bash
# ============================================================
# 基于 tmux 的持久化课程学习训练管理脚本
# 用法:
#   ./sh/start_curriculum.sh          启动训练 (后台/前台取决于 debug.background)
#   ./sh/start_curriculum.sh attach   连接到正在运行的训练会话
#   ./sh/start_curriculum.sh stop     终止训练
#   ./sh/start_curriculum.sh status   查看训练是否在运行
#
# 前台/后台由 sh/config/auto_train.yaml 中 debug.background 控制:
#   background: true  → 通过 tmux 后台运行 (默认，适合长期训练)
#   background: false → 直接在终端前台运行 (适合调试改动)
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
WORKSPACE_ROOT="$SCRIPT_DIR/.."
CONFIG_YAML="$SCRIPT_DIR/config/auto_train.yaml"
SESSION_NAME="pipe_train"
TB_SESSION_NAME="pipe_tb"
CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")"
CONDA_INIT="source $CONDA_BASE/etc/profile.d/conda.sh && conda activate isaaclab"

cd "$WORKSPACE_ROOT" || exit

export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v "miniconda3" | tr '\n' ':' | sed 's/:$//')

# 从 yaml 中读取 debug.background 的值 (默认 true)
USE_BACKGROUND=$(python -c "
import yaml, sys
try:
    with open('$CONFIG_YAML', 'r') as f:
        cfg = yaml.safe_load(f)
    val = cfg.get('debug', {}).get('background', True)
    print('true' if val else 'false')
except Exception:
    print('true')
" 2>/dev/null)

case "${1:-start}" in
    start)
        if [ "$USE_BACKGROUND" = "false" ]; then
            # === 前台模式：直接在终端运行，输出可见 ===
            echo -e "\033[1;36m[INFO] 前台调试模式：训练输出将直接显示在当前终端.\033[0m"
            echo -e "\033[1;36m       按 Ctrl+C 可随时中断训练.\033[0m"
            echo ""

            # TensorBoard 仍在后台启动 (可选)
            if ! tmux has-session -t "$TB_SESSION_NAME" 2>/dev/null; then
                echo "[INFO] 后台启动 TensorBoard 会话: $TB_SESSION_NAME"
                tmux new-session -d -s "$TB_SESSION_NAME" -c "$WORKSPACE_ROOT" \
                    "$CONDA_INIT && python $SCRIPT_DIR/view_tensorboard.py"
            fi

            TAILSCALE_IP=$(tailscale ip -4 2>/dev/null)
            if [ -n "$TAILSCALE_IP" ]; then
                echo -e "  \033[1;31mTensorBoard: http://${TAILSCALE_IP}:6006\033[0m"
            fi

            echo ""
            echo "========================================"
            echo ""

            # 前台直接运行训练
            eval "$CONDA_INIT && python $SCRIPT_DIR/auto_train_loop.py"
            echo ""
            echo "[DONE] 前台训练已结束."
        else
            # === 后台模式：通过 tmux 持久化运行 ===
            if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
                echo -e "\033[1;33m[WARNING] 训练会话 '$SESSION_NAME' 已在运行中.\033[0m"
                echo "  查看:  ./sh/start_curriculum.sh attach"
                echo "  停止:  ./sh/start_curriculum.sh stop"
                exit 1
            fi

            echo -e "\033[1;32m[INFO] 启动后台训练会话: $SESSION_NAME\033[0m"
            tmux new-session -d -s "$SESSION_NAME" -c "$WORKSPACE_ROOT" \
                "$CONDA_INIT && python $SCRIPT_DIR/auto_train_loop.py; echo '[DONE] 训练已结束. 按任意键关闭此窗口.'; read"

            if ! tmux has-session -t "$TB_SESSION_NAME" 2>/dev/null; then
                echo "[INFO] 同时启动后台 TensorBoard 会话: $TB_SESSION_NAME"
                tmux new-session -d -s "$TB_SESSION_NAME" -c "$WORKSPACE_ROOT" \
                    "$CONDA_INIT && python $SCRIPT_DIR/view_tensorboard.py"
            fi

            echo ""
            echo -e "\033[1;32m[OK] 训练已在后台启动. 断开 SSH/Tailscale 后训练将继续运行.\033[0m"
            echo ""
            echo "  常用操作:"
            echo "    查看训练进度:    ./sh/start_curriculum.sh attach"
            echo "    查看训练状态:    ./sh/start_curriculum.sh status"
            echo "    停止训练:        ./sh/start_curriculum.sh stop"
            echo "    查看TensorBoard: ./sh/start_curriculum.sh tb"
            TAILSCALE_IP=$(tailscale ip -4 2>/dev/null)
            if [ -n "$TAILSCALE_IP" ]; then
                echo -e "    \033[1;31mTensorBoard: http://${TAILSCALE_IP}:6006\033[0m"
            fi
            echo ""
        fi
        ;;

    attach|a)
        if [ "$USE_BACKGROUND" = "false" ]; then
            echo "[INFO] 当前为前台模式，无需 attach。训练输出已在当前终端。"
            exit 0
        fi
        if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
            echo "[INFO] 没有正在运行的训练会话."
            exit 1
        fi
        echo "[INFO] 连接到训练会话 (Ctrl+B 然后按 D 可安全脱离而不中断训练)"
        tmux attach-session -t "$SESSION_NAME"
        ;;

    tb)
        if ! tmux has-session -t "$TB_SESSION_NAME" 2>/dev/null; then
            echo "[INFO] TensorBoard 未运行, 正在启动..."
            tmux new-session -d -s "$TB_SESSION_NAME" -c "$WORKSPACE_ROOT" \
                "$CONDA_INIT && python $SCRIPT_DIR/view_tensorboard.py"
        fi
        echo "[INFO] 连接到 TensorBoard 会话 (Ctrl+B 然后按 D 可安全脱离)"
        tmux attach-session -t "$TB_SESSION_NAME"
        ;;

    stop|kill)
        if [ "$USE_BACKGROUND" = "false" ]; then
            # 前台模式下没有 tmux 训练会话，只需停 TensorBoard
            if tmux has-session -t "$TB_SESSION_NAME" 2>/dev/null; then
                tmux kill-session -t "$TB_SESSION_NAME" 2>/dev/null
                echo "[INFO] TensorBoard 会话已终止."
            else
                echo "[INFO] 没有运行中的会话."
            fi
            exit 0
        fi

        if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
            tmux send-keys -t "$SESSION_NAME" C-c
            sleep 2
            tmux kill-session -t "$SESSION_NAME" 2>/dev/null
            echo -e "\033[1;31m[INFO] 训练会话已终止.\033[0m"
        else
            echo "[INFO] 没有正在运行的训练会话."
        fi

        if tmux has-session -t "$TB_SESSION_NAME" 2>/dev/null; then
            tmux kill-session -t "$TB_SESSION_NAME" 2>/dev/null
            echo "[INFO] TensorBoard 会话已终止."
        fi
        ;;

    status|s)
        echo "=== 运行模式 ==="
        if [ "$USE_BACKGROUND" = "false" ]; then
            echo -e "  \033[1;36m前台调试模式 (debug.background: false)\033[0m"
        else
            echo -e "  \033[1;32m后台 tmux 模式 (debug.background: true)\033[0m"
        fi

        echo "=== 训练会话 ==="
        if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
            echo -e "  \033[1;32m● 运行中\033[0m ($SESSION_NAME)"
        else
            echo -e "  \033[1;31m○ 未运行\033[0m"
        fi

        echo "=== TensorBoard ==="
        if tmux has-session -t "$TB_SESSION_NAME" 2>/dev/null; then
            TAILSCALE_IP=$(tailscale ip -4 2>/dev/null)
            echo -e "  \033[1;32m● 运行中\033[0m ($TB_SESSION_NAME)"
            if [ -n "$TAILSCALE_IP" ]; then
                echo -e "  \033[1;31m  http://${TAILSCALE_IP}:6006\033[0m"
            fi
        else
            echo -e "  \033[1;31m○ 未运行\033[0m"
        fi
        ;;

    *)
        echo "用法: $0 {start|attach|tb|stop|status}"
        echo "  start   - 启动训练 (前台/后台取决于 debug.background)"
        echo "  attach  - 连接到训练终端查看进度 (仅后台模式)"
        echo "  tb      - 连接到 TensorBoard 终端"
        echo "  stop    - 终止训练和 TensorBoard"
        echo "  status  - 查看运行状态及当前模式"
        exit 1
        ;;
esac