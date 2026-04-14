#!/bin/bash
# Tmux会话启动完整实验 - 防止SSH中断

SESSION_NAME="exp-k2k8-14b"
SCRIPT_DIR="/root/autodl-tmp/draft"
LOG_FILE="$SCRIPT_DIR/experiments/outputs_local_k2k8/tmux_experiment.log"

# 创建日志目录
mkdir -p "$SCRIPT_DIR/experiments/outputs_local_k2k8"

echo "=============================================="
echo "启动完整实验 (K=2,4,6,8 + 网络模拟)"
echo "=============================================="
echo "Session: $SESSION_NAME"
echo "Log: $LOG_FILE"
echo ""

# 检查验证服务器
echo "[1/3] 检查验证服务器..."
if curl -s http://localhost:6006/health 2>/dev/null | grep -q "healthy"; then
    echo "✓ 验证服务器已运行"
    curl -s http://localhost:6006/health | python -m json.tool 2>/dev/null | head -10
else
    echo "✗ 验证服务器未运行！"
    echo "请先启动验证服务器："
    echo "  cd /root/autodl-tmp/ubuntu-verify"
    echo "  source .venv/bin/activate && export PYTHONPATH=\"\${PYTHONPATH}:\$(pwd)/src\""
    echo "  python src/api_server.py --host 0.0.0.0 --port 6006"
    exit 1
fi

echo ""
echo "[2/3] 创建tmux会话..."

# 如果会话已存在，先删除
tmux kill-session -t $SESSION_NAME 2>/dev/null || true

# 创建新会话，在后台运行实验
tmux new-session -d -s $SESSION_NAME -n "experiment"

# 在tmux会话中执行实验
tmux send-keys -t $SESSION_NAME:0 "
cd $SCRIPT_DIR
source /root/autodl-tmp/ubuntu-verify/.venv/bin/activate

export AINFRA_DRAFT_MODEL_PATH=\"/root/autodl-tmp/ubuntu-verify/models/Qwen2.5-1.5B-Instruct\"
export AINFRA_GPU_MEMORY_UTILIZATION=0.4
export AINFRA_MAX_MODEL_LEN=4096
export AINFRA_TEMPERATURE=0.0

echo \"==============================================\"
echo \"实验开始: \$(date)\"
echo \"Draft: 1.5B | Verify: 14B AWQ\"
echo \"K values: 2, 4, 6, 8\"
echo \"Network: good, medium, bad, bursty\"
echo \"==============================================\"

python experiments/experiment_local_k2k8.py 2>&1 | tee $LOG_FILE

echo \"\"
echo \"==============================================\"
echo \"实验完成: \$(date)\"
echo \"结果目录: experiments/outputs_local_k2k8/\"
echo \"==============================================\"

# 实验完成后保持会话打开
exec bash
" C-m

echo "✓ Tmux会话已创建"
echo ""
echo "[3/3] 实验已启动！"
echo ""
echo "=============================================="
echo "查看实验进度："
echo "  tmux attach-session -t $SESSION_NAME"
echo ""
echo "查看日志（不进入tmux）："
echo "  tail -f $LOG_FILE"
echo ""
echo "分离tmux（保持运行）："
echo "  按 Ctrl+B 然后按 D"
echo ""
echo "实验预计时间：2-3小时"
echo "=============================================="
