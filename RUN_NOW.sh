#!/bin/bash
# 立即运行实验的脚本 - 假设验证服务器已经在运行

set -e

echo "========================================="
echo "快速实验启动脚本"
echo "========================================="
echo ""

# 检查验证服务器
echo "[1/4] 检查验证服务器状态..."
if curl -s http://localhost:6006/health 2>/dev/null | grep -q "healthy"; then
    echo "✓ 验证服务器正在运行"
    curl -s http://localhost:6006/health | python -m json.tool 2>/dev/null || true
else
    echo "✗ 验证服务器未运行"
    echo ""
    echo "请先启动验证服务器："
    echo "  cd /root/autodl-tmp/ubuntu-verify"
    echo "  source .venv/bin/activate"
    echo "  export PYTHONPATH=\"\${PYTHONPATH}:\$(pwd)/src\""
    echo "  export HF_HUB_OFFLINE=1"
    echo "  nohup python src/api_server.py --host 0.0.0.0 --port 6006 > verify_server.log 2>&1 &"
    echo ""
    echo "等待2-3分钟让服务器完全启动，然后再运行此脚本"
    exit 1
fi

echo ""
echo "[2/4] 设置环境变量..."
export AINFRA_DRAFT_MODEL_PATH="/root/autodl-tmp/ubuntu-verify/models/Qwen2.5-1.5B-Instruct"
export AINFRA_GPU_MEMORY_UTILIZATION=0.2
export AINFRA_MAX_MODEL_LEN=2048
export AINFRA_TEMPERATURE=0.0

echo "  Draft model: $AINFRA_DRAFT_MODEL_PATH"
echo "  GPU util: $AINFRA_GPU_MEMORY_UTILIZATION"
echo "  Max len: $AINFRA_MAX_MODEL_LEN"

echo ""
echo "[3/4] 激活虚拟环境..."
cd /root/autodl-tmp/draft
source /root/autodl-tmp/ubuntu-verify/.venv/bin/activate

echo ""
echo "[4/4] 运行快速测试..."
echo "  测试: 2 easy + 2 hard prompts"
echo "  方法: Direct, K=2, K=4"
echo "  预计时间: 5-10分钟"
echo ""

mkdir -p outputs_quick
python quick_test.py 2>&1 | tee outputs_quick/run_$(date +%Y%m%d_%H%M%S).log

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ 测试完成！"
    echo ""
    echo "结果文件："
    echo "  outputs_quick/quick_test_results.json"
    echo ""
    echo "如果测试成功，可以运行完整实验："
    echo "  python experiments/experiment_local_k2k8.py"
else
    echo "✗ 测试失败"
    echo "请检查日志文件"
fi
echo "========================================="

exit $EXIT_CODE
