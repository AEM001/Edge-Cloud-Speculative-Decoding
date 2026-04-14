# 实验问题解决方案

## 问题分析
32B GPTQ模型 + 1.5B模型无法同时在31GB GPU上运行，即使降低GPU内存利用率也不行。

## 解决方案

### 方案1：先运行验证服务器，再运行实验（推荐）

验证服务器已经在运行中（端口6006），直接运行实验即可：

```bash
cd /root/autodl-tmp/draft

# 设置环境变量
export AINFRA_DRAFT_MODEL_PATH="/root/autodl-tmp/ubuntu-verify/models/Qwen2.5-1.5B-Instruct"
export AINFRA_GPU_MEMORY_UTILIZATION=0.2
export AINFRA_MAX_MODEL_LEN=2048

# 激活虚拟环境
source /root/autodl-tmp/ubuntu-verify/.venv/bin/activate

# 运行快速测试（2 easy + 2 hard，K=2,4）
python quick_test.py

# 或运行完整实验（10 easy + 10 hard，K=2,4,6,8，4种网络条件）
python experiments/experiment_local_k2k8.py
```

### 方案2：使用tmux分别启动

```bash
# 终端1：启动验证服务器
cd /root/autodl-tmp/ubuntu-verify
source .venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
export HF_HUB_OFFLINE=1
python src/api_server.py --host 0.0.0.0 --port 6006

# 等待服务器启动完成（约2-3分钟）

# 终端2：运行实验
cd /root/autodl-tmp/draft
source /root/autodl-tmp/ubuntu-verify/.venv/bin/activate
export AINFRA_DRAFT_MODEL_PATH="/root/autodl-tmp/ubuntu-verify/models/Qwen2.5-1.5B-Instruct"
export AINFRA_GPU_MEMORY_UTILIZATION=0.2
export AINFRA_MAX_MODEL_LEN=2048
python quick_test.py
```

### 方案3：检查当前验证服务器状态

```bash
# 检查服务器是否运行
curl http://localhost:6006/health

# 如果返回healthy，直接运行实验
cd /root/autodl-tmp/draft
source /root/autodl-tmp/ubuntu-verify/.venv/bin/activate
python quick_test.py
```

## 当前状态

验证服务器应该已经在运行（从之前的日志看），检查：
```bash
curl -s http://localhost:6006/health | python -m json.tool
```

如果返回：
```json
{
    "status": "healthy",
    "model_info": {
        "model_name": "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4",
        ...
    }
}
```

说明服务器正常，可以直接运行实验。

## 快速测试命令

```bash
cd /root/autodl-tmp/draft
source /root/autodl-tmp/ubuntu-verify/.venv/bin/activate
export AINFRA_DRAFT_MODEL_PATH="/root/autodl-tmp/ubuntu-verify/models/Qwen2.5-1.5B-Instruct"
export AINFRA_GPU_MEMORY_UTILIZATION=0.2
export AINFRA_MAX_MODEL_LEN=2048
python quick_test.py 2>&1 | tee outputs_quick/test.log
```

## 预期结果

快速测试会：
1. 加载1.5B draft模型（约30-60秒）
2. 测试2个easy + 2个hard prompts
3. 对比Direct vs K=2 vs K=4
4. 验证speculative是否比direct快
5. 输出结果到`outputs_quick/quick_test_results.json`

如果成功，会显示：
```
✓ VERIFIED: Speculative (K=2) is faster than Direct!
```

然后可以运行完整实验。
