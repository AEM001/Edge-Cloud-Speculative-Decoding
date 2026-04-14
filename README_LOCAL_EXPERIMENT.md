# 本地实验设置：网络模拟 + Speculative Decoding

## 概述

在本地机器上同时运行draft模型和verification模型，使用网络模拟测试不同网络条件下的speculative decoding性能。

## 当前配置

### 模型配置
- **Draft模型**: Qwen2.5-1.5B-Instruct (3GB)
- **Verify模型**: Qwen2.5-32B-Instruct-GPTQ-Int4 (19GB)
- **GPU总量**: 31GB

### 问题
32B验证模型太大，无法和1.5B draft模型同时加载到GPU。

## 推荐的模型组合

### 方案1：7B GPTQ量化（推荐）
- **Draft**: Qwen2.5-1.5B-Instruct (3GB)
- **Verify**: Qwen2.5-7B-Instruct-GPTQ-Int4 (7GB)
- **总显存**: ~10GB
- **优势**: 量化后显存占用小，速度快，适合本地实验

### 方案2：7B非量化
- **Draft**: Qwen2.5-1.5B-Instruct (3GB)
- **Verify**: Qwen2.5-7B-Instruct (14GB)
- **总显存**: ~17GB
- **优势**: 准确性更高，显存仍然足够

### 方案3：3B
- **Draft**: Qwen2.5-1.5B-Instruct (3GB)
- **Verify**: Qwen2.5-3B-Instruct (6GB)
- **总显存**: ~9GB
- **优势**: 最小显存占用
- **劣势**: 验证模型和draft模型差距太小，acceptance率可能很高

## 如何更换验证模型

### 下载7B GPTQ模型

```bash
cd /root/autodl-tmp/ubuntu-verify/models

# 使用huggingface-cli下载（需要网络）
huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4 \
  --local-dir Qwen2.5-7B-Instruct-GPTQ-Int4 \
  --local-dir-use-symlinks False

# 或使用Python
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4',
    local_dir='/root/autodl-tmp/ubuntu-verify/models/Qwen2.5-7B-Instruct-GPTQ-Int4',
    local_dir_use_symlinks=False
)
"
```

### 更新配置

修改 `/root/autodl-tmp/ubuntu-verify/.env`:

```bash
# Verification (target) model settings
VERIFY_MODEL_NAME=Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4
VERIFY_GPU_MEMORY_UTILIZATION=0.4  # 可以提高到0.4
VERIFY_MAX_MODEL_LEN=8192  # 可以提高
VERIFY_QUANTIZATION=gptq
```

修改 `/root/autodl-tmp/draft/config_local.py`:

```python
# 可以提高draft模型的GPU利用率
GPU_MEMORY_UTILIZATION = _env_float("AINFRA_GPU_MEMORY_UTILIZATION", 0.3)
MAX_MODEL_LEN = _env_int("AINFRA_MAX_MODEL_LEN", 4096)
```

### 重启服务器

```bash
# 停止旧服务器
pkill -f "api_server.py"

# 启动新服务器
cd /root/autodl-tmp/ubuntu-verify
source .venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
export HF_HUB_OFFLINE=1
python src/api_server.py --host 0.0.0.0 --port 6006
```

## 运行实验

### 快速测试（2+2 prompts）

```bash
cd /root/autodl-tmp/draft
bash RUN_NOW.sh
```

### 完整实验（10+10 prompts, K=2,4,6,8, 4种网络条件）

```bash
cd /root/autodl-tmp/draft
source /root/autodl-tmp/ubuntu-verify/.venv/bin/activate
export AINFRA_DRAFT_MODEL_PATH="/root/autodl-tmp/ubuntu-verify/models/Qwen2.5-1.5B-Instruct"
export AINFRA_GPU_MEMORY_UTILIZATION=0.3
export AINFRA_MAX_MODEL_LEN=4096
python experiments/experiment_local_k2k8.py
```

## 实验参数

### K值
- K=2: 保守策略
- K=4: 平衡策略
- K=6: 激进策略
- K=8: 非常激进策略

### 网络条件
| 条件 | RTT (ms) | 带宽 (Mbps) | 描述 |
|------|----------|-------------|------|
| good | 10 | 100 | 优秀网络 |
| medium | 40 | 20 | 平均网络 |
| bad | 100 | 5 | 较差网络 |
| bursty | 20-120 | 5-50 | 波动网络（每20秒切换） |

### 测试集
- Easy prompts: 10个（对话类）
- Hard prompts: 10个（推理/数学/代码类）

## 预期结果

### 性能指标
- **Tokens per Second**: 吞吐量
- **Time per Output Token**: 每个token的时间
- **Acceptance Rate**: 接受率（accepted / drafted）
- **Rounds**: 验证轮数

### 预期性能（基于1.5B draft + 7B verify）
- Direct: ~40-60 tok/s
- K=2: ~30-50 tok/s, 接受率 ~50-60%
- K=4: ~35-55 tok/s, 接受率 ~65-75%
- K=6: ~30-50 tok/s, 接受率 ~70-80%
- K=8: ~25-45 tok/s, 接受率 ~75-85%

## 文件说明

| 文件 | 说明 |
|------|------|
| `config_local.py` | 本地配置（localhost，降低GPU使用） |
| `quick_test.py` | 快速测试脚本 |
| `experiment_local_k2k8.py` | 完整实验脚本 |
| `RUN_NOW.sh` | 一键运行脚本 |
| `run_local_experiment.sh` | 完整实验启动脚本 |
| `experiments/run_local_experiment_tmux.sh` | Tmux后台运行 |

## 故障排除

### GPU内存不足
```bash
# 检查GPU使用
nvidia-smi

# 降低GPU内存利用率
# 在 .env 和 config_local.py 中调整
VERIFY_GPU_MEMORY_UTILIZATION=0.35
AINFRA_GPU_MEMORY_UTILIZATION=0.25
```

### 服务器无法启动
```bash
# 检查日志
tail -50 /root/autodl-tmp/ubuntu-verify/verify_server.log

# 确保使用本地模型
export HF_HUB_OFFLINE=1
```

### 网络模拟不生效
确保在实验脚本中使用了 `create_network_wrapped_client`。

## 下一步

1. 下载7B GPTQ验证模型（推荐）
2. 更新配置文件
3. 重启验证服务器
4. 运行快速测试验证
5. 运行完整实验

## 参考资料

- [ARCHITECTURE_AND_EXPERIMENTS.md](ARCHITECTURE_AND_EXPERIMENTS.md) - 架构和实验详细说明
- [SOLUTION.md](SOLUTION.md) - 问题解决方案
