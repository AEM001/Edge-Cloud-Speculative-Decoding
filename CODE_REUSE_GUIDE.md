# 代码复用指南：Mac → Ubuntu → 组服务器

## 复用策略

```
Mac (mac-draft/)          Ubuntu (你的服务器)          组服务器 (云)
├─ client/                  ├─ edge/ (复用+修改)        ├─ cloud/ (已有)
│  ├─ draft_generator.py ──>│  ├─ draft_generator.py    │  ├─ cloud_server.py
│  ├─ edge_client.py ──────>│  ├─ edge_client.py        │  ├─ model_manager.py
│  ├─ http_cloud_client.py──│  ├─ http_cloud_client.py   │  └─ (保持)
│  └─ model_manager.py ────>│  └─ model_manager.py      │
│                            │    (PyTorch版本)          │
├─ protocol.py ────────────>├─ protocol.py (直接复用)    ├─ protocol.py (已有)
├─ benchmarks/ ────────────>├─ benchmarks/ (直接复用)    │
├─ config.py ──────────────>├─ config.py (修改路径)      │
└─ run_experiment.py ──────>├─ run_experiment.py         │
                              (修改import)               │
```

## 文件复用清单

### ✅ 直接复用（无需修改）

| 文件 | 从 mac-draft 复制到 | 说明 |
|------|-------------------|------|
| `protocol.py` | `ubuntu-verify/edge/protocol.py` | 数据结构定义 |
| `benchmarks/workloads.py` | `ubuntu-verify/edge/benchmarks/` | prompt加载 |
| `benchmarks/metrics_logger.py` | `ubuntu-verify/edge/benchmarks/` | metrics记录 |
| `benchmarks/prompts_easy.json` | `ubuntu-verify/edge/data/` | easy prompts |
| `benchmarks/prompts_hard.json` | `ubuntu-verify/edge/data/` | hard prompts |
| `data/prompts_easy.json` | `ubuntu-verify/edge/data/` | (如果workloads引用) |
| `data/prompts_hard.json` | `ubuntu-verify/edge/data/` | (如果workloads引用) |

### ⚠️ 需要修改（核心改动）

| 文件 | 修改内容 | 原因 |
|------|---------|------|
| `client/draft_generator.py` | MLX → PyTorch | Ubuntu用x86，不支持MLX |
| `client/model_manager.py` | 删除MLX依赖 | 改用HF transformers |
| `client/edge_client.py` | 修改import路径 | 相对路径→绝对路径 |
| `run_experiment.py` | 修改import路径 + 使用PyTorch draft | 主入口脚本 |
| `config.py` | 修改模型路径 | 本地路径不同 |

### ❌ 不需要复制

| 文件 | 原因 |
|------|------|
| `client/network_wrapper.py` | Ubuntu不需要模拟网络 |
| `server/` | Ubuntu作为edge，不需要server代码 |
| `policies/` | 如果需要可以复制，但可能用不到 |
| `main.py` | Mac的demo脚本，不需要 |

## 具体修改步骤

### 1. 创建目录结构

在 `ubuntu-verify/` 下创建：

```bash
cd /path/to/ubuntu-verify
mkdir -p edge/client edge/benchmarks edge/data
mkdir -p cloud  # 如果组服务器代码要放这里管理
```

### 2. 复制文件

```bash
# 从 mac-draft 复制到 ubuntu-verify/edge/
cd /path/to/mac-draft

# 直接复制的文件
cp protocol.py ../ubuntu-verify/edge/
cp benchmarks/workloads.py ../ubuntu-verify/edge/benchmarks/
cp benchmarks/metrics_logger.py ../ubuntu-verify/edge/benchmarks/
cp data/prompts_easy.json ../ubuntu-verify/edge/data/ 2>/dev/null || true
cp data/prompts_hard.json ../ubuntu-verify/edge/data/ 2>/dev/null || true

# 需要修改的文件
cp client/edge_client.py ../ubuntu-verify/edge/client/
cp client/http_cloud_client.py ../ubuntu-verify/edge/client/
cp config.py ../ubuntu-verify/edge/
cp run_experiment.py ../ubuntu-verify/edge/
```

### 3. 修改 draft_generator.py（核心改动）

**原文件**: `mac-draft/client/draft_generator.py` (MLX版本)

**新文件**: `ubuntu-verify/edge/client/draft_generator.py` (PyTorch版本)

```python
"""Draft token generation using PyTorch (for Ubuntu x86)."""
from typing import List, Tuple, Dict
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

# 复用 protocol.py 的导入
import sys
sys.path.append(str(Path(__file__).parent.parent))
from protocol import TokenInfo, DraftRequest, DraftResponse

logger = logging.getLogger(__name__)


class DraftGenerator:
    """Generates draft tokens using PyTorch with KV cache."""
    
    def __init__(self, model_path: str):
        """Initialize with PyTorch model."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # 加载模型到GPU（如果有）或CPU
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,  # 半精度节省内存
            device_map="auto",  # 自动选择GPU/CPU
            trust_remote_code=True
        )
        self.model.eval()  # 推理模式
        
        # 检查是否有GPU
        self.device = next(self.model.parameters()).device
        logger.info(f"Draft model loaded on {self.device}")
    
    def generate_draft_tokens(
        self, 
        request: 'DraftRequest',
        temperature: float = 0.8,
        top_p: float = 0.95
    ) -> 'DraftResponse':
        """Generate K draft tokens with confidence stats."""
        prefix = request.verified_prefix
        k = request.num_draft_tokens
        
        # 转换为tensor
        input_ids = torch.tensor([prefix], device=self.device)
        
        # 使用generate API（内部有KV cache）
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=k,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                return_dict_in_generate=True,
                output_scores=True,  # 返回每个token的logits
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # 提取新生成的token IDs
        full_sequence = outputs.sequences[0].tolist()
        generated_ids = full_sequence[len(prefix):]
        
        # 确保长度正确
        generated_ids = generated_ids[:k]
        
        # 提取每个生成位置的logits计算confidence stats
        draft_tokens = []
        
        for i, token_id in enumerate(generated_ids):
            # outputs.scores[i] 是第i个新token的logits
            logits = outputs.scores[i][0]  # [vocab_size]
            
            # 计算概率分布
            probs = torch.softmax(logits, dim=-1)
            probs_np = probs.cpu().numpy()
            
            # token概率
            token_prob = float(probs[token_id].item())
            token_logprob = float(torch.log(probs[token_id]).item())
            
            # confidence stats
            max_prob = float(probs_np.max())
            
            # entropy
            log_probs = np.log(probs_np + 1e-10)
            entropy = float(-(probs_np * log_probs).sum())
            
            # top margin
            top_2 = np.argpartition(probs_np, -2)[-2:]
            top_2_probs = np.sort(probs_np[top_2])
            top_margin = float(top_2_probs[1] - top_2_probs[0]) if len(top_2_probs) == 2 else 0.0
            
            draft_tokens.append(TokenInfo(
                token_id=token_id,
                logprob=token_logprob,
                probability=token_prob,
                max_prob=max_prob,
                entropy=entropy,
                top_margin=top_margin
            ))
        
        return DraftResponse(
            draft_token_ids=[t.token_id for t in draft_tokens],
            logprobs=[t.logprob for t in draft_tokens],
            probabilities=[t.probability for t in draft_tokens],
            confidence_stats={
                "max_probs": [t.max_prob for t in draft_tokens],
                "entropies": [t.entropy for t in draft_tokens],
                "top_margins": [t.top_margin for t in draft_tokens]
            }
        )
    
    def decode_tokens(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)
```

### 4. 修改 model_manager.py

**新文件**: `ubuntu-verify/edge/client/model_manager.py`

```python
"""Model management for PyTorch draft model."""
from pathlib import Path
from typing import Optional, Tuple
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages PyTorch model loading and caching."""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
    
    def load_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load model and tokenizer."""
        logger.info(f"Loading model from {self.model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        
        # 添加pad token如果缺失
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        logger.info("Model loaded successfully")
        return self.model, self.tokenizer
    
    def is_loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None
```

### 5. 修改 edge_client.py 的 import

**原文件**: `mac-draft/client/edge_client.py`

修改顶部 import：

```python
# 修改前 (Mac版本)
from client.draft_generator import DraftGenerator
from client.model_manager import ModelManager
from protocol import TokenInfo, DraftRequest, DraftResponse, EdgeRequest, CloudResponse

# 修改后 (Ubuntu版本)
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from client.draft_generator import DraftGenerator  # PyTorch版本
from client.model_manager import ModelManager    # PyTorch版本
from protocol import TokenInfo, DraftRequest, DraftResponse, EdgeRequest, CloudResponse
```

### 6. 修改 run_experiment.py

**主要修改**:

```python
#!/usr/bin/env python3
"""Experiment runner for Ubuntu edge + Cloud GPU."""
import sys
from pathlib import Path
# 添加路径
sys.path.insert(0, str(Path(__file__).parent))

from edge.client.edge_client import EdgeClient
from edge.client.http_cloud_client import HTTPCloudClient
from edge.benchmarks.metrics_logger import MetricsLogger
from edge.benchmarks.workloads import load_workloads
from edge.config import CLOUD_URL, MODEL_PATH

# ... 其余代码保持不变 ...
```

### 7. 修改 config.py

**新文件**: `ubuntu-verify/edge/config.py`

```python
"""Configuration for Ubuntu edge + Cloud GPU setup."""
from pathlib import Path

# Cloud GPU URL (组服务器的地址)
# 选项1: 如果组服务器有公网IP
CLOUD_URL = "http://<group-server-ip>:8000"

# 选项2: 如果通过FRP
# CLOUD_URL = "http://49.234.57.210:8006"

# Draft model path (本地下载的模型)
MODEL_PATH = Path.home() / ".models" / "Qwen2.5-0.5B-Instruct"

# 或者使用1.5B模型（如果有GPU）
# MODEL_PATH = Path.home() / ".models" / "Qwen2.5-1.5B-Instruct"

# 实验参数
MAX_DRAFT_TOKENS = 5
TEMPERATURE = 0.8
TOP_P = 0.95
MAX_TOKENS = 128

# 输出目录
OUTPUT_DIR = Path("results")
```

## 组服务器 (Cloud) 部分

### 复用现有代码

组服务器代码已经在 `ubuntu-verify/src/` 中，**不需要从Mac复制**。只需要确保：

1. **更新 model path** 加载更大的模型 (7B/14B/32B)
2. **启动服务** 监听正确的端口

### 修改 cloud_server.py (如果需要)

```python
# src/cloud_server.py

# 修改模型路径为更大的模型
MODEL_PATH = "/path/to/Qwen2.5-7B-Instruct"  # 或14B/32B

# 确保使用4-bit量化节省显存
# vLLM会自动处理
```

## 一键复制脚本

创建 `migrate_code.sh`：

```bash
#!/bin/bash
# 一键复制可复用代码

MAC_DRAFT="/Users/Mac/code/research/Infra/mac-draft"
UBUNTU_VERIFY="/Users/Mac/code/research/Infra/ubuntu-verify"

# 创建目录
mkdir -p $UBUNTU_VERIFY/edge/client
mkdir -p $UBUNTU_VERIFY/edge/benchmarks
mkdir -p $UBUNTU_VERIFY/edge/data

# 直接复制的文件
cp $MAC_DRAFT/protocol.py $UBUNTU_VERIFY/edge/
cp $MAC_DRAFT/benchmarks/workloads.py $UBUNTU_VERIFY/edge/benchmarks/
cp $MAC_DRAFT/benchmarks/metrics_logger.py $UBUNTU_VERIFY/edge/benchmarks/
cp $MAC_DRAFT/client/http_cloud_client.py $UBUNTU_VERIFY/edge/client/
cp $MAC_DRAFT/data/*.json $UBUNTU_VERIFY/edge/data/ 2>/dev/null || true

echo "=== 文件复制完成 ==="
echo "需要手动修改的文件:"
echo "  1. edge/client/draft_generator.py (MLX → PyTorch)"
echo "  2. edge/client/model_manager.py (MLX → PyTorch)"
echo "  3. edge/client/edge_client.py (修改import)"
echo "  4. edge/config.py (修改路径)"
echo "  5. edge/run_experiment.py (创建新文件)"
```

运行：
```bash
chmod +x migrate_code.sh
./migrate_code.sh
```

## 验证清单

- [ ] `protocol.py` 可以 import 成功
- [ ] `workloads.py` 可以加载 prompts
- [ ] PyTorch draft generator 可以加载模型
- [ ] Edge client 可以连接 cloud server
- [ ] 单条 prompt 端到端测试通过
- [ ] Metrics 记录正常

## 快速测试命令

```bash
cd ubuntu-verify/edge

# 1. 测试 imports
python3 -c "from protocol import DraftRequest; print('✓ protocol OK')"
python3 -c "from client.draft_generator import DraftGenerator; print('✓ draft_generator OK')"

# 2. 测试 prompt 加载
python3 -c "from benchmarks.workloads import load_workloads; w=load_workloads(); print(f'✓ Loaded {len(w[\"easy\"])} easy prompts')"

# 3. 测试模型加载 (需要下载模型后)
python3 -c "from client.model_manager import ModelManager; m=ModelManager('~/.models/Qwen2.5-0.5B-Instruct'); m.load_model(); print('✓ Model loaded')"

# 4. 测试 cloud 连接
python3 -c "from client.http_cloud_client import HTTPCloudClient; c=HTTPCloudClient('http://cloud-ip:8000'); print(c.health_check()); print('✓ Cloud connected')"
```

## 常见问题

### Q: 为什么不用 MLX 在 Ubuntu 上？
A: MLX 只支持 Apple Silicon，Ubuntu 服务器是 x86，必须用 PyTorch。

### Q: PyTorch 版本会比 MLX 慢吗？
A: 在 GPU 上 PyTorch 和 MLX 速度相当，在 CPU 上 PyTorch 可能稍慢但仍可用。

### Q: 可以保留 Mac 的代码吗？
A: 可以！Mac 和 Ubuntu 代码可以共存，用于不同场景。

### Q: 组服务器需要做什么修改？
A: 主要是更新 model path 到更大的模型 (7B+)，其他代码基本不变。

## 下一步

1. 运行上面的复制脚本
2. 按照修改指南更新核心文件
3. 在 Ubuntu 上下载 draft model
4. 在组服务器上部署/启动 cloud server
5. 跑单条 prompt 测试
