# 架构 V2: Ubuntu Edge + 租用 Cloud GPU

## 架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                          EXPERIMENT SETUP                          │
└─────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────┐              ┌──────────────────────────┐
  │   UBUNTU SERVER     │              │    RENTED CLOUD GPU       │
  │   (Your Machine)    │              │    (AutoDL/阿里云/...)    │
  │                     │   HTTP/FRP   │                           │
  │  ┌───────────────┐  │ ◄──────────► │  ┌──────────────────────┐ │
  │  │ Draft Model   │  │              │  │ Target Model         │ │
  │  │ • 0.5B/1.5B   │  │   Draft      │  │ • 7B/14B/32B         │ │
  │  │ • PyTorch     │  │   Tokens     │  │ • vLLM               │ │
  │  │ • ~20ms/gen   │  │              │  │ • ~30-80ms/verify    │ │
  │  └───────────────┘  │              │  └──────────────────────┘ │
  │         │           │              │           │              │
  │  ┌───────────────┐  │   Verify     │  ┌──────────────────────┐ │
  │  │ Edge Client   │◄─┼──────────────┼──│ Cloud Server         │ │
  │  │ • Generate    │  │   Result     │  │ • Verify Drafts      │ │
  │  │ • Send HTTP   │  │              │  │ • Return Accepted    │ │
  │  │ • Collect     │  │              │  │   Tokens             │ │
  │  │   Metrics     │  │              │  └──────────────────────┘ │
  │  └───────────────┘  │              └──────────────────────────┘
  │         │           
  │  ┌───────────────┐  
  │  │ Experiment    │  
  │  │ Runner        │  
  │  │ • Loop K      │  
  │  │ • Loop        │  
  │  │   Workloads   │  
  │  │ • Log Results │  
  │  └───────────────┘  
  └─────────────────────┘

Network: Internet or FRP Tunnel
Latency Target: < 50ms round-trip
```

## 组件说明

### 1. Ubuntu Edge (你的服务器)

**硬件**: 
- 你的现有 Ubuntu 服务器
- 可选: 加装入门级 GPU (GTX 1060/RTX 3060) 加速 draft

**软件栈**:
```
PyTorch + Transformers
    ↓
Draft Generator (0.5B/1.5B)
    ↓
HTTP Client → FRP Tunnel → Cloud GPU
```

**关键指标**:
- Draft generation: < 30ms for K=2
- Memory: < 4GB for 0.5B model
- Can run on CPU if no GPU available

### 2. Cloud GPU (租用的)

**推荐配置**:

| 级别 | GPU | Model | 用途 |
|------|-----|-------|------|
| 入门 | RTX 3090/4090 | 7B 4-bit | 验证实验 |
| 标准 | A100 40G | 14B 4-bit | 论文级 |
| 高端 | A100 80G | 32B 4-bit | 大模型研究 |

**软件栈**:
```
vLLM
    ↓
Target Model (7B+/4-bit)
    ↓
FastAPI Server
    ↓
FRP Client (frpc)
```

**关键指标**:
- Verification: 30-80ms depending on model size
- Throughput: 100+ requests/sec
- VRAM: 24-80GB

### 3. 通信协议

保持现有的 `protocol.py`，无需修改:

```python
EdgeRequest:
  - verified_prefix: List[int]
  - draft_ids: List[int]
  - draft_logprobs: List[float]
  
CloudResponse:
  - accepted_length: int
  - accepted_ids: List[int]
  - new_draft_tokens: List[int]  # Optional
  - server_verify_time_ms: float
```

## 数据流

### 单次 Speculative Decoding 流程

```
Time →

0ms     Ubuntu Edge                    Cloud GPU
        │                              │
        │  1. Draft Gen (K=2)          │
        │  ──────────────────────►     │
        │  Time: ~20ms                 │
        │                              │
20ms    │  2. Send Draft + Prefix      │
        │  ─────────────────────────►  │
        │  Payload: ~100 bytes         │
        │  Network: ~10ms                │
        │                              │
30ms    │                              │  3. Verify Draft
        │                              │  ──────────────────────►
        │                              │  Time: ~30ms (7B)
        │                              │
60ms    │                              │  4. Return Result
        │  ◄─────────────────────────  │
        │  Accepted: 1-2 tokens        │
        │                              │
60ms    │  5. Update Prefix            │
        │  Loop if not EOS             │
```

**每轮总计**: ~60ms, 生成 1-2 accepted tokens
**128 tokens 总时间**: ~60ms × (128 / 1.5) ≈ **5s**
**Baseline (直接生成)**: 128 × 30ms = **3.8s**
**Speedup**: 2-3x (with good acceptance ratio)

## 对比: V1 vs V2

### V1: Mac Edge + Ubuntu Cloud (失败)

```
MacBook Air M1 (0.5B 4-bit) ──► Ubuntu (3B 4-bit)
        │                              │
        │ 200ms                        │ 45ms
        │ (Draft too slow!)            │
        │                              │
Result: 5x SLOWER than baseline ❌
```

### V2: Ubuntu Edge + Cloud GPU (预期成功)

```
Ubuntu (0.5B) ──► Cloud GPU (7B+)
       │                │
       │ ~20ms          │ ~30ms
       │ (Fast!)        │
       │                │
Result: 2-3x FASTER than baseline ✅
```

## 性能模型

### 理论 Speedup 计算

```
T_direct = N × T_target

T_spec = N × [
    (1 - α) × (T_draft + T_network + T_target) +
    α × (T_draft + T_network)
] / (1 + α×K)

Where:
- N = total tokens to generate (128)
- T_target = target model time per token (~30ms)
- T_draft = draft model time per token (~10ms)
- T_network = network latency (~20ms)
- α = acceptance ratio (~0.4 for similar models)
- K = draft tokens per round (2-4)

Example:
T_direct = 128 × 30ms = 3840ms (3.8s)
T_spec ≈ 128 × 0.6 × (10 + 20 + 30) / 1.8 ≈ 2560ms (2.5s)
Speedup = 3840 / 2560 ≈ 1.5x

With better draft model (1.5B):
α ≈ 0.6, T_draft ≈ 15ms
Speedup ≈ 2.5x
```

## 实验变量

保持原有的实验矩阵:

| Variable | Values |
|----------|--------|
| K (draft tokens) | 2, 4, 6 |
| Workload | easy, hard |
| Network | good, medium, bad, bursty |

新增变量:
| Variable | Values |
|----------|--------|
| Draft Model | 0.5B, 1.5B |
| Cloud Model | 7B, 14B, 32B |

## 部署清单

### Ubuntu Edge 部署

- [ ] 安装 Python 3.11
- [ ] 安装 PyTorch + CUDA (如果有 GPU)
- [ ] 下载 draft model (0.5B or 1.5B)
- [ ] 实现 edge_client_torch.py
- [ ] 测试 draft generation < 30ms
- [ ] 配置 FRP 客户端

### Cloud GPU 部署

- [ ] 租用 GPU (RTX 4090 / A100)
- [ ] 安装 vLLM
- [ ] 下载 target model (7B/14B/32B)
- [ ] 启动 vLLM server
- [ ] 配置 FRP 服务端/客户端
- [ ] 测试 verify endpoint < 100ms

### 网络配置

- [ ] FRP server 配置 (使用现有 49.234.57.210)
- [ ] Cloud GPU frpc 配置
- [ ] Ubuntu edge 可以访问 FRP
- [ ] 端到端 ping < 50ms

## 文件结构

```
ubuntu-verify/
├── src/
│   ├── cloud_server.py          # 保持不变
│   ├── model_manager.py         # 加载更大的模型
│   └── ...
├── edge/
│   ├── edge_client_torch.py     # 新的 PyTorch edge client
│   ├── draft_generator_torch.py # PyTorch draft generator
│   └── run_experiment.py        # Ubuntu edge 实验脚本
├── MIGRATION_GUIDE.md           # 迁移指南
├── ARCHITECTURE_V2.md           # 本文档
└── ...
```

## 关键代码片段

### Draft Generator (PyTorch)

```python
class DraftGeneratorTorch:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.model.eval()
    
    def generate(self, prefix: List[int], k: int) -> DraftResponse:
        inputs = torch.tensor([prefix], device=self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=k,
                do_sample=True,
                temperature=0.8,
                return_dict_in_generate=True,
                output_scores=True,
            )
        
        # Extract and return stats
        ...
```

### Cloud Server (vLLM)

```bash
# 启动命令
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --quantization awq \
    --tensor-parallel-size 1 \
    --max-num-batched-tokens 4096 \
    --port 8000
```

## 预期结果

| Metric | V1 (Mac) | V2 (Ubuntu+Cloud) | Improvement |
|--------|----------|-------------------|-------------|
| Draft Time | 200ms | 20ms | **10x** |
| Verify Time | 45ms | 30ms | similar |
| Per-Round | 245ms | 50ms | **5x** |
| 128 tokens | 29s | 5s | **6x** |
| Speedup | 0.2x | 2-3x | **10-15x** |

## 风险评估

| 风险 | 可能性 | 缓解措施 |
|------|--------|----------|
| Cloud GPU 延迟过高 | Medium | 选择地理位置近的机房 |
| Draft 仍然太慢 | Low | 可以加小 GPU 到 Ubuntu |
| Network 不稳定 | Medium | 用 FRP + retry logic |
| 成本过高 | Low | 用 RTX 4090 替代 A100 |

## 下一步行动

1. **立即**: 租用 Cloud GPU (RTX 4090 on AutoDL)
2. **今天**: 在 Cloud GPU 上部署 vLLM (7B)
3. **今天**: 在 Ubuntu 上实现 PyTorch draft generator
4. **明天**: 端到端测试单条 prompt
5. **本周**: 跑完整实验

## 参考

- V1 问题分析: `mac-draft/docs/EXPERIMENT_STATUS.md`
- 迁移步骤: `MIGRATION_GUIDE.md`
- FRP 配置: `FRP_SETUP.md`
