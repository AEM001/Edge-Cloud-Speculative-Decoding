# 实验状态与问题分析

## 已完成的工作

### 1. Bug 修复 (全部完成)
- ✅ Mac 端相对导入修复 (8 个文件)
- ✅ EOS 检测 bug 修复 (edge_client.py)
- ✅ Ubuntu 端 Logprob 比较 bug 修复 (cloud_server.py)
- ✅ Prompt 路径修复 (workloads.py)
- ✅ MetricsLogger API 修复 (run_experiment.py)

### 2. 性能诊断
- 网络 RTT: ~50-150ms
- Server verify: ~30-50ms (vLLM 3B 4-bit on Ubuntu)
- Draft generation: ~200ms (MLX 0.5B 4-bit on Mac)

## 核心问题

### 问题：Draft Model 太慢

**数据对比:**
| 操作 | 耗时 | 说明 |
|------|------|------|
| Draft generation (K=2) | ~200ms | Mac MLX 0.5B 4-bit |
| Server verify | ~45ms | Ubuntu vLLM 3B 4-bit |
| **比例** | **4.4:1** | Draft 比 Server 慢 4.4 倍 |

**理想情况:**
- Draft 应该比 Server 快 **10-100 倍**
- 实际: Draft 比 Server 慢 **4.4 倍**

**结果:**
- Speculative decoding 比直接调用慢 **5 倍**
- 128 tokens: 直接调用 ~6s, SpecDec ~29s

### 为什么 Draft 这么慢？

1. **硬件限制**: MacBook Air M1 8GB
   - GPU 算力有限
   - 8GB 统一内存带宽受限
   - 4-bit 量化仍有大量位运算

2. **模型限制**: 0.5B 已经是最小模型
   - 尝试了 2-bit 版本，反而更慢 (350ms vs 215ms)
   - 没有更小的 Qwen2.5 模型可用

3. **网络因素**: 
   - 代理增加了延迟 (~100ms RTT)
   - 但这不是主要瓶颈

## MacBook Air M1 8GB 适合端边云协同 Speculative Decoding 研究吗？

### 结论: **不适合** ❌

**原因:**

1. **算力不足**
   - M1 GPU 只有 8 核心
   - 0.5B 模型生成 2 tokens 需要 200ms
   - 理论上需要 <20ms 才能看到加速效果

2. **内存带宽瓶颈**
   - 8GB 统一内存需要同时服务 CPU 和 GPU
   - 4-bit 量化模型虽然小，但内存带宽受限

3. **缺乏优化空间**
   - 已经是最小模型 (0.5B)
   - 已经是最优量化 (4-bit)
   - 已经使用 KV cache

### 什么样的硬件适合？

| 配置 | 预期 Draft 速度 | 是否合适 |
|------|----------------|---------|
| MacBook Air M1 8GB | 200ms | ❌ 太慢 |
| MacBook Pro M3 18GB | ~50ms | ⚠️ 勉强 |
| Mac Studio M2 Ultra 64GB | ~10ms | ✅ 合适 |
| 专用 AI 芯片 (如 TPU/NPU) | ~5ms | ✅ 理想 |

### 研究建议

**如果必须用 MacBook Air M1:**
1. **降低期望**: 不追求速度提升，只验证算法正确性
2. **缩小规模**: 用 `--num-prompts 3` 跑小规模实验
3. **记录数据**: 重点收集 acceptance ratio、latency breakdown 等指标
4. **论文角度**: 强调算法可行性而非性能优化

**更好的选择:**
1. **用更强悍的 Mac**: M3 Pro/Max 或 M2 Ultra
2. **用 Linux + NVIDIA GPU**: 边缘设备用 Jetson Orin
3. **模拟环境**: 在云端跑两个模型，模拟边缘-云架构

## 下一步建议

### 选项 A: 继续小规模实验 (推荐)
- 跑 3 prompts × 24 conditions
- 预计时间: ~1 小时
- 目标: 验证算法正确性，收集 metrics

### 选项 B: 放弃 Speculative Decoding
- 直接调用云端生成
- 更快，但失去研究意义

### 选项 C: 换硬件
- 需要额外资源

## 当前阻塞

等待用户决定:
1. 是否继续小规模实验？
2. 是否尝试其他方案？
