# 实验状态

## 当前运行状态

✅ **实验正在运行中** - 在tmux会话中，不会因SSH断开而中断

### 实验配置
- **测试方法**: Direct, K=2, K=4, K=6
- **Prompt类型**: Easy (10个) + Hard (10个)
- **总测试数**: 20 prompts × 4 methods = 80 tests
- **预计时间**: 20-40分钟

### 模型配置
- **Draft模型**: Qwen2.5-3B-Instruct (本地Mac) ⚠️ 已从1.5B升级到3B
- **Verify模型**: Qwen2.5-32B-Instruct-GPTQ-Int4 (远程RTX 5090)
- **SSH端口**: 20514 (新端口)
- **隧道端口**: localhost:6006 -> remote:6006
- **Max Model Len**: 8192 (从32768降低以适应3B模型)
- **GPU Memory**: 0.7 (从0.6提升)

### 性能指标 (初步结果)
- **Direct**: ~62.7 tok/s
- **K=2**: ~5.4 tok/s, 接受率 44.1% ✅
- **K=4**: ~9.8 tok/s, 接受率 67.4% ✅
- **K=6**: 测试中...

## 有用的命令

### 查看实验进度
```bash
# 查看实时日志
tail -f /home/albert/code/AInfra/draft/experiments/comprehensive_log.txt

# 进入tmux会话查看
tmux attach-session -t benchmark-experiment
# 退出: Ctrl+B 然后按 D

# 检查是否还在运行
pgrep -f comprehensive_benchmark.py
```

### SSH隧道管理
```bash
# 查看隧道状态
tmux attach-session -t ssh-tunnel

# 检查隧道健康
curl http://localhost:6006/health

# 重启隧道（如果需要）
tmux kill-session -t ssh-tunnel
bash /home/albert/code/AInfra/draft/start_ssh_tunnel.sh
```

### 查看所有tmux会话
```bash
tmux list-sessions
```

## 结果文件

实验完成后，结果将保存在：
- **JSON结果**: `/home/albert/code/AInfra/draft/experiments/comprehensive_results.json`
- **详细日志**: `/home/albert/code/AInfra/draft/experiments/comprehensive_log.txt`

## 更新内容

### 配置更新
1. ✅ SSH端口从 12272 更新为 20514
2. ✅ 更新了 `config.py` 中的 `CLOUD_SSH_PORT`
3. ✅ 更新了 `check_remote_server.sh` 中的所有SSH命令
4. ✅ 更新了 `simple_test.py` 中的SSH隧道注释

### 新增脚本
1. ✅ `start_ssh_tunnel.sh` - 自动重连的SSH隧道管理器
2. ✅ `experiments/run_experiment.sh` - 在tmux中运行实验，防止SSH断开

### 代码修复
1. ✅ 修复了 `client/edge_client.py` 中的导入路径
   - `DraftGenerator` → `VLLMDraftGenerator`
   - `ModelManager` → `VLLMModelManager`

## 问题修复记录

### 问题1: JSON序列化错误 ✅ 已修复
- **错误**: `Out of range float values are not JSON compliant: -inf`
- **原因**: logprobs中包含`-inf`值无法序列化为JSON
- **修复**: 在`protocol.py`的`EdgeRequest.to_dict()`中将`-inf`替换为`-1e10`

### 问题2: 接受率过低 ✅ 已修复
- **问题**: 1.5B模型接受率仅12.6% (K=2) 和 3.8% (K=4)
- **原因**: 1.5B和32B模型差距太大
- **解决**: 切换到3B模型，接受率提升至44.1% (K=2) 和 67.4% (K=4)

### 问题3: 3B模型显存不足 ✅ 已修复
- **错误**: KV cache需要1.12 GiB，但只有0.5 GiB可用
- **解决**: 
  - 降低`MAX_MODEL_LEN`: 32768 → 8192
  - 提升`GPU_MEMORY_UTILIZATION`: 0.6 → 0.7

## 实验开始时间
2026-04-14 12:02:06 (重启后)

## 注意事项
- ⚠️ 两个tmux会话都在运行，即使关闭终端也不会中断
- ⚠️ SSH隧道会自动重连，如果断开会在5秒后重试
- ⚠️ 实验数据会实时写入日志文件
