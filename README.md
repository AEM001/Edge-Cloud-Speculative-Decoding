for experiment
已根据 `spec/experiment.md` 的研究目标补了一套“逐步准备实验”的脚本，产物统一规划到：

`outputs/guidance_reuse/<run_id>/`

新增/修改重点：

- [prepare_guidance_reuse_configs.py](/Users/Mac/code/research/Infra/draft/scripts/experiments/prepare_guidance_reuse_configs.py:1)：生成 `R=1,2,4,8,16,32` 的 sweep 配置和 `manifest.json`
- [run_guidance_reuse_sweep.py](/Users/Mac/code/research/Infra/draft/scripts/experiments/run_guidance_reuse_sweep.py:1)：调用现有 `quick_test.py` 跑 sweep，并把 raw result 归档到 run 目录
- [analyze_guidance_reuse.py](/Users/Mac/code/research/Infra/draft/scripts/experiments/analyze_guidance_reuse.py:1)：输出 `R vs accepted length / acceptance ratio / latency per accepted token / oracle overlap`
- [analyze_update_signals.py](/Users/Mac/code/research/Infra/draft/scripts/experiments/analyze_update_signals.py:1)：做 edge-side signals 的相关性分析
- [simulate_adaptive_update_policy.py](/Users/Mac/code/research/Infra/draft/scripts/experiments/simulate_adaptive_update_policy.py:1)：离线模拟简单 threshold adaptive update policy
- [guidance_reuse_README.md](/Users/Mac/code/research/Infra/draft/scripts/experiments/guidance_reuse_README.md:1)：写了完整运行顺序和输出路径
- [specextend_edge_client.py](/Users/Mac/code/research/Infra/draft/scripts/client/specextend_edge_client.py:48) 和 [quick_test.py](/Users/Mac/code/research/Infra/draft/scripts/experiments/quick_test.py:371)：加了 `retrieve_on_first_round` 可选开关，让 `reuse_R1` 更适合作为 every-step/oracle proxy；默认不变。

推荐运行顺序：

```bash
python3 scripts/experiments/prepare_guidance_reuse_configs.py \
  --reuse-windows 1,2,4,8,16,32 \
  --prompt-count 3 \
  --max-tokens 256 \
  --prompt-input-tokens 2048

bash start_verify.sh

python3 scripts/experiments/run_guidance_reuse_sweep.py \
  outputs/guidance_reuse/<run_id>/configs/guidance_reuse_sweep.json

python3 scripts/experiments/analyze_guidance_reuse.py \
  outputs/guidance_reuse/<run_id>/raw/quick_test_results_<timestamp>.json

python3 scripts/experiments/analyze_update_signals.py \
  outputs/guidance_reuse/<run_id>/analysis/guidance_reuse_rounds.csv

python3 scripts/experiments/simulate_adaptive_update_policy.py \
  outputs/guidance_reuse/<run_id>/analysis/guidance_reuse_rounds.csv
```

验证：已跑 `py_compile`，并用一个极小 fake quick result 顺序测试了三个分析脚本的 CSV 输出。没有启动实际模型/server。当前工作区还有一些和本次无关的已有改动/删除：`.gitignore`、`README.md`、`scripts/download_data.py` 等，我没有处理它们。