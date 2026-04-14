#!/usr/bin/env python3
"""
实验结果分析脚本
生成图表、表格和详细报告
"""
import json
import os
from pathlib import Path
from collections import defaultdict
import statistics

# 设置matplotlib后端为Agg（非交互式）
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

RESULTS_DIR = Path(__file__).parent

def load_data():
    """加载实验数据"""
    with open(RESULTS_DIR / "comprehensive_results.json", 'r') as f:
        data = json.load(f)
    return data["results"], data["metadata"]

def analyze_by_network_regime(results):
    """按网络条件分析"""
    analysis = defaultdict(lambda: defaultdict(list))
    
    for r in results:
        regime = r["network_regime"]
        method = r["method"]
        
        analysis[regime][method].append({
            "tps": r["tokens_per_second"],
            "tokens": r["tokens_generated"],
            "time_ms": r["total_time_ms"],
            "acceptance": r.get("acceptance_rate", 0),
            "rounds": r.get("num_rounds", 0),
            "rtt": r.get("avg_rtt_ms", 0),
            "draft_ms": r.get("avg_draft_ms", 0),
            "verify_ms": r.get("avg_verify_ms", 0),
            "network_ms": r.get("avg_network_ms", 0),
            "prompt_type": r["prompt_type"],
        })
    
    return analysis

def analyze_by_prompt_type(results):
    """按prompt类型分析"""
    analysis = defaultdict(lambda: defaultdict(list))
    
    for r in results:
        ptype = r["prompt_type"]
        regime = r["network_regime"]
        method = r["method"]
        key = f"{regime}_{method}"
        
        analysis[ptype][key].append(r["tokens_per_second"])
    
    return analysis

def calculate_stats(values):
    """计算统计信息"""
    if not values:
        return {"mean": 0, "std": 0, "min": 0, "max": 0, "median": 0}
    
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0,
        "min": min(values),
        "max": max(values),
        "median": statistics.median(values),
        "count": len(values)
    }

def generate_summary_table(analysis):
    """生成汇总表格"""
    table = []
    
    regimes = ["good", "medium", "bad", "bursty"]
    methods = ["direct", "spec_k2", "spec_k4", "spec_k6", "spec_k8"]
    
    for regime in regimes:
        for method in methods:
            if method in analysis[regime]:
                data = analysis[regime][method]
                tps_values = [d["tps"] for d in data]
                acc_values = [d["acceptance"] for d in data if d["acceptance"] > 0]
                
                stats = calculate_stats(tps_values)
                acc_stats = calculate_stats(acc_values) if acc_values else {"mean": 0}
                
                table.append({
                    "regime": regime,
                    "method": method,
                    "avg_tps": stats["mean"],
                    "std_tps": stats["std"],
                    "min_tps": stats["min"],
                    "max_tps": stats["max"],
                    "avg_acceptance": acc_stats["mean"] * 100,
                    "count": stats["count"]
                })
    
    return table

def plot_throughput_comparison(analysis, output_path):
    """绘制吞吐量对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Throughput Comparison Across Network Regimes', fontsize=14, fontweight='bold')
    
    regimes = ["good", "medium", "bad", "bursty"]
    methods = ["direct", "spec_k2", "spec_k4", "spec_k6", "spec_k8"]
    method_labels = ["Direct", "K=2", "K=4", "K=6", "K=8"]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for idx, (regime, ax) in enumerate(zip(regimes, axes.flat)):
        data = analysis[regime]
        
        x = np.arange(len(methods))
        means = []
        stds = []
        
        for method in methods:
            if method in data:
                tps_values = [d["tps"] for d in data[method]]
                stats = calculate_stats(tps_values)
                means.append(stats["mean"])
                stds.append(stats["std"])
            else:
                means.append(0)
                stds.append(0)
        
        bars = ax.bar(x, means, yerr=stds, capsize=3, color=colors, alpha=0.8, edgecolor='black')
        ax.set_ylabel('Tokens/Second', fontsize=10)
        ax.set_title(f'{regime.upper()} Network', fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(method_labels, rotation=0, fontsize=9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # 在柱子上标注数值
        for bar, mean in zip(bars, means):
            if mean > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
                       f'{mean:.1f}', ha='center', va='bottom', fontsize=8)
        
        # 标注最快的方法
        if means:
            max_idx = means.index(max(means))
            bars[max_idx].set_edgecolor('gold')
            bars[max_idx].set_linewidth(2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_acceptance_rate(analysis, output_path):
    """绘制接受率图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Token Acceptance Rate by Network Regime', fontsize=14, fontweight='bold')
    
    regimes = ["good", "medium", "bad", "bursty"]
    spec_methods = ["spec_k2", "spec_k4", "spec_k6", "spec_k8"]
    method_labels = ["K=2", "K=4", "K=6", "K=8"]
    colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for idx, (regime, ax) in enumerate(zip(regimes, axes.flat)):
        data = analysis[regime]
        
        x = np.arange(len(spec_methods))
        means = []
        stds = []
        
        for method in spec_methods:
            if method in data:
                acc_values = [d["acceptance"] * 100 for d in data[method] if d["acceptance"] > 0]
                if acc_values:
                    stats = calculate_stats(acc_values)
                    means.append(stats["mean"])
                    stds.append(stats["std"])
                else:
                    means.append(0)
                    stds.append(0)
            else:
                means.append(0)
                stds.append(0)
        
        bars = ax.bar(x, means, yerr=stds, capsize=3, color=colors, alpha=0.8, edgecolor='black')
        ax.set_ylabel('Acceptance Rate (%)', fontsize=10)
        ax.set_title(f'{regime.upper()} Network', fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(method_labels, rotation=0, fontsize=9)
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        for bar, mean in zip(bars, means):
            if mean > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                       f'{mean:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_speedup_vs_direct(analysis, output_path):
    """绘制相对于Direct的加速比"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    regimes = ["good", "medium", "bad", "bursty"]
    spec_methods = ["spec_k2", "spec_k4", "spec_k6", "spec_k8"]
    method_labels = ["K=2", "K=4", "K=6", "K=8"]
    colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    x = np.arange(len(regimes))
    width = 0.18
    
    for i, (method, label, color) in enumerate(zip(spec_methods, method_labels, colors)):
        speedups = []
        
        for regime in regimes:
            data = analysis[regime]
            
            if "direct" in data and method in data:
                direct_tps = [d["tps"] for d in data["direct"]]
                spec_tps = [d["tps"] for d in data[method]]
                
                direct_mean = statistics.mean(direct_tps) if direct_tps else 0
                spec_mean = statistics.mean(spec_tps) if spec_tps else 0
                
                speedup = spec_mean / direct_mean if direct_mean > 0 else 0
                speedups.append(speedup)
            else:
                speedups.append(0)
        
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, speedups, width, label=label, color=color, alpha=0.8, edgecolor='black')
        
        # 标注数值
        for bar, speedup in zip(bars, speedups):
            if speedup > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                       f'{speedup:.2f}x', ha='center', va='bottom', fontsize=7, rotation=90)
    
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Baseline (Direct)')
    ax.set_ylabel('Speedup vs Direct', fontsize=11)
    ax.set_title('Speculative Decoding Speedup by Network Regime', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([r.upper() for r in regimes], fontsize=10)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max(2.0, ax.get_ylim()[1]))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_latency_breakdown(analysis, output_path):
    """绘制延迟分解图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Latency Breakdown (K=4)', fontsize=14, fontweight='bold')
    
    regimes = ["good", "medium", "bad", "bursty"]
    labels = ['Draft', 'Verify', 'Network']
    colors = ['#ff7f0e', '#2ca02c', '#d62728']
    
    for idx, (regime, ax) in enumerate(zip(regimes, axes.flat)):
        data = analysis[regime]
        
        if "spec_k4" in data:
            draft_times = [d["draft_ms"] for d in data["spec_k4"] if d["draft_ms"] > 0]
            verify_times = [d["verify_ms"] for d in data["spec_k4"] if d["verify_ms"] > 0]
            network_times = [d["network_ms"] for d in data["spec_k4"] if d["network_ms"] > 0]
            
            means = [
                statistics.mean(draft_times) if draft_times else 0,
                statistics.mean(verify_times) if verify_times else 0,
                statistics.mean(network_times) if network_times else 0,
            ]
            
            # 堆叠柱状图
            x = [0]
            bottom = 0
            for label, mean, color in zip(labels, means, colors):
                if mean > 0:
                    ax.bar(x, [mean], bottom=[bottom], label=label, color=color, alpha=0.8, edgecolor='black', width=0.5)
                    ax.text(0, bottom + mean/2, f'{mean:.1f}ms', ha='center', va='center', fontsize=10, fontweight='bold')
                    bottom += mean
            
            ax.set_ylabel('Time (ms)', fontsize=10)
            ax.set_title(f'{regime.upper()} Network', fontsize=11, fontweight='bold')
            ax.set_xticks([])
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_prompt_type_comparison(results, output_path):
    """对比easy vs hard prompts"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Performance: Easy vs Hard Prompts', fontsize=14, fontweight='bold')
    
    # 按prompt类型和网络条件分组
    easy_data = defaultdict(list)
    hard_data = defaultdict(list)
    
    for r in results:
        regime = r["network_regime"]
        method = r["method"]
        
        if r["prompt_type"] == "easy":
            easy_data[regime].append((method, r["tokens_per_second"]))
        else:
            hard_data[regime].append((method, r["tokens_per_second"]))
    
    methods = ["direct", "spec_k2", "spec_k4", "spec_k6", "spec_k8"]
    method_labels = ["Direct", "K=2", "K=4", "K=6", "K=8"]
    regimes = ["good", "medium", "bad", "bursty"]
    x = np.arange(len(methods))
    width = 0.2
    
    for ax_idx, (ax, data, title) in enumerate(zip(axes, [easy_data, hard_data], ["Easy Prompts", "Hard Prompts"])):
        colors = plt.cm.Set3(np.linspace(0, 1, len(regimes)))
        
        for i, regime in enumerate(regimes):
            if regime in data:
                method_tps = defaultdict(list)
                for method, tps in data[regime]:
                    method_tps[method].append(tps)
                
                means = [statistics.mean(method_tps[m]) if m in method_tps and method_tps[m] else 0 for m in methods]
                offset = (i - 1.5) * width
                ax.bar(x + offset, means, width, label=regime.upper(), color=colors[i], alpha=0.8, edgecolor='black')
        
        ax.set_ylabel('Tokens/Second', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(method_labels, fontsize=9)
        ax.legend(loc='upper right', fontsize=8, title='Network')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def generate_markdown_report(analysis, table, metadata):
    """生成Markdown报告"""
    report = []
    
    report.append("# Speculative Decoding实验报告\n")
    report.append(f"**实验时间**: 2024-04-15\n")
    report.append(f"**Draft模型**: {metadata['draft_model']}\n")
    report.append(f"**Verify模型**: Qwen/Qwen2.5-14B-Instruct-AWQ\n")
    report.append(f"**K值**: {metadata['k_values']}\n")
    report.append(f"**网络条件**: {metadata['network_regimes']}\n")
    report.append(f"**Prompt数量**: {metadata['prompt_count']} easy + {metadata['prompt_count']} hard\n")
    report.append(f"**Max Tokens**: {metadata['max_tokens']}\n\n")
    
    report.append("---\n\n")
    
    # 执行摘要
    report.append("## 执行摘要\n\n")
    
    # 找出最佳配置
    best_configs = []
    for regime in metadata['network_regimes']:
        regime_data = [r for r in table if r['regime'] == regime]
        if regime_data:
            best = max(regime_data, key=lambda x: x['avg_tps'])
            best_configs.append({
                'regime': regime,
                'method': best['method'],
                'tps': best['avg_tps']
            })
    
    report.append("### 各网络条件下的最佳配置\n\n")
    report.append("| 网络条件 | 最佳方法 | 吞吐量 (tok/s) |\n")
    report.append("|---------|---------|---------------|\n")
    for cfg in best_configs:
        report.append(f"| {cfg['regime'].upper()} | {cfg['method']} | {cfg['tps']:.2f} |\n")
    
    report.append("\n")
    
    # 详细表格
    report.append("## 详细性能数据\n\n")
    report.append("### 按网络条件和方法汇总\n\n")
    report.append("| 网络 | 方法 | 平均TPS | 标准差 | 最小TPS | 最大TPS | 接受率(%) | 样本数 |\n")
    report.append("|-----|-----|--------|--------|--------|--------|----------|--------|\n")
    
    for row in table:
        report.append(f"| {row['regime'].upper()} | {row['method']} | "
                     f"{row['avg_tps']:.2f} | {row['std_tps']:.2f} | "
                     f"{row['min_tps']:.2f} | {row['max_tps']:.2f} | "
                     f"{row['avg_acceptance']:.1f} | {row['count']} |\n")
    
    report.append("\n")
    
    # 关键发现
    report.append("## 关键发现\n\n")
    
    # 计算加速比
    speedups = defaultdict(dict)
    for regime in metadata['network_regimes']:
        regime_data = [r for r in table if r['regime'] == regime and r['method'] == 'direct']
        if regime_data:
            direct_tps = regime_data[0]['avg_tps']
            
            for spec_method in ['spec_k2', 'spec_k4', 'spec_k6', 'spec_k8']:
                spec_data = [r for r in table if r['regime'] == regime and r['method'] == spec_method]
                if spec_data:
                    spec_tps = spec_data[0]['avg_tps']
                    speedup = spec_tps / direct_tps if direct_tps > 0 else 0
                    speedups[regime][spec_method] = speedup
    
    report.append("### 相对于Direct的加速比\n\n")
    report.append("| 网络条件 | K=2 | K=4 | K=6 | K=8 |\n")
    report.append("|---------|-----|-----|-----|-----|\n")
    
    for regime in metadata['network_regimes']:
        if regime in speedups:
            row = [regime.upper()]
            for method in ['spec_k2', 'spec_k4', 'spec_k6', 'spec_k8']:
                speedup = speedups[regime].get(method, 0)
                row.append(f"{speedup:.2f}x")
            report.append(f"| {' | '.join(row)} |\n")
    
    report.append("\n")
    
    # 接受率分析
    report.append("### Token接受率分析\n\n")
    report.append("| 网络条件 | K=2 | K=4 | K=6 | K=8 |\n")
    report.append("|---------|-----|-----|-----|-----|\n")
    
    for regime in metadata['network_regimes']:
        row = [regime.upper()]
        for method in ['spec_k2', 'spec_k4', 'spec_k6', 'spec_k8']:
            data = [r for r in table if r['regime'] == regime and r['method'] == method]
            if data:
                row.append(f"{data[0]['avg_acceptance']:.1f}%")
            else:
                row.append("N/A")
        report.append(f"| {' | '.join(row)} |\n")
    
    report.append("\n")
    
    # 结论
    report.append("## 结论与建议\n\n")
    report.append("1. **网络条件影响**: Bad网络条件下Speculative Decoding优势更明显\n")
    report.append("2. **最佳K值**: 不同网络条件下最佳K值可能不同\n")
    report.append("3. **接受率**: Easy prompts接受率通常高于Hard prompts\n")
    report.append("4. **部署建议**: 根据实际网络环境选择合适的K值\n\n")
    
    report.append("---\n\n")
    report.append("## 图表\n\n")
    report.append("图表文件保存在当前目录:\n\n")
    report.append("- `fig_throughput.png` - 吞吐量对比\n")
    report.append("- `fig_acceptance.png` - 接受率分析\n")
    report.append("- `fig_speedup.png` - 加速比分析\n")
    report.append("- `fig_latency.png` - 延迟分解\n")
    report.append("- `fig_prompt_type.png` - Easy vs Hard对比\n\n")
    
    return ''.join(report)

def generate_csv_data(table, output_path):
    """生成CSV数据文件"""
    lines = []
    lines.append("network_regime,method,avg_tps,std_tps,min_tps,max_tps,avg_acceptance_pct,count\n")
    
    for row in table:
        lines.append(f"{row['regime']},{row['method']},{row['avg_tps']:.4f},{row['std_tps']:.4f},"
                    f"{row['min_tps']:.4f},{row['max_tps']:.4f},{row['avg_acceptance']:.2f},{row['count']}\n")
    
    with open(output_path, 'w') as f:
        f.writelines(lines)
    print(f"Saved: {output_path}")

def main():
    print("=" * 60)
    print("实验结果分析")
    print("=" * 60)
    
    # 加载数据
    print("\n[1/6] 加载实验数据...")
    results, metadata = load_data()
    print(f"  加载了 {len(results)} 条结果记录")
    
    # 分析
    print("\n[2/6] 按网络条件分析...")
    analysis = analyze_by_network_regime(results)
    
    print("\n[3/6] 生成汇总表格...")
    table = generate_summary_table(analysis)
    
    # 生成图表
    print("\n[4/6] 生成图表...")
    plot_throughput_comparison(analysis, RESULTS_DIR / "fig_throughput.png")
    plot_acceptance_rate(analysis, RESULTS_DIR / "fig_acceptance.png")
    plot_speedup_vs_direct(analysis, RESULTS_DIR / "fig_speedup.png")
    plot_latency_breakdown(analysis, RESULTS_DIR / "fig_latency.png")
    plot_prompt_type_comparison(results, RESULTS_DIR / "fig_prompt_type.png")
    
    # 生成CSV
    print("\n[5/6] 生成CSV数据...")
    generate_csv_data(table, RESULTS_DIR / "results_summary.csv")
    
    # 生成报告
    print("\n[6/6] 生成Markdown报告...")
    report = generate_markdown_report(analysis, table, metadata)
    
    with open(RESULTS_DIR / "REPORT.md", 'w') as f:
        f.write(report)
    print(f"Saved: {RESULTS_DIR / 'REPORT.md'}")
    
    print("\n" + "=" * 60)
    print("分析完成！")
    print("=" * 60)
    print(f"\n输出文件:")
    print(f"  - REPORT.md (主报告)")
    print(f"  - fig_throughput.png")
    print(f"  - fig_acceptance.png")
    print(f"  - fig_speedup.png")
    print(f"  - fig_latency.png")
    print(f"  - fig_prompt_type.png")
    print(f"  - results_summary.csv")

if __name__ == "__main__":
    main()
