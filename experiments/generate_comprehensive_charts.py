"""
Generate comprehensive charts for Easy vs Hard prompts with K=2, K=4, K=6, and Direct.
Tests: 50 easy + 50 hard prompts × 4 methods = 400 tests
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Setup paths
script_dir = Path(__file__).parent
outputs_dir = script_dir / 'outputs'
outputs_dir.mkdir(exist_ok=True)

# Load results
results_file = outputs_dir / 'comprehensive_results.json'
with open(results_file, 'r') as f:
    data = json.load(f)

results = data['results']
summary = data['summary']

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12

# Colors
direct_color = '#3498db'
k2_color = '#e67e22'
k4_color = '#27ae60'
k6_color = '#9b59b6'
easy_color = '#3498db'
hard_color = '#e74c3c'

# Extract data by type and method
def get_data(prompt_type, method):
    return [r for r in results if r['prompt_type'] == prompt_type and r['method'] == method]

easy_direct = get_data('easy', 'direct')
easy_k2 = get_data('easy', 'spec_k2')
easy_k4 = get_data('easy', 'spec_k4')
easy_k6 = get_data('easy', 'spec_k6')
hard_direct = get_data('hard', 'direct')
hard_k2 = get_data('hard', 'spec_k2')
hard_k4 = get_data('hard', 'spec_k4')
hard_k6 = get_data('hard', 'spec_k6')

# Pre-calculate acceptance rate and rounds data (needed for multiple charts)
easy_k2_acc = [r['acceptance_rate']*100 for r in easy_k2]
easy_k4_acc = [r['acceptance_rate']*100 for r in easy_k4]
easy_k6_acc = [r['acceptance_rate']*100 for r in easy_k6]
hard_k2_acc = [r['acceptance_rate']*100 for r in hard_k2]
hard_k4_acc = [r['acceptance_rate']*100 for r in hard_k4]
hard_k6_acc = [r['acceptance_rate']*100 for r in hard_k6]

easy_k2_rounds = [r['num_rounds'] for r in easy_k2]
easy_k4_rounds = [r['num_rounds'] for r in easy_k4]
easy_k6_rounds = [r['num_rounds'] for r in easy_k6]
hard_k2_rounds = [r['num_rounds'] for r in hard_k2]
hard_k4_rounds = [r['num_rounds'] for r in hard_k4]
hard_k6_rounds = [r['num_rounds'] for r in hard_k6]

# Figure 2: Detailed Analysis
fig2, axes = plt.subplots(2, 3, figsize=(16, 10))
fig2.suptitle('Detailed Performance Analysis', fontsize=14, fontweight='bold', y=0.98)

# 1. Time per Round Breakdown - Easy
ax = axes[0, 0]
components = ['Draft\n(K=2)', 'Verify\n(K=2)', 'Draft\n(K=4)', 'Verify\n(K=4)', 'Draft\n(K=6)', 'Verify\n(K=6)']
easy_k2_draft = np.mean([r['avg_draft_ms'] for r in easy_k2])
easy_k2_verify = np.mean([r['avg_verify_ms'] for r in easy_k2])
easy_k4_draft = np.mean([r['avg_draft_ms'] for r in easy_k4])
easy_k4_verify = np.mean([r['avg_verify_ms'] for r in easy_k4])
easy_k6_draft = np.mean([r['avg_draft_ms'] for r in easy_k6])
easy_k6_verify = np.mean([r['avg_verify_ms'] for r in easy_k6])
times = [easy_k2_draft, easy_k2_verify, easy_k4_draft, easy_k4_verify, easy_k6_draft, easy_k6_verify]
colors_time = [k2_color, k2_color, k4_color, k4_color, k6_color, k6_color]
bars = ax.bar(components, times, color=colors_time, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Time per Round (ms)')
ax.set_title('EASY: Time Breakdown per Round', fontweight='bold')
for bar, time_val in zip(bars, times):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2,
            f'{time_val:.0f}ms', ha='center', va='bottom', fontsize=8, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# 2. Hard Time Breakdown
ax = axes[0, 1]
hard_k2_draft = np.mean([r['avg_draft_ms'] for r in hard_k2])
hard_k2_verify = np.mean([r['avg_verify_ms'] for r in hard_k2])
hard_k4_draft = np.mean([r['avg_draft_ms'] for r in hard_k4])
hard_k4_verify = np.mean([r['avg_verify_ms'] for r in hard_k4])
hard_k6_draft = np.mean([r['avg_draft_ms'] for r in hard_k6])
hard_k6_verify = np.mean([r['avg_verify_ms'] for r in hard_k6])
times = [hard_k2_draft, hard_k2_verify, hard_k4_draft, hard_k4_verify, hard_k6_draft, hard_k6_verify]
bars = ax.bar(components, times, color=colors_time, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Time per Round (ms)')
ax.set_title('HARD: Time Breakdown per Round', fontweight='bold')
for bar, time_val in zip(bars, times):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2,
            f'{time_val:.0f}ms', ha='center', va='bottom', fontsize=8, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# 3. Speed Distribution - Easy (Box plot for 50 prompts)
ax = axes[0, 2]
easy_direct_tps = [r['tokens_per_second'] for r in easy_direct]
easy_k2_tps = [r['tokens_per_second'] for r in easy_k2]
easy_k4_tps = [r['tokens_per_second'] for r in easy_k4]
easy_k6_tps = [r['tokens_per_second'] for r in easy_k6]
bp = ax.boxplot([easy_direct_tps, easy_k2_tps, easy_k4_tps, easy_k6_tps],
                labels=['Direct', 'K=2', 'K=4', 'K=6'],
                patch_artist=True, widths=0.6)
bp['boxes'][0].set_facecolor(direct_color)
bp['boxes'][1].set_facecolor(k2_color)
bp['boxes'][2].set_facecolor(k4_color)
bp['boxes'][3].set_facecolor(k6_color)
ax.set_ylabel('Tokens per Second')
ax.set_title('EASY: Speed Distribution (50 prompts)', fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# 4. Speed Distribution - Hard
ax = axes[1, 0]
hard_direct_tps = [r['tokens_per_second'] for r in hard_direct]
hard_k2_tps = [r['tokens_per_second'] for r in hard_k2]
hard_k4_tps = [r['tokens_per_second'] for r in hard_k4]
hard_k6_tps = [r['tokens_per_second'] for r in hard_k6]
bp = ax.boxplot([hard_direct_tps, hard_k2_tps, hard_k4_tps, hard_k6_tps],
                labels=['Direct', 'K=2', 'K=4', 'K=6'],
                patch_artist=True, widths=0.6)
bp['boxes'][0].set_facecolor(direct_color)
bp['boxes'][1].set_facecolor(k2_color)
bp['boxes'][2].set_facecolor(k4_color)
bp['boxes'][3].set_facecolor(k6_color)
ax.set_ylabel('Tokens per Second')
ax.set_title('HARD: Speed Distribution (50 prompts)', fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# 5. Acceptance Rate Comparison (Easy vs Hard with K=6)
ax = axes[1, 1]
categories = ['K=2', 'K=4', 'K=6']
easy_acc_avg = [np.mean(easy_k2_acc), np.mean(easy_k4_acc), np.mean(easy_k6_acc)]
hard_acc_avg = [np.mean(hard_k2_acc), np.mean(hard_k4_acc), np.mean(hard_k6_acc)]
x = np.arange(len(categories))
width = 0.35
bars1 = ax.bar(x - width/2, easy_acc_avg, width, label='Easy', color=easy_color, edgecolor='black')
bars2 = ax.bar(x + width/2, hard_acc_avg, width, label='Hard', color=hard_color, edgecolor='black')
ax.set_ylabel('Acceptance Rate (%)')
ax.set_title('Acceptance Rate: Easy vs Hard', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend(loc='upper right')
ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars1 + bars2, easy_acc_avg + hard_acc_avg):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

# 6. Summary Table as Text (Dynamic)
ax = axes[1, 2]
ax.axis('off')

easy_direct_tps = np.mean([r['tokens_per_second'] for r in easy_direct]) if easy_direct else 0
easy_k2_tps = np.mean([r['tokens_per_second'] for r in easy_k2]) if easy_k2 else 0
easy_k4_tps = np.mean([r['tokens_per_second'] for r in easy_k4]) if easy_k4 else 0
easy_k6_tps = np.mean([r['tokens_per_second'] for r in easy_k6]) if easy_k6 else 0

hard_direct_tps = np.mean([r['tokens_per_second'] for r in hard_direct]) if hard_direct else 0
hard_k2_tps = np.mean([r['tokens_per_second'] for r in hard_k2]) if hard_k2 else 0
hard_k4_tps = np.mean([r['tokens_per_second'] for r in hard_k4]) if hard_k4 else 0
hard_k6_tps = np.mean([r['tokens_per_second'] for r in hard_k6]) if hard_k6 else 0

summary_text = f"""
PERFORMANCE SUMMARY (50 Prompts Each)

EASY PROMPTS
┌─────────┬────────┬────────┬────────┐
│ Method  │ Tok/s  │ Accept │ Rounds │
├─────────┼────────┼────────┼────────┤
│ Direct  │ {easy_direct_tps:5.2f} │   --   │   --   │
│ K=2     │ {easy_k2_tps:5.2f} │ {np.mean(easy_k2_acc):5.1f}% │ {np.mean(easy_k2_rounds):5.1f} │
│ K=4     │ {easy_k4_tps:5.2f} │ {np.mean(easy_k4_acc):5.1f}% │ {np.mean(easy_k4_rounds):5.1f} │
│ K=6     │ {easy_k6_tps:5.2f} │ {np.mean(easy_k6_acc):5.1f}% │ {np.mean(easy_k6_rounds):5.1f} │
└─────────┴────────┴────────┴────────┘

HARD PROMPTS
┌─────────┬────────┬────────┬────────┐
│ Method  │ Tok/s  │ Accept │ Rounds │
├─────────┼────────┼────────┼────────┤
│ Direct  │ {hard_direct_tps:5.2f} │   --   │   --   │
│ K=2     │ {hard_k2_tps:5.2f} │ {np.mean(hard_k2_acc):5.1f}% │ {np.mean(hard_k2_rounds):5.1f} │
│ K=4     │ {hard_k4_tps:5.2f} │ {np.mean(hard_k4_acc):5.1f}% │ {np.mean(hard_k4_rounds):5.1f} │
│ K=6     │ {hard_k6_tps:5.2f} │ {np.mean(hard_k6_acc):5.1f}% │ {np.mean(hard_k6_rounds):5.1f} │
└─────────┴────────┴────────┴────────┘

KEY FINDINGS:
• Direct is ~{(easy_direct_tps/easy_k4_tps if easy_k4_tps > 0 else 0):.1f}x faster than Speculative
• K=6 trades acceptance rate for fewer rounds
• Hard prompts: higher acceptance on K=2, drops with K
"""
ax.text(0.1, 0.5, summary_text, transform=ax.transAxes, fontsize=9,
        verticalalignment='center', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.savefig(outputs_dir / 'comprehensive_detailed.png', dpi=200, bbox_inches='tight')
print("Saved: outputs/comprehensive_detailed.png")

# Figure 3: Network Latency & Timing Analysis
fig3, axes = plt.subplots(1, 2, figsize=(14, 5))
fig3.suptitle('Network Latency & Timing Breakdown', fontsize=14, fontweight='bold', y=0.98)

# 1. Draft vs Verify Time Comparison (Stacked Bar)
ax = axes[0]
categories = ['Easy\nK=2', 'Easy\nK=4', 'Easy\nK=6', 'Hard\nK=2', 'Hard\nK=4', 'Hard\nK=6']
draft_times = [
    np.mean([r['avg_draft_ms'] for r in easy_k2]),
    np.mean([r['avg_draft_ms'] for r in easy_k4]),
    np.mean([r['avg_draft_ms'] for r in easy_k6]),
    np.mean([r['avg_draft_ms'] for r in hard_k2]),
    np.mean([r['avg_draft_ms'] for r in hard_k4]),
    np.mean([r['avg_draft_ms'] for r in hard_k6])
]
verify_times = [
    np.mean([r['avg_verify_ms'] for r in easy_k2]),
    np.mean([r['avg_verify_ms'] for r in easy_k4]),
    np.mean([r['avg_verify_ms'] for r in easy_k6]),
    np.mean([r['avg_verify_ms'] for r in hard_k2]),
    np.mean([r['avg_verify_ms'] for r in hard_k4]),
    np.mean([r['avg_verify_ms'] for r in hard_k6])
]
x = np.arange(len(categories))
width = 0.6
bars1 = ax.bar(x, draft_times, width, label='Draft Time', color=k2_color, edgecolor='black')
bars2 = ax.bar(x, verify_times, width, bottom=draft_times, label='Verify Time', color=k4_color, edgecolor='black')
ax.set_ylabel('Time per Round (ms)')
ax.set_title('Draft vs Verify Time Breakdown', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=9)
ax.legend(loc='upper left')
ax.grid(axis='y', alpha=0.3)
# Add total time labels
total_times = [d + v for d, v in zip(draft_times, verify_times)]
for i, total in enumerate(total_times):
    ax.text(i, total + 5, f'{total:.0f}ms', ha='center', va='bottom', fontsize=8, fontweight='bold')

# 2. Total Latency vs Number of Rounds (Scatter)
ax = axes[1]
# Collect all speculative data points
all_spec = easy_k2 + easy_k4 + easy_k6 + hard_k2 + hard_k4 + hard_k6
latencies = [r['total_time_ms']/1000 for r in all_spec]  # Convert to seconds
rounds = [r['num_rounds'] for r in all_spec]
colors_scatter = [k2_color if 'K=2' in r['method'] else k4_color if 'K=4' in r['method'] else k6_color for r in all_spec]
ax.scatter(rounds, latencies, c=colors_scatter, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
ax.set_xlabel('Number of Rounds')
ax.set_ylabel('Total Latency (seconds)')
ax.set_title('Latency vs Rounds (Each point = 1 test)', fontweight='bold')
ax.grid(alpha=0.3)
# Add trend line
z = np.polyfit(rounds, latencies, 1)
p = np.poly1d(z)
ax.plot(sorted(rounds), p(sorted(rounds)), "r--", alpha=0.8, linewidth=2, label=f'Trend: {z[0]:.2f}s/round')
ax.legend(loc='upper left')

plt.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.savefig(outputs_dir / 'latency_analysis.png', dpi=200, bbox_inches='tight')
print("Saved: outputs/latency_analysis.png")

print("\nCharts generated successfully!")
print("  - outputs/comprehensive_detailed.png (Detailed analysis)")
print("  - outputs/latency_analysis.png (Network latency breakdown)")
