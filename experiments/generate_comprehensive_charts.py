"""
Generate comprehensive charts for Easy vs Hard prompts with K=2, K=4, K=6, and Direct.
Tests: 50 easy + 50 hard prompts × 4 methods = 400 tests
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load results
with open('comprehensive_results.json', 'r') as f:
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

# Figure 1: Main Comparison
fig1, axes = plt.subplots(2, 3, figsize=(16, 10))
fig1.suptitle('Comprehensive Benchmark: Easy vs Hard Prompts (50 each)\nDirect vs K=2 vs K=4 vs K=6', 
              fontsize=14, fontweight='bold', y=0.98)

# 1. Speed Comparison - Easy
ax = axes[0, 0]
methods = ['Direct', 'K=2', 'K=4', 'K=6']
easy_speeds = [
    np.mean([r['tokens_per_second'] for r in easy_direct]),
    np.mean([r['tokens_per_second'] for r in easy_k2]),
    np.mean([r['tokens_per_second'] for r in easy_k4]),
    np.mean([r['tokens_per_second'] for r in easy_k6])
]
bars = ax.bar(methods, easy_speeds, color=[direct_color, k2_color, k4_color, k6_color], 
              edgecolor='black', linewidth=1.5)
ax.set_ylabel('Tokens per Second')
ax.set_title('EASY Prompts: Generation Speed', fontweight='bold')
for bar, speed in zip(bars, easy_speeds):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
            f'{speed:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_ylim(0, 25)
ax.grid(axis='y', alpha=0.3)

# 2. Speed Comparison - Hard
ax = axes[0, 1]
hard_speeds = [
    np.mean([r['tokens_per_second'] for r in hard_direct]),
    np.mean([r['tokens_per_second'] for r in hard_k2]),
    np.mean([r['tokens_per_second'] for r in hard_k4]),
    np.mean([r['tokens_per_second'] for r in hard_k6])
]
bars = ax.bar(methods, hard_speeds, color=[direct_color, k2_color, k4_color, k6_color], 
              edgecolor='black', linewidth=1.5)
ax.set_ylabel('Tokens per Second')
ax.set_title('HARD Prompts: Generation Speed', fontweight='bold')
for bar, speed in zip(bars, hard_speeds):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
            f'{speed:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_ylim(0, 25)
ax.grid(axis='y', alpha=0.3)

# 3. Speedup vs Direct
ax = axes[0, 2]
x = np.arange(2)
width = 0.25
k2_speedups = [easy_speeds[1]/easy_speeds[0], hard_speeds[1]/hard_speeds[0]]
k4_speedups = [easy_speeds[2]/easy_speeds[0], hard_speeds[2]/hard_speeds[0]]
k6_speedups = [easy_speeds[3]/easy_speeds[0], hard_speeds[3]/hard_speeds[0]]
bars1 = ax.bar(x - width, k2_speedups, width, label='K=2', color=k2_color, edgecolor='black')
bars2 = ax.bar(x, k4_speedups, width, label='K=4', color=k4_color, edgecolor='black')
bars3 = ax.bar(x + width, k6_speedups, width, label='K=6', color=k6_color, edgecolor='black')
ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Direct baseline')
ax.set_ylabel('Speedup Factor')
ax.set_title('Speedup vs Direct (1.0 = same speed)', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['Easy', 'Hard'])
ax.legend(loc='upper right')
ax.set_ylim(0, 1.5)
ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars1, k2_speedups):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
            f'{val:.2f}x', ha='center', va='bottom', fontsize=9)
for bar, val in zip(bars2, k4_speedups):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
            f'{val:.2f}x', ha='center', va='bottom', fontsize=9)
for bar, val in zip(bars3, k6_speedups):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
            f'{val:.2f}x', ha='center', va='bottom', fontsize=9)

# 4. Acceptance Rate Distribution - Easy
ax = axes[1, 0]
easy_k2_acc = [r['acceptance_rate']*100 for r in easy_k2]
easy_k4_acc = [r['acceptance_rate']*100 for r in easy_k4]
easy_k6_acc = [r['acceptance_rate']*100 for r in easy_k6]
bp = ax.boxplot([easy_k2_acc, easy_k4_acc, easy_k6_acc], labels=['K=2', 'K=4', 'K=6'],
                patch_artist=True, widths=0.6)
bp['boxes'][0].set_facecolor(k2_color)
bp['boxes'][1].set_facecolor(k4_color)
bp['boxes'][2].set_facecolor(k6_color)
ax.axhline(y=np.mean(easy_k2_acc), color=k2_color, linestyle='--', alpha=0.7, label=f'K=2 Avg: {np.mean(easy_k2_acc):.1f}%')
ax.axhline(y=np.mean(easy_k4_acc), color=k4_color, linestyle='--', alpha=0.7, label=f'K=4 Avg: {np.mean(easy_k4_acc):.1f}%')
ax.axhline(y=np.mean(easy_k6_acc), color=k6_color, linestyle='--', alpha=0.7, label=f'K=6 Avg: {np.mean(easy_k6_acc):.1f}%')
ax.set_ylabel('Acceptance Rate (%)')
ax.set_title('EASY: Acceptance Rate Distribution (50 prompts)', fontweight='bold')
ax.legend(loc='upper right', fontsize=8)
ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.3)

# 5. Acceptance Rate Distribution - Hard
ax = axes[1, 1]
hard_k2_acc = [r['acceptance_rate']*100 for r in hard_k2]
hard_k4_acc = [r['acceptance_rate']*100 for r in hard_k4]
hard_k6_acc = [r['acceptance_rate']*100 for r in hard_k6]
bp = ax.boxplot([hard_k2_acc, hard_k4_acc, hard_k6_acc], labels=['K=2', 'K=4', 'K=6'],
                patch_artist=True, widths=0.6)
bp['boxes'][0].set_facecolor(k2_color)
bp['boxes'][1].set_facecolor(k4_color)
bp['boxes'][2].set_facecolor(k6_color)
ax.axhline(y=np.mean(hard_k2_acc), color=k2_color, linestyle='--', alpha=0.7, label=f'K=2 Avg: {np.mean(hard_k2_acc):.1f}%')
ax.axhline(y=np.mean(hard_k4_acc), color=k4_color, linestyle='--', alpha=0.7, label=f'K=4 Avg: {np.mean(hard_k4_acc):.1f}%')
ax.axhline(y=np.mean(hard_k6_acc), color=k6_color, linestyle='--', alpha=0.7, label=f'K=6 Avg: {np.mean(hard_k6_acc):.1f}%')
ax.set_ylabel('Acceptance Rate (%)')
ax.set_title('HARD: Acceptance Rate Distribution (50 prompts)', fontweight='bold')
ax.legend(loc='upper right', fontsize=8)
ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.3)

# 6. Number of Rounds
ax = axes[1, 2]
easy_k2_rounds = [r['num_rounds'] for r in easy_k2]
easy_k4_rounds = [r['num_rounds'] for r in easy_k4]
easy_k6_rounds = [r['num_rounds'] for r in easy_k6]
hard_k2_rounds = [r['num_rounds'] for r in hard_k2]
hard_k4_rounds = [r['num_rounds'] for r in hard_k4]
hard_k6_rounds = [r['num_rounds'] for r in hard_k6]
x = np.arange(6)
labels = ['Easy K=2', 'Easy K=4', 'Easy K=6', 'Hard K=2', 'Hard K=4', 'Hard K=6']
avg_rounds = [np.mean(easy_k2_rounds), np.mean(easy_k4_rounds), np.mean(easy_k6_rounds),
              np.mean(hard_k2_rounds), np.mean(hard_k4_rounds), np.mean(hard_k6_rounds)]
colors_bar = [k2_color, k4_color, k6_color, k2_color, k4_color, k6_color]
bars = ax.bar(x, avg_rounds, color=colors_bar, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Average Rounds')
ax.set_title('Average Rounds per Configuration', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
for bar, val in zip(bars, avg_rounds):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
            f'{val:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.savefig('comprehensive_main.png', dpi=200, bbox_inches='tight')
print("Saved: comprehensive_main.png")

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
plt.savefig('comprehensive_detailed.png', dpi=200, bbox_inches='tight')
print("Saved: comprehensive_detailed.png")

print("\nAll charts generated successfully!")
print("  - comprehensive_main.png (Main comparison - 50 prompts, K=2/4/6 + Direct)")
print("  - comprehensive_detailed.png (Detailed analysis with distributions)")
