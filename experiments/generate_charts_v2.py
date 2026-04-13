"""
Generate professional charts for Direct vs Speculative Decoding comparison.
Optimized layout for better visibility.
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load results
with open('results.json', 'r') as f:
    data = json.load(f)

direct_results = data['direct']
spec_results = data['speculative']
summary = data['summary']

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9

# Colors
direct_color = '#3498db'
spec_color = '#e74c3c'
draft_color = '#27ae60'
verify_color = '#f39c12'
network_color = '#9b59b6'

# Figure 1: Main Comparison (2x2 layout, larger)
fig1 = plt.figure(figsize=(16, 12))
fig1.suptitle('Direct vs Speculative Decoding: Performance Comparison', 
              fontsize=14, fontweight='bold', y=0.98)

# 1. Speed Comparison
ax1 = plt.subplot(2, 2, 1)
methods = ['Direct', 'Speculative']
speeds = [summary['direct_avg_tps'], summary['spec_avg_tps']]
bars = ax1.bar(methods, speeds, color=[direct_color, spec_color], 
               width=0.5, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Tokens per Second', fontweight='bold')
ax1.set_title('Generation Speed', fontweight='bold', pad=15)
ax1.set_ylim(0, 25)
for bar, speed in zip(bars, speeds):
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
             f'{speed:.2f}', ha='center', va='bottom', fontsize=14, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)
speedup = summary['direct_avg_tps'] / summary['spec_avg_tps']
ax1.text(0.5, 20, f'Direct is {speedup:.1f}x faster', 
         ha='center', fontsize=12, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.5))

# 2. Acceptance Rate
ax2 = plt.subplot(2, 2, 2)
acceptance_rates = [r['acceptance_rate'] * 100 for r in spec_results]
test_labels = [f'Test {i+1}' for i in range(len(spec_results))]
colors_acc = [plt.cm.RdYlGn(rate/100) for rate in acceptance_rates]
bars = ax2.bar(test_labels, acceptance_rates, color=colors_acc, 
               edgecolor='black', linewidth=1.5)
ax2.axhline(y=37.4, color='red', linestyle='--', linewidth=2, 
            label=f'Average: 37.4%')
ax2.axhline(y=65, color='green', linestyle=':', linewidth=2, 
            label='Target: 65%')
ax2.set_xlabel('Test', fontweight='bold')
ax2.set_ylabel('Acceptance Rate (%)', fontweight='bold')
ax2.set_title('Speculative Decoding Acceptance Rate', fontweight='bold', pad=15)
ax2.set_ylim(0, 100)
ax2.legend(loc='upper right', frameon=True)
ax2.grid(axis='y', alpha=0.3)

# 3. Time Breakdown
ax3 = plt.subplot(2, 2, 3)
avg_draft = np.mean([r['avg_draft_ms'] for r in spec_results])
avg_verify = np.mean([r['avg_verify_ms'] for r in spec_results])
avg_round_time = np.mean([r['total_time_ms']/r['num_rounds'] for r in spec_results])
avg_network = avg_round_time - avg_draft - avg_verify

components = ['Draft\nGeneration', 'Server\nVerification', 'Network\nLatency']
times = [avg_draft, avg_verify, avg_network]
colors_comp = [draft_color, verify_color, network_color]
bars = ax3.bar(components, times, color=colors_comp, 
               edgecolor='black', linewidth=1.5)
ax3.set_ylabel('Time per Round (ms)', fontweight='bold')
ax3.set_title('Speculative Decoding: Time per Round', fontweight='bold', pad=15)
for bar, time_val, label in zip(bars, times, components):
    ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2,
             f'{time_val:.1f}ms', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)

# 4. Total Time Comparison
ax4 = plt.subplot(2, 2, 4)
x = np.arange(len(direct_results))
width = 0.35
direct_total = [r['total_time_ms']/1000 for r in direct_results]
spec_total = [r['total_time_ms']/1000 for r in spec_results]
ax4.bar(x - width/2, direct_total, width, label='Direct', 
        color=direct_color, edgecolor='black', linewidth=1.5)
ax4.bar(x + width/2, spec_total, width, label='Speculative', 
        color=spec_color, edgecolor='black', linewidth=1.5)
ax4.set_xlabel('Test', fontweight='bold')
ax4.set_ylabel('Total Time (seconds)', fontweight='bold')
ax4.set_title('Total Generation Time', fontweight='bold', pad=15)
ax4.set_xticks(x)
ax4.set_xticklabels([f'Test {i+1}' for i in range(len(direct_results))])
ax4.legend(loc='upper left', frameon=True)
ax4.grid(axis='y', alpha=0.3)

# Add summary text
summary_text = (
    f"Summary: Direct={summary['direct_avg_tps']:.1f} tok/s | "
    f"Speculative={summary['spec_avg_tps']:.1f} tok/s | "
    f"Acceptance={summary['spec_avg_acceptance']*100:.1f}% | "
    f"Speedup={speedup:.1f}x"
)
fig1.text(0.5, 0.01, summary_text, ha='center', fontsize=11, 
          bbox=dict(boxstyle='round,pad=0.8', facecolor='lightblue', alpha=0.5),
          fontweight='bold')

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.savefig('charts_main.png', dpi=200, bbox_inches='tight')
print("Saved: charts_main.png")

# Figure 2: Detailed Analysis
fig2 = plt.figure(figsize=(14, 10))
fig2.suptitle('Detailed Performance Metrics', fontsize=14, fontweight='bold', y=0.98)

# 1. Tokens Generated
ax1 = plt.subplot(2, 3, 1)
direct_tokens = [r['tokens_generated'] for r in direct_results]
spec_tokens = [r['tokens_generated'] for r in spec_results]
x = np.arange(len(direct_results))
width = 0.35
ax1.bar(x - width/2, direct_tokens, width, label='Direct', 
        color=direct_color, edgecolor='black')
ax1.bar(x + width/2, spec_tokens, width, label='Speculative', 
        color=spec_color, edgecolor='black')
ax1.set_xlabel('Test')
ax1.set_ylabel('Tokens Generated')
ax1.set_title('Tokens per Test')
ax1.set_xticks(x)
ax1.set_xticklabels([f'{i+1}' for i in range(len(direct_results))])
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# 2. Number of Rounds
ax2 = plt.subplot(2, 3, 2)
rounds = [r['num_rounds'] for r in spec_results]
ax2.bar(test_labels, rounds, color=spec_color, edgecolor='black')
ax2.set_xlabel('Test')
ax2.set_ylabel('Number of Rounds')
ax2.set_title('Speculative: Rounds per Test')
ax2.grid(axis='y', alpha=0.3)

# 3. Speed per Test
ax3 = plt.subplot(2, 3, 3)
direct_tps = [r['tokens_per_second'] for r in direct_results]
spec_tps = [r['tokens_per_second'] for r in spec_results]
ax3.bar(x - width/2, direct_tps, width, label='Direct', color=direct_color, edgecolor='black')
ax3.bar(x + width/2, spec_tps, width, label='Speculative', color=spec_color, edgecolor='black')
ax3.set_xlabel('Test')
ax3.set_ylabel('Tokens per Second')
ax3.set_title('Speed per Test')
ax3.set_xticks(x)
ax3.set_xticklabels([f'{i+1}' for i in range(len(direct_results))])
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

# 4. Draft vs Verify Time
ax4 = plt.subplot(2, 3, 4)
draft_times = [r['avg_draft_ms'] for r in spec_results]
verify_times = [r['avg_verify_ms'] for r in spec_results]
ax4.bar(x - width/2, draft_times, width, label='Draft', color=draft_color, edgecolor='black')
ax4.bar(x + width/2, verify_times, width, label='Verify', color=verify_color, edgecolor='black')
ax4.set_xlabel('Test')
ax4.set_ylabel('Time per Round (ms)')
ax4.set_title('Draft vs Verify Time')
ax4.set_xticks(x)
ax4.set_xticklabels([f'{i+1}' for i in range(len(spec_results))])
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

# 5. Time Distribution (Box plot)
ax5 = plt.subplot(2, 3, 5)
bp = ax5.boxplot([direct_total, spec_total], tick_labels=['Direct', 'Speculative'],
                  patch_artist=True, widths=0.5)
bp['boxes'][0].set_facecolor(direct_color)
bp['boxes'][1].set_facecolor(spec_color)
for box in bp['boxes']:
    box.set_edgecolor('black')
    box.set_linewidth(1.5)
ax5.set_ylabel('Time (seconds)')
ax5.set_title('Time Distribution')
ax5.grid(axis='y', alpha=0.3)

# 6. Acceptance Rate Trend
ax6 = plt.subplot(2, 3, 6)
ax6.plot(test_labels, acceptance_rates, marker='o', color=spec_color, 
         linewidth=2, markersize=10, markerfacecolor='white', markeredgewidth=2)
ax6.axhline(y=37.4, color='red', linestyle='--', linewidth=2, label='Average')
ax6.axhline(y=65, color='green', linestyle=':', linewidth=2, label='Target (65%)')
ax6.set_xlabel('Test')
ax6.set_ylabel('Acceptance Rate (%)')
ax6.set_title('Acceptance Rate Trend')
ax6.set_ylim(0, 100)
ax6.legend()
ax6.grid(alpha=0.3)

plt.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.savefig('charts_detailed.png', dpi=200, bbox_inches='tight')
print("Saved: charts_detailed.png")

print("\nCharts generated successfully!")
print("  - charts_main.png (Main comparison)")
print("  - charts_detailed.png (Detailed metrics)")
