"""
Generate professional charts for Direct vs Speculative Decoding comparison.
All labels and text in English.
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
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# Create figure with 2x3 layout
fig = plt.figure(figsize=(18, 12))
fig.suptitle('Direct Request vs Speculative Decoding Performance Comparison', 
             fontsize=16, fontweight='bold', y=0.98)

# Colors
direct_color = '#3498db'  # Blue
spec_color = '#e74c3c'    # Red
draft_color = '#2ecc71'   # Green
verify_color = '#f39c12'  # Orange
network_color = '#9b59b6' # Purple

# 1. Tokens per Second Comparison (Bar Chart)
ax1 = fig.add_subplot(2, 3, 1)
methods = ['Direct', 'Speculative']
speeds = [summary['direct_avg_tps'], summary['spec_avg_tps']]
bars = ax1.bar(methods, speeds, color=[direct_color, spec_color], width=0.5, edgecolor='black', linewidth=1.2)
ax1.set_ylabel('Tokens per Second', fontweight='bold')
ax1.set_title('Generation Speed Comparison', fontweight='bold', pad=10)
ax1.set_ylim(0, max(speeds) * 1.2)
for bar, speed in zip(bars, speeds):
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
             f'{speed:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# Add speedup annotation
speedup = summary['direct_avg_tps'] / summary['spec_avg_tps']
ax1.annotate(f'Direct is {speedup:.1f}x faster', 
               xy=(0.5, max(speeds) * 0.5), xycoords='axes fraction',
               ha='center', fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))

# 2. Time Distribution per Test (Box Plot)
ax2 = fig.add_subplot(2, 3, 2)
direct_times = [r['total_time_ms']/1000 for r in direct_results]
spec_times = [r['total_time_ms']/1000 for r in spec_results]
bp = ax2.boxplot([direct_times, spec_times], tick_labels=['Direct', 'Speculative'],
                  patch_artist=True, widths=0.5)
bp['boxes'][0].set_facecolor(direct_color)
bp['boxes'][1].set_facecolor(spec_color)
for box in bp['boxes']:
    box.set_edgecolor('black')
    box.set_linewidth(1.5)
ax2.set_ylabel('Time (seconds)', fontweight='bold')
ax2.set_title('Time Distribution per Test', fontweight='bold', pad=10)
ax2.grid(axis='y', alpha=0.3)

# 3. Tokens Generated per Test (Grouped Bar)
ax3 = fig.add_subplot(2, 3, 3)
x = np.arange(len(direct_results))
width = 0.35
direct_tokens = [r['tokens_generated'] for r in direct_results]
spec_tokens = [r['tokens_generated'] for r in spec_results]
bars1 = ax3.bar(x - width/2, direct_tokens, width, label='Direct', 
                color=direct_color, edgecolor='black', linewidth=1)
bars2 = ax3.bar(x + width/2, spec_tokens, width, label='Speculative',
                color=spec_color, edgecolor='black', linewidth=1)
ax3.set_xlabel('Test Prompt', fontweight='bold')
ax3.set_ylabel('Tokens Generated', fontweight='bold')
ax3.set_title('Tokens Generated per Test', fontweight='bold', pad=10)
ax3.set_xticks(x)
ax3.set_xticklabels([f'Test {i+1}' for i in range(len(direct_results))])
ax3.legend(loc='upper right', frameon=True, fancybox=True)
ax3.grid(axis='y', alpha=0.3)

# 4. Speculative Decoding Time Breakdown (Stacked Bar)
ax4 = fig.add_subplot(2, 3, 4)
avg_draft = np.mean([r['avg_draft_ms'] for r in spec_results])
avg_verify = np.mean([r['avg_verify_ms'] for r in spec_results])
# Estimate network time from total minus draft and verify
avg_network = np.mean([r['total_time_ms']/r['num_rounds'] - r['avg_draft_ms'] - r['avg_verify_ms'] 
                       for r in spec_results])
components = ['Draft', 'Verify', 'Network']
times = [avg_draft, avg_verify, avg_network]
colors = [draft_color, verify_color, network_color]
bars = ax4.bar(components, times, color=colors, edgecolor='black', linewidth=1.2)
ax4.set_ylabel('Time per Round (ms)', fontweight='bold')
ax4.set_title('Speculative Decoding Time Breakdown', fontweight='bold', pad=10)
for bar, time_val in zip(bars, times):
    ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
             f'{time_val:.1f}ms', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax4.grid(axis='y', alpha=0.3)

# 5. Acceptance Rate by Test (Bar Chart)
ax5 = fig.add_subplot(2, 3, 5)
acceptance_rates = [r['acceptance_rate'] * 100 for r in spec_results]
test_labels = [f'Test {i+1}' for i in range(len(spec_results))]
colors_acc = [plt.cm.RdYlGn(rate/100) for rate in acceptance_rates]
bars = ax5.bar(test_labels, acceptance_rates, color=colors_acc, edgecolor='black', linewidth=1)
ax5.axhline(y=37.4, color='red', linestyle='--', linewidth=2, 
            label=f'Avg: {summary["spec_avg_acceptance"]*100:.1f}%')
ax5.axhline(y=65, color='green', linestyle=':', linewidth=2, 
            label='Target: 65%')
ax5.set_xlabel('Test Prompt', fontweight='bold')
ax5.set_ylabel('Acceptance Rate (%)', fontweight='bold')
ax5.set_title('Speculative Decoding Acceptance Rate', fontweight='bold', pad=10)
ax5.set_ylim(0, 100)
ax5.legend(loc='upper right', frameon=True)
ax5.grid(axis='y', alpha=0.3)
for bar, rate in zip(bars, acceptance_rates):
    ax5.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
             f'{rate:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

# 6. Speedup Factor with Thresholds
ax6 = fig.add_subplot(2, 3, 6)
speedup_value = summary['direct_avg_tps'] / summary['spec_avg_tps']
# Create gauge-like visualization
bars = ax6.barh(['Current'], [speedup_value], color=[spec_color], height=0.3, 
                edgecolor='black', linewidth=2)
ax6.axvline(x=1.0, color='green', linestyle='--', linewidth=2, label='Break-even (1x)')
ax6.axvspan(0, 1, alpha=0.2, color='red', label='Slower (Direct wins)')
ax6.axvspan(1, speedup_value*1.2, alpha=0.2, color='green', label='Faster (Spec wins)')
ax6.set_xlabel('Speedup Factor (Direct / Speculative)', fontweight='bold')
ax6.set_title('Performance Ratio', fontweight='bold', pad=10)
ax6.set_xlim(0, speedup_value * 1.2)
ax6.text(speedup_value/2, 0, f'{speedup_value:.1f}x', ha='center', va='center',
         fontsize=20, fontweight='bold', color='white')
ax6.legend(loc='upper right', frameon=True)
ax6.grid(axis='x', alpha=0.3)

# Add summary text box at the bottom
summary_text = (
    f"Key Metrics:\n"
    f"• Direct Speed: {summary['direct_avg_tps']:.2f} tokens/s\n"
    f"• Speculative Speed: {summary['spec_avg_tps']:.2f} tokens/s\n"
    f"• Average Acceptance Rate: {summary['spec_avg_acceptance']*100:.1f}%\n"
    f"• Average Rounds per Test: {np.mean([r['num_rounds'] for r in spec_results]):.1f}\n"
    f"• Direct is {speedup_value:.1f}x faster than Speculative"
)
fig.text(0.5, 0.02, summary_text, ha='center', fontsize=11, 
         bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.3),
         family='monospace')

plt.tight_layout(rect=[0, 0.08, 1, 0.96])
plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
print("Chart saved: performance_comparison.png")

# Create a second figure for detailed metrics
fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
fig2.suptitle('Detailed Performance Analysis', fontsize=14, fontweight='bold', y=0.98)

# 1. Tokens per Second by Test
ax1 = axes[0, 0]
x = np.arange(len(direct_results))
width = 0.35
direct_tps = [r['tokens_per_second'] for r in direct_results]
spec_tps = [r['tokens_per_second'] for r in spec_results]
ax1.bar(x - width/2, direct_tps, width, label='Direct', color=direct_color, edgecolor='black')
ax1.bar(x + width/2, spec_tps, width, label='Speculative', color=spec_color, edgecolor='black')
ax1.set_xlabel('Test Prompt')
ax1.set_ylabel('Tokens per Second')
ax1.set_title('Generation Speed per Test')
ax1.set_xticks(x)
ax1.set_xticklabels([f'Test {i+1}' for i in range(len(direct_results))])
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# 2. Number of Rounds
ax2 = axes[0, 1]
rounds = [r['num_rounds'] for r in spec_results]
ax2.bar(test_labels, rounds, color=spec_color, edgecolor='black')
ax2.set_xlabel('Test Prompt')
ax2.set_ylabel('Number of Rounds')
ax2.set_title('Speculative Decoding: Rounds per Test')
ax2.grid(axis='y', alpha=0.3)
for i, (bar, val) in enumerate(zip(ax2.patches, rounds)):
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
             f'{val}', ha='center', va='bottom', fontsize=10)

# 3. Time per Round (Draft vs Verify)
ax3 = axes[1, 0]
x = np.arange(len(spec_results))
width = 0.35
draft_times = [r['avg_draft_ms'] for r in spec_results]
verify_times = [r['avg_verify_ms'] for r in spec_results]
ax3.bar(x - width/2, draft_times, width, label='Draft', color=draft_color, edgecolor='black')
ax3.bar(x + width/2, verify_times, width, label='Verify', color=verify_color, edgecolor='black')
ax3.set_xlabel('Test Prompt')
ax3.set_ylabel('Time per Round (ms)')
ax3.set_title('Draft vs Verify Time per Round')
ax3.set_xticks(x)
ax3.set_xticklabels([f'Test {i+1}' for i in range(len(spec_results))])
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

# 4. Total Time Comparison
ax4 = axes[1, 1]
direct_total = [r['total_time_ms']/1000 for r in direct_results]
spec_total = [r['total_time_ms']/1000 for r in spec_results]
x = np.arange(len(direct_results))
width = 0.35
ax4.bar(x - width/2, direct_total, width, label='Direct', color=direct_color, edgecolor='black')
ax4.bar(x + width/2, spec_total, width, label='Speculative', color=spec_color, edgecolor='black')
ax4.set_xlabel('Test Prompt')
ax4.set_ylabel('Total Time (seconds)')
ax4.set_title('Total Generation Time')
ax4.set_xticks(x)
ax4.set_xticklabels([f'Test {i+1}' for i in range(len(direct_results))])
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.savefig('detailed_analysis.png', dpi=300, bbox_inches='tight')
print("Chart saved: detailed_analysis.png")

print("\nAll charts generated successfully!")
