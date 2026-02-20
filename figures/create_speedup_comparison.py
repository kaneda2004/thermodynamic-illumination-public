#!/usr/bin/env python3
"""
Generate Speedup Comparison Figure for Cycle 5-6 Results
Shows how two-stage fixed N=150 beats all alternative approaches
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Create output directory
output_dir = Path(__file__).parent
output_dir.mkdir(exist_ok=True)

# Data from Cycle 5-6 experiments
# Two-stage variants (RES-224)
two_stage_variants = [
    ('Single-stage\nbaseline', 1.00, 0.0),
    ('Two-stage\nN=50', 88.73, 6.2),
    ('Two-stage\nN=100', 90.96, 8.0),
    ('Two-stage\nN=150\n(OPTIMAL)', 92.22, 1.4),
    ('Two-stage\nN=200', 84.96, 0.3),
]

# Three-stage variants (RES-229)
three_stage_variants = [
    ('Three-stage\n(50,50)', 80.28, 4.5),
    ('Three-stage\n(75,25)', 72.36, 5.1),
    ('Three-stage\n(25,75)', 72.36, 5.1),
]

# Refinement attempts (RES-230, RES-231)
refinement_variants = [
    ('Adaptive\nThreshold', 0.96, 8.2),
    ('Hybrid\nMulti-manifold', 84.62, 3.1),
]

# Combine all data
methods = []
speedups = []
errors = []
colors = []
labels_detailed = []

# Two-stage baseline group (blue)
for label, speedup, error in two_stage_variants:
    methods.append(label)
    speedups.append(speedup)
    errors.append(error)
    if 'N=150' in label:
        colors.append('#2E86AB')  # Dark blue for optimal
        labels_detailed.append('Two-Stage\nOptimal')
    else:
        colors.append('#A23B72')  # Purple for variants
        labels_detailed.append('Two-Stage\nVariant')

# Three-stage group (orange)
for label, speedup, error in three_stage_variants:
    methods.append(label)
    speedups.append(speedup)
    errors.append(error)
    colors.append('#F18F01')  # Orange
    labels_detailed.append('Three-Stage')

# Refinement attempts (red)
for label, speedup, error in refinement_variants:
    methods.append(label)
    speedups.append(speedup)
    errors.append(error)
    colors.append('#C1121F')  # Red for failed attempts
    labels_detailed.append('Refinement\nFailed')

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# --- Subplot 1: Full Comparison ---
x_pos = np.arange(len(methods))
bars1 = ax1.bar(x_pos, speedups, yerr=errors, capsize=5, color=colors,
                edgecolor='black', linewidth=1.5, alpha=0.8, error_kw={'linewidth': 2})

# Highlight the optimal
optimal_idx = 3
bars1[optimal_idx].set_linewidth(3)
bars1[optimal_idx].set_edgecolor('#2E86AB')
ax1.axhline(y=92.22, color='#2E86AB', linestyle='--', linewidth=2, alpha=0.5, label='Optimal (92.22×)')

ax1.set_ylabel('Speedup Factor (×)', fontsize=13, fontweight='bold')
ax1.set_xlabel('Sampling Strategy', fontsize=13, fontweight='bold')
ax1.set_title('Speedup Comparison: Why Two-Stage N=150 is Optimal\n(Cycle 5-6 Results)',
              fontsize=14, fontweight='bold', pad=20)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(methods, fontsize=10)
ax1.set_ylim(0, 105)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.legend(fontsize=11, loc='upper right')

# Add value labels on bars
for i, (bar, speedup, error) in enumerate(zip(bars1, speedups, errors)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + error + 2,
            f'{speedup:.1f}×', ha='center', va='bottom', fontsize=10, fontweight='bold')

# --- Subplot 2: Performance Distribution (Violin-style categorization) ---
categories = {
    'Two-Stage\nVariants': [88.73, 90.96, 92.22, 84.96],
    'Three-Stage\nApproaches': [80.28, 72.36, 72.36],
    'Refinement\nAttempts': [0.96, 84.62]
}

cat_positions = np.arange(len(categories))
cat_labels = list(categories.keys())
cat_colors_list = ['#A23B72', '#F18F01', '#C1121F']

# Plot as grouped distributions
for i, (cat_label, cat_speedups) in enumerate(categories.items()):
    x_vals = np.random.normal(i, 0.04, len(cat_speedups))
    ax2.scatter(x_vals, cat_speedups, s=150, alpha=0.6, color=cat_colors_list[i],
               edgecolor='black', linewidth=1.5)

    # Add mean line
    mean_speedup = np.mean(cat_speedups)
    ax2.hlines(mean_speedup, i - 0.2, i + 0.2, colors='black', linewidth=2.5, label=f'Mean: {mean_speedup:.1f}×')

# Add optimal reference line
ax2.axhline(y=92.22, color='#2E86AB', linestyle='--', linewidth=3, alpha=0.7, label='Optimal: 92.22×')

ax2.set_ylabel('Speedup Factor (×)', fontsize=13, fontweight='bold')
ax2.set_title('Speedup Distribution by Strategy Class\n(Higher is Better)',
              fontsize=14, fontweight='bold', pad=20)
ax2.set_xticks(cat_positions)
ax2.set_xticklabels(cat_labels, fontsize=11, fontweight='bold')
ax2.set_ylim(-5, 105)
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.legend(fontsize=10, loc='upper right')

# Add text annotation
ax2.text(0, 95, 'Two-Stage Fixed Budget\nOutperforms All Alternatives',
        ha='center', fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='#2E86AB', alpha=0.2))

plt.tight_layout()
plt.savefig(output_dir / 'speedup_comparison.pdf', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'speedup_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Generated: figures/speedup_comparison.pdf and .png")

plt.close()

# --- Create summary statistics table ---
print("\n=== Speedup Comparison Summary ===")
print(f"Optimal (Two-Stage N=150): 92.22× ± 1.4")
print(f"Second Best (Two-Stage N=100): 90.96× ± 8.0")
print(f"Third Best (Two-Stage N=50): 88.73× ± 6.2")
print(f"Best Three-Stage: 80.28× ± 4.5 (-13.0% vs optimal)")
print(f"Best Refinement: 84.62× ± 3.1 (-8.2% vs optimal)")
print(f"Worst Case: 0.96× (Adaptive threshold - complete failure)")
