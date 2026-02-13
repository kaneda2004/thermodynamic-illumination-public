#!/usr/bin/env python3
"""
Create weight space collapse figure (3-panel mechanistic visualization).

Visualizes:
- Panel A: Effective dimensionality vs order (showing 4.12D → 1.45D collapse)
- Panel B: Scaling exponent by regime (showing α jump 0.41 → 3.02)
- Panel C: Sample difficulty curve with phase transition marked
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load RES-218 data (weight space dimensionality)
res_218_path = Path('/Users/matt/Development/monochrome_noise_converger/results/weight_space_dimensionality/results.json')
with open(res_218_path, 'r') as f:
    res_218 = json.load(f)

# Load RES-215 data (phase transition)
res_215_path = Path('/Users/matt/Development/monochrome_noise_converger/results/threshold_scaling/res_215_final.json')
with open(res_215_path, 'r') as f:
    res_215 = json.load(f)

# Set up figure
fig = plt.figure(figsize=(14, 4.5))
gs = fig.add_gridspec(1, 3, wspace=0.35)

# ========== PANEL A: Effective Dimensionality vs Order ==========
ax_a = fig.add_subplot(gs[0, 0])

# Extract data from RES-218
orders = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
eff_dims = []
for order in orders:
    order_key = str(order) if order != 0.3 else "0.30000000000000004"
    eff_dim = res_218['concentration_results'][order_key]['effective_dim']
    eff_dims.append(eff_dim)

# Plot with smooth interpolation
orders_smooth = np.linspace(0, 0.5, 100)
eff_dims_interp = np.interp(orders_smooth, orders, eff_dims)

ax_a.fill_between(orders_smooth, eff_dims_interp, alpha=0.3, color='#e74c3c')
ax_a.plot(orders_smooth, eff_dims_interp, 'o-', color='#c0392b', linewidth=3, markersize=0, label='Effective dimension')
ax_a.scatter(orders, eff_dims, s=100, color='#c0392b', edgecolors='black', linewidth=1.5, zorder=5)

# Annotate key points
ax_a.annotate('4.12D\n(low order)', xy=(0.0, eff_dims[0]), xytext=(0.08, 3.8),
              fontsize=10, fontweight='bold', ha='left',
              arrowprops=dict(arrowstyle='->', color='#c0392b', lw=2))
ax_a.annotate('1.45D\n(high order)', xy=(0.5, eff_dims[-1]), xytext=(0.32, 1.8),
              fontsize=10, fontweight='bold', ha='left',
              arrowprops=dict(arrowstyle='->', color='#c0392b', lw=2))

ax_a.set_xlabel('CPPN Order', fontsize=12, fontweight='bold')
ax_a.set_ylabel('Effective Dimensionality (PCA)', fontsize=12, fontweight='bold')
ax_a.set_title('Panel A: Weight Space Collapse', fontsize=13, fontweight='bold')
ax_a.grid(True, alpha=0.3, linestyle='--')
ax_a.set_xlim(-0.05, 0.55)
ax_a.set_ylim(1.2, 4.3)

# ========== PANEL B: Scaling Exponent by Regime ==========
ax_b = fig.add_subplot(gs[0, 1])

regimes = ['Early\n(P10-P50)', 'Late\n(P50-P90)']
exponents = [res_215['power_law']['low_slope'], res_215['power_law']['high_slope']]
colors = ['#3498db', '#e74c3c']

bars = ax_b.bar(regimes, exponents, color=colors, edgecolor='black', linewidth=2, alpha=0.8, width=0.6)

# Add value labels on bars
for i, (bar, exp) in enumerate(zip(bars, exponents)):
    height = bar.get_height()
    ax_b.text(bar.get_x() + bar.get_width()/2., height + 0.15,
              f'{exp:.2f}',
              ha='center', va='bottom', fontsize=12, fontweight='bold')

# Add transition magnitude annotation
transition = res_215['power_law']['transition_magnitude']
ax_b.annotate('', xy=(0.5, exponents[0]), xytext=(0.5, exponents[1]),
              arrowprops=dict(arrowstyle='<->', color='#2c3e50', lw=3))
ax_b.text(0.7, (exponents[0] + exponents[1])/2, f'Δα = {transition:.2f}\np = 0.026',
          fontsize=11, fontweight='bold', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

ax_b.set_ylabel('Scaling Exponent (α)', fontsize=12, fontweight='bold')
ax_b.set_title('Panel B: Phase Transition', fontsize=13, fontweight='bold')
ax_b.set_ylim(0, 3.5)
ax_b.grid(True, alpha=0.3, axis='y', linestyle='--')

# Add horizontal lines for reference
ax_b.axhline(y=1.0, color='gray', linestyle=':', linewidth=2, alpha=0.5, label='Linear (α=1)')
ax_b.legend(loc='upper left', fontsize=10)

# ========== PANEL C: Sample Difficulty Curve ==========
ax_c = fig.add_subplot(gs[0, 2])

# Create sample difficulty curve from RES-215 data
percentiles = np.array(res_215['percentiles'])  # [10, 25, 50, 75, 90]
sample_counts = np.array(res_215['sample_counts'])  # [1.0, 1.25, 1.95, 4.56, 12.8]

# Fit early and late regimes separately
early_mask = percentiles <= 50
late_mask = percentiles >= 50

if sum(early_mask) >= 2:
    p_early = np.polyfit(np.log10(percentiles[early_mask]), np.log10(sample_counts[early_mask]), 1)
    p_early_fn = np.poly1d(p_early)

if sum(late_mask) >= 2:
    p_late = np.polyfit(np.log10(percentiles[late_mask]), np.log10(sample_counts[late_mask]), 1)
    p_late_fn = np.poly1d(p_late)

# Plot curves
percentiles_early = np.linspace(10, 50, 50)
percentiles_late = np.linspace(50, 90, 50)

samples_early = 10 ** p_early_fn(np.log10(percentiles_early))
samples_late = 10 ** p_late_fn(np.log10(percentiles_late))

ax_c.loglog(percentiles_early, samples_early, '-', color='#3498db', linewidth=3, label=f'Early (α={p_early[0]:.2f})')
ax_c.loglog(percentiles_late, samples_late, '-', color='#e74c3c', linewidth=3, label=f'Late (α={p_late[0]:.2f})')
ax_c.scatter(percentiles, sample_counts, s=150, color='black', edgecolors='white', linewidth=2, zorder=5)

# Mark transition point
transition_percentile = 50
transition_samples = 10 ** p_early_fn(np.log10(transition_percentile))
ax_c.axvline(transition_percentile, color='#2c3e50', linestyle='--', linewidth=2, alpha=0.7, label='Phase transition')
ax_c.scatter([transition_percentile], [transition_samples], s=300, marker='*',
             color='gold', edgecolors='black', linewidth=2, zorder=10)

ax_c.set_xlabel('Percentile of Order Distribution', fontsize=12, fontweight='bold')
ax_c.set_ylabel('Samples Required (log scale)', fontsize=12, fontweight='bold')
ax_c.set_title('Panel C: Sampling Difficulty', fontsize=13, fontweight='bold')
ax_c.legend(loc='upper left', fontsize=10)
ax_c.grid(True, alpha=0.3, which='both', linestyle='--')

# Add main title
fig.suptitle('Weight Space Collapse & Phase Transition in Sampling Difficulty',
             fontsize=14, fontweight='bold', y=1.00)

plt.tight_layout()
plt.savefig('/Users/matt/Development/monochrome_noise_converger/figures/weight_space_collapse.pdf',
            bbox_inches='tight', dpi=300)
plt.savefig('/Users/matt/Development/monochrome_noise_converger/figures/weight_space_collapse.png',
            bbox_inches='tight', dpi=300)
print("✓ Saved weight_space_collapse.pdf and .png")
plt.close()
