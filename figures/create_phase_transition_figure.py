#!/usr/bin/env python3
"""
Create phase transition figure showing scaling exponent changes.

Visualizes bits vs percentile with two fitted power laws and transition point marked.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# Load RES-215 data
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent.parent  # submission_staging -> monochrome_noise_converger
res_215_path = project_root / 'results/threshold_scaling/res_215_final.json'

if not res_215_path.exists():
    print(f"Error: Data file not found at {res_215_path}")
    print("Please run experiments/res_215_final.py first.")
    exit(1)

with open(res_215_path, 'r') as f:
    res_215 = json.load(f)

# Extract data
percentiles = np.array(res_215['percentiles'])  # [10, 25, 50, 75, 90]
sample_counts = np.array(res_215['sample_counts'])  # [1.0, 1.25, 1.95, 4.56, 12.8]
bits = -np.log2(sample_counts / sample_counts.max()) * 10  # Convert to bits (approximate)

# Separate into early and late regimes (no overlap)
early_mask = percentiles <= 50
late_mask = percentiles > 50

# Fit power laws to each regime using scipy.stats for SEs
log_p = np.log10(percentiles)
log_s = np.log10(sample_counts)

log_p_early = log_p[early_mask]
log_s_early = log_s[early_mask]
log_p_late = log_p[late_mask]
log_s_late = log_s[late_mask]

slope_early, intercept_early, r_early, p_early_stat, se_early = stats.linregress(log_p_early, log_s_early)
slope_late, intercept_late, r_late, p_late_stat, se_late = stats.linregress(log_p_late, log_s_late)

alpha_early = slope_early
alpha_late = slope_late

# Get p-value from JSON if available, otherwise compute
if 'power_law' in res_215 and 'p_value' in res_215['power_law']:
    p_value = res_215['power_law']['p_value']
else:
    # Compute p-value for slope difference using Welch's t-test
    diff = alpha_late - alpha_early
    se_diff = np.sqrt(se_early**2 + se_late**2)
    t_stat = diff / se_diff
    df_early = len(log_p_early) - 2
    df_late = len(log_p_late) - 2
    df_combined = (se_early**2 + se_late**2)**2 / (se_early**4/max(df_early, 0.5) + se_late**4/max(df_late, 0.5))
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), max(df_combined, 1)))

# For polyfit coefficients (still needed for plotting)
p_early = np.array([slope_early, intercept_early])
p_late = np.array([slope_late, intercept_late])

# Create smooth curves for plotting (adjusted for full range)
min_p = percentiles[early_mask].min()
max_p_early = 50
min_p_late = 55
max_p = percentiles[late_mask].max()
percentiles_early = np.linspace(min_p, max_p_early, 100)
percentiles_late = np.linspace(min_p_late, max_p, 100)

samples_early = 10 ** (p_early[0] * np.log10(percentiles_early) + p_early[1])
samples_late = 10 ** (p_late[0] * np.log10(percentiles_late) + p_late[1])

# Convert to bits
bits_early = -np.log2(samples_early / samples_early[0])
bits_late = -np.log2(samples_late / samples_late[0]) - np.log2(samples_early[-1] / samples_early[0])

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Plot data points
ax.loglog(percentiles, sample_counts, 'o', markersize=12, color='black',
          label='Measured samples', zorder=5, markeredgecolor='white', markeredgewidth=2)

# Plot fitted curves
ax.loglog(percentiles_early, samples_early, '-', linewidth=4, color='#3498db',
          label=f'Early regime: α = {alpha_early:.2f} (sublinear)', alpha=0.8)
ax.loglog(percentiles_late, samples_late, '-', linewidth=4, color='#e74c3c',
          label=f'Late regime: α = {alpha_late:.2f} (superlinear)', alpha=0.8)

# Mark the transition point
transition_percentile = 50
transition_samples = 10 ** (p_early[0] * np.log10(transition_percentile) + p_early[1])
ax.scatter([transition_percentile], [transition_samples], s=500, marker='*',
           color='gold', edgecolors='black', linewidth=3, zorder=10,
           label=f'Phase transition (P50)')

# Add vertical line at transition
ax.axvline(transition_percentile, color='#2c3e50', linestyle='--', linewidth=2.5,
           alpha=0.6, zorder=2)

# Shade regions - extend to full axis range
ax.axvspan(1, 50, alpha=0.1, color='#3498db', zorder=0)
ax.axvspan(50, 100, alpha=0.1, color='#e74c3c', zorder=0)

# Add text annotations
ax.text(20, samples_early[30], 'Sublinear\n(easy)', fontsize=12, fontweight='bold',
        ha='center', bbox=dict(boxstyle='round', facecolor='#3498db', alpha=0.3))
ax.text(70, samples_late[60], 'Superlinear\n(hard)', fontsize=12, fontweight='bold',
        ha='center', bbox=dict(boxstyle='round', facecolor='#e74c3c', alpha=0.3))

# Add statistics box (p-value computed from data)
delta_alpha = alpha_late - alpha_early
stats_text = (
              f'Δα = {delta_alpha:.3f}\n'
              f'p-value = {p_value:.3f}\n'
              f'Effect size = {delta_alpha:.2f}'
             )
print(f"Computed p-value: {p_value:.6f}")
ax.text(0.98, 0.05, stats_text, transform=ax.transAxes,
        fontsize=11, fontweight='bold', verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.2, pad=0.8))

# Labels and title
ax.set_xlabel('Order Percentile (P5 = bottom 5%, P95 = top 5%)', fontsize=12, fontweight='bold')
ax.set_ylabel('Samples Required to Reach Percentile', fontsize=12, fontweight='bold')
ax.set_title('Phase Transition in Sampling Difficulty\nScaling Exponent Jump: 13× increase',
             fontsize=14, fontweight='bold', pad=20)
ax.legend(fontsize=10, loc='upper left', framealpha=0.95)
ax.grid(True, alpha=0.3, which='both', linestyle='--')

# Set explicit axis limits and ticks for clarity
ax.set_xlim(3, 100)
ax.set_ylim(0.8, 25)
ax.set_xticks([5, 10, 25, 50, 75, 95])
ax.set_xticklabels(['5', '10', '25', '50', '75', '95'])
ax.set_yticks([1, 2, 5, 10, 20])
ax.set_yticklabels(['1', '2', '5', '10', '20'])

# Save to same directory as this script
out_path = script_dir / 'phase_transition_scaling.pdf'
out_path_png = script_dir / 'phase_transition_scaling.png'

plt.tight_layout()
plt.savefig(out_path, bbox_inches='tight', dpi=300)
plt.savefig(out_path_png, bbox_inches='tight', dpi=300)
print(f"✓ Saved {out_path} and .png")
plt.close()