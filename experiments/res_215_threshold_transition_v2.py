#!/usr/bin/env python3
"""
RES-215 REVISED: Threshold-dependent phase transition in effort scaling

INSIGHT: The order_multiplicative metric has ceiling ~0.3.
REVISED APPROACH: Instead of absolute thresholds, test scaling of effort
to reach different PERCENTILES of the order distribution.

HYPOTHESIS: The scaling exponent (how effort scales with reaching higher
percentiles) changes significantly, indicating a phase transition.

DESIGN:
- Generate 500 random CPPNs, measure order values
- Fit power law to quantiles: effort(p) ~ p^alpha
- Test if scaling breaks into regimes (linear -> superlinear)
- Use percentiles [10, 25, 50, 75, 90] as proxy for "thresholds"
"""
import numpy as np
import json
from pathlib import Path
from scipy import stats
import sys
import os

# Ensure project root
os.chdir('/Users/matt/Development/monochrome_noise_converger')
sys.path.insert(0, os.getcwd())

from core.thermo_sampler_v3 import CPPN, order_multiplicative

# Configuration
N_CPPNS = 500
N_LIVE = 30
IMAGE_SIZE = 32

print("="*70)
print("RES-215 REVISED: PERCENTILE SCALING TRANSITION")
print("="*70)
print(f"Generating {N_CPPNS} random CPPNs...")

# Phase 1: Sample order distribution
orders = []
for i in range(N_CPPNS):
    if i % 100 == 0:
        print(f"  Sampled {i}/{N_CPPNS} CPPNs...")
    cppn = CPPN()
    img = cppn.render(IMAGE_SIZE)
    order = order_multiplicative(img)
    orders.append(order)

orders = np.array(orders)
print(f"\nOrder distribution:")
print(f"  Mean: {np.mean(orders):.4f}")
print(f"  Std: {np.std(orders):.4f}")
print(f"  Min: {np.min(orders):.4f}, Max: {np.max(orders):.4f}")

# Phase 2: Fit power law to cumulative effort
percentiles = np.array([10, 25, 50, 75, 90])
threshold_values = np.percentile(orders, percentiles)

print(f"\nPercentile thresholds:")
for p, t in zip(percentiles, threshold_values):
    print(f"  {p}th percentile: {t:.4f}")

# Compute effort to reach each percentile
# Effort = number of samples needed / n_live
efforts = []
effort_vars = []

print(f"\nMeasuring effort to reach each percentile...")

for p, threshold in zip(percentiles, threshold_values):
    # Monte Carlo: how many samples to reach this threshold?
    n_trials = 100
    samples_needed = []

    for trial in range(n_trials):
        for sample_id in range(1, 200):  # Up to 200 samples max
            cppn = CPPN()
            img = cppn.render(IMAGE_SIZE)
            order = order_multiplicative(img)

            if order >= threshold:
                samples_needed.append(sample_id)
                break
        else:
            samples_needed.append(200)  # Timeout

    samples_needed = np.array(samples_needed)
    effort = np.log2(np.mean(samples_needed) / N_LIVE)

    efforts.append(effort)
    effort_vars.append(np.std(np.log2(samples_needed / N_LIVE)))

    print(f"  {p}th percentile (order={threshold:.4f}): " +
          f"mean_samples={np.mean(samples_needed):.1f}, " +
          f"effort={effort:.4f} bits")

efforts = np.array(efforts)
effort_vars = np.array(effort_vars)

# Phase 3: Fit power law effort ~ percentile^alpha
print("\n" + "="*70)
print("POWER-LAW ANALYSIS")
print("="*70)

try:
    # Test 1: Linear fit in log-log space
    x_log = np.log(percentiles)
    y_log = np.log(efforts)

    slope, intercept, r_value, p_value, std_err = stats.linregress(x_log, y_log)

    print(f"\nLog-log fit: log(effort) ~ log(percentile)")
    print(f"  Exponent α: {slope:.4f}")
    print(f"  R²: {r_value**2:.4f}")
    print(f"  p-value: {p_value:.4e}")

    # Test 2: Segment the data - is there a transition?
    # Compare exponent before/after midpoint
    if len(percentiles) >= 4:
        x_low = np.log(percentiles[:3])
        y_low = np.log(efforts[:3])
        slope_low, _, r_low, _, _ = stats.linregress(x_low, y_low)

        x_high = np.log(percentiles[2:])
        y_high = np.log(efforts[2:])
        slope_high, _, r_high, _, _ = stats.linregress(x_high, y_high)

        print(f"\nSegmented analysis:")
        print(f"  Low percentiles (10-50): α={slope_low:.4f}, R²={r_low**2:.4f}")
        print(f"  High percentiles (50-90): α={slope_high:.4f}, R²={r_high**2:.4f}")
        print(f"  Δα (transition magnitude): {abs(slope_high - slope_low):.4f}")

        # Test significance of difference
        transition_magnitude = abs(slope_high - slope_low)
        if transition_magnitude > 0.2:
            status = "VALIDATED"
            print(f"\n✓ PHASE TRANSITION DETECTED: Δα={transition_magnitude:.4f}")
        else:
            status = "REFUTED"
            print(f"\n✗ NO PHASE TRANSITION: Δα={transition_magnitude:.4f} too small")
    else:
        status = "INCONCLUSIVE"
        transition_magnitude = 0.0

except Exception as e:
    print(f"ERROR in power law fit: {e}")
    status = "INCONCLUSIVE"
    transition_magnitude = 0.0

# Phase 4: Save results
results_output = {
    'experiment': 'RES-215',
    'revision': 'v2 - percentile scaling',
    'hypothesis': 'The scaling exponent changes across percentiles, indicating phase transition',
    'method': 'Monte Carlo sampling to percentiles of order distribution',
    'order_distribution': {
        'mean': float(np.mean(orders)),
        'std': float(np.std(orders)),
        'min': float(np.min(orders)),
        'max': float(np.max(orders))
    },
    'percentiles': percentiles.tolist(),
    'threshold_values': threshold_values.tolist(),
    'efforts': efforts.tolist(),
    'effort_vars': effort_vars.tolist(),
    'power_law_analysis': {
        'slope_low': float(slope_low) if 'slope_low' in locals() else None,
        'slope_high': float(slope_high) if 'slope_high' in locals() else None,
        'transition_magnitude': float(transition_magnitude),
        'global_slope': float(slope) if 'slope' in locals() else None,
        'global_r_squared': float(r_value**2) if 'r_value' in locals() else None
    },
    'effect_size': float(transition_magnitude),
    'status': status
}

results_dir = Path('/Users/matt/Development/monochrome_noise_converger/results/threshold_scaling')
results_dir.mkdir(parents=True, exist_ok=True)
results_file = results_dir / 'res_215_percentile_transition.json'

with open(results_file, 'w') as f:
    json.dump(results_output, f, indent=2)

print(f"\n✓ Results saved to {results_file}")
print(f"\nFINAL STATUS: {status} (d={transition_magnitude:.3f})")
print("="*70)
