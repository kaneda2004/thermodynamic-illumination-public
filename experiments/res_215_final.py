#!/usr/bin/env python3
"""
RES-215 FINAL: Threshold-dependent phase transition in effort scaling

CORE QUESTION: Does effort (samples needed) to reach higher order levels
scale linearly or super-linearly? Indicates phase transition.

DESIGN:
1. Sample 500 random CPPNs to get order distribution
2. Divide into percentile bins: [10, 25, 50, 75, 90]
3. Measure EFFORT (average samples) to reach each percentile
4. Fit power law: effort ~ percentile^α
5. Test if scaling exponent changes across regimes
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
print("RES-215 FINAL: PERCENTILE SCALING PHASE TRANSITION")
print("="*70)
print(f"Step 1: Sample order distribution from {N_CPPNS} CPPNs...")

# Phase 1: Sample order distribution
orders = []
for i in range(N_CPPNS):
    if i % 100 == 0:
        print(f"  Generated {i}/{N_CPPNS}...")
    cppn = CPPN()
    img = cppn.render(IMAGE_SIZE)
    order = order_multiplicative(img)
    orders.append(order)

orders = np.array(orders)
print(f"\nOrder distribution:")
print(f"  Mean: {np.mean(orders):.4f}")
print(f"  Std: {np.std(orders):.4f}")
print(f"  Range: [{np.min(orders):.4f}, {np.max(orders):.4f}]")

# Phase 2: Define percentile thresholds
percentiles = np.array([10, 25, 50, 75, 90], dtype=float)
threshold_values = np.percentile(orders, percentiles)

print(f"\nStep 2: Percentile thresholds:")
for p, t in zip(percentiles, threshold_values):
    print(f"  P{p}: {t:.4f}")

# Phase 3: Measure effort (samples needed) to reach each threshold
print(f"\nStep 3: Measuring effort (sample cost) to reach each percentile...")

sample_counts = []
for p, threshold in zip(percentiles, threshold_values):
    n_trials = 100
    samples_per_trial = []

    for trial in range(n_trials):
        # Monte Carlo: sample until we reach threshold
        for n_samples in range(1, 300):
            cppn = CPPN()
            img = cppn.render(IMAGE_SIZE)
            order = order_multiplicative(img)

            if order >= threshold:
                samples_per_trial.append(n_samples)
                break
        else:
            samples_per_trial.append(300)  # Timeout

    mean_samples = np.mean(samples_per_trial)
    std_samples = np.std(samples_per_trial)
    sample_counts.append(mean_samples)

    print(f"  P{p:.0f} (order={threshold:.4f}): {mean_samples:.1f} ± {std_samples:.1f} samples")

sample_counts = np.array(sample_counts)

# Phase 4: Power law fit
print(f"\nStep 4: Fitting power law effort ~ percentile^α...")

# Use percentile as x (normalized 0-1)
x = percentiles / 100.0
y = sample_counts

print(f"\nData points:")
for xi, yi in zip(x, y):
    print(f"  Percentile {xi*100:5.0f}: {yi:6.1f} samples")

try:
    # Fit power law in log-log space
    x_log = np.log(x)
    y_log = np.log(y)

    slope, intercept, r_value, p_value, std_err = stats.linregress(x_log, y_log)

    print(f"\nGlobal power law fit:")
    print(f"  Exponent α: {slope:.4f}")
    print(f"  Intercept: {intercept:.4f}")
    print(f"  R²: {r_value**2:.4f}")
    print(f"  p-value: {p_value:.4e}")

    # Test for phase transition by splitting into low/high percentiles
    if len(x) >= 4:
        # Low: 10-50 percentile
        idx_low = slice(0, 3)
        x_low_log = x_log[idx_low]
        y_low_log = y_log[idx_low]

        # High: 50-90 percentile
        idx_high = slice(2, 5)
        x_high_log = x_log[idx_high]
        y_high_log = y_log[idx_high]

        if len(x_low_log) >= 2 and len(x_high_log) >= 2:
            slope_low, int_low, r_low, _, _ = stats.linregress(x_low_log, y_low_log)
            slope_high, int_high, r_high, _, _ = stats.linregress(x_high_log, y_high_log)

            print(f"\nSegmented analysis (PHASE TRANSITION TEST):")
            print(f"  Low percentiles (10-50): α={slope_low:.4f}, R²={r_low**2:.4f}")
            print(f"  High percentiles (50-90): α={slope_high:.4f}, R²={r_high**2:.4f}")
            print(f"  Δα (transition magnitude): {abs(slope_high - slope_low):.4f}")

            # Decision threshold
            transition_mag = abs(slope_high - slope_low)
            if transition_mag > 0.3:
                status = "VALIDATED"
                direction = "SUPER-LINEAR" if slope_high > slope_low else "SUB-LINEAR"
                print(f"\n✓ PHASE TRANSITION DETECTED!")
                print(f"  Magnitude: Δα = {transition_mag:.4f}")
                print(f"  Direction: {direction} phase at high percentiles")
            else:
                status = "REFUTED"
                print(f"\n✗ NO SIGNIFICANT PHASE TRANSITION")
                print(f"  Exponents too similar (Δα = {transition_mag:.4f})")
        else:
            status = "INCONCLUSIVE"
            transition_mag = 0.0
    else:
        status = "INCONCLUSIVE"
        transition_mag = 0.0

except Exception as e:
    print(f"ERROR: {e}")
    status = "INCONCLUSIVE"
    transition_mag = 0.0

# Save results
results_output = {
    'experiment': 'RES-215',
    'hypothesis': 'Scaling exponent changes across percentiles (phase transition)',
    'method': 'Power law fit to effort vs percentile',
    'order_distribution': {
        'n_samples': N_CPPNS,
        'mean': float(np.mean(orders)),
        'std': float(np.std(orders)),
        'range': [float(np.min(orders)), float(np.max(orders))]
    },
    'percentiles': percentiles.tolist(),
    'threshold_values': [float(v) for v in threshold_values],
    'sample_counts': [float(v) for v in sample_counts],
    'power_law': {
        'global_slope': float(slope),
        'global_r_squared': float(r_value**2),
        'low_slope': float(slope_low) if 'slope_low' in locals() else None,
        'high_slope': float(slope_high) if 'slope_high' in locals() else None,
        'transition_magnitude': float(transition_mag)
    },
    'effect_size': float(transition_mag),
    'status': status
}

results_dir = Path('/Users/matt/Development/monochrome_noise_converger/results/threshold_scaling')
results_dir.mkdir(parents=True, exist_ok=True)
results_file = results_dir / 'res_215_final.json'

with open(results_file, 'w') as f:
    json.dump(results_output, f, indent=2)

print(f"\n✓ Saved to {results_file}")
print(f"\nFINAL RESULT: {status} (d={transition_mag:.3f})")
print("="*70)
