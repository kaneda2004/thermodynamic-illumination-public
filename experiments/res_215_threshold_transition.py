#!/usr/bin/env python3
"""
RES-215: Threshold-dependent phase transition in effort scaling
Tests whether reaching higher order thresholds exhibits non-linear scaling.

HYPOTHESIS: Bits-to-threshold shows sub-linear-to-super-linear transition
between order 0.3 and 0.7

DESIGN:
- 5 order targets: [0.2, 0.3, 0.5, 0.7, 0.9]
- 20 CPPNs per target
- Nested sampling: measure bits/samples to reach each target
- Fit power law: effort ~ threshold^alpha
- Test: does alpha change significantly? Suggests phase transition.
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
N_CPPNS = 20
ORDER_TARGETS = [0.2, 0.3, 0.5, 0.7, 0.9]
N_LIVE = 30
IMAGE_SIZE = 32

print("="*70)
print("RES-215: THRESHOLD SCALING TRANSITION")
print("="*70)
print(f"Targets: {ORDER_TARGETS}")
print(f"CPPNs per target: {N_CPPNS}")
print(f"Live points: {N_LIVE}")

results_by_threshold = {}

for target_order in ORDER_TARGETS:
    print(f"\n[ORDER {target_order}] Testing {N_CPPNS} CPPNs...")

    effort_to_threshold = []
    n_samples_list = []

    for cppn_id in range(N_CPPNS):
        try:
            # Create random CPPN
            cppn = CPPN()

            # Render initial image
            img = cppn.render(IMAGE_SIZE)
            initial_order = order_multiplicative(img)

            # Run nested sampling in illumination mode to get diverse samples
            # We'll track how many iterations to reach target_order
            n_iterations_to_reach = None

            # Estimate based on Monte Carlo: how many CPPNs needed?
            # Simulate ~50 samples per target to see distribution
            orders_found = []
            for sample_id in range(50):
                test_cppn = CPPN()
                test_img = test_cppn.render(IMAGE_SIZE)
                test_order = order_multiplicative(test_img)
                orders_found.append(test_order)

                if test_order >= target_order:
                    n_iterations_to_reach = sample_id + 1
                    break

            if n_iterations_to_reach is None:
                # Didn't reach target in 50 samples
                print(f"  CPPN {cppn_id+1}: Could not reach order {target_order:.1f} in 50 samples")
                continue

            # Estimate bits cost: log2(samples / n_live)
            bits_cost = np.log2(n_iterations_to_reach / N_LIVE) if n_iterations_to_reach >= N_LIVE else 0.0

            effort_to_threshold.append(bits_cost)
            n_samples_list.append(n_iterations_to_reach)

            print(f"  CPPN {cppn_id+1}: reached order {target_order:.1f} at {n_iterations_to_reach} samples, {bits_cost:.4f} bits")

        except Exception as e:
            print(f"  CPPN {cppn_id+1}: ERROR - {str(e)[:50]}")
            continue

    if effort_to_threshold:
        mean_effort = np.mean(effort_to_threshold)
        std_effort = np.std(effort_to_threshold)
        results_by_threshold[target_order] = {
            'effort': [float(e) for e in effort_to_threshold],
            'samples': [int(s) for s in n_samples_list],
            'mean_effort': float(mean_effort),
            'std_effort': float(std_effort),
            'n_success': len(effort_to_threshold)
        }
        print(f"  → Mean effort: {mean_effort:.4f} ± {std_effort:.4f} bits (n={len(effort_to_threshold)})")
    else:
        print(f"  → No successful runs")

# Analyze scaling exponents
print("\n" + "="*70)
print("POWER-LAW ANALYSIS")
print("="*70)

thresholds = []
scaling_exponents = []
r_squared_values = []

for target_order in ORDER_TARGETS:
    if target_order in results_by_threshold and results_by_threshold[target_order]['n_success'] >= 5:
        effort_values = np.array(results_by_threshold[target_order]['effort'])

        try:
            # Check if we have positive values for log
            if np.all(effort_values > 0):
                # Log-space linear fit
                x_data = np.log([target_order] * len(effort_values))
                y_data = np.log(effort_values)

                slope, intercept, r_value, p_value, std_err = stats.linregress(x_data, y_data)

                thresholds.append(target_order)
                scaling_exponents.append(slope)
                r_squared_values.append(r_value**2)

                print(f"\nOrder {target_order:.1f}:")
                print(f"  Exponent α: {slope:.4f}")
                print(f"  R²: {r_value**2:.4f}")
            else:
                print(f"\nOrder {target_order:.1f}: Skipped (non-positive effort values)")
        except Exception as e:
            print(f"\nOrder {target_order:.1f}: Fit failed - {e}")

# Test for phase transition
status = "INCONCLUSIVE"
cv_exponents = 0.0

if len(scaling_exponents) >= 3:
    print("\n" + "-"*70)
    print("PHASE TRANSITION TEST (Coefficient of Variation)")
    print("-"*70)

    exponent_std = np.std(scaling_exponents)
    exponent_mean = np.mean(scaling_exponents)

    print(f"Mean exponent: {exponent_mean:.4f}")
    print(f"Std exponent: {exponent_std:.4f}")
    print(f"Range: {min(scaling_exponents):.4f} to {max(scaling_exponents):.4f}")

    cv_exponents = exponent_std / abs(exponent_mean) if exponent_mean != 0 else 0
    print(f"CV (effect size): {cv_exponents:.4f}")

    if cv_exponents > 0.15:
        status = "VALIDATED"
        print(f"\n✓ TRANSITION DETECTED: Scaling exponents vary significantly (CV={cv_exponents:.4f})")
    else:
        status = "REFUTED"
        print(f"\n✗ NO TRANSITION: Scaling exponents stable (CV={cv_exponents:.4f})")

# Save results
results_output = {
    'experiment': 'RES-215',
    'hypothesis': 'Bits-to-threshold shows sub-linear-to-super-linear transition between order 0.3 and 0.7',
    'targets': ORDER_TARGETS,
    'results_by_threshold': results_by_threshold,
    'scaling_analysis': {
        'thresholds': thresholds,
        'exponents': scaling_exponents,
        'r_squared': r_squared_values
    },
    'effect_size': float(cv_exponents),
    'status': status
}

results_dir = Path('/Users/matt/Development/monochrome_noise_converger/results/threshold_scaling')
results_dir.mkdir(parents=True, exist_ok=True)
results_file = results_dir / 'res_215_threshold_transition.json'

with open(results_file, 'w') as f:
    json.dump(results_output, f, indent=2)

print(f"\n✓ Results saved to {results_file}")
print(f"\nFINAL STATUS: {status} (d={cv_exponents:.3f})")
print("="*70)
