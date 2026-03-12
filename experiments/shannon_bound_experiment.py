#!/usr/bin/env python3
"""
Shannon Entropy Bound Experiment (RES-013)

HYPOTHESIS: Nested sampling achieves bit-cost within a constant factor
of the Shannon entropy lower bound -log2(P(order >= T)).

THEORETICAL BACKGROUND:
The Shannon source coding theorem establishes that the minimum expected
code length for describing events from a distribution is the entropy.
For threshold-based sampling, the minimum bits to specify "any sample
with order >= T" is at least -log2(P(order >= T)), where P is the
probability under the prior.

If nested sampling is information-efficient, its bit-cost should track
this theoretical bound with a constant multiplicative overhead.

NULL HYPOTHESIS: The ratio B / (-log2(P)) is unbounded or varies
erratically with threshold T (no consistent relationship to Shannon limit).

BUILDS ON: RES-002 (scaling laws), RES-004 (prior comparison)
DOMAIN: information_limits
"""

import sys
import os
import numpy as np
from scipy import stats
from pathlib import Path
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.thermo_sampler_v3 import (
    CPPN,
    order_multiplicative,
    nested_sampling_v3,
)


def estimate_prior_probability(threshold: float, n_samples: int = 10000,
                                image_size: int = 32, seed: int = 42) -> dict:
    """
    Estimate P(order >= threshold) by Monte Carlo sampling from CPPN prior.

    Returns:
        dict with probability estimate, standard error, and Wilson CI
    """
    np.random.seed(seed)

    successes = 0
    for _ in range(n_samples):
        cppn = CPPN()
        img = cppn.render(image_size)
        order = order_multiplicative(img)
        if order >= threshold:
            successes += 1

    # Maximum likelihood estimate
    p_hat = successes / n_samples

    # Standard error (binomial)
    se = np.sqrt(p_hat * (1 - p_hat) / n_samples) if p_hat > 0 and p_hat < 1 else 0

    # Wilson score interval (better for rare events)
    z = 1.96  # 95% CI
    denom = 1 + z**2 / n_samples
    center = (p_hat + z**2 / (2 * n_samples)) / denom
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n_samples)) / n_samples) / denom
    ci_low = max(0, center - margin)
    ci_high = min(1, center + margin)

    # Shannon bound: -log2(P)
    # Use continuity correction for zero probability
    if successes == 0:
        # Upper bound: at least one success would give p = 1/n_samples
        shannon_bound = np.log2(n_samples)  # Lower bound on bits
        shannon_bound_ci = (np.log2(n_samples), np.inf)
    else:
        shannon_bound = -np.log2(p_hat)
        # CI for Shannon bound (derived from Wilson CI)
        shannon_bound_ci = (-np.log2(ci_high), -np.log2(max(ci_low, 1e-10)))

    return {
        'threshold': threshold,
        'successes': successes,
        'n_samples': n_samples,
        'p_hat': p_hat,
        'p_se': se,
        'p_ci_low': ci_low,
        'p_ci_high': ci_high,
        'shannon_bound': shannon_bound,
        'shannon_ci_low': shannon_bound_ci[0],
        'shannon_ci_high': shannon_bound_ci[1],
    }


def measure_empirical_bits(threshold: float, n_seeds: int = 5,
                           n_live: int = 50, image_size: int = 32) -> dict:
    """
    Measure empirical bit-cost to reach threshold via nested sampling.

    Returns:
        dict with mean bits, std, and CI
    """
    import io
    import contextlib

    bits_list = []

    for seed in range(n_seeds):
        # Run nested sampling - returns (dead_points, live_points, snapshots)
        # Suppress verbose output
        with contextlib.redirect_stdout(io.StringIO()):
            dead_points, live_points, snapshots = nested_sampling_v3(
                n_live=n_live,
                n_iterations=1500,
                image_size=image_size,
                seed=seed,
                output_dir=f'/tmp/shannon_exp_seed{seed}'
            )

        # Find first iteration where order >= threshold
        for dp in dead_points:
            if dp.order_value >= threshold:
                # Bits = -log_X (in nats) / ln(2) to convert to bits
                bits = -dp.log_X / np.log(2)
                bits_list.append(bits)
                break
        else:
            # Never reached threshold - use maximum
            if dead_points:
                max_bits = -dead_points[-1].log_X / np.log(2)
                bits_list.append(max_bits)

    bits_arr = np.array(bits_list)

    return {
        'threshold': threshold,
        'bits_mean': float(np.mean(bits_arr)),
        'bits_std': float(np.std(bits_arr)),
        'bits_ci_low': float(np.mean(bits_arr) - 1.96 * np.std(bits_arr) / np.sqrt(len(bits_arr))),
        'bits_ci_high': float(np.mean(bits_arr) + 1.96 * np.std(bits_arr) / np.sqrt(len(bits_arr))),
        'n_seeds': n_seeds,
        'raw_bits': bits_list,
    }


def run_experiment():
    """
    Main experiment: Compare empirical bit-cost to Shannon entropy bound.
    """
    print("=" * 70)
    print("SHANNON ENTROPY BOUND EXPERIMENT")
    print("=" * 70)
    print()
    print("HYPOTHESIS: Nested sampling bit-cost tracks Shannon bound -log2(P)")
    print("NULL: No systematic relationship between B and -log2(P)")
    print()

    # Parameters
    thresholds = [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20]
    n_mc_samples = 10000  # For probability estimation
    n_seeds = 5  # For nested sampling
    n_live = 50
    image_size = 32

    results = {
        'thresholds': [],
        'shannon_bounds': [],
        'empirical_bits': [],
        'efficiency_ratios': [],
        'prior_estimates': [],
        'nested_results': [],
    }

    print(f"Testing {len(thresholds)} thresholds: {thresholds}")
    print(f"Monte Carlo samples for P estimate: {n_mc_samples}")
    print(f"Nested sampling seeds: {n_seeds}")
    print()

    # Step 1: Estimate prior probabilities
    print("-" * 70)
    print("Step 1: Estimating P(order >= T) from prior")
    print("-" * 70)

    for T in thresholds:
        print(f"\nThreshold T = {T:.2f}...")
        prior_est = estimate_prior_probability(T, n_mc_samples, image_size)
        results['prior_estimates'].append(prior_est)
        print(f"  P(order >= {T}) = {prior_est['p_hat']:.6f} ({prior_est['successes']}/{n_mc_samples})")
        print(f"  Shannon bound = {prior_est['shannon_bound']:.2f} bits")
        print(f"  95% CI: [{prior_est['shannon_ci_low']:.2f}, {prior_est['shannon_ci_high']:.2f}]")

    # Step 2: Measure empirical bits
    print()
    print("-" * 70)
    print("Step 2: Measuring empirical bits via nested sampling")
    print("-" * 70)

    for T in thresholds:
        print(f"\nThreshold T = {T:.2f}...")
        nested_res = measure_empirical_bits(T, n_seeds, n_live, image_size)
        results['nested_results'].append(nested_res)
        print(f"  Empirical bits = {nested_res['bits_mean']:.2f} +/- {nested_res['bits_std']:.2f}")
        print(f"  95% CI: [{nested_res['bits_ci_low']:.2f}, {nested_res['bits_ci_high']:.2f}]")

    # Step 3: Compute efficiency ratios
    print()
    print("-" * 70)
    print("Step 3: Computing efficiency ratios")
    print("-" * 70)

    for prior_est, nested_res in zip(results['prior_estimates'], results['nested_results']):
        T = prior_est['threshold']
        shannon = prior_est['shannon_bound']
        empirical = nested_res['bits_mean']

        # Skip if P = 0 (infinite Shannon bound)
        if prior_est['successes'] == 0:
            print(f"  T={T:.2f}: P=0, skipping (Shannon bound undefined)")
            continue

        eta = empirical / shannon
        results['thresholds'].append(T)
        results['shannon_bounds'].append(shannon)
        results['empirical_bits'].append(empirical)
        results['efficiency_ratios'].append(eta)

        print(f"  T={T:.2f}: eta = B/(-log2(P)) = {empirical:.2f}/{shannon:.2f} = {eta:.3f}")

    # Step 4: Statistical analysis
    print()
    print("-" * 70)
    print("Step 4: Statistical Analysis")
    print("-" * 70)

    if len(results['shannon_bounds']) < 3:
        print("ERROR: Not enough valid data points for regression")
        return results

    X = np.array(results['shannon_bounds'])
    Y = np.array(results['empirical_bits'])

    # Linear regression: B = slope * (-log2(P)) + intercept
    slope, intercept, r_value, p_value, std_err = stats.linregress(X, Y)

    print(f"\nLinear regression: B = {slope:.3f} * (-log2(P)) + {intercept:.3f}")
    print(f"  R^2 = {r_value**2:.4f}")
    print(f"  p-value = {p_value:.2e}")
    print(f"  Slope SE = {std_err:.3f}")
    print(f"  Slope 95% CI = [{slope - 1.96*std_err:.3f}, {slope + 1.96*std_err:.3f}]")

    # Effect size: correlation coefficient
    r, r_p = stats.pearsonr(X, Y)
    print(f"\nCorrelation coefficient r = {r:.4f} (p = {r_p:.2e})")

    # Mean efficiency ratio
    eta_mean = np.mean(results['efficiency_ratios'])
    eta_std = np.std(results['efficiency_ratios'])
    eta_se = eta_std / np.sqrt(len(results['efficiency_ratios']))

    print(f"\nMean efficiency ratio eta = {eta_mean:.3f} +/- {eta_std:.3f}")
    print(f"  95% CI = [{eta_mean - 1.96*eta_se:.3f}, {eta_mean + 1.96*eta_se:.3f}]")

    # Test if efficiency ratio is consistent (coefficient of variation)
    cv = eta_std / eta_mean if eta_mean > 0 else np.inf
    print(f"  Coefficient of variation = {cv:.3f}")

    # Decision
    print()
    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    # Criteria: p < 0.01, R^2 > 0.5, bounded efficiency ratio
    validated = (p_value < 0.01 and r_value**2 > 0.5 and cv < 0.5)

    if validated:
        print("VALIDATED: Nested sampling bit-cost tracks Shannon bound")
        print(f"  - Strong correlation (R^2 = {r_value**2:.3f})")
        print(f"  - Significant (p = {p_value:.2e})")
        print(f"  - Consistent efficiency ratio (CV = {cv:.3f})")
        print(f"  - Mean overhead factor: {eta_mean:.2f}x theoretical minimum")
    else:
        if p_value >= 0.01:
            print("INCONCLUSIVE: Relationship not statistically significant")
        elif r_value**2 < 0.5:
            print("INCONCLUSIVE: Weak correlation with Shannon bound")
        elif cv >= 0.5:
            print("INCONCLUSIVE: Efficiency ratio too variable")

    # Store final results
    results['analysis'] = {
        'slope': float(slope),
        'intercept': float(intercept),
        'r_squared': float(r_value**2),
        'p_value': float(p_value),
        'slope_se': float(std_err),
        'slope_ci': [float(slope - 1.96*std_err), float(slope + 1.96*std_err)],
        'correlation_r': float(r),
        'correlation_p': float(r_p),
        'eta_mean': float(eta_mean),
        'eta_std': float(eta_std),
        'eta_cv': float(cv),
        'validated': bool(validated),
    }

    # Save results
    output_dir = Path('results/shannon_bound')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert to JSON-serializable format
    json_results = {
        'thresholds': results['thresholds'],
        'shannon_bounds': results['shannon_bounds'],
        'empirical_bits': results['empirical_bits'],
        'efficiency_ratios': results['efficiency_ratios'],
        'analysis': results['analysis'],
    }

    with open(output_dir / 'shannon_bound_results.json', 'w') as f:
        json.dump(json_results, f, indent=2, default=lambda x: bool(x) if isinstance(x, np.bool_) else x)

    print(f"\nResults saved to {output_dir / 'shannon_bound_results.json'}")

    return results


if __name__ == "__main__":
    results = run_experiment()
