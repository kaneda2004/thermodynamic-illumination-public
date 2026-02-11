#!/usr/bin/env python3
"""
EXPERIMENT: Trajectory Autocorrelation Analysis (RES-017)

HYPOTHESIS: Order values during nested sampling exhibit significant positive
autocorrelation at short lags, indicating the sampler moves continuously through
parameter space rather than jumping between disconnected modes.

NULL HYPOTHESIS: Order value sequence is uncorrelated after removing the trend
(which is intrinsic to nested sampling).

METHOD:
1. Run multiple nested sampling trajectories
2. Detrend the order sequence (remove monotonic increase)
3. Compute autocorrelation function at lags 1, 2, 5, 10, 20
4. Test significance using Ljung-Box test
5. Compute "jump" distribution (order change between iterations)

NOVELTY: No existing entry studies the TEMPORAL structure of sampling trajectories.
RES-002 studied bits-to-threshold (endpoint), RES-015 studied static sensitivity.
This tests whether the sampling path is smooth or discontinuous.

DOMAIN: convergence_dynamics

BUILDS ON: RES-002 (scaling laws), RES-015 (order sensitivity)
"""

import sys
import os
import numpy as np
from scipy.stats import spearmanr
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.thermo_sampler_v3 import (
    CPPN, order_multiplicative, elliptical_slice_sample,
    LivePoint, log_prior, set_global_seed
)


def ljung_box_test(acf_values: np.ndarray, n: int, lags: list[int]) -> tuple[float, float]:
    """
    Ljung-Box test for autocorrelation.

    H0: No autocorrelation up to lag h

    Q = n(n+2) * sum_{k=1}^{h} (r_k^2 / (n-k))

    Under H0, Q ~ chi-squared with h degrees of freedom.

    Returns (Q_statistic, p_value)
    """
    from scipy.stats import chi2

    h = len(lags)
    Q = 0
    for i, k in enumerate(lags):
        if k < len(acf_values):
            Q += (acf_values[k] ** 2) / (n - k)

    Q = n * (n + 2) * Q
    p_value = 1 - chi2.cdf(Q, df=h)

    return Q, p_value


def compute_autocorrelation(x: np.ndarray, max_lag: int) -> np.ndarray:
    """
    Compute autocorrelation function for lags 0 to max_lag.
    """
    n = len(x)
    x_centered = x - np.mean(x)
    var = np.var(x)

    if var < 1e-10:
        return np.zeros(max_lag + 1)

    acf = np.zeros(max_lag + 1)
    acf[0] = 1.0  # By definition

    for lag in range(1, max_lag + 1):
        if lag >= n:
            break
        acf[lag] = np.mean(x_centered[:-lag] * x_centered[lag:]) / var

    return acf


def detrend_sequence(order_values: np.ndarray) -> np.ndarray:
    """
    Detrend the order sequence by removing linear trend.

    Nested sampling inherently increases order over time, so we need to
    remove this trend to analyze the residual autocorrelation structure.
    """
    n = len(order_values)
    x = np.arange(n)

    # Linear regression
    slope, intercept = np.polyfit(x, order_values, 1)
    trend = slope * x + intercept

    residuals = order_values - trend
    return residuals


def run_single_trajectory(n_live: int = 50, n_iterations: int = 300,
                          image_size: int = 32, seed: int = 42) -> dict:
    """
    Run a single nested sampling trajectory and extract order sequence.
    """
    set_global_seed(seed)

    # Initialize live points
    live_points = []
    for _ in range(n_live):
        cppn = CPPN()
        img = cppn.render(image_size)
        o = order_multiplicative(img)
        lp = log_prior(cppn)
        live_points.append(LivePoint(cppn, img, o, lp, {}))

    # Track order values at each iteration
    order_sequence = []
    threshold_sequence = []
    jump_sequence = []  # Order change from iteration to iteration

    prev_threshold = 0

    for iteration in range(n_iterations):
        # Find worst point
        worst_idx = min(range(n_live), key=lambda i: live_points[i].order_value)
        worst = live_points[worst_idx]
        threshold = worst.order_value

        order_sequence.append(threshold)
        threshold_sequence.append(threshold)

        if iteration > 0:
            jump_sequence.append(threshold - prev_threshold)
        prev_threshold = threshold

        # Select seed from valid points
        valid_seeds = [i for i in range(n_live)
                       if i != worst_idx and live_points[i].order_value >= threshold]

        if not valid_seeds:
            valid_seeds = [i for i in range(n_live) if i != worst_idx]

        seed_idx = np.random.choice(valid_seeds)

        # Replace using ESS
        new_cppn, new_img, new_order, new_logp, _, success = elliptical_slice_sample(
            live_points[seed_idx].cppn, threshold, image_size, order_multiplicative
        )

        live_points[worst_idx] = LivePoint(new_cppn, new_img, new_order, new_logp, {})

    return {
        'order_sequence': np.array(order_sequence),
        'jump_sequence': np.array(jump_sequence),
        'final_order': threshold_sequence[-1]
    }


def run_experiment(n_trajectories: int = 30, n_live: int = 50,
                   n_iterations: int = 300, image_size: int = 32,
                   base_seed: int = 42):
    """
    Main experiment: analyze autocorrelation structure of nested sampling trajectories.
    """
    print("=" * 70)
    print("EXPERIMENT: Trajectory Autocorrelation Analysis")
    print("=" * 70)
    print()
    print("H0: Order sequence is uncorrelated (after detrending)")
    print("H1: Significant positive autocorrelation at short lags")
    print("    (sampler moves continuously, not jumping between modes)")
    print()
    print(f"Parameters: {n_trajectories} trajectories, {n_live} live points, "
          f"{n_iterations} iterations")
    print()

    # Collect trajectories
    all_trajectories = []
    all_jumps = []

    print("Running trajectories...")
    for i in range(n_trajectories):
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{n_trajectories}")

        result = run_single_trajectory(
            n_live=n_live,
            n_iterations=n_iterations,
            image_size=image_size,
            seed=base_seed + i
        )
        all_trajectories.append(result['order_sequence'])
        all_jumps.extend(result['jump_sequence'].tolist())

    print(f"\nCollected {len(all_trajectories)} trajectories")

    # Compute autocorrelation for each trajectory (on detrended data)
    lags_to_test = [1, 2, 5, 10, 20]
    max_lag = max(lags_to_test)

    acf_by_lag = {lag: [] for lag in lags_to_test}
    ljung_box_stats = []
    ljung_box_pvals = []

    print("\nComputing autocorrelation functions...")

    for i, trajectory in enumerate(all_trajectories):
        # Detrend
        residuals = detrend_sequence(trajectory)

        # Compute ACF
        acf = compute_autocorrelation(residuals, max_lag)

        for lag in lags_to_test:
            if lag < len(acf):
                acf_by_lag[lag].append(acf[lag])

        # Ljung-Box test
        Q, p = ljung_box_test(acf, len(residuals), lags_to_test)
        ljung_box_stats.append(Q)
        ljung_box_pvals.append(p)

    # Convert to arrays
    for lag in lags_to_test:
        acf_by_lag[lag] = np.array(acf_by_lag[lag])

    ljung_box_stats = np.array(ljung_box_stats)
    ljung_box_pvals = np.array(ljung_box_pvals)
    all_jumps = np.array(all_jumps)

    # Print results
    print("\n" + "-" * 60)
    print("AUTOCORRELATION BY LAG (on detrended residuals):")
    print("-" * 60)

    results = {
        'n_trajectories': n_trajectories,
        'n_live': n_live,
        'n_iterations': n_iterations,
        'acf_by_lag': {},
        'tests': {}
    }

    for lag in lags_to_test:
        acf_vals = acf_by_lag[lag]
        mean_acf = np.mean(acf_vals)
        std_acf = np.std(acf_vals)

        # One-sample t-test: is mean ACF > 0?
        from scipy.stats import ttest_1samp
        t_stat, p_val_two = ttest_1samp(acf_vals, 0)
        p_val = p_val_two / 2 if t_stat > 0 else 1 - p_val_two / 2  # One-sided

        # Effect size (Cohen's d)
        cohens_d = mean_acf / std_acf if std_acf > 0 else 0

        print(f"  Lag {lag:2d}: ACF = {mean_acf:+.4f} +/- {std_acf:.4f}, "
              f"t = {t_stat:.2f}, p = {p_val:.2e}, d = {cohens_d:.2f}")

        results['acf_by_lag'][f'lag_{lag}'] = {
            'mean': float(mean_acf),
            'std': float(std_acf),
            't_statistic': float(t_stat),
            'p_value': float(p_val),
            'cohens_d': float(cohens_d)
        }

    # Ljung-Box summary
    print("\n" + "-" * 60)
    print("LJUNG-BOX TEST (joint test for autocorrelation):")
    print("-" * 60)

    mean_Q = np.mean(ljung_box_stats)
    mean_p = np.mean(ljung_box_pvals)
    n_significant = np.sum(ljung_box_pvals < 0.01)
    frac_significant = n_significant / len(ljung_box_pvals)

    print(f"  Mean Q-statistic: {mean_Q:.2f}")
    print(f"  Mean p-value: {mean_p:.4f}")
    print(f"  Fraction with p < 0.01: {frac_significant:.1%} ({n_significant}/{len(ljung_box_pvals)})")

    results['tests']['ljung_box'] = {
        'mean_Q': float(mean_Q),
        'mean_p': float(mean_p),
        'frac_significant': float(frac_significant),
        'n_significant': int(n_significant)
    }

    # Jump distribution analysis
    print("\n" + "-" * 60)
    print("JUMP DISTRIBUTION (order change between iterations):")
    print("-" * 60)

    mean_jump = np.mean(all_jumps)
    std_jump = np.std(all_jumps)
    median_jump = np.median(all_jumps)

    # What fraction of jumps are positive (order increasing)?
    frac_positive = np.mean(all_jumps > 0)

    # Jump magnitude distribution
    abs_jumps = np.abs(all_jumps)
    p25, p50, p75, p95 = np.percentile(abs_jumps, [25, 50, 75, 95])

    print(f"  Mean jump: {mean_jump:.4f}")
    print(f"  Std jump: {std_jump:.4f}")
    print(f"  Median jump: {median_jump:.4f}")
    print(f"  Fraction positive jumps: {frac_positive:.1%}")
    print(f"  |Jump| percentiles: p25={p25:.4f}, p50={p50:.4f}, p75={p75:.4f}, p95={p95:.4f}")

    results['jump_distribution'] = {
        'mean': float(mean_jump),
        'std': float(std_jump),
        'median': float(median_jump),
        'frac_positive': float(frac_positive),
        'abs_percentiles': {
            'p25': float(p25),
            'p50': float(p50),
            'p75': float(p75),
            'p95': float(p95)
        }
    }

    # Test: is jump size correlated with current order level?
    # Build matched pairs: (order_at_iteration, jump_from_iteration)
    order_at_jump = []
    jumps_matched = []
    for trajectory in all_trajectories:
        for i in range(len(trajectory) - 1):
            order_at_jump.append(trajectory[i])
            jumps_matched.append(trajectory[i+1] - trajectory[i])

    order_at_jump = np.array(order_at_jump)
    jumps_matched = np.array(jumps_matched)

    r_jump_order, p_jump_order = spearmanr(order_at_jump, jumps_matched)

    print(f"\n  Correlation (order level vs jump): rho = {r_jump_order:.3f}, p = {p_jump_order:.2e}")

    results['jump_order_correlation'] = {
        'spearman_rho': float(r_jump_order),
        'p_value': float(p_jump_order)
    }

    # Determine status
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("=" * 60)

    # Primary criterion: significant positive autocorrelation at lag 1
    lag1_result = results['acf_by_lag']['lag_1']
    primary_success = (lag1_result['p_value'] < 0.01 and
                       lag1_result['mean'] > 0 and
                       lag1_result['cohens_d'] > 0.5)

    # Secondary criterion: Ljung-Box significant in majority of trajectories
    secondary_success = frac_significant > 0.5

    if primary_success:
        status = 'validated'
        confidence = 'high'
        summary = (f"Strong positive autocorrelation at lag 1 "
                   f"(ACF={lag1_result['mean']:.3f}, d={lag1_result['cohens_d']:.2f}, p<0.01). "
                   f"Sampling trajectory is smooth - the sampler moves continuously through "
                   f"parameter space rather than jumping between disconnected modes. "
                   f"Ljung-Box significant in {frac_significant:.0%} of trajectories.")

        print(f"\nRESULT: VALIDATED")
        print(f"  - Significant positive lag-1 autocorrelation (ACF = {lag1_result['mean']:.3f})")
        print(f"  - Cohen's d = {lag1_result['cohens_d']:.2f} (large effect)")
        print(f"  - Ljung-Box significant in {frac_significant:.0%} of trajectories")
        print(f"  - Implication: ESS maintains continuous exploration")

    elif secondary_success or (lag1_result['p_value'] < 0.01):
        status = 'inconclusive'
        confidence = 'medium'
        summary = (f"Some evidence for autocorrelation but effect size below threshold. "
                   f"Lag-1 ACF = {lag1_result['mean']:.3f}, d = {lag1_result['cohens_d']:.2f}. "
                   f"Ljung-Box significant in {frac_significant:.0%} of trajectories.")

        print(f"\nRESULT: INCONCLUSIVE")
        print(f"  - Evidence for autocorrelation but effect below threshold")

    else:
        status = 'refuted'
        confidence = 'high'
        summary = (f"No significant autocorrelation after detrending. "
                   f"Lag-1 ACF = {lag1_result['mean']:.3f}, p = {lag1_result['p_value']:.2e}. "
                   f"The sampling trajectory is effectively uncorrelated - each step is "
                   f"independent of previous steps (after accounting for monotonic increase).")

        print(f"\nRESULT: REFUTED")
        print(f"  - No significant autocorrelation")
        print(f"  - Order sequence is effectively random after detrending")

    results['status'] = status
    results['confidence'] = confidence
    results['summary'] = summary

    # Save results
    output_dir = Path("results/trajectory_autocorrelation")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "autocorrelation_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}/autocorrelation_results.json")

    return results


if __name__ == "__main__":
    results = run_experiment(
        n_trajectories=30,
        n_live=50,
        n_iterations=300,
        image_size=32,
        base_seed=42
    )
