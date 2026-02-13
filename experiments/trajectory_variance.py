#!/usr/bin/env python3
"""
RES-072: Test whether order variance across live points decreases monotonically.

Hypothesis: Order variance across nested sampling live points decreases
monotonically as iterations progress, reflecting convergence to high-order regions.
"""

import numpy as np
from scipy import stats
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import nested_sampling_v3, order_multiplicative


def run_trajectory_variance_experiment(n_trials=5, n_live=50, n_iterations=200, image_size=16):
    """Run nested sampling and track variance trajectory."""

    all_variance_trajectories = []
    all_monotonicity_scores = []

    for trial in range(n_trials):
        print(f"\n=== Trial {trial+1}/{n_trials} ===")

        dead_points, live_points, snapshots = nested_sampling_v3(
            n_live=n_live,
            n_iterations=n_iterations,
            image_size=image_size,
            order_fn=order_multiplicative,
            sampling_mode="measure",
            track_metrics=True,
            output_dir=f"/tmp/traj_var_trial_{trial}",
            seed=42 + trial
        )

        # Extract variance trajectory from snapshots
        variances = [s.var_order for s in snapshots]
        iterations = [s.iteration for s in snapshots]

        all_variance_trajectories.append(variances)

        # Compute monotonicity: fraction of steps where variance decreases
        decreases = sum(1 for i in range(1, len(variances)) if variances[i] < variances[i-1])
        monotonicity = decreases / (len(variances) - 1) if len(variances) > 1 else 0
        all_monotonicity_scores.append(monotonicity)

        print(f"  Variance: {variances[0]:.4f} -> {variances[-1]:.4f}")
        print(f"  Monotonicity score: {monotonicity:.3f}")

    # Statistical analysis
    mean_monotonicity = np.mean(all_monotonicity_scores)
    std_monotonicity = np.std(all_monotonicity_scores)

    # Test if monotonicity > 0.5 (random expectation)
    t_stat, p_value = stats.ttest_1samp(all_monotonicity_scores, 0.5)

    # Compute overall trend using Spearman correlation
    # (correlation between iteration number and variance)
    all_spearman = []
    for variances in all_variance_trajectories:
        iters = list(range(len(variances)))
        rho, _ = stats.spearmanr(iters, variances)
        all_spearman.append(rho)

    mean_spearman = np.mean(all_spearman)

    # Effect size (Cohen's d)
    cohens_d = (mean_monotonicity - 0.5) / std_monotonicity if std_monotonicity > 0 else 0

    # Compute variance reduction ratio
    reduction_ratios = []
    for variances in all_variance_trajectories:
        if variances[0] > 0:
            reduction_ratios.append(variances[-1] / variances[0])
    mean_reduction = np.mean(reduction_ratios)

    results = {
        'n_trials': n_trials,
        'mean_monotonicity': mean_monotonicity,
        'std_monotonicity': std_monotonicity,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'mean_spearman_rho': mean_spearman,
        'mean_variance_reduction': mean_reduction,
        'individual_monotonicity': all_monotonicity_scores,
        'individual_spearman': all_spearman,
    }

    return results


if __name__ == "__main__":
    print("=" * 70)
    print("RES-072: Order Variance Trajectory Analysis")
    print("=" * 70)

    results = run_trajectory_variance_experiment(
        n_trials=5,
        n_live=50,
        n_iterations=200,
        image_size=16
    )

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Mean monotonicity score: {results['mean_monotonicity']:.3f} ± {results['std_monotonicity']:.3f}")
    print(f"  (random expectation = 0.5)")
    print(f"P-value (H0: monotonicity = 0.5): {results['p_value']:.4e}")
    print(f"Cohen's d: {results['cohens_d']:.2f}")
    print(f"Mean Spearman rho (iter vs variance): {results['mean_spearman_rho']:.3f}")
    print(f"Mean variance reduction (final/initial): {results['mean_variance_reduction']:.3f}")

    # Determine status
    if results['p_value'] < 0.01 and results['cohens_d'] > 0.5 and results['mean_monotonicity'] > 0.6:
        status = "VALIDATED"
    elif results['p_value'] > 0.05:
        status = "REFUTED"
    else:
        status = "INCONCLUSIVE"

    print(f"\nSTATUS: {status}")
    print(f"\nIndividual trials:")
    for i, (m, s) in enumerate(zip(results['individual_monotonicity'], results['individual_spearman'])):
        print(f"  Trial {i+1}: monotonicity={m:.3f}, spearman={s:.3f}")
