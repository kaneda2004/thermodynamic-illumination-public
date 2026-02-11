"""
RES-191: Order gain per ESS contraction increases as NS progresses to higher-order regions

Hypothesis: The order gain achieved per ESS contraction increases as nested sampling
progresses to higher-order regions.

Background:
- RES-177 showed order velocity per ITERATION decreases (d=-2.23)
- RES-145 showed gradient magnitude increases during NS (d=1.07)
- RES-121 showed more contractions are needed per iteration

This tests whether individual contractions become more effective despite overall slowdown.
The steeper gradients at high order might mean each contraction yields larger order gains.

Null hypothesis: Order gain per contraction is constant or decreasing across NS phases.
"""

import numpy as np
import sys
import os
import json
from scipy import stats
from dataclasses import dataclass, field
from typing import Optional, Callable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.thermo_sampler_v3 import (
    CPPN, PRIOR_SIGMA, order_multiplicative, log_prior, set_global_seed
)


def elliptical_slice_sample_tracked(
    cppn: CPPN,
    threshold: float,
    image_size: int,
    order_fn: Callable,
    max_contractions: int = 100,
    max_restarts: int = 5
) -> tuple[CPPN, np.ndarray, float, float, int, bool, float]:
    """
    ESS with order gain tracking.

    Returns: (cppn, image, order, log_prior, n_contractions, success, order_gain)
    """
    current_w = cppn.get_weights()
    n_params = len(current_w)
    total_contractions = 0

    for restart in range(max_restarts):
        nu = np.random.randn(n_params) * PRIOR_SIGMA
        phi = np.random.uniform(0, 2 * np.pi)
        phi_min = phi - 2 * np.pi
        phi_max = phi
        n_contractions = 0

        while n_contractions < max_contractions:
            proposal_w = current_w * np.cos(phi) + nu * np.sin(phi)
            proposal_cppn = cppn.copy()
            proposal_cppn.set_weights(proposal_w)
            proposal_img = proposal_cppn.render(image_size)
            proposal_order = order_fn(proposal_img)

            if proposal_order >= threshold:
                order_gain = proposal_order - threshold
                return (proposal_cppn, proposal_img, proposal_order,
                       log_prior(proposal_cppn), total_contractions + n_contractions + 1,
                       True, order_gain)

            if phi < 0:
                phi_min = phi
            else:
                phi_max = phi

            phi = np.random.uniform(phi_min, phi_max)
            n_contractions += 1

            if phi_max - phi_min < 1e-10:
                break

        total_contractions += n_contractions

    # Failure
    current_img = cppn.render(image_size)
    return (cppn, current_img, order_fn(current_img), log_prior(cppn),
            total_contractions, False, 0.0)


@dataclass
class LivePoint:
    cppn: CPPN
    image: np.ndarray
    order_value: float


def run_experiment(
    n_runs: int = 10,
    n_live: int = 50,
    n_iterations: int = 300,
    image_size: int = 32,
    seed: int = 42
):
    """
    Run NS and track order gain per contraction across phases.
    """
    print(f"Running {n_runs} NS runs with {n_iterations} iterations each...")

    all_results = []

    for run in range(n_runs):
        set_global_seed(seed + run)

        # Initialize live points
        live_points = []
        for _ in range(n_live):
            cppn = CPPN()
            img = cppn.render(image_size)
            o = order_multiplicative(img)
            live_points.append(LivePoint(cppn, img, o))

        # Track per-iteration data
        iteration_data = []

        for iteration in range(n_iterations):
            # Find worst point
            worst_idx = min(range(n_live), key=lambda i: live_points[i].order_value)
            worst = live_points[worst_idx]
            threshold = worst.order_value

            # Select random seed for replacement
            seed_idx = np.random.choice([i for i in range(n_live) if i != worst_idx])
            seed_cppn = live_points[seed_idx].cppn.copy()

            # ESS with tracking
            new_cppn, new_img, new_order, _, n_contractions, success, order_gain = \
                elliptical_slice_sample_tracked(
                    seed_cppn, threshold, image_size, order_multiplicative
                )

            if success and n_contractions > 0:
                gain_per_contraction = order_gain / n_contractions
                iteration_data.append({
                    'iteration': iteration,
                    'threshold': threshold,
                    'new_order': new_order,
                    'order_gain': order_gain,
                    'n_contractions': n_contractions,
                    'gain_per_contraction': gain_per_contraction
                })

                # Update live points
                live_points[worst_idx] = LivePoint(new_cppn, new_img, new_order)
            elif success:
                # 0 contractions - instant success
                live_points[worst_idx] = LivePoint(new_cppn, new_img, new_order)

        all_results.append(iteration_data)
        print(f"  Run {run+1}/{n_runs}: {len(iteration_data)} valid iterations")

    return all_results


def analyze_results(all_results: list) -> dict:
    """
    Analyze order gain per contraction across NS phases.
    """
    # Combine all runs
    all_data = []
    for run_data in all_results:
        all_data.extend(run_data)

    if len(all_data) < 100:
        return {'error': 'Insufficient data'}

    # Extract arrays
    thresholds = np.array([d['threshold'] for d in all_data])
    gains_per_contraction = np.array([d['gain_per_contraction'] for d in all_data])
    iterations = np.array([d['iteration'] for d in all_data])

    # Define phases by threshold quartiles
    q25, q50, q75 = np.percentile(thresholds, [25, 50, 75])

    low_order_mask = thresholds < q25
    mid_order_mask = (thresholds >= q25) & (thresholds < q75)
    high_order_mask = thresholds >= q75

    low_gains = gains_per_contraction[low_order_mask]
    mid_gains = gains_per_contraction[mid_order_mask]
    high_gains = gains_per_contraction[high_order_mask]

    # Statistics
    low_mean = np.mean(low_gains) if len(low_gains) > 0 else np.nan
    mid_mean = np.mean(mid_gains) if len(mid_gains) > 0 else np.nan
    high_mean = np.mean(high_gains) if len(high_gains) > 0 else np.nan

    low_std = np.std(low_gains) if len(low_gains) > 0 else np.nan
    mid_std = np.std(mid_gains) if len(mid_gains) > 0 else np.nan
    high_std = np.std(high_gains) if len(high_gains) > 0 else np.nan

    print(f"\nOrder gain per contraction by phase:")
    print(f"  Low order (threshold < {q25:.4f}): {low_mean:.6f} ± {low_std:.6f} (n={len(low_gains)})")
    print(f"  Mid order ({q25:.4f} <= threshold < {q75:.4f}): {mid_mean:.6f} ± {mid_std:.6f} (n={len(mid_gains)})")
    print(f"  High order (threshold >= {q75:.4f}): {high_mean:.6f} ± {high_std:.6f} (n={len(high_gains)})")

    # Cohen's d (high vs low)
    pooled_std = np.sqrt((low_std**2 + high_std**2) / 2)
    cohens_d = (high_mean - low_mean) / pooled_std if pooled_std > 0 else 0

    # T-test (high vs low)
    if len(low_gains) > 1 and len(high_gains) > 1:
        t_stat, p_value = stats.ttest_ind(high_gains, low_gains, equal_var=False)
    else:
        t_stat, p_value = np.nan, np.nan

    # Spearman correlation: threshold vs gain_per_contraction
    rho, rho_p = stats.spearmanr(thresholds, gains_per_contraction)

    # Pearson correlation
    r, r_p = stats.pearsonr(thresholds, gains_per_contraction)

    print(f"\nCorrelations with threshold:")
    print(f"  Spearman rho = {rho:.4f} (p = {rho_p:.2e})")
    print(f"  Pearson r = {r:.4f} (p = {r_p:.2e})")

    print(f"\nHigh vs Low phase comparison:")
    print(f"  Cohen's d = {cohens_d:.4f}")
    print(f"  t-test p = {p_value:.2e}")

    # Also test by iteration (early vs late)
    n_iter_max = max(iterations)
    early_mask = iterations < n_iter_max * 0.33
    late_mask = iterations >= n_iter_max * 0.67

    early_gains = gains_per_contraction[early_mask]
    late_gains = gains_per_contraction[late_mask]

    early_mean = np.mean(early_gains) if len(early_gains) > 0 else np.nan
    late_mean = np.mean(late_gains) if len(late_gains) > 0 else np.nan

    print(f"\nBy iteration:")
    print(f"  Early (iter < {n_iter_max*0.33:.0f}): {early_mean:.6f} (n={len(early_gains)})")
    print(f"  Late (iter >= {n_iter_max*0.67:.0f}): {late_mean:.6f} (n={len(late_gains)})")

    # Iteration correlation
    iter_rho, iter_rho_p = stats.spearmanr(iterations, gains_per_contraction)
    print(f"  Spearman rho (iteration) = {iter_rho:.4f} (p = {iter_rho_p:.2e})")

    # Determine verdict
    # Hypothesis: gain per contraction INCREASES with threshold/iteration
    hypothesis_validated = (cohens_d > 0.5 and p_value < 0.01 and rho > 0)

    results = {
        'n_samples': len(all_data),
        'threshold_quartiles': [float(q25), float(q50), float(q75)],
        'low_order': {
            'mean': float(low_mean),
            'std': float(low_std),
            'n': int(len(low_gains))
        },
        'mid_order': {
            'mean': float(mid_mean),
            'std': float(mid_std),
            'n': int(len(mid_gains))
        },
        'high_order': {
            'mean': float(high_mean),
            'std': float(high_std),
            'n': int(len(high_gains))
        },
        'cohens_d': float(cohens_d),
        't_stat': float(t_stat) if not np.isnan(t_stat) else None,
        'p_value': float(p_value) if not np.isnan(p_value) else None,
        'spearman_rho': float(rho),
        'spearman_p': float(rho_p),
        'pearson_r': float(r),
        'pearson_p': float(r_p),
        'iteration_spearman': float(iter_rho),
        'iteration_spearman_p': float(iter_rho_p),
        'early_mean': float(early_mean),
        'late_mean': float(late_mean),
        'hypothesis_validated': bool(hypothesis_validated)
    }

    return results


def main():
    print("=" * 70)
    print("RES-191: Order gain per ESS contraction vs NS phase")
    print("=" * 70)
    print()
    print("Hypothesis: Order gain per contraction INCREASES as NS progresses")
    print("to higher-order regions (steeper gradients → larger jumps)")
    print()

    # Run experiment
    all_results = run_experiment(
        n_runs=15,
        n_live=50,
        n_iterations=300,
        image_size=32,
        seed=42
    )

    # Analyze
    results = analyze_results(all_results)

    # Save results
    os.makedirs('results/order_gain_per_contraction', exist_ok=True)
    with open('results/order_gain_per_contraction/results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print()
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)

    if results.get('hypothesis_validated'):
        print("VALIDATED: Order gain per contraction increases with NS progress")
    else:
        # Check direction
        if results['cohens_d'] < -0.5:
            print("REFUTED: Order gain per contraction DECREASES (opposite hypothesis)")
        elif abs(results['cohens_d']) < 0.5:
            print(f"REFUTED: Effect size d={results['cohens_d']:.3f} below threshold (|d|<0.5)")
        elif results['p_value'] is not None and results['p_value'] >= 0.01:
            print(f"INCONCLUSIVE: p={results['p_value']:.4f} not significant (p>=0.01)")
        else:
            print(f"REFUTED: d={results['cohens_d']:.3f}, p={results.get('p_value', 'N/A')}")

    print()
    print(f"Effect size (Cohen's d): {results['cohens_d']:.4f}")
    print(f"Threshold correlation (Spearman): {results['spearman_rho']:.4f}")
    print(f"p-value: {results.get('p_value', 'N/A')}")

    return results


if __name__ == '__main__':
    main()
