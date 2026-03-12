#!/usr/bin/env python3
"""
RES-200: ESS step size (displacement magnitude) decreases as NS approaches high-order regions

Hypothesis: The weight-space step size (L2 displacement between consecutive accepted samples)
decreases as nested sampling approaches higher-order regions. This would explain why NS
slows down (RES-177 velocity decrease, RES-191 order gain decrease) - not just harder to
find valid proposals, but also taking smaller steps when successful.

Method:
- Run NS with trajectory logging
- Compute displacement = ||w_{t+1} - w_t||_2 for each accepted step
- Bin by order threshold and compute mean displacement per bin
- Test correlation between order threshold and step size

Prior findings suggest:
- RES-165: Trajectory dimension decreases with order
- RES-177: Order velocity decreases through phases
- RES-191: Order gain per contraction decreases 3.6x
- RES-161: Curvature is constant (~2.1 rad) regardless of order

If step size decreases while curvature stays constant, NS moves through tighter
spirals in high-order regions.
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import (
    CPPN, order_multiplicative, PRIOR_SIGMA, log_prior
)
from scipy import stats
import json
from pathlib import Path

def ess_with_trajectory(
    cppn: CPPN,
    threshold: float,
    image_size: int = 32,
    max_contractions: int = 100,
    max_restarts: int = 5
):
    """
    ESS that returns the trajectory of weight vectors for step size analysis.
    Returns: (final_cppn, trajectory_weights, accepted_orders, success)
    """
    current_w = cppn.get_weights()
    n_params = len(current_w)

    trajectory = [current_w.copy()]  # Include starting point

    for restart in range(max_restarts):
        nu = np.random.randn(n_params) * PRIOR_SIGMA

        phi = np.random.uniform(0, 2 * np.pi)
        phi_min = phi - 2 * np.pi
        phi_max = phi

        for _ in range(max_contractions):
            proposal_w = current_w * np.cos(phi) + nu * np.sin(phi)

            proposal_cppn = cppn.copy()
            proposal_cppn.set_weights(proposal_w)
            proposal_img = proposal_cppn.render(image_size)
            proposal_order = order_multiplicative(proposal_img)

            if proposal_order >= threshold:
                trajectory.append(proposal_w.copy())
                return proposal_cppn, np.array(trajectory), proposal_order, True

            if phi < 0:
                phi_min = phi
            else:
                phi_max = phi

            phi = np.random.uniform(phi_min, phi_max)

            if phi_max - phi_min < 1e-10:
                break

    return cppn, np.array(trajectory), order_multiplicative(cppn.render(image_size)), False


def run_ns_with_step_tracking(n_live=50, n_iterations=300, seed=42):
    """
    Run NS and track step sizes at each iteration.
    Returns list of (threshold, step_size, order_gained) tuples.
    """
    np.random.seed(seed)

    # Initialize live points
    live_points = []
    for _ in range(n_live):
        cppn = CPPN()
        img = cppn.render(32)
        order = order_multiplicative(img)
        live_points.append({
            'cppn': cppn,
            'weights': cppn.get_weights().copy(),
            'order': order
        })

    step_records = []

    for iteration in range(n_iterations):
        # Find lowest order point
        orders = [lp['order'] for lp in live_points]
        worst_idx = np.argmin(orders)
        threshold = orders[worst_idx]

        # Select random seed (not the worst)
        valid_indices = [i for i in range(n_live) if i != worst_idx]
        seed_idx = np.random.choice(valid_indices)
        seed_cppn = live_points[seed_idx]['cppn'].copy()
        seed_weights = live_points[seed_idx]['weights'].copy()
        seed_order = live_points[seed_idx]['order']

        # Run ESS
        new_cppn, trajectory, new_order, success = ess_with_trajectory(
            seed_cppn, threshold
        )

        if success and len(trajectory) >= 2:
            # Compute step size (L2 displacement from seed to accepted proposal)
            new_weights = new_cppn.get_weights()
            step_size = np.linalg.norm(new_weights - seed_weights)
            order_gain = new_order - seed_order

            step_records.append({
                'iteration': iteration,
                'threshold': float(threshold),
                'step_size': float(step_size),
                'order_gain': float(order_gain),
                'new_order': float(new_order),
                'seed_order': float(seed_order)
            })

            # Replace worst point
            live_points[worst_idx] = {
                'cppn': new_cppn,
                'weights': new_weights.copy(),
                'order': new_order
            }

        if iteration % 50 == 0:
            current_orders = [lp['order'] for lp in live_points]
            print(f"Iter {iteration}: threshold={threshold:.4f}, "
                  f"mean_order={np.mean(current_orders):.4f}")

    return step_records


def analyze_step_sizes(step_records):
    """Analyze how step size correlates with order threshold."""

    if len(step_records) < 20:
        return None

    thresholds = np.array([r['threshold'] for r in step_records])
    step_sizes = np.array([r['step_size'] for r in step_records])

    # Correlation analysis
    rho, p_spearman = stats.spearmanr(thresholds, step_sizes)
    r, p_pearson = stats.pearsonr(thresholds, step_sizes)

    # Bin by order threshold
    n_bins = 5
    bins = np.linspace(thresholds.min(), thresholds.max(), n_bins + 1)
    bin_means = []
    bin_centers = []

    for i in range(n_bins):
        mask = (thresholds >= bins[i]) & (thresholds < bins[i+1])
        if mask.sum() > 5:
            bin_means.append(np.mean(step_sizes[mask]))
            bin_centers.append((bins[i] + bins[i+1]) / 2)

    # Compare low vs high order phases
    median_threshold = np.median(thresholds)
    low_order_steps = step_sizes[thresholds < median_threshold]
    high_order_steps = step_sizes[thresholds >= median_threshold]

    if len(low_order_steps) > 5 and len(high_order_steps) > 5:
        t_stat, p_ttest = stats.ttest_ind(low_order_steps, high_order_steps)
        cohens_d = (np.mean(low_order_steps) - np.mean(high_order_steps)) / \
                   np.sqrt((np.var(low_order_steps) + np.var(high_order_steps)) / 2)
        mann_whitney, p_mw = stats.mannwhitneyu(low_order_steps, high_order_steps, alternative='two-sided')
    else:
        t_stat, p_ttest, cohens_d, p_mw = np.nan, np.nan, np.nan, np.nan

    return {
        'n_samples': len(step_records),
        'spearman_rho': rho,
        'spearman_p': p_spearman,
        'pearson_r': r,
        'pearson_p': p_pearson,
        'low_order_mean_step': float(np.mean(low_order_steps)) if len(low_order_steps) > 0 else np.nan,
        'high_order_mean_step': float(np.mean(high_order_steps)) if len(high_order_steps) > 0 else np.nan,
        'cohens_d': cohens_d,
        'p_ttest': p_ttest,
        'p_mannwhitney': p_mw,
        'threshold_range': [float(thresholds.min()), float(thresholds.max())],
        'step_size_range': [float(step_sizes.min()), float(step_sizes.max())],
        'bin_centers': bin_centers,
        'bin_means': bin_means
    }


def main():
    print("RES-200: ESS step size vs order threshold")
    print("=" * 60)
    print()
    print("Hypothesis: Step size decreases as NS approaches high-order regions")
    print()

    # Run multiple seeds for robustness
    all_records = []
    results_by_seed = []

    for seed in range(5):
        print(f"\n--- Run {seed+1}/5 (seed={seed}) ---")
        records = run_ns_with_step_tracking(n_live=50, n_iterations=300, seed=seed)
        all_records.extend(records)

        analysis = analyze_step_sizes(records)
        if analysis:
            results_by_seed.append(analysis)
            print(f"  Spearman rho: {analysis['spearman_rho']:.4f} (p={analysis['spearman_p']:.2e})")
            print(f"  Low-order mean step: {analysis['low_order_mean_step']:.4f}")
            print(f"  High-order mean step: {analysis['high_order_mean_step']:.4f}")

    # Combined analysis
    print("\n" + "=" * 60)
    print("COMBINED ANALYSIS (all runs)")
    print("=" * 60)

    combined = analyze_step_sizes(all_records)

    print(f"\nTotal samples: {combined['n_samples']}")
    print(f"Threshold range: [{combined['threshold_range'][0]:.4f}, {combined['threshold_range'][1]:.4f}]")
    print(f"Step size range: [{combined['step_size_range'][0]:.4f}, {combined['step_size_range'][1]:.4f}]")
    print()
    print(f"Spearman rho: {combined['spearman_rho']:.4f} (p={combined['spearman_p']:.2e})")
    print(f"Pearson r: {combined['pearson_r']:.4f} (p={combined['pearson_p']:.2e})")
    print()
    print(f"Low-order phase mean step: {combined['low_order_mean_step']:.4f}")
    print(f"High-order phase mean step: {combined['high_order_mean_step']:.4f}")
    print(f"Ratio (low/high): {combined['low_order_mean_step']/combined['high_order_mean_step']:.2f}x")
    print(f"Cohen's d: {combined['cohens_d']:.4f}")
    print(f"Mann-Whitney p: {combined['p_mannwhitney']:.2e}")
    print()

    # Binned means
    if combined['bin_centers']:
        print("Binned step sizes by order threshold:")
        for center, mean in zip(combined['bin_centers'], combined['bin_means']):
            print(f"  Order ~{center:.3f}: step_size = {mean:.4f}")

    # Determine result
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)

    significant = combined['spearman_p'] < 0.01 and combined['p_mannwhitney'] < 0.01
    effect_strong = abs(combined['cohens_d']) > 0.5
    direction_correct = combined['spearman_rho'] < 0  # Negative = step size decreases with order

    if significant and effect_strong and direction_correct:
        status = "VALIDATED"
        summary = (f"ESS step size DECREASES with order threshold (rho={combined['spearman_rho']:.3f}, "
                  f"d={combined['cohens_d']:.2f}). Low-order steps are {combined['low_order_mean_step']/combined['high_order_mean_step']:.1f}x "
                  f"larger than high-order steps. Combined with constant curvature (RES-161), "
                  f"this means NS traverses tighter spirals in high-order regions.")
    elif significant and effect_strong and not direction_correct:
        status = "REFUTED"
        summary = (f"Step size INCREASES with order (rho={combined['spearman_rho']:.3f}, d={combined['cohens_d']:.2f}), "
                  f"opposite to hypothesis. High-order phase has {combined['high_order_mean_step']/combined['low_order_mean_step']:.1f}x "
                  f"larger steps than low-order. NS takes bigger jumps in high-order regions, not smaller.")
    elif not significant:
        status = "REFUTED"
        summary = (f"Step size shows NO significant correlation with order (rho={combined['spearman_rho']:.3f}, "
                  f"p={combined['spearman_p']:.2e}, d={combined['cohens_d']:.2f}). ESS step size is "
                  f"approximately constant regardless of threshold level.")
    else:
        status = "INCONCLUSIVE"
        summary = (f"Weak effect: rho={combined['spearman_rho']:.3f}, d={combined['cohens_d']:.2f}. "
                  f"p={combined['spearman_p']:.2e}")

    print(f"\nStatus: {status}")
    print(f"\nSummary: {summary}")

    # Save results
    output = {
        'experiment_id': 'RES-200',
        'hypothesis': 'ESS step size (displacement magnitude) decreases as NS approaches high-order regions',
        'domain': 'sampling_geometry',
        'status': status.lower(),
        'combined_analysis': combined,
        'per_seed_results': results_by_seed,
        'summary': summary
    }

    output_path = Path('/Users/matt/Development/monochrome_noise_converger/results/res200_step_size.json')
    output_path.parent.mkdir(exist_ok=True)

    # Convert numpy types
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(output_path, 'w') as f:
        json.dump(convert(output), f, indent=2)

    print(f"\nResults saved to {output_path}")

    return status.lower(), summary


if __name__ == '__main__':
    status, summary = main()
