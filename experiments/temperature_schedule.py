"""
RES-067: Temperature Schedule Affects Order Ceiling via Exploration-Exploitation Tradeoff

Hypothesis: The effective "temperature schedule" in nested sampling (controlled by n_live)
determines the final achievable order. Slower schedules (more live points) allow better
exploration before the constraint tightens, leading to higher final order.

In nested sampling:
- threshold rises by ~1/n_live per iteration
- More live points = slower threshold rise = more exploration at each level
- Fewer live points = faster threshold rise = faster but possibly premature convergence

This tests if there's an optimal schedule steepness for achieving high order.
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import (
    CPPN, order_multiplicative, elliptical_slice_sample,
    log_prior, set_global_seed, LivePoint
)
import random
from scipy import stats


def run_nested_sampling_fixed_budget(n_live: int, total_samples: int, seed: int = 42):
    """
    Run nested sampling with fixed total sample budget but varying n_live.

    More live points = slower temperature rise but fewer iterations
    Fewer live points = faster temperature rise but more iterations
    """
    set_global_seed(seed)
    image_size = 32

    n_iterations = total_samples // n_live  # Fixed total budget

    # Initialize live points
    live_points = []
    for _ in range(n_live):
        cppn = CPPN()
        img = cppn.render(image_size)
        o = order_multiplicative(img)
        lp = log_prior(cppn)
        live_points.append(LivePoint(cppn, img, o, lp, {}))

    # Track trajectory
    thresholds = []
    max_orders = []

    for iteration in range(n_iterations):
        # Find worst
        worst_idx = min(range(n_live), key=lambda i: live_points[i].order_value)
        threshold = live_points[worst_idx].order_value
        thresholds.append(threshold)
        max_orders.append(max(lp.order_value for lp in live_points))

        # Select seed
        valid_seeds = [i for i in range(n_live)
                       if i != worst_idx and live_points[i].order_value >= threshold]
        if not valid_seeds:
            valid_seeds = [i for i in range(n_live) if i != worst_idx]
        seed_idx = random.choice(valid_seeds)

        # Replace using ESS
        new_cppn, new_img, new_order, new_logp, _, success = elliptical_slice_sample(
            live_points[seed_idx].cppn, threshold, image_size, order_multiplicative
        )

        if success:
            live_points[worst_idx] = LivePoint(new_cppn, new_img, new_order, new_logp, {})

    # Final statistics
    final_orders = [lp.order_value for lp in live_points]
    return {
        'n_live': n_live,
        'n_iterations': n_iterations,
        'final_max': max(final_orders),
        'final_mean': np.mean(final_orders),
        'final_std': np.std(final_orders),
        'threshold_trajectory': thresholds,
        'max_trajectory': max_orders,
        'final_threshold': thresholds[-1] if thresholds else 0,
    }


def main():
    print("=" * 70)
    print("RES-067: Temperature Schedule vs Order Ceiling")
    print("=" * 70)

    # Fixed total sample budget
    TOTAL_BUDGET = 5000  # Total samples across all experiments

    # Different "temperature schedules" via n_live
    # Larger n_live = slower threshold rise
    n_live_values = [20, 50, 100, 200, 500]

    n_seeds = 5  # Multiple seeds for statistics

    results_by_nlive = {n: [] for n in n_live_values}

    print(f"\nTotal sample budget: {TOTAL_BUDGET}")
    print(f"Testing n_live values: {n_live_values}")
    print(f"Seeds per condition: {n_seeds}")
    print()

    for n_live in n_live_values:
        n_iter = TOTAL_BUDGET // n_live
        print(f"n_live={n_live:3d}: {n_iter} iterations (threshold_rise_rate ~ 1/{n_live})")

        for seed in range(n_seeds):
            result = run_nested_sampling_fixed_budget(n_live, TOTAL_BUDGET, seed=seed)
            results_by_nlive[n_live].append(result)
            print(f"  Seed {seed}: final_max={result['final_max']:.4f}, final_mean={result['final_mean']:.4f}")

    print("\n" + "=" * 70)
    print("SUMMARY: Final Order by Temperature Schedule")
    print("=" * 70)

    summary_data = []
    for n_live in n_live_values:
        results = results_by_nlive[n_live]
        final_maxes = [r['final_max'] for r in results]
        final_means = [r['final_mean'] for r in results]
        final_thresholds = [r['final_threshold'] for r in results]

        summary_data.append({
            'n_live': n_live,
            'rise_rate': 1/n_live,
            'max_order_mean': np.mean(final_maxes),
            'max_order_std': np.std(final_maxes),
            'mean_order_mean': np.mean(final_means),
            'mean_order_std': np.std(final_means),
            'threshold_mean': np.mean(final_thresholds),
        })

        print(f"n_live={n_live:3d} (rate=1/{n_live:3d}): "
              f"max_order={np.mean(final_maxes):.4f}±{np.std(final_maxes):.4f}, "
              f"mean_order={np.mean(final_means):.4f}±{np.std(final_means):.4f}")

    # Statistical test: correlation between n_live and final order
    n_live_arr = np.array([s['n_live'] for s in summary_data])
    max_order_arr = np.array([s['max_order_mean'] for s in summary_data])
    mean_order_arr = np.array([s['mean_order_mean'] for s in summary_data])

    # Use log(n_live) since effect might be logarithmic
    log_nlive = np.log(n_live_arr)

    r_max, p_max = stats.pearsonr(log_nlive, max_order_arr)
    r_mean, p_mean = stats.pearsonr(log_nlive, mean_order_arr)

    print(f"\nCorrelation with log(n_live):")
    print(f"  Max order:  r={r_max:.3f}, p={p_max:.4f}")
    print(f"  Mean order: r={r_mean:.3f}, p={p_mean:.4f}")

    # Compare extreme conditions
    slow_results = results_by_nlive[max(n_live_values)]
    fast_results = results_by_nlive[min(n_live_values)]

    slow_maxes = [r['final_max'] for r in slow_results]
    fast_maxes = [r['final_max'] for r in fast_results]

    t_stat, t_pval = stats.ttest_ind(slow_maxes, fast_maxes)
    effect_size = (np.mean(slow_maxes) - np.mean(fast_maxes)) / np.sqrt(
        (np.std(slow_maxes)**2 + np.std(fast_maxes)**2) / 2
    )

    print(f"\nSlow (n_live={max(n_live_values)}) vs Fast (n_live={min(n_live_values)}):")
    print(f"  Slow max order: {np.mean(slow_maxes):.4f}±{np.std(slow_maxes):.4f}")
    print(f"  Fast max order: {np.mean(fast_maxes):.4f}±{np.std(fast_maxes):.4f}")
    print(f"  t-test: t={t_stat:.3f}, p={t_pval:.4f}")
    print(f"  Effect size (Cohen's d): {effect_size:.3f}")

    # Determine validation status
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    # Hypothesis: slower schedule (larger n_live) leads to higher final order
    if r_max > 0.5 and p_max < 0.05:
        if effect_size > 0.5:
            status = "VALIDATED"
            conclusion = "Slower temperature schedules achieve significantly higher order"
        else:
            status = "VALIDATED (weak effect)"
            conclusion = "Slower schedules help but effect size is modest"
    elif r_max < -0.5 and p_max < 0.05:
        status = "REFUTED"
        conclusion = "Contrary to hypothesis: faster schedules achieve higher order"
    else:
        status = "INCONCLUSIVE"
        conclusion = "No clear relationship between schedule steepness and order ceiling"

    print(f"Status: {status}")
    print(f"Conclusion: {conclusion}")

    # Metrics for log
    metrics = {
        'correlation_r': float(r_max),
        'correlation_p': float(p_max),
        'effect_size': float(effect_size),
        't_stat': float(t_stat),
        't_pval': float(t_pval),
        'slow_max': float(np.mean(slow_maxes)),
        'fast_max': float(np.mean(fast_maxes)),
    }

    print(f"\nMetrics: {metrics}")

    return status, metrics, conclusion


if __name__ == "__main__":
    status, metrics, conclusion = main()
