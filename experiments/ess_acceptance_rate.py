"""
RES-121: ESS acceptance rate vs nested sampling progress

Hypothesis: Elliptical slice sampling acceptance rate decreases as nested
sampling progresses to higher order thresholds (harder to find samples
that exceed increasingly high thresholds).

Metrics:
- Track contractions per ESS call at different threshold levels
- Acceptance rate = 1 / (1 + contractions)
- Correlation between threshold and contractions
"""

import numpy as np
from scipy import stats
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import (
    CPPN, order_multiplicative, elliptical_slice_sample
)

def run_experiment(n_trials: int = 50, image_size: int = 32):
    """
    Run nested-sampling-like progression and track ESS metrics.

    We simulate nested sampling by starting with random CPPNs and
    progressively increasing the threshold, tracking ESS difficulty.
    """

    # Threshold levels to test (simulating nested sampling progression)
    # Based on actual CPPN order distribution: mean ~0.05, max ~0.3
    # Use ESS to climb through these thresholds
    thresholds = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.22]

    results = {t: [] for t in thresholds}
    successes = {t: 0 for t in thresholds}
    failures = {t: 0 for t in thresholds}

    print(f"Running {n_trials} trials across {len(thresholds)} threshold levels...")
    print(f"Thresholds: {thresholds}")

    for trial in range(n_trials):
        if (trial + 1) % 10 == 0:
            print(f"  Trial {trial + 1}/{n_trials}")

        # Start with fresh CPPN for each trial
        cppn = CPPN()
        img = cppn.render(image_size)
        current_order = order_multiplicative(img)

        # Use ESS to climb through thresholds sequentially
        for threshold in thresholds:
            # Try to find a sample above this threshold
            new_cppn, new_img, new_order, _, n_contractions, success = elliptical_slice_sample(
                cppn, threshold, image_size, order_multiplicative
            )

            results[threshold].append(n_contractions)
            if success:
                successes[threshold] += 1
                # Update for next threshold
                cppn = new_cppn
                current_order = new_order
            else:
                failures[threshold] += 1
                # Can't proceed to higher thresholds if we failed here
                break

    return results, successes, failures, thresholds


def analyze_results(results, successes, failures, thresholds):
    """Compute statistics and correlation."""

    print("\n" + "="*60)
    print("ESS Acceptance Rate Analysis")
    print("="*60)

    threshold_list = []
    mean_contractions = []
    acceptance_rates = []

    # Collect ALL individual observations for regression
    all_thresholds = []
    all_contractions = []

    for t in thresholds:
        if len(results[t]) > 0:
            mean_c = np.mean(results[t])
            total = successes[t] + failures[t]
            acc_rate = successes[t] / total if total > 0 else 0

            threshold_list.append(t)
            mean_contractions.append(mean_c)
            acceptance_rates.append(acc_rate)

            # Add individual observations
            all_thresholds.extend([t] * len(results[t]))
            all_contractions.extend(results[t])

            print(f"\nThreshold {t:.2f}:")
            print(f"  N samples: {len(results[t])}")
            print(f"  Mean contractions: {mean_c:.2f} ± {np.std(results[t]):.2f}")
            print(f"  Success rate: {acc_rate:.2%} ({successes[t]}/{total})")

    # Correlation analysis on individual observations (high power)
    if len(all_thresholds) >= 10:
        print("\n" + "-"*60)
        print("Individual-Level Correlation Analysis:")
        print(f"  Total observations: {len(all_thresholds)}")

        # Spearman correlation on all individual observations
        r_individual, p_individual = stats.spearmanr(all_thresholds, all_contractions)
        print(f"  Threshold vs Contractions (all): r={r_individual:.3f}, p={p_individual:.2e}")

        # Also do Mann-Whitney U test: low thresholds vs high thresholds
        low_thresh = thresholds[:3]  # Bottom 3
        high_thresh = thresholds[-3:]  # Top 3

        low_contractions = []
        high_contractions = []
        for t in low_thresh:
            low_contractions.extend(results[t])
        for t in high_thresh:
            high_contractions.extend(results[t])

        U, p_mann_whitney = stats.mannwhitneyu(low_contractions, high_contractions, alternative='less')
        print(f"  Mann-Whitney U (low vs high thresh): U={U:.0f}, p={p_mann_whitney:.2e}")

        # Effect size (Cohen's d for first vs last threshold)
        data_low = results[thresholds[0]]
        data_high = results[thresholds[-1]]

        pooled_std = np.sqrt((np.var(data_low) + np.var(data_high)) / 2)
        cohens_d = (np.mean(data_high) - np.mean(data_low)) / pooled_std if pooled_std > 0 else 0
        print(f"  Cohen's d (lowest vs highest threshold): {cohens_d:.3f}")

        # Mean-level correlation (original analysis)
        r_means, p_means = stats.spearmanr(threshold_list, mean_contractions)
        print(f"\n  Aggregate correlation (means): r={r_means:.3f}, p={p_means:.4f}")

        return {
            'r_individual': r_individual,
            'p_individual': p_individual,
            'p_mann_whitney': p_mann_whitney,
            'cohens_d': cohens_d,
            'r_means': r_means,
            'p_means': p_means,
            'threshold_list': threshold_list,
            'mean_contractions': mean_contractions,
            'acceptance_rates': acceptance_rates,
            'n_observations': len(all_thresholds)
        }

    return None


def main():
    print("RES-121: ESS Acceptance Rate vs Nested Sampling Progress")
    print("="*60)

    np.random.seed(42)

    results, successes, failures, thresholds = run_experiment(n_trials=200, image_size=32)
    stats_results = analyze_results(results, successes, failures, thresholds)

    if stats_results:
        print("\n" + "="*60)
        print("CONCLUSION:")

        # Check hypothesis using individual-level analysis (high power)
        p_val = stats_results['p_individual']
        r_val = stats_results['r_individual']

        if p_val < 0.01 and r_val > 0:
            print("  VALIDATED: ESS contractions significantly increase with threshold")
            print(f"  Individual correlation: r={r_val:.3f}, p={p_val:.2e}")
            print(f"  Effect size (Cohen's d): {stats_results['cohens_d']:.2f}")
            status = "validated"
        elif p_val < 0.05 and r_val > 0:
            print("  INCONCLUSIVE: Weak positive correlation (p < 0.05 but >= 0.01)")
            status = "inconclusive"
        else:
            print("  REFUTED: No significant increase in ESS difficulty with threshold")
            status = "refuted"

        return status, stats_results

    return "inconclusive", None


if __name__ == "__main__":
    status, stats_results = main()
    print(f"\nFinal status: {status}")
