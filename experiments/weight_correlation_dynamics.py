"""
RES-136: Weight changes during optimization are correlated across connections

Hypothesis: During optimization toward high-order images, weight changes are
correlated across connections (weights move together in coordinated patterns).

This tests whether optimization follows structured paths in weight space
(correlated updates) or explores independently (uncorrelated updates).
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import CPPN, order_multiplicative
from scipy import stats


def optimize_for_order(n_steps=50, lr=0.1, n_trials=100):
    """
    Optimize CPPN weights toward high-order images using gradient-free
    hill climbing. Track weight change correlations.
    """
    correlations_high_order = []  # Trials that reach high order
    correlations_low_order = []   # Trials that stay low order

    for trial in range(n_trials):
        cppn = CPPN()
        weights_history = [cppn.get_weights().copy()]
        order_history = [order_multiplicative(cppn.render(32))]

        # Simple hill-climbing optimization
        for step in range(n_steps):
            best_weights = cppn.get_weights().copy()
            best_order = order_history[-1]

            # Try random perturbations
            for _ in range(10):
                candidate = cppn.copy()
                delta = np.random.randn(len(best_weights)) * lr
                candidate.set_weights(best_weights + delta)
                new_order = order_multiplicative(candidate.render(32))

                if new_order > best_order:
                    best_order = new_order
                    best_weights = candidate.get_weights().copy()

            cppn.set_weights(best_weights)
            weights_history.append(best_weights.copy())
            order_history.append(best_order)

        # Compute weight change correlations
        weights_history = np.array(weights_history)
        deltas = np.diff(weights_history, axis=0)  # [n_steps, n_weights]

        # Correlation matrix of weight changes
        if deltas.shape[0] > 5:  # Need enough steps
            # Filter out zero-variance columns
            nonzero_mask = np.std(deltas, axis=0) > 1e-10
            if np.sum(nonzero_mask) < 2:
                continue  # Skip if not enough non-zero deltas

            filtered_deltas = deltas[:, nonzero_mask]
            n_weights = filtered_deltas.shape[1]
            corr_matrix = np.corrcoef(filtered_deltas.T)  # [n_weights, n_weights]

            # Mean absolute off-diagonal correlation
            mask = ~np.eye(n_weights, dtype=bool)
            mean_corr = np.nanmean(np.abs(corr_matrix[mask]))

            if np.isnan(mean_corr):
                continue

            final_order = order_history[-1]
            if final_order > 0.3:  # High order threshold
                correlations_high_order.append(mean_corr)
            else:
                correlations_low_order.append(mean_corr)

    return correlations_high_order, correlations_low_order


def main():
    np.random.seed(42)

    print("RES-136: Weight change correlation during optimization")
    print("=" * 60)

    high_corr, low_corr = optimize_for_order(n_steps=50, lr=0.1, n_trials=200)

    print(f"\nTrials reaching high order (>0.3): {len(high_corr)}")
    print(f"Trials staying low order: {len(low_corr)}")

    if len(high_corr) < 10 or len(low_corr) < 10:
        print("\nInsufficient samples for comparison")
        return

    high_mean = np.mean(high_corr)
    low_mean = np.mean(low_corr)
    high_std = np.std(high_corr)
    low_std = np.std(low_corr)

    print(f"\nWeight change correlation (high-order paths): {high_mean:.4f} +/- {high_std:.4f}")
    print(f"Weight change correlation (low-order paths):  {low_mean:.4f} +/- {low_std:.4f}")

    # Statistical test
    t_stat, p_value = stats.ttest_ind(high_corr, low_corr)
    pooled_std = np.sqrt((high_std**2 + low_std**2) / 2)
    cohens_d = (high_mean - low_mean) / pooled_std if pooled_std > 0 else 0

    print(f"\nStatistical comparison:")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.6f}")
    print(f"  Cohen's d: {cohens_d:.4f}")

    # Additional: correlation sign analysis
    print("\n" + "=" * 60)
    print("Interpretation:")
    if cohens_d > 0.5 and p_value < 0.01:
        print("  VALIDATED: High-order paths have more correlated weight changes")
    elif cohens_d < -0.5 and p_value < 0.01:
        print("  REFUTED: High-order paths have LESS correlated weight changes")
    else:
        print("  INCONCLUSIVE: No significant difference in weight change correlation")

    return {
        'high_order_corr_mean': high_mean,
        'low_order_corr_mean': low_mean,
        'effect_size': cohens_d,
        'p_value': p_value,
        'n_high': len(high_corr),
        'n_low': len(low_corr)
    }


if __name__ == '__main__':
    results = main()
