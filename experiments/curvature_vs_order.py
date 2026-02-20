#!/usr/bin/env python
"""
RES-129: Local curvature of order landscape varies with order level.

Extends RES-049 (Fisher information) by measuring Hessian eigenvalues
at different order levels to determine if high-order regions have
systematically different curvature (sharper peaks vs flatter basins).

Method:
1. Generate CPPNs across a range of order values
2. Compute numerical Hessian of order function at each point
3. Analyze eigenvalue distribution vs order level
"""

import numpy as np
from scipy import stats
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import CPPN, order_multiplicative


def compute_hessian_eigenvalues(cppn: CPPN, eps: float = 0.01) -> np.ndarray:
    """
    Compute eigenvalues of numerical Hessian of order function.
    Uses central differences for second derivatives.
    """
    weights = cppn.get_weights()
    n = len(weights)

    # Base order
    img = cppn.render(32)
    base_order = order_multiplicative(img)

    # Compute Hessian diagonals (most informative for curvature)
    # Full Hessian is O(n^2) evaluations, diagonals are O(n)
    diag_hess = np.zeros(n)

    for i in range(n):
        w_plus = weights.copy()
        w_minus = weights.copy()
        w_plus[i] += eps
        w_minus[i] -= eps

        cppn.set_weights(w_plus)
        f_plus = order_multiplicative(cppn.render(32))

        cppn.set_weights(w_minus)
        f_minus = order_multiplicative(cppn.render(32))

        # Second derivative: (f(x+h) - 2f(x) + f(x-h)) / h^2
        diag_hess[i] = (f_plus - 2*base_order + f_minus) / (eps**2)

    # Restore original weights
    cppn.set_weights(weights)

    return diag_hess


def sample_diverse_cppns(n_samples: int = 100) -> list[tuple[CPPN, float]]:
    """Sample diverse CPPNs and return with their order values."""
    samples = []
    for _ in range(n_samples):
        cppn = CPPN()
        img = cppn.render(32)
        order = order_multiplicative(img)
        samples.append((cppn.copy(), order))
    return samples


def main():
    np.random.seed(42)

    # Sample diverse CPPNs and analyze curvature vs order
    n_samples = 100

    print("Sampling CPPNs and computing Hessian eigenvalues...")
    print("-" * 60)

    samples = sample_diverse_cppns(n_samples)

    results = []
    for i, (cppn, order) in enumerate(samples):
        # Compute Hessian diagonal eigenvalues
        hess_diag = compute_hessian_eigenvalues(cppn, eps=0.02)

        # Curvature metric: mean absolute Hessian diagonal
        mean_curvature = np.mean(np.abs(hess_diag))
        max_curvature = np.max(np.abs(hess_diag))

        results.append({
            'order': order,
            'mean_curvature': mean_curvature,
            'max_curvature': max_curvature,
            'hess_diag': hess_diag
        })

        if (i + 1) % 20 == 0:
            print(f"Processed {i+1}/{n_samples} samples...")

    # Statistical analysis
    print("\n" + "=" * 60)
    print("STATISTICAL ANALYSIS")
    print("=" * 60)

    orders = np.array([r['order'] for r in results])
    curvatures = np.array([r['mean_curvature'] for r in results])

    print(f"\nOrder range: [{orders.min():.3f}, {orders.max():.3f}]")
    print(f"Curvature range: [{curvatures.min():.3f}, {curvatures.max():.3f}]")

    # Pearson correlation
    r, p_corr = stats.pearsonr(orders, curvatures)
    print(f"\nCorrelation (order vs curvature): r = {r:.4f}, p = {p_corr:.2e}")

    # Compare low vs high order groups using terciles
    sorted_orders = np.sort(orders)
    low_threshold = sorted_orders[len(sorted_orders)//3]
    high_threshold = sorted_orders[2*len(sorted_orders)//3]

    low_mask = orders <= low_threshold
    high_mask = orders >= high_threshold

    low_curvature = curvatures[low_mask]
    high_curvature = curvatures[high_mask]

    t_stat, p_ttest = stats.ttest_ind(low_curvature, high_curvature)
    pooled_std = np.sqrt((np.var(low_curvature) + np.var(high_curvature)) / 2)
    cohens_d = (np.mean(high_curvature) - np.mean(low_curvature)) / pooled_std

    print(f"\nLow-order tercile curvature: {np.mean(low_curvature):.4f} +/- {np.std(low_curvature):.4f} (n={len(low_curvature)})")
    print(f"High-order tercile curvature: {np.mean(high_curvature):.4f} +/- {np.std(high_curvature):.4f} (n={len(high_curvature)})")
    print(f"T-test: t = {t_stat:.3f}, p = {p_ttest:.2e}")
    print(f"Effect size (Cohen's d): {cohens_d:.3f}")

    # Spearman rank correlation (robust to outliers)
    rho, p_spearman = stats.spearmanr(orders, curvatures)
    print(f"\nSpearman rank correlation: rho = {rho:.4f}, p = {p_spearman:.2e}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if p_corr < 0.01 and abs(cohens_d) > 0.5:
        if r > 0:
            status = "VALIDATED"
            conclusion = "High-order regions have HIGHER curvature (sharper peaks)"
        else:
            status = "REFUTED (inverted)"
            conclusion = "High-order regions have LOWER curvature (flatter regions)"
    elif p_corr < 0.01:
        status = "WEAK EFFECT"
        conclusion = f"Significant but small effect (d={cohens_d:.2f})"
    else:
        status = "INCONCLUSIVE"
        conclusion = "No significant relationship found"

    print(f"Status: {status}")
    print(f"Conclusion: {conclusion}")
    print(f"\nMetrics:")
    print(f"  correlation: {r:.3f}")
    print(f"  p_value: {p_corr:.2e}")
    print(f"  effect_size: {abs(cohens_d):.3f}")

    return {
        'status': status,
        'correlation': r,
        'p_value': p_corr,
        'effect_size': abs(cohens_d),
        'conclusion': conclusion
    }


if __name__ == "__main__":
    main()
