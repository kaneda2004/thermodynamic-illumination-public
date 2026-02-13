#!/usr/bin/env python3
"""
RES-084: Test if gradient magnitude ||∇O|| correlates with current order level.

Hypothesis: High order regions have smaller gradients (near optimum),
low order regions have larger gradients (steep descent possible).

Related: RES-049 showed Fisher info 157x higher at high order.
"""

import numpy as np
from scipy import stats
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import CPPN, order_multiplicative

def compute_gradient_magnitude(cppn, epsilon=1e-4):
    """Compute ||∇O|| via finite differences in weight space."""
    weights = cppn.get_weights()
    base_img = cppn.render(32)
    base_order = order_multiplicative(base_img)

    gradients = []
    for i in range(len(weights)):
        weights_plus = weights.copy()
        weights_plus[i] += epsilon
        cppn.set_weights(weights_plus)
        img_plus = cppn.render(32)
        order_plus = order_multiplicative(img_plus)
        grad_i = (order_plus - base_order) / epsilon
        gradients.append(grad_i)

    # Restore original weights
    cppn.set_weights(weights)
    return np.linalg.norm(gradients), base_order

def main():
    np.random.seed(42)
    n_samples = 200

    orders = []
    grad_mags = []

    for i in range(n_samples):
        # Create fresh CPPN with random weights from prior
        cppn = CPPN()
        # Randomize weights more aggressively to sample diverse order levels
        weights = cppn.get_weights()
        weights = np.random.randn(len(weights)) * 2.0
        cppn.set_weights(weights)

        grad_mag, order = compute_gradient_magnitude(cppn)
        orders.append(order)
        grad_mags.append(grad_mag)
        if (i + 1) % 50 == 0:
            print(f"Sample {i+1}/{n_samples}")

    orders = np.array(orders)
    grad_mags = np.array(grad_mags)

    # Correlation analysis
    r, p = stats.pearsonr(orders, grad_mags)
    rho, p_spearman = stats.spearmanr(orders, grad_mags)

    # Bin by order level
    low_order = grad_mags[orders < np.percentile(orders, 33)]
    mid_order = grad_mags[(orders >= np.percentile(orders, 33)) & (orders < np.percentile(orders, 66))]
    high_order = grad_mags[orders >= np.percentile(orders, 66)]

    print("\n=== RES-084: Gradient Magnitude vs Order ===")
    print(f"Samples: {n_samples}")
    print(f"\nCorrelation:")
    print(f"  Pearson r = {r:.4f}, p = {p:.2e}")
    print(f"  Spearman ρ = {rho:.4f}, p = {p_spearman:.2e}")

    print(f"\nGradient magnitude by order tercile:")
    print(f"  Low order:  mean={np.mean(low_order):.4f}, std={np.std(low_order):.4f}")
    print(f"  Mid order:  mean={np.mean(mid_order):.4f}, std={np.std(mid_order):.4f}")
    print(f"  High order: mean={np.mean(high_order):.4f}, std={np.std(high_order):.4f}")

    # Effect size (Cohen's d between low and high)
    pooled_std = np.sqrt((np.var(low_order) + np.var(high_order)) / 2)
    if pooled_std > 0:
        cohens_d = (np.mean(low_order) - np.mean(high_order)) / pooled_std
    else:
        cohens_d = 0.0

    print(f"\nEffect size (low vs high): Cohen's d = {cohens_d:.3f}")

    # Validation criteria
    significant = p < 0.01
    large_effect = abs(cohens_d) > 0.5

    print(f"\n=== VALIDATION ===")
    print(f"p < 0.01: {significant} (p={p:.2e})")
    print(f"|d| > 0.5: {large_effect} (d={cohens_d:.3f})")

    if significant and large_effect:
        if r < 0:
            print("STATUS: VALIDATED - Negative correlation (high order = small gradients)")
        else:
            print("STATUS: VALIDATED - Positive correlation (high order = large gradients)")
    elif significant:
        print("STATUS: INCONCLUSIVE - Significant but small effect")
    else:
        print("STATUS: REFUTED - No significant correlation")

    return {
        'r': r, 'p': p, 'rho': rho, 'cohens_d': cohens_d,
        'low_mean': np.mean(low_order), 'high_mean': np.mean(high_order)
    }

if __name__ == "__main__":
    main()
