#!/usr/bin/env python3
"""
RES-051: Latent Interpolation Order Trajectory

Hypothesis: Order along CPPN latent interpolation paths follows concave trajectory
(order dips in middle of interpolation between two high-order endpoints).

Method:
1. Sample pairs of CPPN weight vectors that produce high-order images
2. Linearly interpolate between them at 21 points (t=0, 0.05, ..., 1)
3. Measure order at each interpolation point
4. Fit quadratic to determine convexity (concave if a2 < 0)
5. Statistical test across many pairs
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import CPPN, order_multiplicative, set_global_seed
from scipy import stats

def sample_high_order_cppn(threshold: float = 0.1, max_attempts: int = 1000) -> CPPN:
    """Sample a CPPN that produces order above threshold."""
    for _ in range(max_attempts):
        cppn = CPPN()
        img = cppn.render(32)
        if order_multiplicative(img) >= threshold:
            return cppn
    raise ValueError(f"Could not find CPPN with order >= {threshold} in {max_attempts} attempts")


def interpolate_weights(w1: np.ndarray, w2: np.ndarray, t: float) -> np.ndarray:
    """Linear interpolation: w1 * (1-t) + w2 * t"""
    return w1 * (1 - t) + w2 * t


def measure_interpolation_path(cppn1: CPPN, cppn2: CPPN, n_points: int = 21) -> tuple:
    """
    Measure order along interpolation path between two CPPNs.

    Returns:
        t_values: array of interpolation parameters
        orders: array of order values at each point
    """
    w1 = cppn1.get_weights()
    w2 = cppn2.get_weights()

    t_values = np.linspace(0, 1, n_points)
    orders = []

    # Create a template CPPN for rendering
    cppn_interp = cppn1.copy()

    for t in t_values:
        w_interp = interpolate_weights(w1, w2, t)
        cppn_interp.set_weights(w_interp)
        img = cppn_interp.render(32)
        orders.append(order_multiplicative(img))

    return t_values, np.array(orders)


def fit_quadratic(t_values: np.ndarray, orders: np.ndarray) -> tuple:
    """
    Fit quadratic: order(t) = a0 + a1*t + a2*t^2

    Returns:
        coeffs: (a0, a1, a2)
        curvature: a2 (negative = concave, positive = convex)
    """
    # Fit polynomial
    coeffs = np.polyfit(t_values, orders, 2)  # Returns [a2, a1, a0]
    a2, a1, a0 = coeffs
    return (a0, a1, a2), a2


def compute_path_statistics(t_values: np.ndarray, orders: np.ndarray) -> dict:
    """Compute statistics about the interpolation path."""
    # Endpoints
    start_order = orders[0]
    end_order = orders[-1]
    midpoint_order = orders[len(orders) // 2]

    # Expected midpoint if linear
    expected_mid = (start_order + end_order) / 2

    # Is actual midpoint below expected? (concave signature)
    mid_deviation = midpoint_order - expected_mid

    # Minimum order along path
    min_order = np.min(orders)
    min_t = t_values[np.argmin(orders)]

    # Quadratic fit
    _, curvature = fit_quadratic(t_values, orders)

    return {
        'start_order': start_order,
        'end_order': end_order,
        'midpoint_order': midpoint_order,
        'expected_midpoint': expected_mid,
        'mid_deviation': mid_deviation,
        'min_order': min_order,
        'min_t': min_t,
        'curvature': curvature,
        'is_concave': curvature < 0
    }


def run_experiment(n_pairs: int = 50, order_threshold: float = 0.1, seed: int = 42):
    """
    Run latent interpolation experiment.

    Args:
        n_pairs: Number of CPPN pairs to test
        order_threshold: Minimum order for endpoint CPPNs
        seed: Random seed
    """
    set_global_seed(seed)

    print(f"RES-051: Latent Interpolation Order Trajectory")
    print(f"=" * 60)
    print(f"Testing {n_pairs} CPPN pairs with order threshold >= {order_threshold}")
    print()

    curvatures = []
    mid_deviations = []
    is_concave_list = []
    all_stats = []

    for i in range(n_pairs):
        try:
            # Sample two high-order CPPNs
            cppn1 = sample_high_order_cppn(order_threshold)
            cppn2 = sample_high_order_cppn(order_threshold)

            # Measure interpolation path
            t_values, orders = measure_interpolation_path(cppn1, cppn2)

            # Compute statistics
            path_stats = compute_path_statistics(t_values, orders)
            all_stats.append(path_stats)

            curvatures.append(path_stats['curvature'])
            mid_deviations.append(path_stats['mid_deviation'])
            is_concave_list.append(path_stats['is_concave'])

            if (i + 1) % 10 == 0:
                print(f"  Completed {i + 1}/{n_pairs} pairs")

        except ValueError as e:
            print(f"  Warning: {e}")
            continue

    curvatures = np.array(curvatures)
    mid_deviations = np.array(mid_deviations)
    is_concave = np.array(is_concave_list)

    # Statistical tests
    # H0: mean curvature = 0 (linear paths)
    # H1: mean curvature < 0 (concave paths)
    t_stat, p_value_two = stats.ttest_1samp(curvatures, 0)
    p_value = p_value_two / 2 if t_stat < 0 else 1 - p_value_two / 2  # One-tailed

    # Effect size (Cohen's d)
    cohens_d = np.mean(curvatures) / np.std(curvatures)

    # Proportion concave with binomial test
    n_concave = np.sum(is_concave)
    n_total = len(is_concave)
    prop_concave = n_concave / n_total
    binom_result = stats.binomtest(n_concave, n_total, 0.5, alternative='greater')
    binom_pval = binom_result.pvalue

    print()
    print("RESULTS")
    print("-" * 60)
    print(f"Curvature Statistics:")
    print(f"  Mean curvature: {np.mean(curvatures):.4f}")
    print(f"  Std curvature:  {np.std(curvatures):.4f}")
    print(f"  Median:         {np.median(curvatures):.4f}")
    print()
    print(f"Concavity Test (H0: linear, H1: concave):")
    print(f"  t-statistic:    {t_stat:.4f}")
    print(f"  p-value:        {p_value:.6f}")
    print(f"  Cohen's d:      {cohens_d:.4f}")
    print()
    print(f"Proportion Concave:")
    print(f"  Concave paths:  {n_concave}/{n_total} ({prop_concave:.1%})")
    print(f"  Binomial p:     {binom_pval:.6f}")
    print()
    print(f"Midpoint Deviation:")
    print(f"  Mean deviation: {np.mean(mid_deviations):.4f}")
    print(f"  (negative = order dips below linear expectation)")
    print()

    # Determine status
    if p_value < 0.01 and abs(cohens_d) > 0.5 and prop_concave > 0.6:
        status = "VALIDATED"
        if cohens_d < 0:
            conclusion = "Order trajectories are CONCAVE (dip in middle)"
        else:
            conclusion = "Order trajectories are CONVEX (peak in middle)"
    elif p_value < 0.05:
        status = "INCONCLUSIVE"
        conclusion = "Weak evidence for non-linear trajectories"
    else:
        status = "REFUTED"
        conclusion = "No evidence paths differ from linear"

    print(f"STATUS: {status}")
    print(f"CONCLUSION: {conclusion}")

    return {
        'status': status,
        'hypothesis': 'Order along CPPN latent interpolation paths follows concave trajectory',
        'n_pairs': n_total,
        'mean_curvature': float(np.mean(curvatures)),
        'std_curvature': float(np.std(curvatures)),
        'cohens_d': float(cohens_d),
        'p_value': float(p_value),
        'prop_concave': float(prop_concave),
        'binom_pval': float(binom_pval),
        'mean_mid_deviation': float(np.mean(mid_deviations)),
        'conclusion': conclusion
    }


if __name__ == '__main__':
    results = run_experiment(n_pairs=50, order_threshold=0.1, seed=42)

    print()
    print("=" * 60)
    print("SUMMARY FOR RESEARCH LOG")
    print("=" * 60)
    print(f"effect_size: {abs(results['cohens_d']):.2f}")
    print(f"p_value: {results['p_value']:.6f}")
    print(f"prop_concave: {results['prop_concave']:.2f}")
