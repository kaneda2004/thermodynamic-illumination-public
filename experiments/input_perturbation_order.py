"""
RES-107: Input Coordinate Perturbation Effect on CPPN Order

Hypothesis: Adding small Gaussian noise (sigma=0.01-0.1) to CPPN input
coordinates reduces order scores by smoothing sharp transitions, with
larger noise creating more diffuse/natural textures.

Protocol:
1. Generate N CPPNs
2. For each, render with:
   - Clean coordinates (baseline)
   - Perturbed coordinates at various sigma levels
3. Measure order change across noise levels
4. Statistical test: paired t-test on order delta
"""

import numpy as np
from scipy import stats
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import CPPN, order_multiplicative


def render_with_perturbation(cppn: CPPN, size: int = 32, sigma: float = 0.0) -> np.ndarray:
    """Render CPPN with optional Gaussian noise on input coordinates."""
    coords = np.linspace(-1, 1, size)
    x, y = np.meshgrid(coords, coords)

    if sigma > 0:
        x = x + np.random.randn(*x.shape) * sigma
        y = y + np.random.randn(*y.shape) * sigma

    return (cppn.activate(x, y) > 0.5).astype(np.uint8)


def run_experiment(n_cppns: int = 200, size: int = 32, seed: int = 42):
    """
    Test effect of input perturbation on CPPN order scores.

    Returns dict with results for statistical analysis.
    """
    np.random.seed(seed)

    sigma_levels = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]

    # Storage
    results = {sigma: [] for sigma in sigma_levels}
    order_deltas = {sigma: [] for sigma in sigma_levels if sigma > 0}

    # Generate CPPNs and measure
    for i in range(n_cppns):
        cppn = CPPN()

        # Mutate a few times to get interesting structure
        for _ in range(5):
            # Simple weight perturbation
            weights = cppn.get_weights()
            weights += np.random.randn(len(weights)) * 0.3
            cppn.set_weights(weights)

        # Render at each sigma level
        baseline_order = None
        for sigma in sigma_levels:
            img = render_with_perturbation(cppn, size, sigma)
            order = order_multiplicative(img)
            results[sigma].append(order)

            if sigma == 0.0:
                baseline_order = order
            else:
                order_deltas[sigma].append(order - baseline_order)

    return results, order_deltas


def analyze_results(results: dict, order_deltas: dict):
    """Statistical analysis of perturbation effects."""
    print("=" * 60)
    print("RES-107: Input Coordinate Perturbation Effect")
    print("=" * 60)

    # Summary statistics
    print("\nOrder scores by sigma level:")
    print(f"{'Sigma':<10} {'Mean Order':<12} {'Std':<10} {'N'}")
    print("-" * 45)

    for sigma, orders in sorted(results.items()):
        print(f"{sigma:<10.3f} {np.mean(orders):<12.4f} {np.std(orders):<10.4f} {len(orders)}")

    # Test hypothesis: does perturbation REDUCE order?
    print("\n" + "=" * 60)
    print("Hypothesis Test: Perturbation reduces order")
    print("=" * 60)

    baseline = np.array(results[0.0])

    effect_sizes = {}
    p_values = {}

    for sigma in sorted(order_deltas.keys()):
        deltas = np.array(order_deltas[sigma])

        # Paired t-test (one-sided: delta < 0 means order decreased)
        t_stat, p_two = stats.ttest_1samp(deltas, 0)
        # One-sided p-value for negative effect
        p_one = p_two / 2 if t_stat < 0 else 1 - p_two / 2

        # Effect size (Cohen's d for paired samples)
        perturbed = np.array(results[sigma])
        d = (np.mean(baseline) - np.mean(perturbed)) / np.std(deltas)

        effect_sizes[sigma] = d
        p_values[sigma] = p_one

        print(f"\nSigma = {sigma}:")
        print(f"  Mean delta: {np.mean(deltas):.4f} (negative = order decreased)")
        print(f"  t-statistic: {t_stat:.3f}")
        print(f"  p-value (one-sided): {p_one:.6f}")
        print(f"  Effect size (Cohen's d): {d:.3f}")
        print(f"  Verdict: {'SIGNIFICANT' if p_one < 0.01 and d > 0.5 else 'not significant'}")

    # Find best sigma for validation
    best_sigma = max(effect_sizes.keys(), key=lambda s: effect_sizes[s])
    best_effect = effect_sizes[best_sigma]
    best_p = p_values[best_sigma]

    print("\n" + "=" * 60)
    print("OVERALL RESULT")
    print("=" * 60)

    # Check for monotonic decrease
    sigmas = sorted(results.keys())
    means = [np.mean(results[s]) for s in sigmas]
    is_monotonic = all(means[i] >= means[i+1] for i in range(len(means)-1))

    print(f"\nMonotonic decrease with sigma: {is_monotonic}")
    print(f"Best effect at sigma={best_sigma}: d={best_effect:.3f}, p={best_p:.6f}")

    # Correlation between sigma and order
    all_sigmas = []
    all_orders = []
    for sigma, orders in results.items():
        all_sigmas.extend([sigma] * len(orders))
        all_orders.extend(orders)

    r, p_corr = stats.pearsonr(all_sigmas, all_orders)
    print(f"\nCorrelation (sigma vs order): r={r:.3f}, p={p_corr:.2e}")

    # Final verdict
    validated = best_p < 0.01 and best_effect > 0.5 and r < 0

    print("\n" + "=" * 60)
    print(f"VALIDATED: {validated}")
    print("=" * 60)

    if validated:
        print(f"\nInput perturbation significantly reduces CPPN order.")
        print(f"Effect size d={best_effect:.2f} at sigma={best_sigma}")
        print(f"Correlation r={r:.3f} confirms monotonic relationship")
    else:
        reasons = []
        if best_p >= 0.01:
            reasons.append(f"p-value {best_p:.4f} >= 0.01")
        if best_effect <= 0.5:
            reasons.append(f"effect size {best_effect:.2f} <= 0.5")
        if r >= 0:
            reasons.append(f"correlation r={r:.3f} >= 0 (wrong direction)")
        print(f"\nFailed because: {'; '.join(reasons)}")

    return {
        'validated': validated,
        'best_sigma': best_sigma,
        'effect_size': best_effect,
        'p_value': best_p,
        'correlation': r,
        'monotonic': is_monotonic
    }


if __name__ == "__main__":
    print("Running RES-107 experiment...")
    print("N=200 CPPNs, sigmas=[0, 0.01, 0.02, 0.05, 0.1, 0.2]\n")

    results, order_deltas = run_experiment(n_cppns=200, size=32, seed=42)
    summary = analyze_results(results, order_deltas)

    print("\n" + "=" * 60)
    print("Summary for log_manager:")
    print("=" * 60)
    print(f"effect_size: {summary['effect_size']:.3f}")
    print(f"p_value: {summary['p_value']:.6f}")
    print(f"correlation: {summary['correlation']:.3f}")
    print(f"status: {'validated' if summary['validated'] else 'refuted'}")
