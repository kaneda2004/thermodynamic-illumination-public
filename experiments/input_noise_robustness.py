"""
RES-101: Input Noise Robustness

Hypothesis: High-order CPPN images exhibit greater robustness to input coordinate
noise than low-order ones - small perturbations to (x,y) coordinates cause smaller
changes in order score for structured images.

Methodology:
1. Generate CPPNs and measure their baseline order scores
2. Add Gaussian noise to input coordinates (x, y) at multiple noise levels
3. Measure the order change (delta_order = |order_noisy - order_baseline|)
4. Correlate baseline order with noise robustness (lower delta = more robust)

If validated: High-order structures are "stable attractors" that tolerate perturbations.
If refuted: Order is equally fragile regardless of baseline structure.
"""

import numpy as np
from scipy import stats
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import CPPN, order_multiplicative, set_global_seed

def add_coordinate_noise(x: np.ndarray, y: np.ndarray, noise_level: float) -> tuple:
    """Add Gaussian noise to coordinate grids."""
    x_noisy = x + np.random.randn(*x.shape) * noise_level
    y_noisy = y + np.random.randn(*y.shape) * noise_level
    return x_noisy, y_noisy


def render_with_noise(cppn: CPPN, size: int, noise_level: float) -> np.ndarray:
    """Render CPPN with noisy input coordinates."""
    coords = np.linspace(-1, 1, size)
    x, y = np.meshgrid(coords, coords)
    if noise_level > 0:
        x, y = add_coordinate_noise(x, y, noise_level)
    return (cppn.activate(x, y) > 0.5).astype(np.uint8)


def measure_noise_robustness(cppn: CPPN, size: int = 32, noise_levels: list = None,
                               n_trials: int = 10) -> dict:
    """
    Measure how robust a CPPN's order score is to input coordinate noise.

    Returns:
        dict with baseline_order, and mean/std of order changes at each noise level
    """
    if noise_levels is None:
        noise_levels = [0.01, 0.02, 0.05, 0.1, 0.2]

    # Baseline (no noise)
    baseline_img = cppn.render(size)
    baseline_order = order_multiplicative(baseline_img)

    results = {
        'baseline_order': baseline_order,
        'noise_levels': noise_levels,
        'mean_delta': [],
        'std_delta': [],
        'mean_order': [],
    }

    for nl in noise_levels:
        deltas = []
        orders = []
        for _ in range(n_trials):
            noisy_img = render_with_noise(cppn, size, nl)
            noisy_order = order_multiplicative(noisy_img)
            deltas.append(abs(noisy_order - baseline_order))
            orders.append(noisy_order)
        results['mean_delta'].append(np.mean(deltas))
        results['std_delta'].append(np.std(deltas))
        results['mean_order'].append(np.mean(orders))

    return results


def run_experiment(n_samples: int = 200, seed: int = 42):
    """
    Main experiment: correlate baseline order with noise robustness.

    Hypothesis: High-order CPPNs have smaller delta_order under noise.
    """
    set_global_seed(seed)
    print("="*60)
    print("RES-101: Input Noise Robustness Experiment")
    print("="*60)

    noise_levels = [0.01, 0.02, 0.05, 0.1]

    # Generate diverse CPPNs
    baseline_orders = []
    robustness_scores = []  # Average delta across noise levels (lower = more robust)
    all_results = []

    print(f"\nGenerating {n_samples} CPPNs and measuring noise robustness...")

    for i in range(n_samples):
        cppn = CPPN()  # Random initialization
        # Optionally add hidden nodes for diversity
        if np.random.rand() < 0.3:
            from core.thermo_sampler_v3 import Node, Connection, PRIOR_SIGMA
            hidden_id = 5 + i % 3
            activation = np.random.choice(['sin', 'tanh', 'gauss'])
            cppn.nodes.append(Node(hidden_id, activation, np.random.randn() * PRIOR_SIGMA))
            # Connect from inputs to hidden
            for inp in [0, 1, 2]:
                if np.random.rand() < 0.5:
                    cppn.connections.append(Connection(inp, hidden_id, np.random.randn() * PRIOR_SIGMA))
            # Connect hidden to output
            cppn.connections.append(Connection(hidden_id, 4, np.random.randn() * PRIOR_SIGMA))

        results = measure_noise_robustness(cppn, noise_levels=noise_levels, n_trials=5)

        baseline_orders.append(results['baseline_order'])
        # Robustness = average delta (lower is more robust)
        avg_delta = np.mean(results['mean_delta'])
        robustness_scores.append(avg_delta)
        all_results.append(results)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{n_samples} CPPNs")

    baseline_orders = np.array(baseline_orders)
    robustness_scores = np.array(robustness_scores)

    # Analysis 1: Overall correlation between baseline order and delta (robustness)
    # If hypothesis holds: negative correlation (higher order -> smaller delta)
    r, p = stats.pearsonr(baseline_orders, robustness_scores)

    print("\n" + "="*60)
    print("RESULTS: Correlation between baseline order and noise sensitivity")
    print("="*60)
    print(f"\nPearson correlation (order vs delta): r = {r:.4f}, p = {p:.2e}")
    print("  (Negative r means high-order is MORE robust)")
    print("  (Positive r means high-order is LESS robust)")

    # Analysis 2: Compare high-order vs low-order groups
    median_order = np.median(baseline_orders)
    high_order_mask = baseline_orders > median_order
    low_order_mask = baseline_orders <= median_order

    high_order_deltas = robustness_scores[high_order_mask]
    low_order_deltas = robustness_scores[low_order_mask]

    t_stat, t_pval = stats.ttest_ind(high_order_deltas, low_order_deltas)
    effect_size = (np.mean(low_order_deltas) - np.mean(high_order_deltas)) / np.std(robustness_scores)

    print("\n" + "-"*60)
    print("Group comparison: High-order vs Low-order CPPNs")
    print("-"*60)
    print(f"Median order threshold: {median_order:.4f}")
    print(f"High-order group (n={sum(high_order_mask)}): mean delta = {np.mean(high_order_deltas):.4f}")
    print(f"Low-order group (n={sum(low_order_mask)}): mean delta = {np.mean(low_order_deltas):.4f}")
    print(f"T-test: t = {t_stat:.4f}, p = {t_pval:.2e}")
    print(f"Effect size (Cohen's d): {effect_size:.4f}")
    print("  (Positive d means high-order is MORE robust)")

    # Analysis 3: Per-noise-level breakdown
    print("\n" + "-"*60)
    print("Per-noise-level analysis:")
    print("-"*60)
    for i, nl in enumerate(noise_levels):
        deltas_at_nl = np.array([r['mean_delta'][i] for r in all_results])
        r_nl, p_nl = stats.pearsonr(baseline_orders, deltas_at_nl)
        print(f"  Noise {nl:.2f}: r = {r_nl:.4f}, p = {p_nl:.2e}")

    # Analysis 4: Distribution stats
    print("\n" + "-"*60)
    print("Distribution statistics:")
    print("-"*60)
    print(f"Baseline order: mean={np.mean(baseline_orders):.4f}, std={np.std(baseline_orders):.4f}")
    print(f"  Range: [{np.min(baseline_orders):.4f}, {np.max(baseline_orders):.4f}]")
    print(f"Delta scores: mean={np.mean(robustness_scores):.4f}, std={np.std(robustness_scores):.4f}")

    # Verdict
    print("\n" + "="*60)
    print("VERDICT")
    print("="*60)

    # Criteria for validation:
    # 1. Significant negative correlation (p < 0.01)
    # 2. Effect size |d| > 0.5

    validated = (r < 0 and p < 0.01 and effect_size > 0.5)
    refuted = (r > 0 and p < 0.01)

    if validated:
        status = "VALIDATED"
        conclusion = "High-order CPPNs ARE more robust to input coordinate noise."
    elif refuted:
        status = "REFUTED"
        conclusion = "High-order CPPNs are actually LESS robust to noise."
    else:
        status = "INCONCLUSIVE"
        if abs(r) < 0.1:
            conclusion = "No meaningful relationship between order and noise robustness."
        else:
            conclusion = f"Trend observed (r={r:.3f}) but insufficient significance or effect size."

    print(f"\nStatus: {status}")
    print(f"Conclusion: {conclusion}")

    return {
        'status': status,
        'correlation': float(r),
        'p_value': float(p),
        'effect_size': float(effect_size),
        't_pval': float(t_pval),
        'n_samples': n_samples,
        'high_order_mean_delta': float(np.mean(high_order_deltas)),
        'low_order_mean_delta': float(np.mean(low_order_deltas)),
    }


if __name__ == "__main__":
    results = run_experiment(n_samples=200, seed=42)
    print("\n" + "="*60)
    print("Summary for log:")
    print("="*60)
    print(f"r={results['correlation']:.3f}, p={results['p_value']:.2e}, d={results['effect_size']:.2f}")
