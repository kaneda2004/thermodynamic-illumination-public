"""
RES-062: Activation Distribution Shape Analysis

Hypothesis: High-order CPPNs exhibit bimodal activation distributions (peaks near -1 and +1)
while low-order CPPNs show unimodal distributions centered at 0.

The reasoning is that high-order images require crisp boundaries (saturated activations at
extreme values after sigmoid), while low-order (noisy) images have activations that stay
near 0 (unsaturated, leading to 50/50 threshold crossings = noise).
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import CPPN, ACTIVATIONS, order_multiplicative, set_global_seed
from scipy import stats
from dataclasses import dataclass
from typing import Tuple, List


@dataclass
class ActivationStats:
    """Statistics about activation distribution."""
    order: float
    mean: float
    std: float
    skewness: float
    kurtosis: float
    bimodality_coefficient: float  # Ashman's D or similar
    peak_count: int
    entropy: float


def compute_bimodality_coefficient(data: np.ndarray) -> float:
    """
    Compute bimodality coefficient using Sarle's formula.
    BC = (skewness^2 + 1) / (kurtosis + 3)
    BC > 5/9 suggests bimodality.
    """
    n = len(data)
    if n < 4:
        return 0.0
    skew = stats.skew(data)
    kurt = stats.kurtosis(data)  # excess kurtosis
    # Sarle's bimodality coefficient
    bc = (skew**2 + 1) / (kurt + 3)
    return bc


def count_peaks_kde(data: np.ndarray, n_points: int = 200) -> int:
    """Count number of peaks in KDE of data."""
    if len(data) < 10:
        return 0
    try:
        kde = stats.gaussian_kde(data)
        x = np.linspace(data.min(), data.max(), n_points)
        y = kde(x)
        # Find local maxima
        peaks = 0
        for i in range(1, len(y) - 1):
            if y[i] > y[i-1] and y[i] > y[i+1]:
                # Only count significant peaks (>10% of max)
                if y[i] > 0.1 * y.max():
                    peaks += 1
        return peaks
    except Exception:
        return 0


def compute_entropy(data: np.ndarray, n_bins: int = 50) -> float:
    """Compute entropy of discretized distribution."""
    hist, _ = np.histogram(data, bins=n_bins, density=True)
    hist = hist + 1e-10  # Avoid log(0)
    hist = hist / hist.sum()  # Normalize
    return -np.sum(hist * np.log2(hist))


def get_pre_threshold_activations(cppn: CPPN, size: int = 32) -> np.ndarray:
    """
    Get the raw activation values BEFORE thresholding at 0.5.
    These are the output of the sigmoid (or other output activation).
    """
    coords = np.linspace(-1, 1, size)
    x, y = np.meshgrid(coords, coords)

    # Get raw activations (the output of activate() before the > 0.5 in render())
    raw = cppn.activate(x, y)
    return raw.flatten()


def sample_cppns_with_order(n_samples: int, seed: int = 42) -> List[Tuple[CPPN, np.ndarray, float]]:
    """
    Generate random CPPNs and compute their order.
    Returns list of (cppn, image, order).
    """
    set_global_seed(seed)
    results = []

    for i in range(n_samples):
        cppn = CPPN()  # Random initialization from prior
        img = cppn.render(size=32)
        order = order_multiplicative(img)
        results.append((cppn, img, order))

    return results


def analyze_activation_distribution(cppn: CPPN, order: float) -> ActivationStats:
    """Compute comprehensive statistics about activation distribution."""
    activations = get_pre_threshold_activations(cppn)

    return ActivationStats(
        order=order,
        mean=float(np.mean(activations)),
        std=float(np.std(activations)),
        skewness=float(stats.skew(activations)),
        kurtosis=float(stats.kurtosis(activations)),
        bimodality_coefficient=compute_bimodality_coefficient(activations),
        peak_count=count_peaks_kde(activations),
        entropy=compute_entropy(activations)
    )


def main():
    print("=" * 70)
    print("RES-062: Activation Distribution Shape Analysis")
    print("=" * 70)
    print()

    # Sample many CPPNs
    n_samples = 2000
    print(f"Sampling {n_samples} random CPPNs...")
    samples = sample_cppns_with_order(n_samples, seed=42)

    # Compute stats for each
    print("Analyzing activation distributions...")
    all_stats = []
    for cppn, img, order in samples:
        stats_obj = analyze_activation_distribution(cppn, order)
        all_stats.append(stats_obj)

    # Split into high vs low order
    orders = np.array([s.order for s in all_stats])
    median_order = np.median(orders)

    # Use more extreme cutoffs for clearer separation
    low_threshold = np.percentile(orders, 25)
    high_threshold = np.percentile(orders, 75)

    low_order = [s for s in all_stats if s.order <= low_threshold]
    high_order = [s for s in all_stats if s.order >= high_threshold]

    print(f"\nOrder distribution: min={orders.min():.4f}, median={median_order:.4f}, max={orders.max():.4f}")
    print(f"Low-order group (<=25th percentile, order<={low_threshold:.4f}): {len(low_order)} samples")
    print(f"High-order group (>=75th percentile, order>={high_threshold:.4f}): {len(high_order)} samples")

    # Compare key metrics
    print("\n" + "=" * 70)
    print("COMPARISON: Low-Order vs High-Order CPPNs")
    print("=" * 70)

    metrics = ['mean', 'std', 'bimodality_coefficient', 'peak_count', 'entropy', 'kurtosis']

    results = {}
    for metric in metrics:
        low_vals = np.array([getattr(s, metric) for s in low_order])
        high_vals = np.array([getattr(s, metric) for s in high_order])

        # Two-sample t-test (Welch's)
        t_stat, p_value = stats.ttest_ind(low_vals, high_vals, equal_var=False)

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((low_vals.std()**2 + high_vals.std()**2) / 2)
        cohens_d = (high_vals.mean() - low_vals.mean()) / (pooled_std + 1e-10)

        results[metric] = {
            'low_mean': float(low_vals.mean()),
            'low_std': float(low_vals.std()),
            'high_mean': float(high_vals.mean()),
            'high_std': float(high_vals.std()),
            't_stat': float(t_stat),
            'p_value': float(p_value),
            'cohens_d': float(cohens_d)
        }

        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
        print(f"\n{metric}:")
        print(f"  Low-order:  {low_vals.mean():.4f} +/- {low_vals.std():.4f}")
        print(f"  High-order: {high_vals.mean():.4f} +/- {high_vals.std():.4f}")
        print(f"  t={t_stat:.3f}, p={p_value:.2e}, d={cohens_d:.3f} {sig}")

    # Primary hypothesis test: bimodality coefficient
    print("\n" + "=" * 70)
    print("PRIMARY HYPOTHESIS TEST: Bimodality Coefficient")
    print("=" * 70)

    bc_result = results['bimodality_coefficient']

    # Bimodality threshold is traditionally 5/9 = 0.555
    # If high-order CPPNs have HIGHER BC, hypothesis is supported
    print(f"\nBimodality coefficient (BC > 0.555 suggests bimodality):")
    print(f"  Low-order mean BC:  {bc_result['low_mean']:.4f}")
    print(f"  High-order mean BC: {bc_result['high_mean']:.4f}")
    print(f"  Effect size (d):    {bc_result['cohens_d']:.4f}")
    print(f"  p-value:            {bc_result['p_value']:.2e}")

    # Secondary test: distribution spread (std)
    print("\n" + "=" * 70)
    print("SECONDARY TEST: Activation Spread (std)")
    print("=" * 70)
    std_result = results['std']
    print(f"\nActivation standard deviation:")
    print(f"  Low-order mean std:  {std_result['low_mean']:.4f}")
    print(f"  High-order mean std: {std_result['high_mean']:.4f}")
    print(f"  Effect size (d):     {std_result['cohens_d']:.4f}")
    print(f"  p-value:             {std_result['p_value']:.2e}")

    # Correlation analysis
    print("\n" + "=" * 70)
    print("CORRELATION ANALYSIS: Order vs Bimodality")
    print("=" * 70)

    all_orders = np.array([s.order for s in all_stats])
    all_bc = np.array([s.bimodality_coefficient for s in all_stats])
    all_std = np.array([s.std for s in all_stats])

    r_bc, p_bc = stats.pearsonr(all_orders, all_bc)
    r_std, p_std = stats.pearsonr(all_orders, all_std)

    print(f"Order vs Bimodality Coefficient: r={r_bc:.4f}, p={p_bc:.2e}")
    print(f"Order vs Activation Std:         r={r_std:.4f}, p={p_std:.2e}")

    # Final verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    # Hypothesis: high-order CPPNs have HIGHER bimodality (more extreme activations)
    bc_diff = bc_result['high_mean'] - bc_result['low_mean']
    bc_significant = bc_result['p_value'] < 0.01 and abs(bc_result['cohens_d']) > 0.5
    std_diff = std_result['high_mean'] - std_result['low_mean']
    std_significant = std_result['p_value'] < 0.01 and abs(std_result['cohens_d']) > 0.5

    # Check direction: hypothesis says high-order -> MORE bimodal
    bc_direction_correct = bc_diff > 0

    if bc_significant and bc_direction_correct:
        verdict = "VALIDATED"
        summary = f"High-order CPPNs have significantly higher bimodality (d={bc_result['cohens_d']:.2f})"
    elif bc_significant and not bc_direction_correct:
        verdict = "REFUTED"
        summary = f"Effect is opposite: low-order CPPNs are MORE bimodal (d={bc_result['cohens_d']:.2f})"
    else:
        # Check if std tells a different story
        if std_significant:
            if std_diff > 0:
                verdict = "PARTIAL"
                summary = f"No bimodality difference, but high-order has wider spread (d={std_result['cohens_d']:.2f})"
            else:
                verdict = "REFUTED"
                summary = f"Low-order CPPNs have wider activation spread (d={std_result['cohens_d']:.2f})"
        else:
            verdict = "INCONCLUSIVE"
            summary = f"No significant difference in activation distributions (BC d={bc_result['cohens_d']:.2f})"

    print(f"\nStatus: {verdict}")
    print(f"Summary: {summary}")

    # Output structured results
    print("\n" + "=" * 70)
    print("STRUCTURED OUTPUT")
    print("=" * 70)

    output = {
        'hypothesis': 'High-order CPPNs exhibit bimodal activation distributions while low-order show unimodal',
        'verdict': verdict,
        'n_samples': n_samples,
        'low_order_n': len(low_order),
        'high_order_n': len(high_order),
        'bimodality': {
            'low_mean': bc_result['low_mean'],
            'high_mean': bc_result['high_mean'],
            'effect_size': bc_result['cohens_d'],
            'p_value': bc_result['p_value']
        },
        'std': {
            'low_mean': std_result['low_mean'],
            'high_mean': std_result['high_mean'],
            'effect_size': std_result['cohens_d'],
            'p_value': std_result['p_value']
        },
        'correlation': {
            'order_vs_bimodality_r': r_bc,
            'order_vs_bimodality_p': p_bc,
            'order_vs_std_r': r_std,
            'order_vs_std_p': p_std
        }
    }

    import json
    print(json.dumps(output, indent=2))

    return output


if __name__ == '__main__':
    main()
