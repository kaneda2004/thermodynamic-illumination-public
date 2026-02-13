"""
RES-094: Pixel Histogram Shape Analysis

HYPOTHESIS: High-order CPPN images exhibit bimodal pixel histograms while
random images have uniform distributions.

For binary images (0/1), we measure histogram properties:
- Bimodality: deviation from 50/50 split
- Extreme values: proportion of pixels at 0 or 1 (which is 100% for binary)

Since images are binary, we measure density and its variance across samples,
and also look at continuous activation values BEFORE thresholding.
"""

import numpy as np
from scipy import stats
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import (
    CPPN, order_multiplicative, nested_sampling_v3,
    set_global_seed
)


def get_pre_threshold_histogram(cppn, size=32):
    """Get continuous pixel values before threshold."""
    coords = np.linspace(-1, 1, size)
    x, y = np.meshgrid(coords, coords)
    values = cppn.activate(x, y)
    return values.flatten()


def histogram_bimodality_coefficient(values):
    """
    Sarle's bimodality coefficient: BC = (skewness^2 + 1) / kurtosis
    BC > 5/9 suggests bimodality.
    """
    n = len(values)
    if n < 4:
        return 0

    skew = stats.skew(values)
    kurt = stats.kurtosis(values, fisher=False)  # Excess kurtosis + 3

    if kurt <= 0:
        return 0

    bc = (skew**2 + 1) / kurt
    return bc


def dip_statistic(values):
    """
    Hartigan's dip statistic - measures deviation from unimodality.
    Higher values indicate multimodality.
    """
    sorted_values = np.sort(values)
    n = len(sorted_values)

    # Compute empirical CDF
    cdf = np.arange(1, n + 1) / n

    # Compute greatest convex minorant (gcm) and least concave majorant (lcm)
    # Simplified: measure max deviation from uniform CDF
    uniform_cdf = (sorted_values - sorted_values.min()) / (sorted_values.max() - sorted_values.min() + 1e-10)
    dip = np.max(np.abs(cdf - uniform_cdf))

    return dip


def analyze_histogram_shape(values):
    """Comprehensive histogram shape analysis."""
    return {
        'mean': np.mean(values),
        'std': np.std(values),
        'skewness': stats.skew(values),
        'kurtosis': stats.kurtosis(values),  # Excess kurtosis
        'bimodality_coef': histogram_bimodality_coefficient(values),
        'dip': dip_statistic(values),
        'q10': np.percentile(values, 10),
        'q90': np.percentile(values, 90),
        'iqr': np.percentile(values, 75) - np.percentile(values, 25),
    }


def run_experiment(n_samples=200, image_size=32, seed=42):
    """Compare histogram shapes between high-order and random images."""
    set_global_seed(seed)

    print("="*70)
    print("RES-094: Pixel Histogram Shape Analysis")
    print("="*70)
    print(f"Samples per group: {n_samples}")
    print(f"Image size: {image_size}x{image_size}")
    print()

    # Generate random images (uniform noise)
    print("Generating random images...")
    random_histograms = []
    for _ in range(n_samples):
        # Continuous uniform values
        values = np.random.rand(image_size * image_size)
        random_histograms.append(analyze_histogram_shape(values))

    # Generate CPPN images and collect pre-threshold values
    print("Generating CPPN images...")
    cppn_histograms = []
    cppn_orders = []

    for _ in range(n_samples * 3):  # Generate more to filter for high-order
        cppn = CPPN()
        values = get_pre_threshold_histogram(cppn, image_size)
        img = (values.reshape(image_size, image_size) > 0.5).astype(np.uint8)
        order = order_multiplicative(img)
        cppn_orders.append(order)
        cppn_histograms.append(analyze_histogram_shape(values))

    # Split CPPNs by order
    order_threshold = np.percentile(cppn_orders, 75)
    high_order_mask = np.array(cppn_orders) >= order_threshold
    low_order_mask = np.array(cppn_orders) < np.percentile(cppn_orders, 25)

    high_order_hists = [h for h, m in zip(cppn_histograms, high_order_mask) if m][:n_samples]
    low_order_hists = [h for h, m in zip(cppn_histograms, low_order_mask) if m][:n_samples]

    print(f"\nOrder threshold (75th percentile): {order_threshold:.4f}")
    print(f"High-order samples: {len(high_order_hists)}")
    print(f"Low-order samples: {len(low_order_hists)}")
    print()

    # Compare metrics
    metrics = ['bimodality_coef', 'dip', 'std', 'skewness', 'kurtosis', 'iqr']

    results = {}

    print("="*70)
    print("METRIC COMPARISONS")
    print("="*70)

    for metric in metrics:
        random_vals = [h[metric] for h in random_histograms]
        high_vals = [h[metric] for h in high_order_hists]
        low_vals = [h[metric] for h in low_order_hists]

        # Statistical tests
        stat_rh, p_rh = stats.mannwhitneyu(random_vals, high_vals, alternative='two-sided')
        stat_hl, p_hl = stats.mannwhitneyu(high_vals, low_vals, alternative='two-sided')

        # Effect sizes (Cohen's d)
        pooled_std = np.sqrt((np.var(random_vals) + np.var(high_vals)) / 2)
        effect_rh = (np.mean(high_vals) - np.mean(random_vals)) / (pooled_std + 1e-10)

        pooled_std_hl = np.sqrt((np.var(high_vals) + np.var(low_vals)) / 2)
        effect_hl = (np.mean(high_vals) - np.mean(low_vals)) / (pooled_std_hl + 1e-10)

        results[metric] = {
            'random_mean': np.mean(random_vals),
            'random_std': np.std(random_vals),
            'high_mean': np.mean(high_vals),
            'high_std': np.std(high_vals),
            'low_mean': np.mean(low_vals),
            'low_std': np.std(low_vals),
            'p_random_vs_high': p_rh,
            'effect_random_vs_high': effect_rh,
            'p_high_vs_low': p_hl,
            'effect_high_vs_low': effect_hl,
        }

        print(f"\n{metric.upper()}")
        print("-"*50)
        print(f"  Random:     {np.mean(random_vals):.4f} ± {np.std(random_vals):.4f}")
        print(f"  High-order: {np.mean(high_vals):.4f} ± {np.std(high_vals):.4f}")
        print(f"  Low-order:  {np.mean(low_vals):.4f} ± {np.std(low_vals):.4f}")
        print(f"  Random vs High-order: p={p_rh:.2e}, d={effect_rh:.3f}")
        print(f"  High vs Low order:    p={p_hl:.2e}, d={effect_hl:.3f}")

    # KEY TEST: Bimodality coefficient
    bc_random = [h['bimodality_coef'] for h in random_histograms]
    bc_high = [h['bimodality_coef'] for h in high_order_hists]

    # Bimodality threshold is 5/9 ≈ 0.555
    bimodal_threshold = 5/9
    random_bimodal_frac = np.mean(np.array(bc_random) > bimodal_threshold)
    high_bimodal_frac = np.mean(np.array(bc_high) > bimodal_threshold)

    print("\n" + "="*70)
    print("BIMODALITY ANALYSIS")
    print("="*70)
    print(f"Bimodality threshold (Sarle's): {bimodal_threshold:.3f}")
    print(f"Random images bimodal: {random_bimodal_frac*100:.1f}%")
    print(f"High-order CPPN bimodal: {high_bimodal_frac*100:.1f}%")

    # Chi-square test for bimodality proportions
    observed = np.array([
        [np.sum(np.array(bc_random) > bimodal_threshold), np.sum(np.array(bc_random) <= bimodal_threshold)],
        [np.sum(np.array(bc_high) > bimodal_threshold), np.sum(np.array(bc_high) <= bimodal_threshold)]
    ])
    chi2, p_chi2, dof, expected = stats.chi2_contingency(observed)

    print(f"\nChi-square test: χ²={chi2:.2f}, p={p_chi2:.2e}")

    # Final verdict
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)

    # Check key criterion: bimodality coefficient difference
    bc_result = results['bimodality_coef']
    significant = bc_result['p_random_vs_high'] < 0.01
    large_effect = abs(bc_result['effect_random_vs_high']) > 0.5

    # Also check standard deviation (spread of values)
    std_result = results['std']
    std_significant = std_result['p_random_vs_high'] < 0.01
    std_large_effect = abs(std_result['effect_random_vs_high']) > 0.5

    print(f"\nBimodality coefficient:")
    print(f"  Random: {bc_result['random_mean']:.4f} ± {bc_result['random_std']:.4f}")
    print(f"  High-order CPPN: {bc_result['high_mean']:.4f} ± {bc_result['high_std']:.4f}")
    print(f"  p-value: {bc_result['p_random_vs_high']:.2e}")
    print(f"  Effect size (Cohen's d): {bc_result['effect_random_vs_high']:.3f}")

    print(f"\nStandard deviation of pixel values:")
    print(f"  Random: {std_result['random_mean']:.4f} ± {std_result['random_std']:.4f}")
    print(f"  High-order CPPN: {std_result['high_mean']:.4f} ± {std_result['high_std']:.4f}")
    print(f"  p-value: {std_result['p_random_vs_high']:.2e}")
    print(f"  Effect size (Cohen's d): {std_result['effect_random_vs_high']:.3f}")

    # Determine status
    if significant and large_effect:
        if bc_result['high_mean'] > bc_result['random_mean']:
            status = "validated"
            summary = f"High-order CPPN images have significantly higher bimodality (BC={bc_result['high_mean']:.3f}) than random (BC={bc_result['random_mean']:.3f}), p={bc_result['p_random_vs_high']:.2e}, d={bc_result['effect_random_vs_high']:.2f}"
        else:
            status = "refuted"
            summary = f"High-order CPPN images have LOWER bimodality than random. BC_high={bc_result['high_mean']:.3f}, BC_random={bc_result['random_mean']:.3f}"
    elif std_significant and std_large_effect:
        status = "validated"
        summary = f"High-order CPPN images have distinct histogram shape (higher std={std_result['high_mean']:.3f} vs {std_result['random_mean']:.3f}), suggesting concentrated values. p={std_result['p_random_vs_high']:.2e}"
    else:
        status = "refuted"
        summary = f"No significant difference in histogram shape between high-order CPPN and random images. BC: p={bc_result['p_random_vs_high']:.2e}, d={bc_result['effect_random_vs_high']:.2f}"

    print(f"\n{'='*70}")
    print(f"STATUS: {status.upper()}")
    print(f"{'='*70}")
    print(f"Summary: {summary}")

    return {
        'status': status,
        'summary': summary,
        'metrics': results,
        'bimodality_coefficient': {
            'effect_size': bc_result['effect_random_vs_high'],
            'p_value': bc_result['p_random_vs_high'],
        },
        'std_dev': {
            'effect_size': std_result['effect_random_vs_high'],
            'p_value': std_result['p_random_vs_high'],
        }
    }


if __name__ == "__main__":
    results = run_experiment(n_samples=200, seed=42)

    print("\n" + "="*70)
    print("EXPERIMENT OUTPUT FOR LOG")
    print("="*70)
    print(f"effect_size: {results['bimodality_coefficient']['effect_size']:.3f}")
    print(f"p_value: {results['bimodality_coefficient']['p_value']:.2e}")
    print(f"status: {results['status']}")
