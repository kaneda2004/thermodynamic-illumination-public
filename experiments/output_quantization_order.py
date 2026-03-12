"""
RES-088: Output Quantization Effects on Order

HYPOTHESIS: Output quantization to discrete levels (2, 4, 8, 16 bins) reduces
order scores due to destruction of fine gradient information.

REVISED APPROACH: Since the order metric operates on binary images, we need
to study how quantization affects the THRESHOLD SWEEP behavior. When you
quantize to N levels, sweeping threshold creates only N distinct binary images.
This "threshold sensitivity" relates to perceptual structure.

We measure:
1. Threshold sensitivity: how order changes across different thresholds
2. Optimal threshold discovery: whether quantization makes it harder to find good thresholds
3. Order variance across thresholds: continuous images should have smoother transitions
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import (
    CPPN, order_multiplicative, compute_compressibility,
    compute_edge_density, compute_spectral_coherence, compute_symmetry,
    set_global_seed
)
from scipy import stats


def render_continuous(cppn: CPPN, size: int = 32) -> np.ndarray:
    """Render CPPN with continuous [0,1] output (before thresholding)."""
    coords = np.linspace(-1, 1, size)
    x, y = np.meshgrid(coords, coords)
    return cppn.activate(x, y)  # Returns float in [0, 1] after sigmoid


def quantize(img: np.ndarray, levels: int) -> np.ndarray:
    """Quantize continuous [0,1] image to discrete levels."""
    # Map to [0, levels-1], round, map back to [0, 1]
    quantized = np.round(img * (levels - 1)) / (levels - 1)
    return quantized


def binarize(img: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convert continuous to binary (what CPPN.render() does)."""
    return (img > threshold).astype(np.uint8)


def run_experiment(n_samples: int = 200, seed: int = 42):
    """Run quantization experiment with threshold sweeping."""
    set_global_seed(seed)

    # Quantization levels to test
    quantization_levels = [2, 4, 8, 16, 32, 64, 256]  # 256 = effectively continuous

    # Thresholds to sweep
    thresholds = np.linspace(0.1, 0.9, 17)

    # Results: for each quantization level, track:
    # - max_order: best order across thresholds
    # - order_variance: variance of order across thresholds
    # - unique_images: how many distinct binary images threshold sweep produces
    results = {
        'max_order': {level: [] for level in quantization_levels},
        'order_variance': {level: [] for level in quantization_levels},
        'unique_images': {level: [] for level in quantization_levels},
        'mean_order': {level: [] for level in quantization_levels},
    }

    for i in range(n_samples):
        # Generate random CPPN
        cppn = CPPN()  # Random init

        # Get continuous output
        continuous = render_continuous(cppn)

        for level in quantization_levels:
            if level == 256:
                # Effectively continuous
                img_float = continuous
            else:
                img_float = quantize(continuous, level)

            # Sweep thresholds and collect orders
            orders = []
            binary_images = []

            for thresh in thresholds:
                img_binary = binarize(img_float, thresh)
                order = order_multiplicative(img_binary)
                orders.append(order)
                binary_images.append(img_binary.tobytes())

            # Compute metrics
            results['max_order'][level].append(np.max(orders))
            results['order_variance'][level].append(np.var(orders))
            results['unique_images'][level].append(len(set(binary_images)))
            results['mean_order'][level].append(np.mean(orders))

        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{n_samples} samples")

    return results, quantization_levels, thresholds


def analyze_results(results, quantization_levels):
    """Statistical analysis of quantization effects."""

    print("\n" + "="*70)
    print("MAX ORDER (best threshold) BY QUANTIZATION LEVEL")
    print("="*70)
    print(f"{'Levels':>8} | {'Mean':>8} | {'Std':>8} | {'Median':>8}")
    print("-"*40)

    for level in quantization_levels:
        label = "cont." if level == 256 else str(level)
        mean = np.mean(results['max_order'][level])
        std = np.std(results['max_order'][level])
        median = np.median(results['max_order'][level])
        print(f"{label:>8} | {mean:>8.4f} | {std:>8.4f} | {median:>8.4f}")

    print("\n" + "="*70)
    print("ORDER VARIANCE (across thresholds) BY QUANTIZATION")
    print("="*70)
    print(f"{'Levels':>8} | {'Mean':>8} | {'Std':>8}")
    print("-"*30)

    for level in quantization_levels:
        label = "cont." if level == 256 else str(level)
        mean = np.mean(results['order_variance'][level])
        std = np.std(results['order_variance'][level])
        print(f"{label:>8} | {mean:>8.4f} | {std:>8.4f}")

    print("\n" + "="*70)
    print("UNIQUE BINARY IMAGES (threshold sweep) BY QUANTIZATION")
    print("="*70)
    print(f"{'Levels':>8} | {'Mean':>8} | {'Std':>8}")
    print("-"*30)

    for level in quantization_levels:
        label = "cont." if level == 256 else str(level)
        mean = np.mean(results['unique_images'][level])
        std = np.std(results['unique_images'][level])
        print(f"{label:>8} | {mean:>8.1f} | {std:>8.2f}")

    # Statistical test: continuous (256) vs 2-level
    continuous_max = results['max_order'][256]
    two_level_max = results['max_order'][2]

    t_stat, p_value = stats.ttest_rel(continuous_max, two_level_max)
    pooled_std = np.sqrt((np.var(continuous_max) + np.var(two_level_max)) / 2)
    effect_size = (np.mean(continuous_max) - np.mean(two_level_max)) / pooled_std

    print("\n" + "="*70)
    print("STATISTICAL TEST: Continuous (256) vs 2-Level MAX ORDER")
    print("="*70)
    print(f"Paired t-test: t={t_stat:.4f}, p={p_value:.2e}")
    print(f"Effect size (Cohen's d): {effect_size:.4f}")
    print(f"Mean continuous: {np.mean(continuous_max):.4f}")
    print(f"Mean 2-level: {np.mean(two_level_max):.4f}")
    print(f"Mean difference: {np.mean(continuous_max) - np.mean(two_level_max):.4f}")

    # Test trend: does max_order increase with more quantization levels?
    means = [np.mean(results['max_order'][l]) for l in quantization_levels]
    corr, p_corr = stats.spearmanr(quantization_levels, means)

    print("\n" + "="*70)
    print("TREND ANALYSIS: Max Order vs Quantization Levels")
    print("="*70)
    print(f"Spearman correlation: r={corr:.4f}, p={p_corr:.2e}")
    print("(Positive = more levels -> higher max order, confirming hypothesis)")

    # Test unique images relationship
    unique_means = [np.mean(results['unique_images'][l]) for l in quantization_levels]
    corr_unique, p_unique = stats.spearmanr(quantization_levels, unique_means)

    print("\n" + "="*70)
    print("UNIQUE IMAGES CORRELATION")
    print("="*70)
    print(f"Spearman (levels vs unique images): r={corr_unique:.4f}, p={p_unique:.2e}")

    return {
        't_stat': t_stat,
        'p_value': p_value,
        'effect_size': effect_size,
        'correlation': corr,
        'p_correlation': p_corr,
        'mean_continuous': np.mean(continuous_max),
        'mean_2level': np.mean(two_level_max),
    }


def main():
    print("RES-088: Output Quantization Effects on Order")
    print("="*70)
    print("HYPOTHESIS: Coarser quantization reduces achievable order scores")
    print("by limiting threshold search space")
    print("="*70)

    results, quantization_levels, thresholds = run_experiment(n_samples=200, seed=42)
    stats_results = analyze_results(results, quantization_levels)

    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)

    # Determine outcome based on whether more quantization levels -> higher max order
    if stats_results['p_value'] < 0.01 and abs(stats_results['effect_size']) > 0.5:
        if stats_results['effect_size'] > 0:
            status = "VALIDATED"
            conclusion = "Finer quantization enables higher achievable order scores"
        else:
            status = "REFUTED"
            conclusion = "Coarser quantization enables HIGHER order (opposite of hypothesis)"
    elif stats_results['p_value'] < 0.01:
        status = "INCONCLUSIVE"
        conclusion = f"Significant but small effect (d={stats_results['effect_size']:.3f})"
    else:
        status = "REFUTED"
        conclusion = "No significant effect of quantization on achievable order"

    print(f"Status: {status}")
    print(f"Conclusion: {conclusion}")
    print(f"\nKey metrics:")
    print(f"  p-value: {stats_results['p_value']:.2e}")
    print(f"  effect_size: {stats_results['effect_size']:.4f}")
    print(f"  correlation: {stats_results['correlation']:.4f}")

    return status, stats_results


if __name__ == "__main__":
    status, stats_results = main()
