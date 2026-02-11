#!/usr/bin/env python
"""
RES-091: Multi-Scale Order Consistency

HYPOTHESIS: High-order CPPN images show consistent order scores across multiple
spatial scales (8x8, 16x16, 32x32 patches), while low-order images show
scale-dependent order variation.

METHOD:
1. Generate CPPN images at various order levels
2. For each image, compute order on 8x8, 16x16, 32x32 non-overlapping patches
3. Aggregate patch scores to compute cross-scale consistency (variance)
4. Compare consistency between high-order and low-order images
"""

import numpy as np
from scipy import stats
import sys
import random
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import (
    CPPN, Node, Connection, ACTIVATIONS, PRIOR_SIGMA,
    order_multiplicative,
    compute_spectral_coherence, compute_edge_density,
    compute_compressibility
)


def create_cppn_with_hidden_nodes(n_hidden: int) -> CPPN:
    """Create a CPPN with specified number of hidden nodes."""
    nodes = [
        Node(0, 'identity', 0.0),  # x
        Node(1, 'identity', 0.0),  # y
        Node(2, 'identity', 0.0),  # r
        Node(3, 'identity', 0.0),  # bias
        Node(4, 'sigmoid', np.random.randn() * PRIOR_SIGMA),  # output
    ]

    connections = []
    activations = list(ACTIVATIONS.keys())
    hidden_ids = []

    for i in range(n_hidden):
        hid = 5 + i
        hidden_ids.append(hid)
        act = random.choice(activations)
        nodes.append(Node(hid, act, np.random.randn() * PRIOR_SIGMA))

    if n_hidden == 0:
        for inp in [0, 1, 2, 3]:
            connections.append(Connection(inp, 4, np.random.randn() * PRIOR_SIGMA))
    else:
        for inp in [0, 1, 2, 3]:
            for hid in hidden_ids:
                if random.random() < 0.7:
                    connections.append(Connection(inp, hid, np.random.randn() * PRIOR_SIGMA))
        for hid in hidden_ids:
            connections.append(Connection(hid, 4, np.random.randn() * PRIOR_SIGMA))

    return CPPN(nodes=nodes, connections=connections)


def compute_order_at_scale(img: np.ndarray, patch_size: int) -> tuple[float, float]:
    """
    Compute order metric over non-overlapping patches of given size.
    Returns: (mean_order, std_order) across patches.
    """
    h, w = img.shape
    patches = []

    for i in range(0, h - patch_size + 1, patch_size):
        for j in range(0, w - patch_size + 1, patch_size):
            patch = img[i:i+patch_size, j:j+patch_size]
            # Use order_multiplicative on each patch
            order = order_multiplicative(patch)
            patches.append(order)

    if len(patches) == 0:
        return 0.0, 0.0

    return np.mean(patches), np.std(patches)


def compute_multi_scale_features(img: np.ndarray) -> dict:
    """
    Compute order at multiple scales and cross-scale consistency.
    """
    # Full image order
    full_order = order_multiplicative(img)

    # Order at different patch scales
    scales = [8, 16, 32]
    scale_orders = {}
    scale_stds = {}

    for s in scales:
        mean_o, std_o = compute_order_at_scale(img, s)
        scale_orders[s] = mean_o
        scale_stds[s] = std_o

    # Cross-scale consistency: how similar are orders across scales
    order_values = list(scale_orders.values())
    cross_scale_variance = np.var(order_values)
    cross_scale_range = max(order_values) - min(order_values)

    # Cross-scale correlation: do high-order patches at one scale correspond to high at another?
    # Using coefficient of variation as consistency measure
    cross_scale_cv = np.std(order_values) / (np.mean(order_values) + 1e-10)

    return {
        'full_order': full_order,
        'order_8x8': scale_orders[8],
        'order_16x16': scale_orders[16],
        'order_32x32': scale_orders[32],
        'std_8x8': scale_stds[8],
        'std_16x16': scale_stds[16],
        'std_32x32': scale_stds[32],
        'cross_scale_variance': cross_scale_variance,
        'cross_scale_range': cross_scale_range,
        'cross_scale_cv': cross_scale_cv,
    }


def main():
    print("=" * 60)
    print("RES-091: Multi-Scale Order Consistency")
    print("=" * 60)

    np.random.seed(42)

    # Generate images at different order levels
    n_images = 100
    image_size = 64

    print(f"\nGenerating {n_images} CPPN images at {image_size}x{image_size}...")

    results = []
    for i in range(n_images):
        cppn = create_cppn_with_hidden_nodes(n_hidden=3)
        img = cppn.render(image_size)
        features = compute_multi_scale_features(img)
        results.append(features)

        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{n_images}")

    # Also generate random noise images for comparison
    print(f"\nGenerating {n_images} random noise images...")
    random_results = []
    for i in range(n_images):
        img = (np.random.rand(image_size, image_size) > 0.5).astype(np.uint8)
        features = compute_multi_scale_features(img)
        random_results.append(features)

    # Convert to arrays for analysis
    full_orders = np.array([r['full_order'] for r in results])
    cross_scale_variances = np.array([r['cross_scale_variance'] for r in results])
    cross_scale_ranges = np.array([r['cross_scale_range'] for r in results])
    cross_scale_cvs = np.array([r['cross_scale_cv'] for r in results])

    random_cross_scale_variances = np.array([r['cross_scale_variance'] for r in random_results])
    random_cross_scale_cvs = np.array([r['cross_scale_cv'] for r in random_results])

    # Split CPPN images into high-order and low-order groups
    median_order = np.median(full_orders)
    high_order_mask = full_orders >= median_order
    low_order_mask = full_orders < median_order

    high_order_cv = cross_scale_cvs[high_order_mask]
    low_order_cv = cross_scale_cvs[low_order_mask]
    high_order_var = cross_scale_variances[high_order_mask]
    low_order_var = cross_scale_variances[low_order_mask]
    high_order_range = cross_scale_ranges[high_order_mask]
    low_order_range = cross_scale_ranges[low_order_mask]

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\nCPPN Image Order Distribution:")
    print(f"  Min: {full_orders.min():.4f}, Max: {full_orders.max():.4f}")
    print(f"  Mean: {full_orders.mean():.4f}, Median: {median_order:.4f}")
    print(f"  High-order group (n={sum(high_order_mask)}): >= {median_order:.4f}")
    print(f"  Low-order group (n={sum(low_order_mask)}): < {median_order:.4f}")

    # Test 1: Cross-scale variance comparison
    print(f"\n--- Cross-Scale Variance (lower = more consistent) ---")
    print(f"  High-order CPPN: {high_order_var.mean():.6f} +/- {high_order_var.std():.6f}")
    print(f"  Low-order CPPN:  {low_order_var.mean():.6f} +/- {low_order_var.std():.6f}")
    print(f"  Random noise:    {random_cross_scale_variances.mean():.6f} +/- {random_cross_scale_variances.std():.6f}")

    # Test 2: Cross-scale CV comparison
    print(f"\n--- Cross-Scale CV (lower = more consistent) ---")
    print(f"  High-order CPPN: {high_order_cv.mean():.4f} +/- {high_order_cv.std():.4f}")
    print(f"  Low-order CPPN:  {low_order_cv.mean():.4f} +/- {low_order_cv.std():.4f}")
    print(f"  Random noise:    {random_cross_scale_cvs.mean():.4f} +/- {random_cross_scale_cvs.std():.4f}")

    # Test 3: Cross-scale range comparison
    print(f"\n--- Cross-Scale Range (lower = more consistent) ---")
    print(f"  High-order CPPN: {high_order_range.mean():.4f} +/- {high_order_range.std():.4f}")
    print(f"  Low-order CPPN:  {low_order_range.mean():.4f} +/- {low_order_range.std():.4f}")

    # Statistical tests
    print("\n" + "=" * 60)
    print("STATISTICAL TESTS")
    print("=" * 60)

    # Primary test: High vs Low order CPPN cross-scale consistency
    t_stat, p_value = stats.ttest_ind(high_order_cv, low_order_cv)
    effect_size = (low_order_cv.mean() - high_order_cv.mean()) / np.sqrt(
        (high_order_cv.var() + low_order_cv.var()) / 2
    )

    print(f"\nHigh-order vs Low-order CPPN (Cross-Scale CV):")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.4e}")
    print(f"  Effect size (Cohen's d): {effect_size:.4f}")

    # Secondary test: CPPN vs Random noise
    t_stat2, p_value2 = stats.ttest_ind(cross_scale_cvs, random_cross_scale_cvs)
    effect_size2 = (random_cross_scale_cvs.mean() - cross_scale_cvs.mean()) / np.sqrt(
        (cross_scale_cvs.var() + random_cross_scale_cvs.var()) / 2
    )

    print(f"\nCPPN vs Random noise (Cross-Scale CV):")
    print(f"  t-statistic: {t_stat2:.4f}")
    print(f"  p-value: {p_value2:.4e}")
    print(f"  Effect size (Cohen's d): {effect_size2:.4f}")

    # Correlation: full order vs cross-scale consistency
    corr, corr_p = stats.pearsonr(full_orders, cross_scale_cvs)
    print(f"\nCorrelation (full order vs cross-scale CV):")
    print(f"  r = {corr:.4f}, p = {corr_p:.4e}")

    # Per-scale analysis
    print("\n" + "=" * 60)
    print("PER-SCALE ORDER ANALYSIS")
    print("=" * 60)

    for scale in [8, 16, 32]:
        key = f'order_{scale}x{scale}'
        cppn_scale_orders = np.array([r[key] for r in results])
        random_scale_orders = np.array([r[key] for r in random_results])

        t, p = stats.ttest_ind(cppn_scale_orders, random_scale_orders)
        d = (cppn_scale_orders.mean() - random_scale_orders.mean()) / np.sqrt(
            (cppn_scale_orders.var() + random_scale_orders.var()) / 2
        )

        print(f"\n{scale}x{scale} patches:")
        print(f"  CPPN mean: {cppn_scale_orders.mean():.4f}, Random mean: {random_scale_orders.mean():.4f}")
        print(f"  Effect size: {d:.4f}, p-value: {p:.4e}")

    # Final verdict
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)

    # Hypothesis: high-order images show MORE consistent cross-scale order (lower CV)
    # So effect_size should be positive (low_order - high_order > 0)
    validated = abs(effect_size) > 0.5 and p_value < 0.01

    if validated and effect_size > 0:
        status = "VALIDATED"
        conclusion = "High-order CPPN images show significantly MORE consistent cross-scale order than low-order images."
    elif validated and effect_size < 0:
        status = "REFUTED"
        conclusion = "High-order CPPN images show LESS consistent cross-scale order (opposite of hypothesis)."
    else:
        status = "INCONCLUSIVE"
        conclusion = f"Effect size ({effect_size:.3f}) or p-value ({p_value:.4e}) did not meet threshold."

    print(f"\nStatus: {status}")
    print(f"Effect size: {effect_size:.4f} (threshold: |d| > 0.5)")
    print(f"P-value: {p_value:.4e} (threshold: p < 0.01)")
    print(f"\n{conclusion}")

    return {
        'status': status.lower(),
        'effect_size': effect_size,
        'p_value': p_value,
        'high_order_cv_mean': high_order_cv.mean(),
        'low_order_cv_mean': low_order_cv.mean(),
        'correlation': corr,
        'correlation_p': corr_p,
    }


if __name__ == '__main__':
    main()
