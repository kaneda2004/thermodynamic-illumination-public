"""
RES-093: Decision Boundary Complexity Analysis

Hypothesis: Higher-order CPPN images have simpler (smoother) 0.5 threshold
decision boundaries than lower-order CPPNs, measured by boundary perimeter
relative to enclosed area (isoperimetric ratio).

The isoperimetric ratio Q = perimeter^2 / (4 * pi * area) equals 1 for a
perfect circle and increases for more complex shapes.
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')
from core.thermo_sampler_v3 import CPPN, order_multiplicative
from scipy import stats
from scipy.ndimage import label, binary_fill_holes
import json


def compute_boundary_complexity(img: np.ndarray) -> dict:
    """
    Compute decision boundary complexity metrics.

    Returns:
        dict with:
        - perimeter: total boundary length
        - area: total foreground area
        - isoperimetric_ratio: perimeter^2 / (4*pi*area), 1=circle
        - num_components: number of connected components
        - boundary_fractal_dim: estimate via box-counting
    """
    # Find contours at 0.5 threshold (but image is already binary)
    # Count boundary pixels (edge between 0 and 1)
    padded = np.pad(img, 1, mode='constant', constant_values=0)

    # Boundary pixels: foreground pixels adjacent to background
    boundary = np.zeros_like(padded, dtype=bool)
    for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
        shifted = np.roll(np.roll(padded, di, axis=0), dj, axis=1)
        boundary |= (padded == 1) & (shifted == 0)

    # Remove padding
    boundary = boundary[1:-1, 1:-1]

    perimeter = np.sum(boundary)
    area = np.sum(img)

    # Isoperimetric ratio (avoiding division by zero)
    if area > 0:
        iso_ratio = (perimeter ** 2) / (4 * np.pi * area)
    else:
        iso_ratio = np.nan

    # Count connected components using scipy.ndimage.label
    labeled, num_components = label(img)

    # Also count holes (background components not connected to border)
    labeled_bg, num_bg_components = label(1 - img)
    # Component labels on border are exterior
    border_labels = set(labeled_bg[0, :]).union(
        set(labeled_bg[-1, :])).union(
        set(labeled_bg[:, 0])).union(
        set(labeled_bg[:, -1]))
    num_holes = num_bg_components - len(border_labels) + 1  # +1 for 0 label

    return {
        'perimeter': float(perimeter),
        'area': float(area),
        'isoperimetric_ratio': float(iso_ratio) if not np.isnan(iso_ratio) else None,
        'num_components': int(num_components),
        'num_holes': max(0, int(num_holes))
    }


def sample_cppn_with_order(n_samples: int = 500, seed: int = 42) -> list:
    """Sample CPPNs and compute order + boundary complexity."""
    np.random.seed(seed)
    results = []

    for i in range(n_samples):
        cppn = CPPN()
        img = cppn.render(size=64)  # Higher res for better boundary estimation

        # Skip trivial images (all 0 or all 1)
        if img.sum() == 0 or img.sum() == img.size:
            continue

        order = order_multiplicative(img)
        boundary = compute_boundary_complexity(img)

        if boundary['isoperimetric_ratio'] is not None:
            results.append({
                'order': order,
                **boundary
            })

    return results


def analyze_results(results: list) -> dict:
    """Compute correlations and test hypothesis."""
    orders = np.array([r['order'] for r in results])
    iso_ratios = np.array([r['isoperimetric_ratio'] for r in results])
    perimeters = np.array([r['perimeter'] for r in results])
    areas = np.array([r['area'] for r in results])
    num_components = np.array([r['num_components'] for r in results])

    # Correlation between order and isoperimetric ratio
    # Hypothesis: higher order -> lower isoperimetric ratio (simpler boundary)
    corr_iso, p_iso = stats.pearsonr(orders, iso_ratios)
    corr_iso_spearman, p_iso_spearman = stats.spearmanr(orders, iso_ratios)

    # Correlation with number of components
    corr_comp, p_comp = stats.spearmanr(orders, num_components)

    # Correlation with perimeter (normalized by area)
    norm_perimeter = perimeters / np.sqrt(areas + 1)
    corr_norm_perim, p_norm_perim = stats.pearsonr(orders, norm_perimeter)

    # Split into high/low order groups for effect size
    median_order = np.median(orders)
    high_order_iso = iso_ratios[orders >= median_order]
    low_order_iso = iso_ratios[orders < median_order]

    # Cohen's d effect size
    pooled_std = np.sqrt((np.var(high_order_iso) + np.var(low_order_iso)) / 2)
    cohens_d = (np.mean(low_order_iso) - np.mean(high_order_iso)) / pooled_std

    # Mann-Whitney U test (non-parametric)
    u_stat, p_mannwhitney = stats.mannwhitneyu(high_order_iso, low_order_iso, alternative='less')

    return {
        'n_samples': len(results),
        'correlation_iso_pearson': float(corr_iso),
        'p_value_iso_pearson': float(p_iso),
        'correlation_iso_spearman': float(corr_iso_spearman),
        'p_value_iso_spearman': float(p_iso_spearman),
        'correlation_components': float(corr_comp),
        'p_value_components': float(p_comp),
        'correlation_norm_perimeter': float(corr_norm_perim),
        'p_value_norm_perimeter': float(p_norm_perim),
        'cohens_d': float(cohens_d),
        'p_value_mannwhitney': float(p_mannwhitney),
        'high_order_iso_mean': float(np.mean(high_order_iso)),
        'low_order_iso_mean': float(np.mean(low_order_iso)),
        'order_range': [float(orders.min()), float(orders.max())],
        'iso_range': [float(iso_ratios.min()), float(iso_ratios.max())]
    }


def main():
    print("RES-093: Decision Boundary Complexity Analysis")
    print("=" * 60)

    # Run multiple seeds for robustness
    all_results = []
    for seed in [42, 123, 456, 789, 1011]:
        results = sample_cppn_with_order(n_samples=400, seed=seed)
        all_results.extend(results)
        print(f"Seed {seed}: {len(results)} valid samples")

    print(f"\nTotal samples: {len(all_results)}")

    analysis = analyze_results(all_results)

    print("\n--- Results ---")
    print(f"Correlation (order vs iso ratio, Pearson): r = {analysis['correlation_iso_pearson']:.4f}, p = {analysis['p_value_iso_pearson']:.2e}")
    print(f"Correlation (order vs iso ratio, Spearman): rho = {analysis['correlation_iso_spearman']:.4f}, p = {analysis['p_value_iso_spearman']:.2e}")
    print(f"Correlation (order vs num components): rho = {analysis['correlation_components']:.4f}, p = {analysis['p_value_components']:.2e}")
    print(f"Cohen's d (high vs low order): {analysis['cohens_d']:.4f}")
    print(f"Mann-Whitney p (high < low iso?): {analysis['p_value_mannwhitney']:.2e}")
    print(f"\nMean iso ratio: high-order = {analysis['high_order_iso_mean']:.2f}, low-order = {analysis['low_order_iso_mean']:.2f}")

    # Determine outcome
    # Hypothesis: higher order -> simpler boundary (lower iso ratio) -> negative correlation
    significant = analysis['p_value_iso_spearman'] < 0.01
    correct_direction = analysis['correlation_iso_spearman'] < 0
    sufficient_effect = abs(analysis['cohens_d']) > 0.3

    if significant and correct_direction and sufficient_effect:
        status = "VALIDATED"
        summary = f"Higher-order CPPNs have simpler decision boundaries (r={analysis['correlation_iso_spearman']:.3f}, d={analysis['cohens_d']:.2f})"
    elif significant and not correct_direction:
        status = "REFUTED"
        summary = f"Higher-order CPPNs have MORE complex boundaries (r={analysis['correlation_iso_spearman']:.3f})"
    else:
        status = "INCONCLUSIVE"
        summary = f"No clear relationship (r={analysis['correlation_iso_spearman']:.3f}, p={analysis['p_value_iso_spearman']:.2e})"

    print(f"\n--- Status: {status} ---")
    print(summary)

    # Save full results
    output = {
        'experiment_id': 'RES-093',
        'hypothesis': 'Higher-order CPPN images have simpler 0.5 threshold decision boundaries',
        'status': status,
        'summary': summary,
        'analysis': analysis
    }

    with open('/Users/matt/Development/monochrome_noise_converger/results/res093_boundary_complexity.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to results/res093_boundary_complexity.json")

    return status, analysis


if __name__ == '__main__':
    main()
