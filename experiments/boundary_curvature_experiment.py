"""
RES-169: Boundary Curvature Analysis

Hypothesis: High-order CPPN images have lower mean boundary curvature than low-order images.

Rationale: High-order CPPNs produce spatially coherent patterns with smooth structures
(see RES-146). Smooth boundaries should have lower curvature (less zig-zagging).
This tests whether the order metric correlates with geometric boundary smoothness.

Approach:
1. Generate CPPN images across order spectrum
2. Extract boundary contours using marching squares
3. Compute mean absolute curvature at each boundary point
4. Correlate curvature with order

Success criteria: |r| > 0.3, d > 0.5, p < 0.01
"""

import numpy as np
import sys
import os
import json
from pathlib import Path
from scipy import ndimage
from scipy.stats import pearsonr, spearmanr, ttest_ind

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.thermo_sampler_v3 import CPPN, order_multiplicative, set_global_seed


def extract_boundary_points(img):
    """Extract boundary pixel coordinates using edge detection."""
    # Pad to avoid edge effects
    padded = np.pad(img, 1, mode='constant', constant_values=0)

    # Find edge pixels (foreground adjacent to background)
    kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    neighbors = ndimage.convolve(padded, kernel, mode='constant')

    # Edge = foreground pixel with at least one background neighbor
    edges = (padded == 1) & (neighbors < 4)

    # Remove padding
    edges = edges[1:-1, 1:-1]

    # Get coordinates
    y_coords, x_coords = np.where(edges)
    return np.column_stack([x_coords, y_coords])


def order_boundary_points(points):
    """Order boundary points to form a contour using nearest neighbor."""
    if len(points) < 3:
        return points

    ordered = [points[0]]
    remaining = list(range(1, len(points)))

    while remaining:
        last = ordered[-1]
        # Find nearest remaining point
        distances = np.sum((points[remaining] - last) ** 2, axis=1)
        nearest_idx = np.argmin(distances)
        ordered.append(points[remaining[nearest_idx]])
        remaining.pop(nearest_idx)

    return np.array(ordered)


def compute_curvature(points):
    """
    Compute curvature at each point using discrete curvature formula.
    Curvature k = |dT/ds| where T is the unit tangent.
    For discrete points: k ≈ 2 * sin(theta/2) / d
    where theta is the turning angle.
    """
    if len(points) < 3:
        return np.array([0])

    curvatures = []
    n = len(points)

    for i in range(n):
        p0 = points[(i - 1) % n]
        p1 = points[i]
        p2 = points[(i + 1) % n]

        # Vectors
        v1 = p1 - p0
        v2 = p2 - p1

        # Magnitudes
        len1 = np.linalg.norm(v1) + 1e-10
        len2 = np.linalg.norm(v2) + 1e-10

        # Turning angle via cross product
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        dot = np.dot(v1, v2)

        # Curvature = turning angle / arc length
        angle = np.arctan2(abs(cross), dot)
        arc_length = (len1 + len2) / 2

        k = angle / arc_length
        curvatures.append(k)

    return np.array(curvatures)


def compute_boundary_curvature_stats(img):
    """Compute boundary curvature statistics for a binary image."""
    # Extract boundary points
    points = extract_boundary_points(img)

    if len(points) < 5:
        return {'mean_curvature': 0, 'max_curvature': 0, 'total_points': len(points)}

    # Order points to form contour
    ordered_points = order_boundary_points(points)

    # Compute curvature
    curvatures = compute_curvature(ordered_points)

    return {
        'mean_curvature': float(np.mean(curvatures)),
        'max_curvature': float(np.max(curvatures)),
        'std_curvature': float(np.std(curvatures)),
        'median_curvature': float(np.median(curvatures)),
        'total_points': len(points)
    }


def main():
    set_global_seed(42)

    n_samples = 500
    size = 32

    print(f"Generating {n_samples} CPPN images...")

    results = []

    for i in range(n_samples):
        cppn = CPPN()
        img = cppn.render(size)
        order = order_multiplicative(img)

        stats = compute_boundary_curvature_stats(img)

        results.append({
            'idx': i,
            'order': order,
            'mean_curvature': stats['mean_curvature'],
            'max_curvature': stats['max_curvature'],
            'std_curvature': stats.get('std_curvature', 0),
            'median_curvature': stats.get('median_curvature', 0),
            'boundary_points': stats['total_points']
        })

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{n_samples}")

    # Convert to arrays for analysis
    orders = np.array([r['order'] for r in results])
    mean_curvatures = np.array([r['mean_curvature'] for r in results])
    max_curvatures = np.array([r['max_curvature'] for r in results])

    # Filter out zero-boundary cases
    valid_mask = mean_curvatures > 0
    orders_valid = orders[valid_mask]
    mean_curv_valid = mean_curvatures[valid_mask]
    max_curv_valid = max_curvatures[valid_mask]

    print(f"\nValid samples (with boundaries): {np.sum(valid_mask)}/{n_samples}")

    # Correlation analysis
    r_mean, p_mean = pearsonr(orders_valid, mean_curv_valid)
    rho_mean, p_rho = spearmanr(orders_valid, mean_curv_valid)
    r_max, p_max = pearsonr(orders_valid, max_curv_valid)

    print(f"\nCorrelation (order vs mean curvature):")
    print(f"  Pearson r = {r_mean:.4f}, p = {p_mean:.2e}")
    print(f"  Spearman rho = {rho_mean:.4f}, p = {p_rho:.2e}")
    print(f"\nCorrelation (order vs max curvature):")
    print(f"  Pearson r = {r_max:.4f}, p = {p_max:.2e}")

    # Split into high/low order groups
    median_order = np.median(orders_valid)
    high_order = orders_valid >= median_order
    low_order = orders_valid < median_order

    high_curv = mean_curv_valid[high_order]
    low_curv = mean_curv_valid[low_order]

    t_stat, t_pval = ttest_ind(high_curv, low_curv)

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.var(high_curv) + np.var(low_curv)) / 2)
    cohens_d = (np.mean(high_curv) - np.mean(low_curv)) / (pooled_std + 1e-10)

    print(f"\nGroup comparison:")
    print(f"  High-order mean curvature: {np.mean(high_curv):.4f} +/- {np.std(high_curv):.4f}")
    print(f"  Low-order mean curvature: {np.mean(low_curv):.4f} +/- {np.std(low_curv):.4f}")
    print(f"  t-stat = {t_stat:.3f}, p = {t_pval:.2e}")
    print(f"  Cohen's d = {cohens_d:.3f}")

    # Validation criteria
    validated = (abs(cohens_d) > 0.5) and (t_pval < 0.01)

    # Direction check (hypothesis: high order -> lower curvature)
    hypothesis_direction = cohens_d < 0  # Negative d means high order has lower curvature

    print(f"\n--- RESULTS ---")
    print(f"Effect size (d): {cohens_d:.3f}")
    print(f"P-value: {t_pval:.2e}")
    print(f"Correlation (r): {r_mean:.3f}")
    print(f"Hypothesis direction (high order -> lower curvature): {hypothesis_direction}")
    print(f"Validated (|d|>0.5, p<0.01): {validated}")

    if validated:
        if hypothesis_direction:
            status = "validated"
            summary = f"High-order CPPN images have lower boundary curvature (d={cohens_d:.2f}, r={r_mean:.2f}, p<{t_pval:.0e})"
        else:
            status = "refuted"
            summary = f"High-order CPPN images have HIGHER boundary curvature (d={cohens_d:.2f}, r={r_mean:.2f}, p<{t_pval:.0e})"
    else:
        if abs(cohens_d) <= 0.5:
            status = "refuted" if t_pval < 0.01 else "inconclusive"
            summary = f"Boundary curvature effect too small (d={cohens_d:.2f}, r={r_mean:.2f}, p={t_pval:.2e})"
        else:
            status = "inconclusive"
            summary = f"Boundary curvature effect inconclusive (d={cohens_d:.2f}, p={t_pval:.2e})"

    print(f"\nStatus: {status}")
    print(f"Summary: {summary}")

    # Save results
    output_dir = Path(__file__).parent.parent / "results" / "boundary_curvature"
    output_dir.mkdir(parents=True, exist_ok=True)

    output = {
        'experiment_id': 'RES-169',
        'hypothesis': 'High-order CPPN images have lower mean boundary curvature than low-order images',
        'domain': 'boundary_analysis',
        'status': status,
        'n_samples': n_samples,
        'n_valid': int(np.sum(valid_mask)),
        'metrics': {
            'pearson_r': float(r_mean),
            'pearson_p': float(p_mean),
            'spearman_rho': float(rho_mean),
            'spearman_p': float(p_rho),
            'cohens_d': float(cohens_d),
            't_pval': float(t_pval),
            'high_order_mean_curvature': float(np.mean(high_curv)),
            'low_order_mean_curvature': float(np.mean(low_curv)),
            'high_order_std': float(np.std(high_curv)),
            'low_order_std': float(np.std(low_curv)),
        },
        'summary': summary
    }

    with open(output_dir / "results.json", 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_dir / 'results.json'}")

    return status, summary, output['metrics']


if __name__ == '__main__':
    main()
