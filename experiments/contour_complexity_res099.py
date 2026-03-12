"""
RES-099: Contour Complexity Analysis - CPPN vs Random Images

Hypothesis: CPPN images have lower contour complexity (perimeter/area ratio)
than random images due to smoother boundaries.

Metrics:
- Perimeter/area ratio for each connected region
- Number of contours per image
- Mean contour smoothness (inverse of jaggedness)
"""

import numpy as np
from scipy import ndimage
from scipy import stats
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.thermo_sampler_v3 import CPPN, set_global_seed


def extract_contour_metrics(img: np.ndarray) -> dict:
    """
    Extract contour complexity metrics from a binary image.

    Returns:
        - perimeter_area_ratio: Mean perimeter/area ratio across all components
        - num_contours: Number of distinct connected components
        - smoothness: Inverse of jaggedness (fewer direction changes = smoother)
    """
    # Label connected components (foreground)
    labeled, num_components_fg = ndimage.label(img)

    # Also count background components for total contours
    labeled_bg, num_components_bg = ndimage.label(1 - img)

    total_contours = num_components_fg + num_components_bg

    if num_components_fg == 0:
        return {
            'perimeter_area_ratio': 0.0,
            'num_contours': total_contours,
            'smoothness': 1.0,
            'mean_area': 0.0
        }

    perimeter_area_ratios = []
    smoothness_scores = []
    areas = []

    for component_id in range(1, num_components_fg + 1):
        component_mask = (labeled == component_id)
        area = np.sum(component_mask)
        areas.append(area)

        if area < 4:  # Skip tiny components
            continue

        # Compute perimeter by counting boundary pixels
        # Dilate and subtract to get boundary
        dilated = ndimage.binary_dilation(component_mask)
        perimeter_mask = dilated & ~component_mask
        perimeter = np.sum(perimeter_mask)

        # Perimeter/area ratio (lower = more compact/smooth)
        pa_ratio = perimeter / np.sqrt(area) if area > 0 else 0
        perimeter_area_ratios.append(pa_ratio)

        # Smoothness: count direction changes along boundary
        # Find boundary pixels and trace
        boundary_pixels = np.argwhere(perimeter_mask)
        if len(boundary_pixels) > 2:
            # Calculate mean curvature proxy: how much boundary deviates from straight
            # Using convex hull area / actual area as smoothness proxy
            from scipy.spatial import ConvexHull
            try:
                if len(boundary_pixels) >= 4:
                    hull = ConvexHull(boundary_pixels)
                    hull_area = hull.volume  # In 2D, volume = area
                    smoothness = area / max(hull_area, 1)  # Closer to 1 = smoother
                    smoothness_scores.append(smoothness)
            except Exception:
                pass

    mean_pa_ratio = np.mean(perimeter_area_ratios) if perimeter_area_ratios else 0.0
    mean_smoothness = np.mean(smoothness_scores) if smoothness_scores else 0.5
    mean_area = np.mean(areas) if areas else 0.0

    return {
        'perimeter_area_ratio': mean_pa_ratio,
        'num_contours': total_contours,
        'smoothness': mean_smoothness,
        'mean_area': mean_area
    }


def generate_cppn_sample(size: int = 32, seed: int = None) -> np.ndarray:
    """Generate a CPPN image."""
    if seed is not None:
        set_global_seed(seed)
    cppn = CPPN()
    return cppn.render(size)


def generate_random_sample(size: int = 32, seed: int = None) -> np.ndarray:
    """Generate a uniform random binary image."""
    if seed is not None:
        np.random.seed(seed)
    return (np.random.random((size, size)) > 0.5).astype(np.uint8)


def run_experiment(n_samples: int = 200, size: int = 32, seed: int = 42):
    """
    Run contour complexity comparison experiment.
    """
    set_global_seed(seed)

    cppn_metrics = {'perimeter_area_ratio': [], 'num_contours': [], 'smoothness': [], 'mean_area': []}
    random_metrics = {'perimeter_area_ratio': [], 'num_contours': [], 'smoothness': [], 'mean_area': []}

    print(f"Generating {n_samples} CPPN and random samples...")

    for i in range(n_samples):
        # CPPN sample
        cppn_img = generate_cppn_sample(size, seed=seed + i)
        cppn_result = extract_contour_metrics(cppn_img)
        for k, v in cppn_result.items():
            cppn_metrics[k].append(v)

        # Random sample
        random_img = generate_random_sample(size, seed=seed + i + 10000)
        random_result = extract_contour_metrics(random_img)
        for k, v in random_result.items():
            random_metrics[k].append(v)

    print("\n" + "="*60)
    print("CONTOUR COMPLEXITY ANALYSIS RESULTS")
    print("="*60)

    results = {}

    for metric_name in ['perimeter_area_ratio', 'num_contours', 'smoothness', 'mean_area']:
        cppn_vals = np.array(cppn_metrics[metric_name])
        random_vals = np.array(random_metrics[metric_name])

        # Remove any NaN/inf values
        cppn_vals = cppn_vals[np.isfinite(cppn_vals)]
        random_vals = random_vals[np.isfinite(random_vals)]

        if len(cppn_vals) < 10 or len(random_vals) < 10:
            print(f"\n{metric_name}: Insufficient valid samples")
            continue

        # Statistical test
        t_stat, p_value = stats.ttest_ind(cppn_vals, random_vals)

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(cppn_vals) + np.var(random_vals)) / 2)
        cohens_d = (np.mean(cppn_vals) - np.mean(random_vals)) / (pooled_std + 1e-10)

        # Mann-Whitney U test (non-parametric)
        u_stat, mw_p = stats.mannwhitneyu(cppn_vals, random_vals, alternative='two-sided')

        results[metric_name] = {
            'cppn_mean': np.mean(cppn_vals),
            'cppn_std': np.std(cppn_vals),
            'random_mean': np.mean(random_vals),
            'random_std': np.std(random_vals),
            't_stat': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'mw_p': mw_p
        }

        print(f"\n{metric_name}:")
        print(f"  CPPN:   {np.mean(cppn_vals):.4f} +/- {np.std(cppn_vals):.4f}")
        print(f"  Random: {np.mean(random_vals):.4f} +/- {np.std(random_vals):.4f}")
        print(f"  Effect size (Cohen's d): {cohens_d:.4f}")
        print(f"  t-test p-value: {p_value:.2e}")
        print(f"  Mann-Whitney p-value: {mw_p:.2e}")

    # Primary hypothesis: perimeter/area ratio
    pa_result = results.get('perimeter_area_ratio', {})

    print("\n" + "="*60)
    print("PRIMARY HYPOTHESIS EVALUATION")
    print("="*60)

    if pa_result:
        hypothesis_supported = (
            pa_result['cppn_mean'] < pa_result['random_mean'] and
            pa_result['p_value'] < 0.01 and
            abs(pa_result['cohens_d']) > 0.5
        )

        print(f"\nHypothesis: CPPN images have LOWER perimeter/area ratio (smoother contours)")
        print(f"CPPN mean: {pa_result['cppn_mean']:.4f}")
        print(f"Random mean: {pa_result['random_mean']:.4f}")
        print(f"Direction: {'CPPN < Random (as predicted)' if pa_result['cppn_mean'] < pa_result['random_mean'] else 'CPPN >= Random (opposite)'}")
        print(f"Effect size: {pa_result['cohens_d']:.4f} (threshold: |d| > 0.5)")
        print(f"P-value: {pa_result['p_value']:.2e} (threshold: p < 0.01)")
        print(f"\nVERDICT: {'VALIDATED' if hypothesis_supported else 'REFUTED'}")

        return {
            'status': 'validated' if hypothesis_supported else 'refuted',
            'primary_metric': 'perimeter_area_ratio',
            'effect_size': pa_result['cohens_d'],
            'p_value': pa_result['p_value'],
            'all_results': results
        }
    else:
        print("ERROR: Could not compute perimeter/area ratio")
        return {'status': 'inconclusive'}


if __name__ == "__main__":
    result = run_experiment(n_samples=200, size=32, seed=42)
    print(f"\n\nFinal status: {result['status']}")
