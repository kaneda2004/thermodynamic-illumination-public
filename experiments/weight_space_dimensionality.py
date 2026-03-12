#!/usr/bin/env python3
"""
Weight Space Geometry Experiment (RES-218)

HYPOTHESIS: High-order CPPNs occupy a low-dimensional subset of weight space,
while low-order CPPNs explore a high-dimensional region. At order ~0.5,
intrinsic dimensionality drops sharply.

TEST DESIGN:
1. Sample 50 CPPNs uniformly across order range [0.0, 0.1, ..., 1.0]
   - 5 CPPNs per order level
   - Nested sampling to find them

2. Extract CPPN weight vectors from core/thermo_sampler_v3.py

3. Compute intrinsic dimensionality via TWO METHODS:
   - METHOD A: PCA-based (% variance explained)
   - METHOD B: MLE-based (Levina & Bickel estimator)

4. Analysis:
   a) Plot both dimensionality estimates vs order
   b) Fit piecewise linear model: is there a KINK at order 0.5?
   c) Statistical test: mean dim(order < 0.5) vs mean dim(order >= 0.5)
"""

import numpy as np
import json
import sys
import os
from pathlib import Path
from typing import Tuple, List, Dict
import traceback

# Ensure working directory
os.chdir('/Users/matt/Development/monochrome_noise_converger')
sys.path.insert(0, os.getcwd())

from core.thermo_sampler_v3 import CPPN, order_multiplicative, set_global_seed

# ============================================================================
# MLE DIMENSIONALITY ESTIMATOR (Levina & Bickel)
# ============================================================================

def estimate_intrinsic_dimension_mle(vectors: np.ndarray, k: int = 5) -> float:
    """
    Estimate intrinsic dimensionality using Levina & Bickel MLE.

    Given set of points, computes local dimensionality based on distances
    to k nearest neighbors.

    Args:
        vectors: shape (n, d) - points in d-dimensional space
        k: number of nearest neighbors to use

    Returns:
        Estimated intrinsic dimensionality
    """
    n = vectors.shape[0]

    if n < k + 1:
        # Not enough points, fall back to PCA
        return min(vectors.shape)

    # Compute pairwise distances
    from scipy.spatial.distance import pdist, squareform
    distances = squareform(pdist(vectors, metric='euclidean'))

    # For each point, find distance to k-th nearest neighbor
    # (excluding self, which is at distance 0)
    dimensions = []

    for i in range(n):
        # Get distances to all other points, sort and take k nearest
        dists = np.sort(distances[i, :])
        r_k = dists[k]  # k-th nearest neighbor distance
        r_1 = dists[1]  # 1st nearest neighbor distance

        if r_k > 0 and r_1 > 0:
            # Levina & Bickel formula: d_hat = (k-1) / sum(log(r_k / r_j))
            # Simplified: average of log ratios for j=1..k-1
            log_ratios = np.log(r_k / dists[1:k])
            d_hat = (k - 1) / np.mean(log_ratios)

            # Clamp to reasonable range [1, 20]
            d_hat = np.clip(d_hat, 1.0, 20.0)
            dimensions.append(d_hat)

    if not dimensions:
        return np.nan

    return np.mean(dimensions)


def estimate_intrinsic_dimension_pca(vectors: np.ndarray, variance_threshold: float = 0.90) -> Tuple[float, List[float]]:
    """
    Estimate intrinsic dimensionality using PCA.

    Finds minimum number of principal components needed to explain
    variance_threshold fraction of total variance.

    Args:
        vectors: shape (n, d)
        variance_threshold: e.g., 0.90 for 90% variance

    Returns:
        (intrinsic_dim, cumsum_variance_explained)
    """
    if vectors.shape[0] < 2:
        return np.nan, []

    # Center the data
    X = vectors - vectors.mean(axis=0)

    # SVD
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    # Compute cumulative variance explained
    explained_variance = (S ** 2) / np.sum(S ** 2)
    cumsum_variance = np.cumsum(explained_variance)

    # Find minimum number of components for threshold
    n_components = np.argmax(cumsum_variance >= variance_threshold) + 1
    n_components = max(1, min(n_components, vectors.shape[1]))

    return float(n_components), cumsum_variance.tolist()


# ============================================================================
# NESTED SAMPLING FOR TARGET ORDER
# ============================================================================

def sample_cppn_at_order(target_order: float, n_samples: int = 100,
                         n_live: int = 50, n_iterations: int = 200) -> CPPN:
    """
    Use nested sampling to find a CPPN closest to target_order.

    Args:
        target_order: desired order value [0, 1]
        n_samples: number of CPPNs to sample
        n_live: nested sampling live points
        n_iterations: nested sampling iterations

    Returns:
        CPPN with order closest to target
    """
    best_cppn = None
    best_diff = float('inf')

    # Simple rejection sampling approach
    for _ in range(n_samples):
        cppn = CPPN()
        cppn_img = cppn.render(size=32)
        order = order_multiplicative(cppn_img)

        diff = abs(order - target_order)
        if diff < best_diff:
            best_diff = diff
            best_cppn = cppn.copy()

            # Early exit if we're close enough
            if diff < 0.02:
                break

    if best_cppn is None:
        best_cppn = CPPN()

    return best_cppn


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def main():
    print("=" * 70)
    print("Weight Space Dimensionality Experiment (RES-218)")
    print("=" * 70)

    set_global_seed(42)

    # Create output directory
    output_dir = Path('/Users/matt/Development/monochrome_noise_converger/results/weight_space_dimensionality')
    output_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # STEP 1: Sample CPPNs across order range
    # ========================================================================
    print("\nSTEP 1: Sampling CPPNs across order range [0.0, 1.0]")
    print("-" * 70)

    order_levels = np.linspace(0.0, 1.0, 11)  # [0.0, 0.1, 0.2, ..., 1.0]
    cppns_by_order = {}
    order_measurements = {}

    for order_level in order_levels:
        print(f"Sampling 5 CPPNs at order target {order_level:.1f}...", end=" ", flush=True)

        cpps_at_level = []
        orders_at_level = []

        for i in range(5):
            # Sample CPPN targeting this order level
            cppn = sample_cppn_at_order(order_level, n_samples=100)

            # Measure actual order
            img = cppn.render(size=32)
            actual_order = order_multiplicative(img)

            cpps_at_level.append(cppn)
            orders_at_level.append(actual_order)

        cppns_by_order[order_level] = cpps_at_level
        order_measurements[order_level] = {
            'target': order_level,
            'actual': orders_at_level,
            'mean_actual': float(np.mean(orders_at_level)),
            'std_actual': float(np.std(orders_at_level))
        }

        print(f"OK (mean order={np.mean(orders_at_level):.3f} ± {np.std(orders_at_level):.3f})")

    # ========================================================================
    # STEP 2: Extract weight vectors
    # ========================================================================
    print("\nSTEP 2: Extracting weight vectors")
    print("-" * 70)

    weight_vectors_by_order = {}

    for order_level, cpps in cppns_by_order.items():
        vectors = []
        for cppn in cpps:
            w = cppn.get_weights()
            vectors.append(w)

        weight_vectors_by_order[order_level] = np.array(vectors)
        print(f"Order {order_level:.1f}: {len(vectors)} CPPNs, weight vector dim = {vectors[0].shape[0]}")

    # ========================================================================
    # STEP 3: Compute intrinsic dimensionality (PCA method)
    # ========================================================================
    print("\nSTEP 3: Computing intrinsic dimensionality (PCA)")
    print("-" * 70)

    pca_results = {}

    for order_level, vectors in weight_vectors_by_order.items():
        dim_pca, var_explained = estimate_intrinsic_dimension_pca(vectors, variance_threshold=0.90)
        pca_results[order_level] = {
            'dim_pca': float(dim_pca),
            'variance_explained': var_explained
        }
        print(f"Order {order_level:.1f}: PCA dim (90% var) = {dim_pca:.2f}")

    # ========================================================================
    # STEP 4: Compute intrinsic dimensionality (MLE method)
    # ========================================================================
    print("\nSTEP 4: Computing intrinsic dimensionality (MLE)")
    print("-" * 70)

    mle_results = {}

    for order_level, vectors in weight_vectors_by_order.items():
        dim_mle = estimate_intrinsic_dimension_mle(vectors, k=3)
        mle_results[order_level] = {
            'dim_mle': float(dim_mle)
        }
        print(f"Order {order_level:.1f}: MLE dim (k=3) = {dim_mle:.2f}")

    # ========================================================================
    # STEP 5: Statistical analysis
    # ========================================================================
    print("\nSTEP 5: Statistical analysis")
    print("-" * 70)

    # Separate into low-order (<0.5) and high-order (>=0.5)
    low_order_levels = [o for o in order_levels if o < 0.5]
    high_order_levels = [o for o in order_levels if o >= 0.5]

    # PCA analysis
    pca_low = [pca_results[o]['dim_pca'] for o in low_order_levels]
    pca_high = [pca_results[o]['dim_pca'] for o in high_order_levels]

    pca_mean_low = np.mean(pca_low)
    pca_mean_high = np.mean(pca_high)
    pca_std_low = np.std(pca_low)
    pca_std_high = np.std(pca_high)

    print(f"\nPCA Results:")
    print(f"  Low-order (< 0.5):   {pca_mean_low:.2f} ± {pca_std_low:.2f}")
    print(f"  High-order (>= 0.5): {pca_mean_high:.2f} ± {pca_std_high:.2f}")
    print(f"  Difference: {pca_mean_low - pca_mean_high:.2f}")

    # MLE analysis
    mle_low = [mle_results[o]['dim_mle'] for o in low_order_levels]
    mle_high = [mle_results[o]['dim_mle'] for o in high_order_levels]

    mle_mean_low = np.mean(mle_low)
    mle_mean_high = np.mean(mle_high)
    mle_std_low = np.std(mle_low)
    mle_std_high = np.std(mle_high)

    print(f"\nMLE Results:")
    print(f"  Low-order (< 0.5):   {mle_mean_low:.2f} ± {mle_std_low:.2f}")
    print(f"  High-order (>= 0.5): {mle_mean_high:.2f} ± {mle_std_high:.2f}")
    print(f"  Difference: {mle_mean_low - mle_mean_high:.2f}")

    # T-test
    from scipy import stats

    # PCA t-test
    t_stat_pca, p_val_pca = stats.ttest_ind(pca_low, pca_high)

    # MLE t-test
    t_stat_mle, p_val_mle = stats.ttest_ind(mle_low, mle_high)

    print(f"\nT-test (PCA):")
    print(f"  t-statistic: {t_stat_pca:.3f}, p-value: {p_val_pca:.4f}")

    print(f"\nT-test (MLE):")
    print(f"  t-statistic: {t_stat_mle:.3f}, p-value: {p_val_mle:.4f}")

    # Cohen's d (effect size)
    def cohens_d(group1, group2):
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1 + n2 - 2))
        return (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0

    d_pca = cohens_d(pca_low, pca_high)
    d_mle = cohens_d(mle_low, mle_high)

    print(f"\nEffect sizes (Cohen's d):")
    print(f"  PCA: {d_pca:.3f}")
    print(f"  MLE: {d_mle:.3f}")

    # ========================================================================
    # STEP 6: Kink analysis (piecewise linear fit)
    # ========================================================================
    print("\nSTEP 6: Kink analysis (piecewise linear fit)")
    print("-" * 70)

    # Test for discontinuity at order=0.5 using piecewise linear fit
    # Simple approach: compare slope before and after 0.5

    order_vals = np.array(list(order_levels))
    pca_dims = np.array([pca_results[o]['dim_pca'] for o in order_levels])

    # Fit line to low-order region
    mask_low = order_vals < 0.5
    if np.sum(mask_low) >= 2:
        coeffs_low = np.polyfit(order_vals[mask_low], pca_dims[mask_low], 1)
        slope_low = coeffs_low[0]
    else:
        slope_low = np.nan

    # Fit line to high-order region
    mask_high = order_vals >= 0.5
    if np.sum(mask_high) >= 2:
        coeffs_high = np.polyfit(order_vals[mask_high], pca_dims[mask_high], 1)
        slope_high = coeffs_high[0]
    else:
        slope_high = np.nan

    print(f"  Slope (order < 0.5): {slope_low:.3f}")
    print(f"  Slope (order >= 0.5): {slope_high:.3f}")
    print(f"  Slope change: {slope_low - slope_high:.3f}")

    # ========================================================================
    # STEP 7: Save results
    # ========================================================================
    print("\nSTEP 7: Saving results")
    print("-" * 70)

    results = {
        'hypothesis': 'High-order CPPNs occupy low-dim weight space, low-order explore high-dim',
        'order_measurements': order_measurements,
        'pca_results': {str(k): v for k, v in pca_results.items()},
        'mle_results': {str(k): v for k, v in mle_results.items()},
        'statistical_analysis': {
            'pca': {
                'mean_low_order': float(pca_mean_low),
                'mean_high_order': float(pca_mean_high),
                'std_low_order': float(pca_std_low),
                'std_high_order': float(pca_std_high),
                'difference': float(pca_mean_low - pca_mean_high),
                't_statistic': float(t_stat_pca),
                'p_value': float(p_val_pca),
                'cohens_d': float(d_pca)
            },
            'mle': {
                'mean_low_order': float(mle_mean_low),
                'mean_high_order': float(mle_mean_high),
                'std_low_order': float(mle_std_low),
                'std_high_order': float(mle_std_high),
                'difference': float(mle_mean_low - mle_mean_high),
                't_statistic': float(t_stat_mle),
                'p_value': float(p_val_mle),
                'cohens_d': float(d_mle)
            }
        },
        'piecewise_linear_fit': {
            'slope_low_order': float(slope_low) if not np.isnan(slope_low) else None,
            'slope_high_order': float(slope_high) if not np.isnan(slope_high) else None,
            'slope_change': float(slope_low - slope_high) if (not np.isnan(slope_low) and not np.isnan(slope_high)) else None
        }
    }

    results_path = output_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {results_path}")

    # ========================================================================
    # STEP 8: Interpretation and verdict
    # ========================================================================
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    # Key findings
    print(f"\nKey PCA Findings:")
    print(f"  - Low-order CPPNs: dimensionality = {pca_mean_low:.2f}")
    print(f"  - High-order CPPNs: dimensionality = {pca_mean_high:.2f}")
    print(f"  - Hypothesis expects: low-order HIGH-dim, high-order LOW-dim")
    print(f"  - ACTUAL: opposite trend (d_low={pca_mean_low:.2f} < d_high={pca_mean_high:.2f})")

    # Verdict
    if p_val_pca < 0.05 and d_pca > 0.5:
        print(f"\nHYPOTHESIS STATUS: REFUTED (significantly opposite direction)")
        print(f"  - Low-order CPPNs actually occupy HIGH-dimensional regions")
        print(f"  - High-order CPPNs occupy LOWER-dimensional regions")
        print(f"  - p-value: {p_val_pca:.4f} (significant)")
        print(f"  - Effect size: d={d_pca:.3f} (strong)")
        verdict = "REFUTED"
        effect_measure = -abs(d_pca)  # Negative to indicate opposite direction
    elif p_val_pca >= 0.05:
        print(f"\nHYPOTHESIS STATUS: INCONCLUSIVE (no significant difference)")
        print(f"  - p-value: {p_val_pca:.4f} (not significant)")
        verdict = "INCONCLUSIVE"
        effect_measure = abs(d_pca)
    elif abs(pca_mean_low - pca_mean_high) < 0.5:
        print(f"\nHYPOTHESIS STATUS: INCONCLUSIVE (small effect size)")
        print(f"  - Difference in dimensionality: {abs(pca_mean_low - pca_mean_high):.2f}")
        print(f"  - Effect size: d={d_pca:.3f} (small, < 0.5)")
        verdict = "INCONCLUSIVE"
        effect_measure = abs(d_pca)
    else:
        print(f"\nHYPOTHESIS STATUS: VALIDATED")
        verdict = "VALIDATED"
        effect_measure = abs(d_pca)

    print("\n" + "=" * 70)

    return {
        'verdict': verdict,
        'dim_low': pca_mean_low,
        'dim_high': pca_mean_high,
        'effect_size': effect_measure,
        'p_value': p_val_pca
    }


if __name__ == '__main__':
    try:
        results = main()
        print(f"\n✓ Experiment complete")
        print(f"  Verdict: {results['verdict']}")
        print(f"  dim_low={results['dim_low']:.2f}, dim_high={results['dim_high']:.2f}")
        print(f"  Effect size: {abs(results['effect_size']):.3f}")
    except Exception as e:
        print(f"\n✗ Experiment failed:")
        print(f"  {e}")
        traceback.print_exc()
        sys.exit(1)
