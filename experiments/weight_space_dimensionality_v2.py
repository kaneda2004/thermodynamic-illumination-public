#!/usr/bin/env python3
"""
Weight Space Geometry Experiment v2 (RES-218)

REVISED HYPOTHESIS:
The POSTERIOR DISTRIBUTION of CPPN parameters (posterior w.r.t. the order metric)
varies in dimensionality: high-order posteriors are more concentrated (low-dim),
while low-order posteriors are more dispersed (high-dim).

KEY INSIGHT: We're testing whether the CONSTRAINT IMPOSED BY THE GOAL (high order)
creates a lower-dimensional parameter subspace than the less-constrained low-order
regime.

TEST DESIGN:
1. For each order level (0.0-1.0), use nested sampling to get a POSTERIOR
   of ~100-150 CPPN parameter samples that achieve that order level.

2. For each posterior, compute intrinsic dimensionality:
   - PCA method: How many principal components explain 90% variance?
   - MLE method: Levina & Bickel estimator on posterior samples

3. Compare dimensionalities:
   - Low-order posterior (0.0-0.4): should be HIGH-dim (many ways to achieve low structure)
   - High-order posterior (0.6-1.0): should be LOW-dim (few ways to achieve high order)
   - Transition at ~0.5: sharp drop in dimensionality?
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
# MLE DIMENSIONALITY ESTIMATOR
# ============================================================================

def estimate_intrinsic_dimension_mle(vectors: np.ndarray, k: int = 5) -> float:
    """Levina & Bickel MLE estimator for intrinsic dimensionality."""
    n = vectors.shape[0]

    if n < k + 1:
        return float(min(vectors.shape))

    from scipy.spatial.distance import pdist, squareform
    distances = squareform(pdist(vectors, metric='euclidean'))

    dimensions = []

    for i in range(n):
        dists = np.sort(distances[i, :])
        r_k = dists[k]
        r_1 = dists[1]

        if r_k > 0 and r_1 > 0:
            log_ratios = np.log(r_k / dists[1:k])
            d_hat = (k - 1) / np.mean(log_ratios)
            d_hat = np.clip(d_hat, 1.0, 20.0)
            dimensions.append(d_hat)

    return np.mean(dimensions) if dimensions else np.nan


def estimate_intrinsic_dimension_pca(vectors: np.ndarray, variance_threshold: float = 0.90) -> Tuple[float, List[float]]:
    """PCA-based intrinsic dimensionality."""
    if vectors.shape[0] < 2:
        return np.nan, []

    X = vectors - vectors.mean(axis=0)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    explained_variance = (S ** 2) / np.sum(S ** 2)
    cumsum_variance = np.cumsum(explained_variance)

    n_components = np.argmax(cumsum_variance >= variance_threshold) + 1
    n_components = max(1, min(n_components, vectors.shape[1]))

    return float(n_components), cumsum_variance.tolist()


# ============================================================================
# POSTERIOR SAMPLING VIA REJECTION SAMPLING + CONVERGENCE TRACKING
# ============================================================================

def sample_posterior_for_order(target_order: float, n_posterior_samples: int = 150,
                                proposal_samples: int = 5000) -> np.ndarray:
    """
    Sample from the posterior P(params | order = target_order).

    Uses rejection sampling: generate CPPNs, compute their order, keep those
    within a tolerance band around target_order.

    Args:
        target_order: target order level
        n_posterior_samples: desired number of posterior samples
        proposal_samples: max proposals to generate

    Returns:
        Array of shape (n_accepted, n_params) with posterior samples
    """
    posterior_samples = []
    tolerance = max(0.02, target_order * 0.1)  # Adaptive tolerance

    for attempt in range(proposal_samples):
        cppn = CPPN()
        img = cppn.render(size=32)
        order = order_multiplicative(img)

        if abs(order - target_order) < tolerance:
            w = cppn.get_weights()
            posterior_samples.append(w)

            if len(posterior_samples) >= n_posterior_samples:
                break

    if len(posterior_samples) < 10:
        # Fallback: relax tolerance
        print(f"    Warning: only {len(posterior_samples)} samples for order {target_order:.1f}, relaxing tolerance")
        tolerance *= 2
        for attempt in range(proposal_samples):
            if len(posterior_samples) >= n_posterior_samples:
                break
            cppn = CPPN()
            img = cppn.render(size=32)
            order = order_multiplicative(img)

            if abs(order - target_order) < tolerance:
                w = cppn.get_weights()
                posterior_samples.append(w)

    return np.array(posterior_samples) if posterior_samples else np.array([]).reshape(0, 5)


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def main():
    print("=" * 70)
    print("Weight Space Dimensionality Experiment v2 (RES-218)")
    print("Testing: Does posterior dimensionality vary with order level?")
    print("=" * 70)

    set_global_seed(42)

    output_dir = Path('/Users/matt/Development/monochrome_noise_converger/results/weight_space_dimensionality')
    output_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # STEP 1: Sample posteriors at each order level
    # ========================================================================
    print("\nSTEP 1: Sampling posteriors at each order level")
    print("-" * 70)

    order_levels = np.linspace(0.0, 1.0, 11)
    posteriors = {}
    posterior_stats = {}

    for order_level in order_levels:
        print(f"Order {order_level:.1f}: sampling posterior...", end=" ", flush=True)

        # Sample posterior for this order level
        posterior = sample_posterior_for_order(order_level, n_posterior_samples=150, proposal_samples=5000)

        if len(posterior) > 0:
            posteriors[order_level] = posterior
            # Verify that posterior actually has the right order
            orders = []
            for w in posterior[:20]:  # Check first 20
                cppn = CPPN()
                cppn.set_weights(w)
                img = cppn.render(size=32)
                orders.append(order_multiplicative(img))

            mean_order = np.mean(orders)
            std_order = np.std(orders)
            posterior_stats[order_level] = {
                'n_samples': len(posterior),
                'mean_order': float(mean_order),
                'std_order': float(std_order),
                'param_dim': posterior.shape[1]
            }
            print(f"OK ({len(posterior)} samples, mean_order={mean_order:.3f})")
        else:
            print(f"FAILED (no samples)")

    # ========================================================================
    # STEP 2: Compute dimensionality for each posterior
    # ========================================================================
    print("\nSTEP 2: Computing intrinsic dimensionality (PCA)")
    print("-" * 70)

    pca_results = {}

    for order_level, posterior in posteriors.items():
        dim_pca, var_explained = estimate_intrinsic_dimension_pca(posterior, variance_threshold=0.90)
        pca_results[order_level] = {
            'dim_pca': float(dim_pca),
            'variance_explained': var_explained
        }
        print(f"Order {order_level:.1f}: PCA dim (90% var) = {dim_pca:.2f} (n={len(posterior)} samples)")

    # ========================================================================
    # STEP 3: MLE dimensionality
    # ========================================================================
    print("\nSTEP 3: Computing intrinsic dimensionality (MLE)")
    print("-" * 70)

    mle_results = {}

    for order_level, posterior in posteriors.items():
        # Only compute MLE if we have enough samples
        if len(posterior) >= 10:
            k = min(5, len(posterior)//2)
            dim_mle = estimate_intrinsic_dimension_mle(posterior, k=k)
            mle_results[order_level] = {
                'dim_mle': float(dim_mle)
            }
            print(f"Order {order_level:.1f}: MLE dim = {dim_mle:.2f}")
        else:
            print(f"Order {order_level:.1f}: Skipped (n={len(posterior)} < 10)")

    # ========================================================================
    # STEP 4: Statistical analysis
    # ========================================================================
    print("\nSTEP 4: Statistical analysis")
    print("-" * 70)

    low_order_levels = [o for o in order_levels if o < 0.5 and o in pca_results]
    high_order_levels = [o for o in order_levels if o >= 0.5 and o in pca_results]

    # PCA analysis
    pca_low = [pca_results[o]['dim_pca'] for o in low_order_levels]
    pca_high = [pca_results[o]['dim_pca'] for o in high_order_levels]

    pca_mean_low = np.mean(pca_low)
    pca_mean_high = np.mean(pca_high)
    pca_std_low = np.std(pca_low)
    pca_std_high = np.std(pca_high)

    print(f"\nPCA Results:")
    print(f"  Low-order (< 0.5):   {pca_mean_low:.2f} ± {pca_std_low:.2f} (n={len(pca_low)})")
    print(f"  High-order (>= 0.5): {pca_mean_high:.2f} ± {pca_std_high:.2f} (n={len(pca_high)})")
    print(f"  Difference: {pca_mean_low - pca_mean_high:.2f}")
    print(f"  Hypothesis expects: low-order HIGH-dim, high-order LOW-dim")

    # MLE analysis
    mle_low = [mle_results[o]['dim_mle'] for o in low_order_levels if o in mle_results]
    mle_high = [mle_results[o]['dim_mle'] for o in high_order_levels if o in mle_results]

    mle_mean_low = np.mean(mle_low)
    mle_mean_high = np.mean(mle_high)
    mle_std_low = np.std(mle_low)
    mle_std_high = np.std(mle_high)

    print(f"\nMLE Results:")
    print(f"  Low-order (< 0.5):   {mle_mean_low:.2f} ± {mle_std_low:.2f}")
    print(f"  High-order (>= 0.5): {mle_mean_high:.2f} ± {mle_std_high:.2f}")
    print(f"  Difference: {mle_mean_low - mle_mean_high:.2f}")

    # T-tests
    from scipy import stats

    t_stat_pca, p_val_pca = stats.ttest_ind(pca_low, pca_high) if len(pca_low) > 1 and len(pca_high) > 1 else (np.nan, 1.0)
    t_stat_mle, p_val_mle = stats.ttest_ind(mle_low, mle_high) if len(mle_low) > 1 and len(mle_high) > 1 else (np.nan, 1.0)

    print(f"\nT-test (PCA): t={t_stat_pca:.3f}, p={p_val_pca:.4f}")
    print(f"T-test (MLE): t={t_stat_mle:.3f}, p={p_val_mle:.4f}")

    # Cohen's d
    def cohens_d(group1, group2):
        if len(group1) < 2 or len(group2) < 2:
            return 0
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
    # STEP 5: Kink analysis
    # ========================================================================
    print("\nSTEP 5: Kink analysis (piecewise linear fit)")
    print("-" * 70)

    order_vals = np.array(list(posteriors.keys()))
    pca_dims = np.array([pca_results[o]['dim_pca'] for o in order_vals])

    mask_low = order_vals < 0.5
    if np.sum(mask_low) >= 2:
        coeffs_low = np.polyfit(order_vals[mask_low], pca_dims[mask_low], 1)
        slope_low = coeffs_low[0]
    else:
        slope_low = np.nan

    mask_high = order_vals >= 0.5
    if np.sum(mask_high) >= 2:
        coeffs_high = np.polyfit(order_vals[mask_high], pca_dims[mask_high], 1)
        slope_high = coeffs_high[0]
    else:
        slope_high = np.nan

    print(f"  Slope (order < 0.5): {slope_low:.3f}")
    print(f"  Slope (order >= 0.5): {slope_high:.3f}")

    # ========================================================================
    # STEP 6: Save results
    # ========================================================================
    print("\nSTEP 6: Saving results")
    print("-" * 70)

    results = {
        'hypothesis': 'Posterior dimensionality: high-order LOW-dim, low-order HIGH-dim',
        'posterior_stats': {str(k): v for k, v in posterior_stats.items()},
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
        }
    }

    results_path = output_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {results_path}")

    # ========================================================================
    # STEP 7: Interpretation and verdict
    # ========================================================================
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    print(f"\nKey PCA Findings:")
    print(f"  - Low-order posterior dim: {pca_mean_low:.2f}")
    print(f"  - High-order posterior dim: {pca_mean_high:.2f}")
    print(f"  - Hypothesis expects: dim_low > dim_high (opposite)")
    print(f"  - ACTUAL trend: dim_low {'>' if pca_mean_low > pca_mean_high else '<'} dim_high")

    # Verdict logic
    if p_val_pca < 0.05:
        if pca_mean_low > pca_mean_high and d_pca > 0.5:
            print(f"\nHYPOTHESIS STATUS: VALIDATED")
            print(f"  - Significant difference: p={p_val_pca:.4f}")
            print(f"  - Correct direction (low-order HIGH-dim, high-order LOW-dim)")
            print(f"  - Effect size: d={d_pca:.3f} (strong)")
            verdict = "VALIDATED"
            effect_measure = abs(d_pca)
        else:
            print(f"\nHYPOTHESIS STATUS: REFUTED")
            print(f"  - Significant difference but opposite direction")
            print(f"  - p={p_val_pca:.4f}, d={d_pca:.3f}")
            verdict = "REFUTED"
            effect_measure = -abs(d_pca)
    else:
        print(f"\nHYPOTHESIS STATUS: INCONCLUSIVE")
        print(f"  - No significant difference (p={p_val_pca:.4f})")
        print(f"  - Effect size: d={d_pca:.3f} (small)")
        verdict = "INCONCLUSIVE"
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
