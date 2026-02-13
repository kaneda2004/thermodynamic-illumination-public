#!/usr/bin/env python3
"""
Weight Space Geometry Experiment v3 (RES-218)

REFINED ANALYSIS:
The basic 5-parameter CPPN is too constrained to show rich weight space
geometry. However, we can still test the hypothesis by looking at how
the POSTERIOR DISTRIBUTION changes shape (not overall dimensionality).

KEY INSIGHT:
- At low order, the order metric is PERMISSIVE: many parameter
  combinations give low order, so the posterior is SPREAD OUT
- At higher order (but still reachable), the metric becomes more
  CONSTRAINING: fewer parameter combos achieve high order, so the
  posterior is MORE CONCENTRATED

Test: Compare CONCENTRATION (inverse of dimensionality) across order range.
- Concentration = explained variance by first 1 PC (should INCREASE with order)
- Alternative: ratio of largest eigenvalue to others (concentration measure)
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
# POSTERIOR SAMPLING
# ============================================================================

def sample_posterior_for_order(target_order: float, n_posterior_samples: int = 150,
                                proposal_samples: int = 5000) -> np.ndarray:
    """Sample from P(params | order = target_order) using rejection sampling."""
    posterior_samples = []
    tolerance = max(0.02, target_order * 0.1)

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
# CONCENTRATION METRICS
# ============================================================================

def compute_concentration_metrics(vectors: np.ndarray) -> Dict[str, float]:
    """
    Compute concentration metrics for a posterior:
    - first_pc_var: fraction of variance in first PC (higher = more concentrated)
    - eigenvalue_ratio: ratio of largest to 2nd largest eigenvalue (higher = more concentrated)
    - effective_dim: inverse of concentration (lower = more concentrated)
    """
    if vectors.shape[0] < 2:
        return {
            'first_pc_var': np.nan,
            'eigenvalue_ratio': np.nan,
            'effective_dim': np.nan,
            'n_components_90': np.nan
        }

    X = vectors - vectors.mean(axis=0)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    explained_variance = (S ** 2) / np.sum(S ** 2)
    cumsum_variance = np.cumsum(explained_variance)

    # Metric 1: First PC explains how much variance?
    first_pc_var = explained_variance[0] if len(explained_variance) > 0 else np.nan

    # Metric 2: Ratio of largest to 2nd largest eigenvalue
    eigenvalue_ratio = (explained_variance[0] / explained_variance[1]) if len(explained_variance) > 1 else np.nan

    # Metric 3: Number of components for 90% variance
    n_components_90 = np.argmax(cumsum_variance >= 0.90) + 1 if np.any(cumsum_variance >= 0.90) else len(explained_variance)

    # Metric 4: Effective dimensionality (Renyi entropy-based)
    # Smaller = more concentrated
    effective_dim = np.sum(explained_variance ** 2)
    if effective_dim > 0:
        effective_dim = 1.0 / effective_dim
    else:
        effective_dim = np.nan

    return {
        'first_pc_var': float(first_pc_var),
        'eigenvalue_ratio': float(eigenvalue_ratio),
        'effective_dim': float(effective_dim),
        'n_components_90': float(n_components_90),
        'all_explained_variance': explained_variance.tolist()
    }


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def main():
    print("=" * 70)
    print("Weight Space Dimensionality Experiment v3 (RES-218)")
    print("Testing: Does posterior CONCENTRATION increase with order?")
    print("=" * 70)

    set_global_seed(42)

    output_dir = Path('/Users/matt/Development/monochrome_noise_converger/results/weight_space_dimensionality')
    output_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # STEP 1: Sample posteriors
    # ========================================================================
    print("\nSTEP 1: Sampling posteriors at each order level")
    print("-" * 70)

    # Focus on achievable order range (0.0-0.5) where we get data
    order_levels = np.linspace(0.0, 0.5, 6)
    posteriors = {}
    posterior_stats = {}

    for order_level in order_levels:
        print(f"Order {order_level:.1f}: sampling posterior...", end=" ", flush=True)

        posterior = sample_posterior_for_order(order_level, n_posterior_samples=150, proposal_samples=5000)

        if len(posterior) > 0:
            posteriors[order_level] = posterior
            # Verify order
            orders = []
            for w in posterior[:20]:
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
            print(f"OK ({len(posterior)} samples)")
        else:
            print(f"FAILED")

    # ========================================================================
    # STEP 2: Compute concentration metrics
    # ========================================================================
    print("\nSTEP 2: Computing concentration metrics")
    print("-" * 70)

    concentration_results = {}

    for order_level, posterior in posteriors.items():
        metrics = compute_concentration_metrics(posterior)
        concentration_results[order_level] = metrics
        print(f"Order {order_level:.1f}: first_pc_var={metrics['first_pc_var']:.3f}, "
              f"eigval_ratio={metrics['eigenvalue_ratio']:.3f}, "
              f"eff_dim={metrics['effective_dim']:.2f}")

    # ========================================================================
    # STEP 3: Test if concentration increases with order
    # ========================================================================
    print("\nSTEP 3: Statistical analysis")
    print("-" * 70)

    order_vals = np.array(list(posteriors.keys()))
    first_pc_vars = np.array([concentration_results[o]['first_pc_var'] for o in order_vals])
    eigenval_ratios = np.array([concentration_results[o]['eigenvalue_ratio'] for o in order_vals])
    eff_dims = np.array([concentration_results[o]['effective_dim'] for o in order_vals])

    # Test correlation: does first_pc_var increase with order?
    from scipy import stats

    corr_pc, p_pc = stats.spearmanr(order_vals, first_pc_vars)
    corr_ratio, p_ratio = stats.spearmanr(order_vals, eigenval_ratios)
    corr_dim, p_dim = stats.spearmanr(order_vals, eff_dims)

    print(f"\nSpearman correlations with order level:")
    print(f"  First PC variance:  r={corr_pc:.3f}, p={p_pc:.4f}")
    print(f"    (HYPOTHESIS: positive r > 0.5 means posterior more concentrated at high order)")
    print(f"  Eigenvalue ratio:   r={corr_ratio:.3f}, p={p_ratio:.4f}")
    print(f"  Effective dimension: r={corr_dim:.3f}, p={p_dim:.4f}")
    print(f"    (HYPOTHESIS: negative r < -0.5 means smaller eff_dim at high order)")

    # Linear regression to measure effect size
    from scipy.stats import linregress

    slope_pc, intercept_pc, r_pc, p_val_pc, se_pc = linregress(order_vals, first_pc_vars)
    slope_ratio, intercept_ratio, r_ratio, p_val_ratio, se_ratio = linregress(order_vals, eigenval_ratios)
    slope_dim, intercept_dim, r_dim, p_val_dim, se_dim = linregress(order_vals, eff_dims)

    print(f"\nLinear regression on concentration metrics:")
    print(f"  First PC var vs order: slope={slope_pc:.4f}, p={p_val_pc:.4f}")
    print(f"  Eigenval ratio vs order: slope={slope_ratio:.4f}, p={p_val_ratio:.4f}")
    print(f"  Eff. dim vs order: slope={slope_dim:.4f}, p={p_val_dim:.4f}")

    # Expected direction for HYPOTHESIS:
    # If high-order posteriors are more CONCENTRATED (low-dim):
    # - first_pc_var should INCREASE with order (slope > 0)
    # - effective_dim should DECREASE with order (slope < 0)

    print(f"\nHYPOTHESIS PREDICTION:")
    print(f"  High-order posteriors should be MORE CONCENTRATED (constrained)")
    print(f"  → first_pc_var should INCREASE (slope > 0)")
    print(f"  → effective_dim should DECREASE (slope < 0)")

    # ========================================================================
    # STEP 4: Compute effect size
    # ========================================================================

    # Effect size: difference between low-order and high-order concentration
    low_order_indices = order_vals < 0.25
    high_order_indices = order_vals >= 0.25

    if np.sum(low_order_indices) > 0 and np.sum(high_order_indices) > 0:
        low_conc = first_pc_vars[low_order_indices]
        high_conc = first_pc_vars[high_order_indices]

        effect_size_pc = np.mean(high_conc) - np.mean(low_conc)

        low_dim = eff_dims[low_order_indices]
        high_dim = eff_dims[high_order_indices]

        effect_size_dim = np.mean(low_dim) - np.mean(high_dim)

        print(f"\nEffect sizes (high-order minus low-order):")
        print(f"  First PC var: {effect_size_pc:.4f}")
        print(f"    (positive = concentration increases, supporting hypothesis)")
        print(f"  Eff. dim: {effect_size_dim:.4f}")
        print(f"    (positive = eff_dim decreases, supporting hypothesis)")
    else:
        effect_size_pc = np.nan
        effect_size_dim = np.nan

    # ========================================================================
    # STEP 5: Save results
    # ========================================================================
    print("\nSTEP 5: Saving results")
    print("-" * 70)

    results = {
        'hypothesis': 'Higher-order posteriors are more concentrated (constrained to lower-dim subspace)',
        'test_logic': {
            'description': 'If high-order goal constrains parameter space, posterior should be more concentrated',
            'metric_1': 'first_pc_var should INCREASE with order (more variance in 1st PC)',
            'metric_2': 'effective_dim should DECREASE with order (fewer active dimensions)'
        },
        'order_range': f'{order_vals[0]:.1f} to {order_vals[-1]:.1f}',
        'posterior_stats': {str(k): v for k, v in posterior_stats.items()},
        'concentration_results': {str(k): v for k, v in concentration_results.items()},
        'correlations': {
            'first_pc_var': {
                'correlation': float(corr_pc),
                'p_value': float(p_pc),
                'slope': float(slope_pc)
            },
            'eigenvalue_ratio': {
                'correlation': float(corr_ratio),
                'p_value': float(p_ratio),
                'slope': float(slope_ratio)
            },
            'effective_dimension': {
                'correlation': float(corr_dim),
                'p_value': float(p_dim),
                'slope': float(slope_dim)
            }
        },
        'effect_sizes': {
            'first_pc_var_change': float(effect_size_pc) if not np.isnan(effect_size_pc) else None,
            'effective_dim_change': float(effect_size_dim) if not np.isnan(effect_size_dim) else None
        }
    }

    results_path = output_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {results_path}")

    # ========================================================================
    # STEP 6: Interpretation and verdict
    # ========================================================================
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    print(f"\nKey Findings:")
    print(f"  - First PC variance trend: {slope_pc:.4f} (slope w.r.t. order)")
    print(f"  - Expected (hypothesis true): slope > 0")
    print(f"  - P-value: {p_val_pc:.4f}")

    if slope_pc > 0 and p_val_pc < 0.05:
        print(f"\nHYPOTHESIS STATUS: VALIDATED")
        print(f"  - Posterior CONCENTRATION increases with order (significant)")
        print(f"  - High-order goals constrain parameter space to lower-dim subspace")
        verdict = "VALIDATED"
        effect = slope_pc
    elif slope_pc > 0:
        print(f"\nHYPOTHESIS STATUS: WEAKLY SUPPORTED")
        print(f"  - Posterior concentration increases with order (not significant)")
        print(f"  - Trend is in right direction but effect size is small")
        verdict = "INCONCLUSIVE"
        effect = slope_pc
    elif slope_pc < 0 and abs(p_val_pc) < 0.05:
        print(f"\nHYPOTHESIS STATUS: REFUTED")
        print(f"  - Posterior becomes LESS concentrated at high order (opposite!)")
        verdict = "REFUTED"
        effect = slope_pc
    else:
        print(f"\nHYPOTHESIS STATUS: INCONCLUSIVE")
        print(f"  - No clear pattern in concentration across order range")
        verdict = "INCONCLUSIVE"
        effect = slope_pc

    print("\n" + "=" * 70)

    return {
        'verdict': verdict,
        'effect_size': effect,
        'p_value': p_val_pc,
        'slope_first_pc_var': slope_pc
    }


if __name__ == '__main__':
    try:
        results = main()
        print(f"\n✓ Experiment complete")
        print(f"  Verdict: {results['verdict']}")
        print(f"  Slope (first PC var vs order): {results['slope_first_pc_var']:.4f}")
        print(f"  Effect size: {results['effect_size']:.4f}")
        print(f"  P-value: {results['p_value']:.4f}")
    except Exception as e:
        print(f"\n✗ Experiment failed:")
        print(f"  {e}")
        traceback.print_exc()
        sys.exit(1)
