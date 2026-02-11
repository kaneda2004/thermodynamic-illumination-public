#!/usr/bin/env python3
"""
RES-013: Reconstruction Predictability Experiment

HYPOTHESIS: Initial image features (spectral coherence, edge density, symmetry)
predict reconstruction difficulty (bits-to-threshold) beyond initial order alone.

BUILDS ON: RES-007 (feature correlations), RES-002 (scaling laws)

NOVELTY: RES-007 measured cross-sectional correlation between features and order.
This measures PREDICTIVE power of initial features for convergence rate.

NULL HYPOTHESIS: Initial features do not predict bits-to-threshold beyond initial order.

METHOD:
1. Run nested sampling with many different seeds
2. Record initial features at iteration 0 and bits-to-threshold
3. Multiple regression: bits ~ initial_order + initial_spectral + initial_edge + initial_symmetry
4. Test if additional predictors improve R^2 (F-test for nested models)

SUCCESS CRITERIA: p < 0.01, effect size f^2 > 0.15
"""

import sys
import os
import numpy as np
from scipy.stats import spearmanr, pearsonr, f as f_dist
from pathlib import Path
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.thermo_sampler_v3 import (
    nested_sampling_v3,
    order_multiplicative,
    compute_symmetry,
    compute_edge_density,
    compute_spectral_coherence,
    compute_compressibility,
    CPPN,
)


def extract_initial_features(seed: int, image_size: int = 32) -> dict:
    """Extract features from a CPPN's initial render (no optimization)."""
    np.random.seed(seed)
    cppn = CPPN()
    img = cppn.render(image_size)

    return {
        'seed': seed,
        'initial_order': order_multiplicative(img),
        'initial_symmetry': compute_symmetry(img),
        'initial_edge_density': compute_edge_density(img),
        'initial_spectral_coherence': compute_spectral_coherence(img),
        'initial_compressibility': compute_compressibility(img),
        'initial_density': np.mean(img),
    }


def run_nested_sampling_for_seed(seed: int, threshold: float,
                                  n_live: int, n_iterations: int,
                                  image_size: int) -> dict:
    """Run nested sampling and measure bits to reach threshold."""
    dead_points, live_final, _ = nested_sampling_v3(
        n_live=n_live,
        n_iterations=n_iterations,
        image_size=image_size,
        order_fn=order_multiplicative,
        seed=seed
    )

    orders = np.array([d.order_value for d in dead_points])
    log_X = np.array([d.log_X for d in dead_points])
    bits = -log_X / np.log(2)

    # Find bits at threshold
    idx = np.searchsorted(orders, threshold)
    if idx < len(bits):
        bits_to_threshold = bits[idx]
        reached = True
        final_order = orders[-1] if len(orders) > 0 else 0
    else:
        bits_to_threshold = bits[-1] if len(bits) > 0 else np.nan
        reached = False
        final_order = orders[-1] if len(orders) > 0 else 0

    return {
        'bits_to_threshold': float(bits_to_threshold),
        'reached_threshold': reached,
        'final_order': float(final_order),
        'max_bits': float(bits[-1]) if len(bits) > 0 else 0,
    }


def multiple_regression(y, X):
    """
    Perform OLS regression: y = X @ beta + epsilon.

    X should include a column of 1s for intercept if desired.
    Returns: beta, residuals, R^2, adjusted R^2, F-statistic, p-value
    """
    n = len(y)
    k = X.shape[1]  # Number of predictors (including intercept)

    # OLS: beta = (X'X)^-1 X'y
    XtX = X.T @ X
    Xty = X.T @ y
    beta = np.linalg.lstsq(XtX, Xty, rcond=None)[0]

    # Predictions and residuals
    y_hat = X @ beta
    residuals = y - y_hat

    # R^2
    SS_res = np.sum(residuals ** 2)
    SS_tot = np.sum((y - np.mean(y)) ** 2)
    R2 = 1 - SS_res / SS_tot if SS_tot > 0 else 0

    # Adjusted R^2
    R2_adj = 1 - (1 - R2) * (n - 1) / (n - k) if n > k else 0

    # F-statistic for overall regression
    SS_reg = SS_tot - SS_res
    df_reg = k - 1  # -1 for intercept
    df_res = n - k
    MS_reg = SS_reg / df_reg if df_reg > 0 else 0
    MS_res = SS_res / df_res if df_res > 0 else 1e-10
    F_stat = MS_reg / MS_res
    p_value = 1 - f_dist.cdf(F_stat, df_reg, df_res) if df_reg > 0 and df_res > 0 else 1.0

    return {
        'beta': beta,
        'R2': R2,
        'R2_adj': R2_adj,
        'F_stat': F_stat,
        'p_value': p_value,
        'df_reg': df_reg,
        'df_res': df_res,
        'SS_res': SS_res,
        'SS_tot': SS_tot,
    }


def compare_nested_models(y, X_reduced, X_full):
    """
    F-test comparing nested models.
    H0: The additional predictors in X_full have no effect.

    Returns F-statistic, p-value, and effect size f^2.
    """
    n = len(y)
    k_reduced = X_reduced.shape[1]
    k_full = X_full.shape[1]

    # Fit both models
    res_reduced = multiple_regression(y, X_reduced)
    res_full = multiple_regression(y, X_full)

    # F-test for nested models
    SS_res_reduced = res_reduced['SS_res']
    SS_res_full = res_full['SS_res']
    df_diff = k_full - k_reduced
    df_res_full = n - k_full

    if df_diff <= 0 or df_res_full <= 0:
        return {'F_stat': 0, 'p_value': 1.0, 'f2': 0}

    F_stat = ((SS_res_reduced - SS_res_full) / df_diff) / (SS_res_full / df_res_full)
    p_value = 1 - f_dist.cdf(F_stat, df_diff, df_res_full)

    # Effect size f^2 = (R2_full - R2_reduced) / (1 - R2_full)
    R2_diff = res_full['R2'] - res_reduced['R2']
    f2 = R2_diff / (1 - res_full['R2']) if res_full['R2'] < 1 else 0

    return {
        'F_stat': F_stat,
        'p_value': p_value,
        'f2': f2,
        'R2_reduced': res_reduced['R2'],
        'R2_full': res_full['R2'],
        'R2_increase': R2_diff,
        'df_diff': df_diff,
        'df_res': df_res_full,
    }


def run_experiment():
    """
    Main experiment: Test if initial features predict bits-to-threshold
    beyond initial order alone.
    """
    print("=" * 70)
    print("RES-013: RECONSTRUCTION PREDICTABILITY")
    print("=" * 70)
    print()
    print("H0: Initial features do not predict bits-to-threshold beyond initial order")
    print("H1: Features (spectral, edge, symmetry) add predictive power")
    print()
    print("Success criteria: p < 0.01, f^2 > 0.15")
    print()

    # Parameters
    n_seeds = 80  # Number of different CPPN initializations
    threshold = 0.15  # Order threshold to reach
    n_live = 50
    n_iterations = 800
    image_size = 32

    print(f"Parameters: n_seeds={n_seeds}, threshold={threshold}")
    print(f"            n_live={n_live}, n_iterations={n_iterations}, size={image_size}")
    print()

    # Collect data
    print("Running nested sampling for each seed...")
    data = []

    for i, seed in enumerate(range(1000, 1000 + n_seeds)):
        # Extract initial features
        features = extract_initial_features(seed, image_size)

        # Run nested sampling
        ns_result = run_nested_sampling_for_seed(
            seed, threshold, n_live, n_iterations, image_size
        )

        # Combine
        record = {**features, **ns_result}
        data.append(record)

        if (i + 1) % 10 == 0:
            print(f"  Completed {i+1}/{n_seeds} seeds")

    # Filter to seeds that reached threshold
    data_reached = [d for d in data if d['reached_threshold']]
    n_reached = len(data_reached)

    print(f"\n{n_reached}/{n_seeds} seeds reached threshold")

    if n_reached < 20:
        print("ERROR: Insufficient data (need at least 20 seeds reaching threshold)")
        return {'error': 'insufficient_data', 'n_reached': n_reached}

    # Prepare arrays
    y = np.array([d['bits_to_threshold'] for d in data_reached])
    initial_order = np.array([d['initial_order'] for d in data_reached])
    initial_spectral = np.array([d['initial_spectral_coherence'] for d in data_reached])
    initial_edge = np.array([d['initial_edge_density'] for d in data_reached])
    initial_symmetry = np.array([d['initial_symmetry'] for d in data_reached])
    initial_compress = np.array([d['initial_compressibility'] for d in data_reached])

    # Standardize predictors for interpretability
    def standardize(x):
        return (x - np.mean(x)) / (np.std(x) + 1e-10)

    y_std = standardize(y)
    order_std = standardize(initial_order)
    spectral_std = standardize(initial_spectral)
    edge_std = standardize(initial_edge)
    symmetry_std = standardize(initial_symmetry)
    compress_std = standardize(initial_compress)

    # Build design matrices
    intercept = np.ones(n_reached)

    # Reduced model: bits ~ intercept + initial_order
    X_reduced = np.column_stack([intercept, order_std])

    # Full model: bits ~ intercept + initial_order + spectral + edge + symmetry
    X_full = np.column_stack([intercept, order_std, spectral_std, edge_std, symmetry_std])

    # Run nested model comparison
    print("\n" + "-" * 60)
    print("MODEL COMPARISON (Nested F-test)")
    print("-" * 60)

    comparison = compare_nested_models(y_std, X_reduced, X_full)

    print(f"\nReduced model (order only):  R^2 = {comparison['R2_reduced']:.4f}")
    print(f"Full model (+ features):     R^2 = {comparison['R2_full']:.4f}")
    print(f"R^2 increase:                     {comparison['R2_increase']:.4f}")
    print()
    print(f"F-statistic: {comparison['F_stat']:.3f}")
    print(f"df: ({comparison['df_diff']}, {comparison['df_res']})")
    print(f"p-value: {comparison['p_value']:.6f}")
    print(f"Effect size f^2: {comparison['f2']:.4f}")
    print()

    # Interpret effect size
    if comparison['f2'] >= 0.35:
        effect_interp = "large"
    elif comparison['f2'] >= 0.15:
        effect_interp = "medium"
    elif comparison['f2'] >= 0.02:
        effect_interp = "small"
    else:
        effect_interp = "negligible"

    print(f"Effect size interpretation: {effect_interp}")

    # Verdict
    print("\n" + "-" * 60)
    print("VERDICT")
    print("-" * 60)

    if comparison['p_value'] < 0.01 and comparison['f2'] > 0.15:
        status = "validated"
        print("VALIDATED: Initial features significantly predict bits-to-threshold")
        print(f"           beyond initial order (p={comparison['p_value']:.4f}, f^2={comparison['f2']:.3f})")
    elif comparison['p_value'] < 0.05:
        status = "inconclusive"
        print("INCONCLUSIVE: Marginally significant but doesn't meet strict criteria")
        print(f"              p={comparison['p_value']:.4f} (need <0.01), f^2={comparison['f2']:.3f}")
    else:
        status = "refuted"
        print("REFUTED: Initial features do not significantly improve prediction")
        print(f"         p={comparison['p_value']:.4f}, f^2={comparison['f2']:.3f}")

    # Additional analysis: individual correlations
    print("\n" + "-" * 60)
    print("INDIVIDUAL CORRELATIONS WITH BITS-TO-THRESHOLD")
    print("-" * 60)

    correlations = {}
    features_list = [
        ('initial_order', initial_order),
        ('initial_spectral_coherence', initial_spectral),
        ('initial_edge_density', initial_edge),
        ('initial_symmetry', initial_symmetry),
        ('initial_compressibility', initial_compress),
    ]

    print(f"{'Feature':<30} {'Spearman r':<12} {'p-value':<12}")
    print("-" * 54)

    for name, values in features_list:
        r, p = spearmanr(y, values)
        correlations[name] = {'r': float(r), 'p': float(p)}
        sig = "*" if p < 0.01 else ""
        print(f"{name:<30} {r:>+.4f}      {p:<.6f} {sig}")

    # Partial correlations (controlling for initial_order)
    print("\n" + "-" * 60)
    print("PARTIAL CORRELATIONS (controlling for initial_order)")
    print("-" * 60)

    def partial_correlation(y, x, z):
        """Partial correlation between y and x, controlling for z."""
        # Regress out z from both
        y_resid = y - np.polyval(np.polyfit(z, y, 1), z)
        x_resid = x - np.polyval(np.polyfit(z, x, 1), z)
        return pearsonr(y_resid, x_resid)

    partial_corrs = {}
    for name, values in features_list[1:]:  # Skip initial_order
        r, p = partial_correlation(y, values, initial_order)
        partial_corrs[name] = {'r': float(r), 'p': float(p)}
        sig = "*" if p < 0.01 else ""
        print(f"{name:<30} {r:>+.4f}      {p:<.6f} {sig}")

    # Save results
    results = {
        'hypothesis': 'Initial image features predict reconstruction difficulty beyond initial order',
        'null_hypothesis': 'Initial features do not predict bits-to-threshold beyond initial order',
        'n_seeds': n_seeds,
        'n_reached_threshold': n_reached,
        'threshold': threshold,
        'model_comparison': {
            'R2_reduced': float(comparison['R2_reduced']),
            'R2_full': float(comparison['R2_full']),
            'R2_increase': float(comparison['R2_increase']),
            'F_stat': float(comparison['F_stat']),
            'df': [int(comparison['df_diff']), int(comparison['df_res'])],
            'p_value': float(comparison['p_value']),
            'f2_effect_size': float(comparison['f2']),
        },
        'correlations': correlations,
        'partial_correlations': partial_corrs,
        'bits_statistics': {
            'mean': float(np.mean(y)),
            'std': float(np.std(y)),
            'min': float(np.min(y)),
            'max': float(np.max(y)),
        },
        'status': status,
    }

    # Save to file
    output_dir = Path("results/reconstruction_predictability")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "predictability_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}/predictability_results.json")

    return results


if __name__ == "__main__":
    results = run_experiment()
