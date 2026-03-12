#!/usr/bin/env python3
"""
RES-016: Reconstruction Quality Experiment

HYPOTHESIS: Initial image features predict reconstruction difficulty beyond order alone

BUILDS ON: RES-007 (feature correlations), RES-002 (scaling laws)

NOVELTY: Tests whether initial image features (spectral coherence, edge density,
entropy) predict bits-needed-for-reconstruction independently of order.

NULL HYPOTHESIS: Initial features do not predict bits-to-threshold beyond initial order.

METHOD:
1. Generate 100 high-order CPPN images
2. For each image, measure initial features:
   - Spectral coherence
   - Edge density
   - Entropy
3. Measure reconstruction difficulty:
   - Bits required to reconstruct from low-order features
4. Test if initial features (beyond order) predict bits needed
5. Use partial correlation: correlation(features, bits | order)
6. Compute effect size and p-value

SUCCESS CRITERIA: p < 0.01, effect size > 0.5
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
        'initial_spectral_coherence': compute_spectral_coherence(img),
        'initial_edge_density': compute_edge_density(img),
        'initial_entropy': float(-np.sum(np.where(img > 0, img, 1) * np.log2(np.where(img > 0, img, 1) + 1e-10))),
        'initial_compressibility': compute_compressibility(img),
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


def partial_correlation(y, x, z):
    """
    Partial correlation between y and x, controlling for z.

    Returns: correlation coefficient, p-value
    """
    # Regress out z from both y and x
    y_resid = y - np.polyval(np.polyfit(z, y, 1), z)
    x_resid = x - np.polyval(np.polyfit(z, x, 1), z)
    return pearsonr(y_resid, x_resid)


def run_experiment():
    """
    Main experiment: Test if initial features predict bits-to-threshold
    beyond initial order alone using partial correlations.
    """
    print("=" * 70)
    print("RES-016: RECONSTRUCTION QUALITY")
    print("=" * 70)
    print()
    print("H0: Initial features do not predict bits-to-threshold beyond initial order")
    print("H1: Features add predictive power beyond order")
    print()
    print("Method: Partial correlations controlling for initial order")
    print()

    # Parameters
    n_seeds = 100  # Generate 100 high-order images
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

    for i, seed in enumerate(range(2000, 2000 + n_seeds)):
        # Extract initial features
        features = extract_initial_features(seed, image_size)

        # Run nested sampling
        ns_result = run_nested_sampling_for_seed(
            seed, threshold, n_live, n_iterations, image_size
        )

        # Combine
        record = {**features, **ns_result}
        data.append(record)

        if (i + 1) % 20 == 0:
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
    initial_entropy = np.array([d['initial_entropy'] for d in data_reached])
    initial_compress = np.array([d['initial_compressibility'] for d in data_reached])

    print("\n" + "-" * 60)
    print("PARTIAL CORRELATIONS (controlling for initial_order)")
    print("-" * 60)
    print(f"{'Feature':<30} {'Partial r':<12} {'p-value':<12}")
    print("-" * 54)

    # Compute partial correlations for each feature
    partial_corrs = {}
    features_list = [
        ('spectral_coherence', initial_spectral),
        ('edge_density', initial_edge),
        ('entropy', initial_entropy),
        ('compressibility', initial_compress),
    ]

    for name, values in features_list:
        r, p = partial_correlation(y, values, initial_order)
        partial_corrs[name] = {'r': float(r), 'p': float(p)}
        sig = "*" if p < 0.01 else ""
        print(f"{name:<30} {r:>+.4f}      {p:<.6f} {sig}")

    # Test if all features together predict beyond order (F-test)
    print("\n" + "-" * 60)
    print("JOINT TEST: Do all features together predict beyond order?")
    print("-" * 60)

    # Standardize for regression
    def standardize(x):
        return (x - np.mean(x)) / (np.std(x) + 1e-10)

    y_std = standardize(y)
    order_std = standardize(initial_order)
    spectral_std = standardize(initial_spectral)
    edge_std = standardize(initial_edge)
    entropy_std = standardize(initial_entropy)
    compress_std = standardize(initial_compress)

    # Build design matrices
    intercept = np.ones(n_reached)

    # Reduced model: bits ~ intercept + order
    X_reduced = np.column_stack([intercept, order_std])

    # Full model: bits ~ intercept + order + spectral + edge + entropy + compress
    X_full = np.column_stack([intercept, order_std, spectral_std, edge_std, entropy_std, compress_std])

    # Fit both models
    n = len(y_std)
    k_reduced = X_reduced.shape[1]
    k_full = X_full.shape[1]

    # Reduced model
    XtX_r = X_reduced.T @ X_reduced
    Xty_r = X_reduced.T @ y_std
    beta_r = np.linalg.lstsq(XtX_r, Xty_r, rcond=None)[0]
    y_hat_r = X_reduced @ beta_r
    SS_res_r = np.sum((y_std - y_hat_r) ** 2)

    # Full model
    XtX_f = X_full.T @ X_full
    Xty_f = X_full.T @ y_std
    beta_f = np.linalg.lstsq(XtX_f, Xty_f, rcond=None)[0]
    y_hat_f = X_full @ beta_f
    SS_res_f = np.sum((y_std - y_hat_f) ** 2)

    # F-test
    df_diff = k_full - k_reduced
    df_res_full = n - k_full
    F_stat = ((SS_res_r - SS_res_f) / df_diff) / (SS_res_f / df_res_full)
    p_value = 1 - f_dist.cdf(F_stat, df_diff, df_res_full)

    # Effect size f^2
    SS_tot = np.sum((y_std - np.mean(y_std)) ** 2)
    R2_r = 1 - SS_res_r / SS_tot
    R2_f = 1 - SS_res_f / SS_tot
    f2 = (R2_f - R2_r) / (1 - R2_f) if R2_f < 1 else 0

    print(f"\nReduced model (order only):  R^2 = {R2_r:.4f}")
    print(f"Full model (+ features):     R^2 = {R2_f:.4f}")
    print(f"R^2 increase:                     {R2_f - R2_r:.4f}")
    print()
    print(f"F-statistic: {F_stat:.3f}")
    print(f"df: ({df_diff}, {df_res_full})")
    print(f"p-value: {p_value:.6f}")
    print(f"Effect size f^2: {f2:.4f}")

    # Interpret effect size
    if f2 >= 0.35:
        effect_interp = "large"
    elif f2 >= 0.15:
        effect_interp = "medium"
    elif f2 >= 0.02:
        effect_interp = "small"
    else:
        effect_interp = "negligible"

    print(f"Effect size interpretation: {effect_interp}")

    # Verdict
    print("\n" + "-" * 60)
    print("VERDICT")
    print("-" * 60)

    if p_value < 0.01 and f2 > 0.15:
        status = "validated"
        print("VALIDATED: Initial features significantly predict bits-to-threshold")
        print(f"           beyond initial order (p={p_value:.4f}, f^2={f2:.3f})")
    elif p_value < 0.05:
        status = "inconclusive"
        print("INCONCLUSIVE: Marginally significant but doesn't meet strict criteria")
        print(f"              p={p_value:.4f} (need <0.01), f^2={f2:.3f}")
    else:
        status = "refuted"
        print("REFUTED: Initial features do not significantly predict beyond order")
        print(f"         p={p_value:.4f}, f^2={f2:.3f}")

    # Save results
    results = {
        'hypothesis': 'Initial image features predict reconstruction difficulty beyond order alone',
        'null_hypothesis': 'Initial features do not predict bits-to-threshold beyond initial order',
        'n_seeds': n_seeds,
        'n_reached_threshold': n_reached,
        'threshold': threshold,
        'joint_test': {
            'R2_reduced': float(R2_r),
            'R2_full': float(R2_f),
            'R2_increase': float(R2_f - R2_r),
            'F_stat': float(F_stat),
            'df': [int(df_diff), int(df_res_full)],
            'p_value': float(p_value),
            'f2_effect_size': float(f2),
        },
        'partial_correlations': partial_corrs,
        'features': {
            'bits_statistics': {
                'mean': float(np.mean(y)),
                'std': float(np.std(y)),
                'min': float(np.min(y)),
                'max': float(np.max(y)),
            },
            'order_statistics': {
                'mean': float(np.mean(initial_order)),
                'std': float(np.std(initial_order)),
            },
        },
        'status': status,
    }

    # Save to file
    output_dir = Path("results/reconstruction_quality")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}/results.json")

    return results


if __name__ == "__main__":
    results = run_experiment()
