"""
Weight Initialization Experiment for CPPN

Hypothesis: Different weight initialization schemes (uniform, normal, Xavier, He, sparse)
produce systematically different order value distributions in CPPN-generated images.

Builds on RES-021: conn/bias ratio matters for order. This experiment tests whether
initialization strategy (beyond just ratio) affects the prior over orders.

Initialization Schemes:
1. Normal (baseline): N(0, 1) - current default
2. Uniform: U(-sqrt(3), sqrt(3)) - same variance as N(0,1)
3. Xavier/Glorot: N(0, sqrt(2/(fan_in+fan_out))) - for balanced gradients
4. He/Kaiming: N(0, sqrt(2/fan_in)) - optimized for ReLU (not used here but common)
5. Sparse: 50% zeros, rest N(0, sqrt(2)) to maintain variance
6. LeCun: N(0, sqrt(1/fan_in)) - for sigmoid activations
7. Orthogonal-like: QR decomposition of random matrix (approximate for CPPN)
8. Small: N(0, 0.1) - very small initialization
9. Large: N(0, 3.0) - large initialization
10. Bias-heavy: Small connections, large biases (inverse of high conn/bias ratio)

Statistical Tests:
- Kruskal-Wallis H-test for overall difference
- Mann-Whitney U pairwise comparisons with Bonferroni correction
- Cohen's d effect sizes
"""

import numpy as np
import sys
import os
import json
from datetime import datetime
from scipy import stats
from dataclasses import dataclass
from typing import Callable, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.thermo_sampler_v3 import (
    CPPN, Node, Connection, ACTIVATIONS, PRIOR_SIGMA,
    order_multiplicative, compute_edge_density, compute_spectral_coherence,
    compute_symmetry, compute_compressibility
)

# ============================================================================
# INITIALIZATION SCHEMES
# ============================================================================

def init_normal(n_weights: int, n_biases: int) -> tuple[np.ndarray, np.ndarray]:
    """Standard normal N(0, 1) - current baseline."""
    weights = np.random.randn(n_weights) * PRIOR_SIGMA
    biases = np.random.randn(n_biases) * PRIOR_SIGMA
    return weights, biases

def init_uniform(n_weights: int, n_biases: int) -> tuple[np.ndarray, np.ndarray]:
    """Uniform U(-sqrt(3), sqrt(3)) - same variance as N(0,1)."""
    scale = np.sqrt(3) * PRIOR_SIGMA
    weights = np.random.uniform(-scale, scale, n_weights)
    biases = np.random.uniform(-scale, scale, n_biases)
    return weights, biases

def init_xavier(n_weights: int, n_biases: int, fan_in: int = 4, fan_out: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Xavier/Glorot: N(0, sqrt(2/(fan_in+fan_out)))."""
    std = np.sqrt(2.0 / (fan_in + fan_out))
    weights = np.random.randn(n_weights) * std
    biases = np.random.randn(n_biases) * std
    return weights, biases

def init_he(n_weights: int, n_biases: int, fan_in: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """He/Kaiming: N(0, sqrt(2/fan_in))."""
    std = np.sqrt(2.0 / fan_in)
    weights = np.random.randn(n_weights) * std
    biases = np.random.randn(n_biases) * std
    return weights, biases

def init_lecun(n_weights: int, n_biases: int, fan_in: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """LeCun: N(0, sqrt(1/fan_in)) - good for sigmoid."""
    std = np.sqrt(1.0 / fan_in)
    weights = np.random.randn(n_weights) * std
    biases = np.random.randn(n_biases) * std
    return weights, biases

def init_sparse(n_weights: int, n_biases: int, sparsity: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Sparse: 50% zeros, rest scaled to maintain variance."""
    # Scale non-zero elements to maintain E[x^2] = 1
    # If P(zero)=0.5, non-zero variance needs to be 2 to get E[x^2]=1
    scale = np.sqrt(2.0) * PRIOR_SIGMA
    weights = np.random.randn(n_weights) * scale
    weights[np.random.rand(n_weights) < sparsity] = 0

    biases = np.random.randn(n_biases) * scale
    biases[np.random.rand(n_biases) < sparsity] = 0
    return weights, biases

def init_small(n_weights: int, n_biases: int) -> tuple[np.ndarray, np.ndarray]:
    """Small initialization: N(0, 0.1)."""
    std = 0.1
    weights = np.random.randn(n_weights) * std
    biases = np.random.randn(n_biases) * std
    return weights, biases

def init_large(n_weights: int, n_biases: int) -> tuple[np.ndarray, np.ndarray]:
    """Large initialization: N(0, 3.0)."""
    std = 3.0
    weights = np.random.randn(n_weights) * std
    biases = np.random.randn(n_biases) * std
    return weights, biases

def init_bias_heavy(n_weights: int, n_biases: int) -> tuple[np.ndarray, np.ndarray]:
    """Bias-heavy: small connections, large biases (inverse of RES-021 finding)."""
    weights = np.random.randn(n_weights) * 0.3  # Small weights
    biases = np.random.randn(n_biases) * 2.0   # Large biases
    return weights, biases

def init_connection_heavy(n_weights: int, n_biases: int) -> tuple[np.ndarray, np.ndarray]:
    """Connection-heavy: large connections, small biases (aligned with RES-021)."""
    weights = np.random.randn(n_weights) * 2.0   # Large weights
    biases = np.random.randn(n_biases) * 0.3    # Small biases
    return weights, biases

INIT_SCHEMES = {
    'normal': init_normal,
    'uniform': init_uniform,
    'xavier': init_xavier,
    'he': init_he,
    'lecun': init_lecun,
    'sparse': init_sparse,
    'small': init_small,
    'large': init_large,
    'bias_heavy': init_bias_heavy,
    'connection_heavy': init_connection_heavy,
}

# ============================================================================
# CPPN WITH CUSTOM INITIALIZATION
# ============================================================================

def create_cppn_with_init(init_fn: Callable, fan_in: int = 4, fan_out: int = 1) -> CPPN:
    """Create a CPPN with custom weight initialization."""
    n_weights = 4  # 4 input connections to output
    n_biases = 1   # 1 output node bias

    # Get initialized values
    if init_fn in [init_xavier, init_he, init_lecun]:
        weights, biases = init_fn(n_weights, n_biases, fan_in)
    elif init_fn == init_sparse:
        weights, biases = init_fn(n_weights, n_biases, sparsity=0.5)
    else:
        weights, biases = init_fn(n_weights, n_biases)

    # Create CPPN manually
    nodes = [
        Node(0, 'identity', 0.0),
        Node(1, 'identity', 0.0),
        Node(2, 'identity', 0.0),
        Node(3, 'identity', 0.0),
        Node(4, 'sigmoid', biases[0]),  # Output node
    ]

    connections = [
        Connection(0, 4, weights[0]),
        Connection(1, 4, weights[1]),
        Connection(2, 4, weights[2]),
        Connection(3, 4, weights[3]),
    ]

    return CPPN(nodes=nodes, connections=connections)

# ============================================================================
# EXPERIMENT
# ============================================================================

def compute_features(img: np.ndarray) -> dict:
    """Compute all features for an image."""
    return {
        'order': order_multiplicative(img),
        'edge_density': compute_edge_density(img),
        'spectral_coherence': compute_spectral_coherence(img),
        'symmetry': compute_symmetry(img),
        'compressibility': compute_compressibility(img),
        'density': np.mean(img),
    }

def run_experiment(
    n_samples: int = 500,
    image_size: int = 32,
    seed: int = 42
) -> dict:
    """Run weight initialization comparison experiment."""

    np.random.seed(seed)

    print("=" * 70)
    print("WEIGHT INITIALIZATION EXPERIMENT")
    print("=" * 70)
    print(f"Samples per scheme: {n_samples}")
    print(f"Image size: {image_size}x{image_size}")
    print(f"Schemes tested: {list(INIT_SCHEMES.keys())}")
    print()

    # Collect data for each scheme
    results = {}
    all_orders = {}

    for scheme_name, init_fn in INIT_SCHEMES.items():
        print(f"Testing {scheme_name}...", end=" ", flush=True)

        orders = []
        features_list = []
        weight_stats = []

        for _ in range(n_samples):
            cppn = create_cppn_with_init(init_fn)
            img = cppn.render(image_size)
            features = compute_features(img)
            orders.append(features['order'])
            features_list.append(features)

            # Track weight statistics
            w = cppn.get_weights()
            conn_weights = w[:4]
            bias = w[4]
            weight_stats.append({
                'conn_l2': np.sqrt(np.sum(conn_weights**2)),
                'bias_l2': np.abs(bias),
                'conn_mean': np.mean(np.abs(conn_weights)),
                'ratio': np.sqrt(np.sum(conn_weights**2)) / (np.abs(bias) + 1e-10),
            })

        orders = np.array(orders)
        all_orders[scheme_name] = orders

        # Summary statistics
        results[scheme_name] = {
            'order_mean': float(np.mean(orders)),
            'order_std': float(np.std(orders)),
            'order_median': float(np.median(orders)),
            'order_q25': float(np.percentile(orders, 25)),
            'order_q75': float(np.percentile(orders, 75)),
            'edge_density_mean': float(np.mean([f['edge_density'] for f in features_list])),
            'spectral_coherence_mean': float(np.mean([f['spectral_coherence'] for f in features_list])),
            'conn_l2_mean': float(np.mean([ws['conn_l2'] for ws in weight_stats])),
            'bias_l2_mean': float(np.mean([ws['bias_l2'] for ws in weight_stats])),
            'ratio_mean': float(np.mean([ws['ratio'] for ws in weight_stats])),
        }

        print(f"order={results[scheme_name]['order_mean']:.4f} +/- {results[scheme_name]['order_std']:.4f}")

    print()

    # ===== STATISTICAL TESTS =====
    print("=" * 70)
    print("STATISTICAL ANALYSIS")
    print("=" * 70)

    # 1. Kruskal-Wallis H-test (non-parametric ANOVA)
    order_arrays = [all_orders[s] for s in INIT_SCHEMES.keys()]
    H_stat, kw_p = stats.kruskal(*order_arrays)
    print(f"\nKruskal-Wallis H-test:")
    print(f"  H = {H_stat:.2f}, p = {kw_p:.2e}")

    # 2. Pairwise Mann-Whitney U tests with Bonferroni correction
    scheme_names = list(INIT_SCHEMES.keys())
    n_pairs = len(scheme_names) * (len(scheme_names) - 1) // 2
    bonferroni_alpha = 0.01 / n_pairs

    pairwise_results = []
    print(f"\nPairwise Mann-Whitney U tests (Bonferroni alpha = {bonferroni_alpha:.4f}):")

    for i in range(len(scheme_names)):
        for j in range(i + 1, len(scheme_names)):
            s1, s2 = scheme_names[i], scheme_names[j]
            o1, o2 = all_orders[s1], all_orders[s2]

            U, p = stats.mannwhitneyu(o1, o2, alternative='two-sided')

            # Cohen's d effect size
            pooled_std = np.sqrt((np.var(o1) + np.var(o2)) / 2)
            d = (np.mean(o1) - np.mean(o2)) / (pooled_std + 1e-10)

            significant = p < bonferroni_alpha

            pairwise_results.append({
                'pair': f"{s1}_vs_{s2}",
                'scheme1': s1,
                'scheme2': s2,
                'U': float(U),
                'p_value': float(p),
                'cohens_d': float(d),
                'significant': bool(significant),
            })

            if significant or abs(d) > 0.5:
                print(f"  {s1} vs {s2}: p={p:.2e}, d={d:.2f} {'***' if significant else ''}")

    # 3. Effect size for normal (baseline) vs each other scheme
    baseline_comparisons = []
    print("\nComparison to baseline (normal):")
    baseline = all_orders['normal']

    for scheme_name, orders in all_orders.items():
        if scheme_name == 'normal':
            continue

        U, p = stats.mannwhitneyu(baseline, orders, alternative='two-sided')
        pooled_std = np.sqrt((np.var(baseline) + np.var(orders)) / 2)
        d = (np.mean(baseline) - np.mean(orders)) / (pooled_std + 1e-10)

        baseline_comparisons.append({
            'scheme': scheme_name,
            'U': float(U),
            'p_value': float(p),
            'cohens_d': float(d),
            'significant': bool(p < 0.01),
        })

        print(f"  normal vs {scheme_name}: mean_diff={np.mean(baseline) - np.mean(orders):.4f}, d={d:.2f}, p={p:.2e}")

    # ===== KEY FINDINGS =====
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    # Rank schemes by mean order
    sorted_schemes = sorted(results.items(), key=lambda x: x[1]['order_mean'], reverse=True)
    print("\nSchemes ranked by mean order:")
    for i, (name, data) in enumerate(sorted_schemes, 1):
        print(f"  {i}. {name}: {data['order_mean']:.4f} +/- {data['order_std']:.4f}")

    best_scheme = sorted_schemes[0][0]
    worst_scheme = sorted_schemes[-1][0]

    # Effect size between best and worst
    d_extremes = (np.mean(all_orders[best_scheme]) - np.mean(all_orders[worst_scheme])) / \
                 np.sqrt((np.var(all_orders[best_scheme]) + np.var(all_orders[worst_scheme])) / 2)

    print(f"\nBest scheme: {best_scheme} (order={results[best_scheme]['order_mean']:.4f})")
    print(f"Worst scheme: {worst_scheme} (order={results[worst_scheme]['order_mean']:.4f})")
    print(f"Effect size (best vs worst): d = {d_extremes:.2f}")

    # Count significant pairs
    n_significant = sum(1 for pr in pairwise_results if pr['significant'])
    n_large_effect = sum(1 for pr in pairwise_results if abs(pr['cohens_d']) > 0.5)

    print(f"\nSignificant pairwise comparisons: {n_significant}/{n_pairs}")
    print(f"Large effect sizes (|d| > 0.5): {n_large_effect}/{n_pairs}")

    # Test hypothesis: Does conn/bias ratio predict order across schemes?
    scheme_ratios = [(name, results[name]['ratio_mean'], results[name]['order_mean'])
                     for name in scheme_names]
    ratios = [r for _, r, _ in scheme_ratios]
    orders = [o for _, _, o in scheme_ratios]

    rho, rho_p = stats.spearmanr(ratios, orders)
    print(f"\nConnection/bias ratio vs order (across schemes):")
    print(f"  Spearman rho = {rho:.3f}, p = {rho_p:.4f}")

    # ===== COMPILE FINAL RESULTS =====
    final_results = {
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'n_samples': n_samples,
            'image_size': image_size,
            'seed': seed,
            'n_schemes': len(INIT_SCHEMES),
        },
        'scheme_results': results,
        'statistical_tests': {
            'kruskal_wallis_H': float(H_stat),
            'kruskal_wallis_p': float(kw_p),
            'n_pairwise_tests': n_pairs,
            'bonferroni_alpha': float(bonferroni_alpha),
            'n_significant_pairs': n_significant,
            'n_large_effect_pairs': n_large_effect,
        },
        'baseline_comparisons': baseline_comparisons,
        'pairwise_results': pairwise_results,
        'ratio_order_correlation': {
            'spearman_rho': float(rho),
            'p_value': float(rho_p),
        },
        'summary': {
            'best_scheme': best_scheme,
            'worst_scheme': worst_scheme,
            'best_order_mean': float(results[best_scheme]['order_mean']),
            'worst_order_mean': float(results[worst_scheme]['order_mean']),
            'd_best_vs_worst': float(d_extremes),
            'overall_significant': bool(kw_p < 0.01),
        }
    }

    # Determine validation status
    if kw_p < 0.01 and abs(d_extremes) > 0.5:
        final_results['validation_status'] = 'validated'
        final_results['validation_reason'] = f'Significant scheme effect (H={H_stat:.1f}, p<0.01) with large effect size (d={d_extremes:.2f})'
    elif kw_p < 0.01:
        final_results['validation_status'] = 'validated'
        final_results['validation_reason'] = f'Significant scheme effect (H={H_stat:.1f}, p<0.01) but small effect size (d={d_extremes:.2f})'
    else:
        final_results['validation_status'] = 'refuted'
        final_results['validation_reason'] = f'No significant scheme effect (p={kw_p:.4f})'

    return final_results


if __name__ == "__main__":
    # Run experiment
    results = run_experiment(n_samples=500, image_size=32, seed=42)

    # Save results
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                              'results', 'weight_initialization')
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, 'initialization_experiment_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    print(f"\nVALIDATION STATUS: {results['validation_status'].upper()}")
    print(f"Reason: {results['validation_reason']}")
