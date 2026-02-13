#!/usr/bin/env python3
"""
EXPERIMENT: Weight Structure Analysis (RES-017)

HYPOTHESIS: High-order CPPN images emerge from weight configurations with
intermediate L2 norm - neither too small (weak signal, near-random output)
nor too large (saturated activations, trivial patterns).

NULL HYPOTHESIS: Weight statistics (L2 norm, variance, sparsity) are
independent of order level - any weight configuration is equally likely
to produce high or low order images.

METHOD:
1. Generate large sample of CPPNs (N=1000)
2. Compute weight statistics: L2 norm, L1 norm, variance, max abs, sparsity
3. Compute order for each CPPN
4. Test: (a) correlation between weight stats and order
        (b) whether optimal order occurs at intermediate L2 norm
5. Separate analysis for connection weights vs biases

NOVELTY: RES-015 studied sensitivity to perturbations (dynamic stability).
This studies static weight properties - does weight space have structure
that predicts order? Different question: "what do high-order weights look like"
vs "how stable are high-order configurations".

BUILDS ON: RES-015 (order sensitivity)
"""

import sys
import os
import numpy as np
from scipy.stats import spearmanr, pearsonr, mannwhitneyu, kruskal
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.thermo_sampler_v3 import CPPN, order_multiplicative


def compute_weight_statistics(cppn: CPPN) -> dict:
    """
    Compute statistical properties of CPPN weights.

    Returns dictionary with:
    - l2_norm: Euclidean norm of all weights
    - l1_norm: Sum of absolute weights (sparsity proxy)
    - mean_abs: Mean absolute weight value
    - variance: Variance of weights
    - max_abs: Maximum absolute weight
    - sparsity: Fraction of weights with |w| < 0.1
    - kurtosis: Excess kurtosis (tailedness)
    - conn_l2: L2 norm of connection weights only
    - bias_l2: L2 norm of biases only
    """
    weights = cppn.get_weights()

    # Separate connection weights from biases
    # In baseline CPPN: 4 connections (input_ids to output), 1 bias (output node)
    n_connections = len([c for c in cppn.connections if c.enabled])
    conn_weights = weights[:n_connections]
    biases = weights[n_connections:]

    # Full weight vector statistics
    l2_norm = np.linalg.norm(weights)
    l1_norm = np.sum(np.abs(weights))
    mean_abs = np.mean(np.abs(weights))
    variance = np.var(weights)
    max_abs = np.max(np.abs(weights))
    sparsity = np.mean(np.abs(weights) < 0.1)

    # Kurtosis (excess kurtosis, 0 for normal)
    if len(weights) > 3 and np.std(weights) > 1e-10:
        kurtosis = ((weights - np.mean(weights))**4).mean() / (np.std(weights)**4) - 3
    else:
        kurtosis = 0.0

    # Separate statistics
    conn_l2 = np.linalg.norm(conn_weights) if len(conn_weights) > 0 else 0
    bias_l2 = np.linalg.norm(biases) if len(biases) > 0 else 0

    return {
        'l2_norm': float(l2_norm),
        'l1_norm': float(l1_norm),
        'mean_abs': float(mean_abs),
        'variance': float(variance),
        'max_abs': float(max_abs),
        'sparsity': float(sparsity),
        'kurtosis': float(kurtosis),
        'conn_l2': float(conn_l2),
        'bias_l2': float(bias_l2),
        'n_weights': len(weights),
        'n_connections': n_connections,
        'n_biases': len(biases)
    }


def run_experiment(n_samples: int = 1000, image_size: int = 32, seed: int = 42):
    """
    Main experiment: analyze weight structure vs order level.
    """
    np.random.seed(seed)

    print("=" * 70)
    print("EXPERIMENT: Weight Structure Analysis")
    print("=" * 70)
    print()
    print("H0: Weight statistics are independent of order level")
    print("H1: High-order CPPNs have characteristic weight structure")
    print("    (specifically, intermediate L2 norm - not too small, not too large)")
    print()

    # Collect data
    print(f"Generating {n_samples} CPPN samples and computing statistics...")

    data = {
        'order': [],
        'l2_norm': [],
        'l1_norm': [],
        'mean_abs': [],
        'variance': [],
        'max_abs': [],
        'sparsity': [],
        'kurtosis': [],
        'conn_l2': [],
        'bias_l2': []
    }

    for i in range(n_samples):
        if (i + 1) % 200 == 0:
            print(f"  Progress: {i+1}/{n_samples}")

        cppn = CPPN()
        img = cppn.render(image_size)
        order = order_multiplicative(img)

        stats = compute_weight_statistics(cppn)

        data['order'].append(order)
        for key in ['l2_norm', 'l1_norm', 'mean_abs', 'variance', 'max_abs',
                    'sparsity', 'kurtosis', 'conn_l2', 'bias_l2']:
            data[key].append(stats[key])

    # Convert to arrays
    for k in data:
        data[k] = np.array(data[k])

    print(f"\nData collected: {len(data['order'])} samples")
    print(f"Order range: [{data['order'].min():.4f}, {data['order'].max():.4f}]")
    print(f"L2 norm range: [{data['l2_norm'].min():.4f}, {data['l2_norm'].max():.4f}]")

    # Results dictionary
    results = {
        'n_samples': n_samples,
        'order_stats': {
            'min': float(data['order'].min()),
            'max': float(data['order'].max()),
            'mean': float(data['order'].mean()),
            'std': float(data['order'].std())
        },
        'l2_stats': {
            'min': float(data['l2_norm'].min()),
            'max': float(data['l2_norm'].max()),
            'mean': float(data['l2_norm'].mean()),
            'std': float(data['l2_norm'].std())
        }
    }

    # =========================================================================
    # ANALYSIS 1: Correlations between weight statistics and order
    # =========================================================================
    print("\n" + "-" * 60)
    print("ANALYSIS 1: Correlations (Weight Statistics vs Order)")
    print("-" * 60)

    correlations = {}
    weight_metrics = ['l2_norm', 'l1_norm', 'mean_abs', 'variance', 'max_abs',
                      'sparsity', 'kurtosis', 'conn_l2', 'bias_l2']

    for metric in weight_metrics:
        r_spearman, p_spearman = spearmanr(data[metric], data['order'])
        r_pearson, p_pearson = pearsonr(data[metric], data['order'])

        correlations[metric] = {
            'spearman_r': float(r_spearman),
            'spearman_p': float(p_spearman),
            'pearson_r': float(r_pearson),
            'pearson_p': float(p_pearson)
        }

        sig_marker = "**" if p_spearman < 0.01 and abs(r_spearman) > 0.15 else ""
        print(f"  {metric:12s}: rho={r_spearman:+.4f}, p={p_spearman:.2e} {sig_marker}")

    results['correlations'] = correlations

    # =========================================================================
    # ANALYSIS 2: Non-monotonic relationship - does order peak at intermediate L2?
    # =========================================================================
    print("\n" + "-" * 60)
    print("ANALYSIS 2: Order vs L2 Norm (binned analysis)")
    print("-" * 60)

    # Bin L2 norm into quintiles
    l2_percentiles = np.percentile(data['l2_norm'], [0, 20, 40, 60, 80, 100])
    bin_labels = ['Q1 (lowest)', 'Q2', 'Q3', 'Q4', 'Q5 (highest)']

    binned_orders = []
    bin_stats = []

    for i in range(5):
        mask = (data['l2_norm'] >= l2_percentiles[i]) & (data['l2_norm'] < l2_percentiles[i+1] + 0.001)
        orders_in_bin = data['order'][mask]
        l2_in_bin = data['l2_norm'][mask]

        binned_orders.append(orders_in_bin)
        stat = {
            'label': bin_labels[i],
            'l2_range': [float(l2_percentiles[i]), float(l2_percentiles[i+1])],
            'n': len(orders_in_bin),
            'order_mean': float(np.mean(orders_in_bin)) if len(orders_in_bin) > 0 else 0,
            'order_std': float(np.std(orders_in_bin)) if len(orders_in_bin) > 0 else 0,
            'l2_mean': float(np.mean(l2_in_bin)) if len(l2_in_bin) > 0 else 0
        }
        bin_stats.append(stat)

        print(f"  {bin_labels[i]:15s}: L2=[{l2_percentiles[i]:.2f}, {l2_percentiles[i+1]:.2f}], "
              f"n={stat['n']}, order_mean={stat['order_mean']:.4f}")

    results['l2_bins'] = bin_stats

    # Find which bin has highest mean order
    order_means = [s['order_mean'] for s in bin_stats]
    peak_bin = np.argmax(order_means)
    print(f"\n  Peak order occurs at: {bin_labels[peak_bin]} (mean order = {order_means[peak_bin]:.4f})")

    # Test if middle bins have higher order than extremes (non-monotonic hypothesis)
    extreme_bins = np.concatenate([binned_orders[0], binned_orders[4]])
    middle_bins = np.concatenate([binned_orders[1], binned_orders[2], binned_orders[3]])

    if len(extreme_bins) >= 20 and len(middle_bins) >= 20:
        stat_mw, p_mw = mannwhitneyu(middle_bins, extreme_bins, alternative='greater')
        pooled_std = np.sqrt((np.var(extreme_bins) + np.var(middle_bins)) / 2)
        cohens_d = (np.mean(middle_bins) - np.mean(extreme_bins)) / pooled_std if pooled_std > 0 else 0

        print(f"\n  Mann-Whitney (middle > extreme L2 norms):")
        print(f"    U = {stat_mw:.0f}, p = {p_mw:.2e}, Cohen's d = {cohens_d:.4f}")
        print(f"    Extreme bins mean order: {np.mean(extreme_bins):.4f}")
        print(f"    Middle bins mean order:  {np.mean(middle_bins):.4f}")

        results['nonmonotonic_test'] = {
            'mann_whitney_U': float(stat_mw),
            'p_value': float(p_mw),
            'cohens_d': float(cohens_d),
            'extreme_mean': float(np.mean(extreme_bins)),
            'middle_mean': float(np.mean(middle_bins)),
            'peak_bin': bin_labels[peak_bin]
        }
    else:
        results['nonmonotonic_test'] = {'error': 'insufficient samples'}

    # Kruskal-Wallis: do bins differ significantly?
    non_empty = [b for b in binned_orders if len(b) >= 5]
    if len(non_empty) >= 3:
        h_stat, p_kw = kruskal(*non_empty)
        print(f"\n  Kruskal-Wallis (order differs across L2 bins):")
        print(f"    H = {h_stat:.2f}, p = {p_kw:.2e}")
        results['kruskal_wallis_l2'] = {'H': float(h_stat), 'p_value': float(p_kw)}

    # =========================================================================
    # ANALYSIS 3: Low vs High order weight comparison
    # =========================================================================
    print("\n" + "-" * 60)
    print("ANALYSIS 3: Weight Statistics by Order Level")
    print("-" * 60)

    low_mask = data['order'] < 0.05
    high_mask = data['order'] >= 0.15

    n_low = np.sum(low_mask)
    n_high = np.sum(high_mask)

    print(f"  Low order (< 0.05): n = {n_low}")
    print(f"  High order (>= 0.15): n = {n_high}")

    if n_low >= 20 and n_high >= 20:
        print(f"\n  {'Metric':12s} | {'Low Mean':>10s} | {'High Mean':>10s} | {'Cohen d':>8s} | {'p-value':>10s}")
        print("  " + "-" * 60)

        order_comparison = {}

        for metric in ['l2_norm', 'l1_norm', 'variance', 'max_abs', 'sparsity']:
            low_vals = data[metric][low_mask]
            high_vals = data[metric][high_mask]

            pooled_std = np.sqrt((np.var(low_vals) + np.var(high_vals)) / 2)
            d = (np.mean(high_vals) - np.mean(low_vals)) / pooled_std if pooled_std > 0 else 0
            _, p = mannwhitneyu(high_vals, low_vals, alternative='two-sided')

            order_comparison[metric] = {
                'low_mean': float(np.mean(low_vals)),
                'high_mean': float(np.mean(high_vals)),
                'cohens_d': float(d),
                'p_value': float(p)
            }

            sig = "*" if p < 0.01 else ""
            print(f"  {metric:12s} | {np.mean(low_vals):10.4f} | {np.mean(high_vals):10.4f} | "
                  f"{d:+8.4f} | {p:10.2e} {sig}")

        results['order_comparison'] = order_comparison
    else:
        print("  Insufficient samples for comparison")
        results['order_comparison'] = {'error': 'insufficient samples'}

    # =========================================================================
    # ANALYSIS 4: Connection weights vs Biases
    # =========================================================================
    print("\n" + "-" * 60)
    print("ANALYSIS 4: Connection Weights vs Biases Contribution")
    print("-" * 60)

    # Which matters more for order?
    r_conn, p_conn = spearmanr(data['conn_l2'], data['order'])
    r_bias, p_bias = spearmanr(data['bias_l2'], data['order'])

    print(f"  Connection weights L2 vs order: rho = {r_conn:+.4f}, p = {p_conn:.2e}")
    print(f"  Bias L2 vs order:               rho = {r_bias:+.4f}, p = {p_bias:.2e}")

    # Ratio of conn to bias L2
    conn_ratio = data['conn_l2'] / (data['bias_l2'] + 1e-10)
    r_ratio, p_ratio = spearmanr(conn_ratio, data['order'])
    print(f"  Conn/Bias ratio vs order:       rho = {r_ratio:+.4f}, p = {p_ratio:.2e}")

    results['conn_vs_bias'] = {
        'conn_correlation': {'rho': float(r_conn), 'p': float(p_conn)},
        'bias_correlation': {'rho': float(r_bias), 'p': float(p_bias)},
        'ratio_correlation': {'rho': float(r_ratio), 'p': float(p_ratio)}
    }

    # =========================================================================
    # SUMMARY AND STATUS
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY:")
    print("=" * 70)

    # Primary hypothesis: non-monotonic relationship (intermediate L2 is best)
    nonmono_test = results.get('nonmonotonic_test', {})
    nonmono_validated = (nonmono_test.get('p_value', 1) < 0.01 and
                         nonmono_test.get('cohens_d', 0) > 0.5)

    # Alternative: any strong correlation
    best_corr = max(correlations.values(),
                    key=lambda x: abs(x['spearman_r']) if x['spearman_p'] < 0.01 else 0)
    best_metric = [k for k, v in correlations.items() if v == best_corr][0]
    strong_corr = abs(best_corr['spearman_r']) > 0.5 and best_corr['spearman_p'] < 0.01

    # Kruskal-Wallis significance
    kw_sig = results.get('kruskal_wallis_l2', {}).get('p_value', 1) < 0.01

    if nonmono_validated:
        status = 'validated'
        confidence = 'high'
        summary = (f"High-order CPPNs emerge from intermediate L2 norm configurations. "
                   f"Middle quintiles have significantly higher order than extremes "
                   f"(d={nonmono_test['cohens_d']:.2f}, p={nonmono_test['p_value']:.2e}). "
                   f"Peak order at {nonmono_test['peak_bin']}.")

        print("\nRESULT: VALIDATED")
        print(f"  - Middle L2 bins have higher order than extremes (d={nonmono_test['cohens_d']:.2f})")
        print(f"  - Peak order at {nonmono_test['peak_bin']}")

    elif strong_corr:
        status = 'validated'
        confidence = 'high'
        direction = 'positive' if best_corr['spearman_r'] > 0 else 'negative'
        summary = (f"Strong {direction} correlation between {best_metric} and order "
                   f"(rho={best_corr['spearman_r']:.3f}, p={best_corr['spearman_p']:.2e}). "
                   f"Weight structure predicts order level.")

        print("\nRESULT: VALIDATED (monotonic relationship)")
        print(f"  - Strong {direction} correlation: {best_metric} vs order")
        print(f"  - rho = {best_corr['spearman_r']:.3f}, p = {best_corr['spearman_p']:.2e}")

    elif kw_sig:
        status = 'inconclusive'
        confidence = 'medium'
        summary = (f"Order differs across L2 bins (Kruskal-Wallis p < 0.01) but "
                   f"effect sizes do not meet threshold (d < 0.5). Some structure exists.")

        print("\nRESULT: INCONCLUSIVE")
        print("  - Significant differences across bins but weak effect size")

    else:
        status = 'refuted'
        confidence = 'high'
        summary = (f"No significant relationship between weight statistics and order. "
                   f"Best correlation: {best_metric} (rho={best_corr['spearman_r']:.3f}, "
                   f"p={best_corr['spearman_p']:.2e}). Weight space appears unstructured "
                   f"with respect to order.")

        print("\nRESULT: REFUTED")
        print("  - No significant relationship between weight structure and order")

    results['status'] = status
    results['confidence'] = confidence
    results['summary'] = summary

    # Save results
    output_dir = Path("results/weight_structure")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "weight_structure_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}/weight_structure_results.json")

    return results


if __name__ == "__main__":
    results = run_experiment(n_samples=1000, image_size=32, seed=42)
