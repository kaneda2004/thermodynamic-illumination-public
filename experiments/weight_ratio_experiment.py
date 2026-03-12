#!/usr/bin/env python3
"""
EXPERIMENT: Connection-to-Bias Weight Ratio Analysis (RES-017)

HYPOTHESIS: High-order CPPN images emerge from weight configurations with
high connection-to-bias L2 norm ratio - strong connection weights relative
to output bias produce more structured patterns.

NULL HYPOTHESIS: The ratio of connection L2 to bias L2 is independent of
order level - connection-bias balance does not predict image structure.

METHOD:
1. Generate large sample of CPPNs (N=1000)
2. Compute connection L2, bias L2, and their ratio
3. Compute order for each CPPN
4. Test correlation between ratio and order
5. Bin by ratio quintiles and compare order distributions
6. Mann-Whitney comparing low-ratio vs high-ratio CPPNs

NOVELTY: Pilot experiment (weight_structure) found:
- L2 norm overall: not predictive (rho=0.016, p=0.62)
- Bias L2: negatively correlated with order (rho=-0.205, p<10^-10)
- Conn/Bias ratio: positively correlated (rho=+0.237, p<10^-13)

This refined experiment focuses on the ratio hypothesis with rigorous testing.

THEORETICAL MOTIVATION: In a minimal CPPN with 4 input connections and 1 output
bias, the output is: sigmoid(w_x*x + w_y*y + w_r*r + w_1*1 + b)
High connection weights relative to bias means the spatial coordinates (x,y,r)
dominate the output, creating spatially-varying patterns. High bias relative
to connections means the output is nearly constant (close to sigmoid(b)).

BUILDS ON: RES-015 (order sensitivity), pilot weight_structure analysis
"""

import sys
import os
import numpy as np
from scipy.stats import spearmanr, pearsonr, mannwhitneyu, kruskal
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.thermo_sampler_v3 import CPPN, order_multiplicative


def compute_weight_ratios(cppn: CPPN) -> dict:
    """
    Compute connection-to-bias weight ratios.

    Returns dictionary with:
    - conn_l2: L2 norm of connection weights
    - bias_l2: L2 norm of biases (abs value for single bias)
    - ratio_l2: conn_l2 / (bias_l2 + epsilon)
    - conn_l1: L1 norm of connection weights
    - bias_l1: L1 norm of biases
    - ratio_l1: conn_l1 / (bias_l1 + epsilon)
    - conn_mean_abs: mean absolute connection weight
    - bias_abs: absolute bias value
    - ratio_mean: conn_mean_abs / (bias_abs + epsilon)
    """
    weights = cppn.get_weights()

    # In baseline CPPN: 4 connections, 1 bias
    n_connections = len([c for c in cppn.connections if c.enabled])
    conn_weights = weights[:n_connections]
    biases = weights[n_connections:]

    eps = 0.01  # Regularization to avoid division by zero

    conn_l2 = np.linalg.norm(conn_weights)
    bias_l2 = np.linalg.norm(biases)
    ratio_l2 = conn_l2 / (bias_l2 + eps)

    conn_l1 = np.sum(np.abs(conn_weights))
    bias_l1 = np.sum(np.abs(biases))
    ratio_l1 = conn_l1 / (bias_l1 + eps)

    conn_mean_abs = np.mean(np.abs(conn_weights))
    bias_abs = np.abs(biases[0]) if len(biases) > 0 else 0
    ratio_mean = conn_mean_abs / (bias_abs + eps)

    return {
        'conn_l2': float(conn_l2),
        'bias_l2': float(bias_l2),
        'ratio_l2': float(ratio_l2),
        'conn_l1': float(conn_l1),
        'bias_l1': float(bias_l1),
        'ratio_l1': float(ratio_l1),
        'conn_mean_abs': float(conn_mean_abs),
        'bias_abs': float(bias_abs),
        'ratio_mean': float(ratio_mean),
        'bias_sign': float(np.sign(biases[0])) if len(biases) > 0 else 0
    }


def run_experiment(n_samples: int = 1000, image_size: int = 32, seed: int = 42):
    """
    Main experiment: analyze connection-to-bias ratio vs order level.
    """
    np.random.seed(seed)

    print("=" * 70)
    print("EXPERIMENT: Connection-to-Bias Weight Ratio Analysis")
    print("=" * 70)
    print()
    print("H0: Conn/Bias ratio is independent of order level")
    print("H1: High conn/bias ratio produces higher order images")
    print("    (strong connections relative to bias = more spatial structure)")
    print()

    # Collect data
    print(f"Generating {n_samples} CPPN samples...")

    data = {
        'order': [],
        'conn_l2': [],
        'bias_l2': [],
        'ratio_l2': [],
        'ratio_l1': [],
        'ratio_mean': [],
        'bias_sign': [],
        'bias_abs': []
    }

    for i in range(n_samples):
        if (i + 1) % 200 == 0:
            print(f"  Progress: {i+1}/{n_samples}")

        cppn = CPPN()
        img = cppn.render(image_size)
        order = order_multiplicative(img)

        ratios = compute_weight_ratios(cppn)

        data['order'].append(order)
        data['conn_l2'].append(ratios['conn_l2'])
        data['bias_l2'].append(ratios['bias_l2'])
        data['ratio_l2'].append(ratios['ratio_l2'])
        data['ratio_l1'].append(ratios['ratio_l1'])
        data['ratio_mean'].append(ratios['ratio_mean'])
        data['bias_sign'].append(ratios['bias_sign'])
        data['bias_abs'].append(ratios['bias_abs'])

    # Convert to arrays
    for k in data:
        data[k] = np.array(data[k])

    print(f"\nData collected: {len(data['order'])} samples")
    print(f"Order range: [{data['order'].min():.4f}, {data['order'].max():.4f}]")
    print(f"Ratio (L2) range: [{data['ratio_l2'].min():.4f}, {data['ratio_l2'].max():.4f}]")

    results = {
        'n_samples': n_samples,
        'order_stats': {
            'min': float(data['order'].min()),
            'max': float(data['order'].max()),
            'mean': float(data['order'].mean()),
            'std': float(data['order'].std())
        },
        'ratio_l2_stats': {
            'min': float(data['ratio_l2'].min()),
            'max': float(data['ratio_l2'].max()),
            'mean': float(data['ratio_l2'].mean()),
            'std': float(data['ratio_l2'].std())
        }
    }

    # =========================================================================
    # ANALYSIS 1: Primary correlation test
    # =========================================================================
    print("\n" + "-" * 60)
    print("ANALYSIS 1: Primary Correlation (Ratio vs Order)")
    print("-" * 60)

    r_spearman, p_spearman = spearmanr(data['ratio_l2'], data['order'])
    r_pearson, p_pearson = pearsonr(data['ratio_l2'], data['order'])

    print(f"\n  Spearman correlation (ratio_l2 vs order):")
    print(f"    rho = {r_spearman:+.4f}")
    print(f"    p-value = {p_spearman:.2e}")

    print(f"\n  Pearson correlation (ratio_l2 vs order):")
    print(f"    r = {r_pearson:+.4f}")
    print(f"    p-value = {p_pearson:.2e}")

    results['primary_correlation'] = {
        'spearman_rho': float(r_spearman),
        'spearman_p': float(p_spearman),
        'pearson_r': float(r_pearson),
        'pearson_p': float(p_pearson)
    }

    # Additional ratio metrics
    for metric in ['ratio_l1', 'ratio_mean']:
        r, p = spearmanr(data[metric], data['order'])
        print(f"\n  {metric} vs order: rho = {r:+.4f}, p = {p:.2e}")
        results[f'{metric}_correlation'] = {'rho': float(r), 'p': float(p)}

    # =========================================================================
    # ANALYSIS 2: Component analysis (conn and bias separately)
    # =========================================================================
    print("\n" + "-" * 60)
    print("ANALYSIS 2: Component Analysis")
    print("-" * 60)

    r_conn, p_conn = spearmanr(data['conn_l2'], data['order'])
    r_bias, p_bias = spearmanr(data['bias_l2'], data['order'])
    r_bias_abs, p_bias_abs = spearmanr(data['bias_abs'], data['order'])

    print(f"\n  Connection L2 vs order: rho = {r_conn:+.4f}, p = {p_conn:.2e}")
    print(f"  Bias L2 vs order:       rho = {r_bias:+.4f}, p = {p_bias:.2e}")
    print(f"  |Bias| vs order:        rho = {r_bias_abs:+.4f}, p = {p_bias_abs:.2e}")

    results['component_correlations'] = {
        'conn_l2': {'rho': float(r_conn), 'p': float(p_conn)},
        'bias_l2': {'rho': float(r_bias), 'p': float(p_bias)},
        'bias_abs': {'rho': float(r_bias_abs), 'p': float(p_bias_abs)}
    }

    # =========================================================================
    # ANALYSIS 3: Quintile analysis
    # =========================================================================
    print("\n" + "-" * 60)
    print("ANALYSIS 3: Order by Ratio Quintiles")
    print("-" * 60)

    ratio_percentiles = np.percentile(data['ratio_l2'], [0, 20, 40, 60, 80, 100])
    bin_labels = ['Q1 (lowest)', 'Q2', 'Q3', 'Q4', 'Q5 (highest)']

    binned_orders = []
    bin_stats = []

    for i in range(5):
        mask = (data['ratio_l2'] >= ratio_percentiles[i]) & \
               (data['ratio_l2'] < ratio_percentiles[i+1] + 0.001)
        orders_in_bin = data['order'][mask]

        binned_orders.append(orders_in_bin)
        stat = {
            'label': bin_labels[i],
            'ratio_range': [float(ratio_percentiles[i]), float(ratio_percentiles[i+1])],
            'n': len(orders_in_bin),
            'order_mean': float(np.mean(orders_in_bin)) if len(orders_in_bin) > 0 else 0,
            'order_std': float(np.std(orders_in_bin)) if len(orders_in_bin) > 0 else 0
        }
        bin_stats.append(stat)

        print(f"  {bin_labels[i]:15s}: ratio=[{ratio_percentiles[i]:.2f}, {ratio_percentiles[i+1]:.2f}], "
              f"n={stat['n']}, order={stat['order_mean']:.4f} +/- {stat['order_std']:.4f}")

    results['quintile_stats'] = bin_stats

    # Monotonic increase test
    order_means = [s['order_mean'] for s in bin_stats]
    monotonic_increases = sum(1 for i in range(4) if order_means[i+1] > order_means[i])
    print(f"\n  Monotonic increases: {monotonic_increases}/4")

    # =========================================================================
    # ANALYSIS 4: Low vs High ratio comparison (primary effect size test)
    # =========================================================================
    print("\n" + "-" * 60)
    print("ANALYSIS 4: Low vs High Ratio Comparison")
    print("-" * 60)

    # Use bottom 30% vs top 30% for cleaner separation
    low_threshold = np.percentile(data['ratio_l2'], 30)
    high_threshold = np.percentile(data['ratio_l2'], 70)

    low_mask = data['ratio_l2'] <= low_threshold
    high_mask = data['ratio_l2'] >= high_threshold

    low_orders = data['order'][low_mask]
    high_orders = data['order'][high_mask]

    n_low = len(low_orders)
    n_high = len(high_orders)

    print(f"\n  Low ratio (bottom 30%, ratio <= {low_threshold:.2f}): n = {n_low}")
    print(f"  High ratio (top 30%, ratio >= {high_threshold:.2f}): n = {n_high}")

    if n_low >= 50 and n_high >= 50:
        stat_mw, p_mw = mannwhitneyu(high_orders, low_orders, alternative='greater')
        pooled_std = np.sqrt((np.var(low_orders) + np.var(high_orders)) / 2)
        cohens_d = (np.mean(high_orders) - np.mean(low_orders)) / pooled_std if pooled_std > 0 else 0

        print(f"\n  Mann-Whitney U (high ratio > low ratio order):")
        print(f"    U = {stat_mw:.0f}")
        print(f"    p-value = {p_mw:.2e}")
        print(f"    Cohen's d = {cohens_d:.4f}")
        print(f"\n  Low ratio mean order:  {np.mean(low_orders):.4f} +/- {np.std(low_orders):.4f}")
        print(f"  High ratio mean order: {np.mean(high_orders):.4f} +/- {np.std(high_orders):.4f}")

        results['low_vs_high'] = {
            'low_threshold': float(low_threshold),
            'high_threshold': float(high_threshold),
            'n_low': n_low,
            'n_high': n_high,
            'mann_whitney_U': float(stat_mw),
            'p_value': float(p_mw),
            'cohens_d': float(cohens_d),
            'low_mean': float(np.mean(low_orders)),
            'low_std': float(np.std(low_orders)),
            'high_mean': float(np.mean(high_orders)),
            'high_std': float(np.std(high_orders))
        }

    # Kruskal-Wallis across all quintiles
    non_empty = [b for b in binned_orders if len(b) >= 5]
    if len(non_empty) >= 3:
        h_stat, p_kw = kruskal(*non_empty)
        print(f"\n  Kruskal-Wallis (order differs across ratio quintiles):")
        print(f"    H = {h_stat:.2f}, p = {p_kw:.2e}")
        results['kruskal_wallis'] = {'H': float(h_stat), 'p_value': float(p_kw)}

    # =========================================================================
    # ANALYSIS 5: Bias sign effect
    # =========================================================================
    print("\n" + "-" * 60)
    print("ANALYSIS 5: Bias Sign Effect")
    print("-" * 60)

    neg_bias_mask = data['bias_sign'] < 0
    pos_bias_mask = data['bias_sign'] > 0

    neg_orders = data['order'][neg_bias_mask]
    pos_orders = data['order'][pos_bias_mask]

    print(f"\n  Negative bias: n = {len(neg_orders)}, mean order = {np.mean(neg_orders):.4f}")
    print(f"  Positive bias: n = {len(pos_orders)}, mean order = {np.mean(pos_orders):.4f}")

    if len(neg_orders) >= 50 and len(pos_orders) >= 50:
        _, p_sign = mannwhitneyu(neg_orders, pos_orders, alternative='two-sided')
        pooled_std = np.sqrt((np.var(neg_orders) + np.var(pos_orders)) / 2)
        d_sign = (np.mean(neg_orders) - np.mean(pos_orders)) / pooled_std if pooled_std > 0 else 0
        print(f"  Mann-Whitney p-value: {p_sign:.2e}, Cohen's d: {d_sign:+.4f}")

        results['bias_sign_effect'] = {
            'neg_mean': float(np.mean(neg_orders)),
            'pos_mean': float(np.mean(pos_orders)),
            'p_value': float(p_sign),
            'cohens_d': float(d_sign)
        }

    # =========================================================================
    # SUMMARY AND STATUS DETERMINATION
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY:")
    print("=" * 70)

    # Primary success criteria:
    # 1. Spearman rho > 0.15 with p < 0.01 (weaker than 0.5 given we're looking at ratios)
    # 2. Mann-Whitney p < 0.01 with Cohen's d > 0.5

    primary_corr_sig = p_spearman < 0.01 and r_spearman > 0.15
    effect_size_sig = (results.get('low_vs_high', {}).get('p_value', 1) < 0.01 and
                       results.get('low_vs_high', {}).get('cohens_d', 0) > 0.5)

    if primary_corr_sig and effect_size_sig:
        status = 'validated'
        confidence = 'high'
        d = results['low_vs_high']['cohens_d']
        summary = (f"High conn/bias ratio predicts higher order images. "
                   f"Spearman rho={r_spearman:.3f} (p={p_spearman:.2e}), "
                   f"Cohen's d={d:.2f} for low vs high ratio comparison. "
                   f"Mechanism: strong connection weights relative to bias allow "
                   f"spatial coordinates to dominate output, creating structure.")

        print("\nRESULT: VALIDATED")
        print(f"  - Significant positive correlation: rho = {r_spearman:.3f}")
        print(f"  - Large effect size: d = {d:.2f}")
        print(f"  - High ratio CPPNs produce higher order images")

    elif primary_corr_sig:
        status = 'inconclusive'
        confidence = 'medium'
        d = results.get('low_vs_high', {}).get('cohens_d', 0)
        summary = (f"Significant correlation (rho={r_spearman:.3f}, p={p_spearman:.2e}) "
                   f"but effect size below threshold (d={d:.2f} < 0.5).")

        print("\nRESULT: INCONCLUSIVE")
        print(f"  - Significant correlation but weak effect size (d = {d:.2f})")

    else:
        status = 'refuted'
        confidence = 'high'
        summary = (f"Conn/bias ratio does not significantly predict order. "
                   f"rho={r_spearman:.3f}, p={p_spearman:.2e}.")

        print("\nRESULT: REFUTED")
        print(f"  - No significant relationship (rho = {r_spearman:.3f})")

    results['status'] = status
    results['confidence'] = confidence
    results['summary'] = summary

    # Save results
    output_dir = Path("results/weight_ratio")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "weight_ratio_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}/weight_ratio_results.json")

    return results


if __name__ == "__main__":
    results = run_experiment(n_samples=1000, image_size=32, seed=42)
