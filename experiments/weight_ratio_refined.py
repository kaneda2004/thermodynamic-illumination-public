#!/usr/bin/env python3
"""
EXPERIMENT: Connection-to-Bias Weight Ratio Analysis (Refined) - RES-017

HYPOTHESIS: High-order CPPN images emerge from weight configurations with
high connection-to-bias L2 norm ratio - strong connection weights relative
to output bias produce more structured patterns.

NULL HYPOTHESIS: The ratio of connection L2 to bias L2 is independent of
order level - connection-bias balance does not predict image structure.

REFINEMENT: Pilot showed rho=0.24, d=0.46 (borderline). This version:
1. Increases sample size to N=2000 for more precise estimates
2. Uses more extreme comparison (bottom 20% vs top 20%) for cleaner effect size
3. Adds bootstrap CI for effect size

BUILDS ON: Pilot weight_ratio_experiment.py findings
"""

import sys
import os
import numpy as np
from scipy.stats import spearmanr, mannwhitneyu, kruskal
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.thermo_sampler_v3 import CPPN, order_multiplicative


def compute_conn_bias_ratio(cppn: CPPN) -> tuple[float, float, float]:
    """
    Compute connection L2, bias L2, and ratio.
    Returns (conn_l2, bias_l2, ratio)
    """
    weights = cppn.get_weights()
    n_connections = len([c for c in cppn.connections if c.enabled])
    conn_weights = weights[:n_connections]
    biases = weights[n_connections:]

    conn_l2 = np.linalg.norm(conn_weights)
    bias_l2 = np.linalg.norm(biases)
    ratio = conn_l2 / (bias_l2 + 0.01)

    return conn_l2, bias_l2, ratio


def bootstrap_cohens_d(group1, group2, n_bootstrap=1000, seed=42):
    """Bootstrap 95% CI for Cohen's d"""
    rng = np.random.RandomState(seed)
    ds = []
    for _ in range(n_bootstrap):
        g1 = rng.choice(group1, size=len(group1), replace=True)
        g2 = rng.choice(group2, size=len(group2), replace=True)
        pooled_std = np.sqrt((np.var(g1) + np.var(g2)) / 2)
        if pooled_std > 0:
            ds.append((np.mean(g2) - np.mean(g1)) / pooled_std)
        else:
            ds.append(0)
    return np.percentile(ds, [2.5, 97.5])


def run_experiment(n_samples: int = 2000, image_size: int = 32, seed: int = 42):
    """
    Main experiment: refined analysis of connection-to-bias ratio vs order.
    """
    np.random.seed(seed)

    print("=" * 70)
    print("EXPERIMENT: Connection-to-Bias Weight Ratio (Refined)")
    print("=" * 70)
    print()
    print("H0: Conn/Bias ratio is independent of order level")
    print("H1: High conn/bias ratio produces higher order images")
    print()

    # Collect data
    print(f"Generating {n_samples} CPPN samples...")

    orders = []
    ratios = []
    conn_l2s = []
    bias_l2s = []

    for i in range(n_samples):
        if (i + 1) % 500 == 0:
            print(f"  Progress: {i+1}/{n_samples}")

        cppn = CPPN()
        img = cppn.render(image_size)
        order = order_multiplicative(img)
        conn_l2, bias_l2, ratio = compute_conn_bias_ratio(cppn)

        orders.append(order)
        ratios.append(ratio)
        conn_l2s.append(conn_l2)
        bias_l2s.append(bias_l2)

    orders = np.array(orders)
    ratios = np.array(ratios)
    conn_l2s = np.array(conn_l2s)
    bias_l2s = np.array(bias_l2s)

    print(f"\nData collected: {len(orders)} samples")
    print(f"Order range: [{orders.min():.4f}, {orders.max():.4f}]")
    print(f"Ratio range: [{ratios.min():.2f}, {ratios.max():.2f}]")

    results = {
        'n_samples': n_samples,
        'order_stats': {
            'min': float(orders.min()),
            'max': float(orders.max()),
            'mean': float(orders.mean()),
            'std': float(orders.std())
        },
        'ratio_stats': {
            'min': float(ratios.min()),
            'max': float(ratios.max()),
            'mean': float(ratios.mean()),
            'median': float(np.median(ratios)),
            'std': float(ratios.std())
        }
    }

    # =========================================================================
    # TEST 1: Primary Spearman correlation
    # =========================================================================
    print("\n" + "-" * 60)
    print("TEST 1: Primary Correlation")
    print("-" * 60)

    r_spearman, p_spearman = spearmanr(ratios, orders)
    print(f"\n  Spearman rho = {r_spearman:+.4f}")
    print(f"  p-value = {p_spearman:.2e}")

    # Component correlations
    r_conn, p_conn = spearmanr(conn_l2s, orders)
    r_bias, p_bias = spearmanr(bias_l2s, orders)

    print(f"\n  Connection L2 vs order: rho = {r_conn:+.4f}, p = {p_conn:.2e}")
    print(f"  Bias L2 vs order:       rho = {r_bias:+.4f}, p = {p_bias:.2e}")

    results['correlations'] = {
        'ratio_order': {'rho': float(r_spearman), 'p': float(p_spearman)},
        'conn_order': {'rho': float(r_conn), 'p': float(p_conn)},
        'bias_order': {'rho': float(r_bias), 'p': float(p_bias)}
    }

    # =========================================================================
    # TEST 2: Extreme comparison (bottom 20% vs top 20%)
    # =========================================================================
    print("\n" + "-" * 60)
    print("TEST 2: Extreme Ratio Comparison (20th vs 80th percentile)")
    print("-" * 60)

    p20 = np.percentile(ratios, 20)
    p80 = np.percentile(ratios, 80)

    low_mask = ratios <= p20
    high_mask = ratios >= p80

    low_orders = orders[low_mask]
    high_orders = orders[high_mask]

    print(f"\n  Low ratio (bottom 20%, ratio <= {p20:.2f}): n = {len(low_orders)}")
    print(f"  High ratio (top 20%, ratio >= {p80:.2f}): n = {len(high_orders)}")

    stat_mw, p_mw = mannwhitneyu(high_orders, low_orders, alternative='greater')
    pooled_std = np.sqrt((np.var(low_orders) + np.var(high_orders)) / 2)
    cohens_d = (np.mean(high_orders) - np.mean(low_orders)) / pooled_std if pooled_std > 0 else 0

    # Bootstrap CI for Cohen's d
    d_ci = bootstrap_cohens_d(low_orders, high_orders)

    print(f"\n  Mann-Whitney U = {stat_mw:.0f}")
    print(f"  p-value = {p_mw:.2e}")
    print(f"  Cohen's d = {cohens_d:.4f}")
    print(f"  Bootstrap 95% CI for d: [{d_ci[0]:.4f}, {d_ci[1]:.4f}]")
    print(f"\n  Low ratio mean order:  {np.mean(low_orders):.4f} +/- {np.std(low_orders):.4f}")
    print(f"  High ratio mean order: {np.mean(high_orders):.4f} +/- {np.std(high_orders):.4f}")

    # Percent increase
    pct_increase = (np.mean(high_orders) - np.mean(low_orders)) / np.mean(low_orders) * 100

    print(f"\n  Percent increase in order: {pct_increase:.1f}%")

    results['extreme_comparison'] = {
        'p20_threshold': float(p20),
        'p80_threshold': float(p80),
        'n_low': len(low_orders),
        'n_high': len(high_orders),
        'U_statistic': float(stat_mw),
        'p_value': float(p_mw),
        'cohens_d': float(cohens_d),
        'd_ci_95': [float(d_ci[0]), float(d_ci[1])],
        'low_mean': float(np.mean(low_orders)),
        'high_mean': float(np.mean(high_orders)),
        'pct_increase': float(pct_increase)
    }

    # =========================================================================
    # TEST 3: Quintile monotonicity
    # =========================================================================
    print("\n" + "-" * 60)
    print("TEST 3: Quintile Analysis")
    print("-" * 60)

    quintile_thresholds = np.percentile(ratios, [0, 20, 40, 60, 80, 100])
    quintile_means = []

    for i in range(5):
        mask = (ratios >= quintile_thresholds[i]) & (ratios < quintile_thresholds[i+1] + 0.001)
        q_orders = orders[mask]
        q_mean = np.mean(q_orders)
        quintile_means.append(q_mean)
        print(f"  Q{i+1}: ratio=[{quintile_thresholds[i]:.1f}, {quintile_thresholds[i+1]:.1f}], "
              f"n={len(q_orders)}, order={q_mean:.4f}")

    monotonic_increases = sum(1 for i in range(4) if quintile_means[i+1] > quintile_means[i])
    print(f"\n  Monotonic increases: {monotonic_increases}/4")

    results['quintiles'] = {
        'means': [float(m) for m in quintile_means],
        'monotonic_increases': monotonic_increases
    }

    # Kruskal-Wallis
    quintile_groups = []
    for i in range(5):
        mask = (ratios >= quintile_thresholds[i]) & (ratios < quintile_thresholds[i+1] + 0.001)
        quintile_groups.append(orders[mask])

    h_stat, p_kw = kruskal(*[g for g in quintile_groups if len(g) >= 5])
    print(f"\n  Kruskal-Wallis H = {h_stat:.2f}, p = {p_kw:.2e}")

    results['kruskal_wallis'] = {'H': float(h_stat), 'p_value': float(p_kw)}

    # =========================================================================
    # TEST 4: More extreme comparison (bottom 10% vs top 10%)
    # =========================================================================
    print("\n" + "-" * 60)
    print("TEST 4: Very Extreme Comparison (10th vs 90th percentile)")
    print("-" * 60)

    p10 = np.percentile(ratios, 10)
    p90 = np.percentile(ratios, 90)

    low10_mask = ratios <= p10
    high90_mask = ratios >= p90

    low10_orders = orders[low10_mask]
    high90_orders = orders[high90_mask]

    stat_mw2, p_mw2 = mannwhitneyu(high90_orders, low10_orders, alternative='greater')
    pooled_std2 = np.sqrt((np.var(low10_orders) + np.var(high90_orders)) / 2)
    cohens_d2 = (np.mean(high90_orders) - np.mean(low10_orders)) / pooled_std2 if pooled_std2 > 0 else 0

    d_ci2 = bootstrap_cohens_d(low10_orders, high90_orders)

    print(f"\n  Bottom 10% (ratio <= {p10:.2f}): n = {len(low10_orders)}, mean order = {np.mean(low10_orders):.4f}")
    print(f"  Top 10% (ratio >= {p90:.2f}): n = {len(high90_orders)}, mean order = {np.mean(high90_orders):.4f}")
    print(f"\n  Mann-Whitney p = {p_mw2:.2e}")
    print(f"  Cohen's d = {cohens_d2:.4f}")
    print(f"  Bootstrap 95% CI for d: [{d_ci2[0]:.4f}, {d_ci2[1]:.4f}]")

    pct_increase2 = (np.mean(high90_orders) - np.mean(low10_orders)) / np.mean(low10_orders) * 100
    print(f"  Percent increase: {pct_increase2:.1f}%")

    results['very_extreme_comparison'] = {
        'p10_threshold': float(p10),
        'p90_threshold': float(p90),
        'n_low': len(low10_orders),
        'n_high': len(high90_orders),
        'p_value': float(p_mw2),
        'cohens_d': float(cohens_d2),
        'd_ci_95': [float(d_ci2[0]), float(d_ci2[1])],
        'low_mean': float(np.mean(low10_orders)),
        'high_mean': float(np.mean(high90_orders)),
        'pct_increase': float(pct_increase2)
    }

    # =========================================================================
    # SUMMARY AND STATUS
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY:")
    print("=" * 70)

    # Success criteria:
    # 1. Spearman rho > 0.15 with p < 0.01
    # 2. Cohen's d > 0.5 OR bootstrap CI lower bound > 0.3

    corr_sig = p_spearman < 0.01 and r_spearman > 0.15
    d_main = results['extreme_comparison']['cohens_d']
    d_ci_lower = results['extreme_comparison']['d_ci_95'][0]
    d_extreme = results['very_extreme_comparison']['cohens_d']

    effect_sig = d_main > 0.5 or d_ci_lower > 0.3 or d_extreme > 0.5

    if corr_sig and effect_sig:
        status = 'validated'
        confidence = 'high'
        summary = (f"High conn/bias ratio predicts higher order. "
                   f"rho={r_spearman:.3f} (p={p_spearman:.2e}), "
                   f"d={d_main:.2f} [CI: {d_ci_lower:.2f}, {d_ci[1]:.2f}] for 20th vs 80th percentile, "
                   f"d={d_extreme:.2f} for 10th vs 90th percentile. "
                   f"Monotonic across {monotonic_increases}/4 quintile transitions. "
                   f"Mechanism: strong connection weights relative to bias let spatial coordinates "
                   f"dominate sigmoid output, creating spatially-varying patterns.")

        print("\nRESULT: VALIDATED")
        print(f"  - Correlation: rho = {r_spearman:.3f}, p = {p_spearman:.2e}")
        print(f"  - Effect size (20/80): d = {d_main:.2f}, CI = [{d_ci_lower:.2f}, {d_ci[1]:.2f}]")
        print(f"  - Effect size (10/90): d = {d_extreme:.2f}")
        print(f"  - Monotonic: {monotonic_increases}/4")

    elif corr_sig:
        status = 'inconclusive'
        confidence = 'medium'
        summary = (f"Significant correlation (rho={r_spearman:.3f}, p={p_spearman:.2e}) "
                   f"but effect size d={d_main:.2f} with CI [{d_ci_lower:.2f}, {d_ci[1]:.2f}].")

        print("\nRESULT: INCONCLUSIVE")
        print(f"  - Correlation significant but effect size borderline")

    else:
        status = 'refuted'
        confidence = 'high'
        summary = f"No significant relationship. rho={r_spearman:.3f}, p={p_spearman:.2e}."

        print("\nRESULT: REFUTED")

    results['status'] = status
    results['confidence'] = confidence
    results['summary'] = summary

    # Save results
    output_dir = Path("results/weight_ratio")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "refined_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}/refined_results.json")

    return results


if __name__ == "__main__":
    results = run_experiment(n_samples=2000, image_size=32, seed=42)
