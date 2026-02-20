#!/usr/bin/env python3
"""
RES-170: CPPN weight sign patterns (all-same vs mixed) correlate with output order

Hypothesis: The sign pattern of CPPN weights (specifically whether all weights have
the same sign vs mixed signs) correlates with output order.

Rationale: The CPPN inputs (x, y, r, bias) have natural symmetries. When weights all
have the same sign, the contribution from different inputs combines constructively.
When signs are mixed, interference patterns emerge. This could affect the resulting
image structure.

Method:
1. Generate many random CPPNs from the prior
2. Classify each by its weight sign pattern (count of positive/negative weights)
3. Compute order for each CPPN
4. Test correlation between sign uniformity (how many weights share the majority sign)
   and order
"""

import numpy as np
import json
import sys
import os
from scipy import stats
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.thermo_sampler_v3 import CPPN, order_multiplicative

def compute_sign_uniformity(weights):
    """
    Compute how uniform the sign pattern is.
    Returns fraction of weights matching the majority sign (0.5 to 1.0).
    """
    signs = np.sign(weights)
    n_positive = np.sum(signs > 0)
    n_negative = np.sum(signs < 0)
    n_total = len(weights)

    if n_total == 0:
        return 0.5

    majority_count = max(n_positive, n_negative)
    return majority_count / n_total

def get_sign_pattern(weights):
    """
    Get categorical sign pattern (e.g., "+++-" for 4 weights).
    """
    return ''.join(['+' if w >= 0 else '-' for w in weights])

def run_experiment(n_samples=1000, seed=42):
    """Run the sign pattern experiment."""
    np.random.seed(seed)

    results = {
        'n_samples': n_samples,
        'seed': seed,
        'uniformities': [],
        'orders': [],
        'sign_patterns': [],
        'n_positive': [],
        'n_negative': [],
        'weights_std': [],
        'weights_mean': [],
    }

    # Generate samples and compute metrics
    for i in range(n_samples):
        cppn = CPPN()  # Fresh random CPPN from prior
        weights = cppn.get_weights()

        # Compute sign metrics
        uniformity = compute_sign_uniformity(weights)
        sign_pattern = get_sign_pattern(weights)
        n_pos = np.sum(weights >= 0)
        n_neg = np.sum(weights < 0)

        # Compute image and order
        img = cppn.render(32)
        order = order_multiplicative(img)

        results['uniformities'].append(uniformity)
        results['orders'].append(order)
        results['sign_patterns'].append(sign_pattern)
        results['n_positive'].append(int(n_pos))
        results['n_negative'].append(int(n_neg))
        results['weights_std'].append(float(np.std(weights)))
        results['weights_mean'].append(float(np.mean(weights)))

        if (i + 1) % 200 == 0:
            print(f"Progress: {i + 1}/{n_samples}")

    # Convert to numpy for analysis
    uniformities = np.array(results['uniformities'])
    orders = np.array(results['orders'])
    n_positives = np.array(results['n_positive'])

    # --- Statistical Analysis ---

    # 1. Correlation between sign uniformity and order
    corr_uniformity, p_uniformity = stats.spearmanr(uniformities, orders)

    # 2. Correlation between absolute count of majority sign and order
    # This checks if having more weights of same sign predicts order

    # 3. Group comparison: all-same vs mixed
    # 5 parameters: if >3 have same sign, call it "uniform", else "mixed"
    uniform_mask = uniformities >= 0.8  # 4+ out of 5 same sign
    mixed_mask = uniformities < 0.8

    orders_uniform = orders[uniform_mask]
    orders_mixed = orders[mixed_mask]

    if len(orders_uniform) > 5 and len(orders_mixed) > 5:
        # Mann-Whitney U test
        u_stat, p_group = stats.mannwhitneyu(orders_uniform, orders_mixed, alternative='two-sided')

        # Effect size (Cohen's d)
        mean_uniform = np.mean(orders_uniform)
        mean_mixed = np.mean(orders_mixed)
        pooled_std = np.sqrt((np.std(orders_uniform)**2 + np.std(orders_mixed)**2) / 2)
        cohens_d = (mean_uniform - mean_mixed) / pooled_std if pooled_std > 0 else 0
    else:
        p_group = 1.0
        cohens_d = 0.0
        mean_uniform = np.nan
        mean_mixed = np.nan

    # 4. Check specific sign patterns
    pattern_counts = Counter(results['sign_patterns'])
    pattern_orders = {}
    for pattern in pattern_counts:
        mask = [p == pattern for p in results['sign_patterns']]
        pattern_orders[pattern] = {
            'count': pattern_counts[pattern],
            'mean_order': float(np.mean(orders[mask])),
            'std_order': float(np.std(orders[mask])),
        }

    # Sort patterns by mean order
    sorted_patterns = sorted(pattern_orders.items(), key=lambda x: -x[1]['mean_order'])

    # 5. All-positive vs all-negative comparison
    all_positive = [i for i, p in enumerate(results['sign_patterns']) if set(p) == {'+'}]
    all_negative = [i for i, p in enumerate(results['sign_patterns']) if set(p) == {'-'}]

    if len(all_positive) > 0 and len(all_negative) > 0:
        mean_all_pos = np.mean(orders[all_positive])
        mean_all_neg = np.mean(orders[all_negative])
    else:
        mean_all_pos = np.nan
        mean_all_neg = np.nan

    # --- Summary ---
    summary = {
        'correlation_uniformity_order': float(corr_uniformity),
        'p_value_uniformity': float(p_uniformity),
        'cohens_d_uniform_vs_mixed': float(cohens_d),
        'p_value_group_comparison': float(p_group),
        'mean_order_uniform_signs': float(mean_uniform) if not np.isnan(mean_uniform) else None,
        'mean_order_mixed_signs': float(mean_mixed) if not np.isnan(mean_mixed) else None,
        'n_uniform_sign_cpps': int(np.sum(uniform_mask)),
        'n_mixed_sign_cpps': int(np.sum(mixed_mask)),
        'mean_all_positive': float(mean_all_pos) if not np.isnan(mean_all_pos) else None,
        'mean_all_negative': float(mean_all_neg) if not np.isnan(mean_all_neg) else None,
        'n_all_positive': len(all_positive),
        'n_all_negative': len(all_negative),
        'top_5_patterns_by_order': sorted_patterns[:5],
        'bottom_5_patterns_by_order': sorted_patterns[-5:],
    }

    # Add raw data for reproducibility
    results['summary'] = summary

    return results, summary

def main():
    print("RES-170: CPPN weight sign patterns vs output order")
    print("=" * 60)

    results, summary = run_experiment(n_samples=1000, seed=42)

    print("\n=== RESULTS ===\n")
    print(f"Correlation (uniformity vs order): r = {summary['correlation_uniformity_order']:.4f}")
    print(f"P-value (uniformity correlation): p = {summary['p_value_uniformity']:.2e}")
    print()
    print(f"Group comparison (uniform signs vs mixed signs):")
    print(f"  Uniform signs (>=80%): n = {summary['n_uniform_sign_cpps']}, mean order = {summary['mean_order_uniform_signs']:.4f}")
    print(f"  Mixed signs (<80%):    n = {summary['n_mixed_sign_cpps']}, mean order = {summary['mean_order_mixed_signs']:.4f}")
    print(f"  Cohen's d: {summary['cohens_d_uniform_vs_mixed']:.4f}")
    print(f"  P-value: {summary['p_value_group_comparison']:.2e}")
    print()
    print(f"All same sign patterns:")
    print(f"  All positive: n = {summary['n_all_positive']}, mean order = {summary['mean_all_positive']}")
    print(f"  All negative: n = {summary['n_all_negative']}, mean order = {summary['mean_all_negative']}")
    print()
    print("Top 5 sign patterns by order:")
    for pattern, data in summary['top_5_patterns_by_order']:
        print(f"  {pattern}: n={data['count']}, order={data['mean_order']:.4f} +/- {data['std_order']:.4f}")
    print()
    print("Bottom 5 sign patterns by order:")
    for pattern, data in summary['bottom_5_patterns_by_order']:
        print(f"  {pattern}: n={data['count']}, order={data['mean_order']:.4f} +/- {data['std_order']:.4f}")

    # Determine status
    d = abs(summary['cohens_d_uniform_vs_mixed'])
    p = summary['p_value_group_comparison']

    print("\n=== CONCLUSION ===")
    if d >= 0.5 and p < 0.01:
        if summary['cohens_d_uniform_vs_mixed'] > 0:
            print("VALIDATED: Uniform sign patterns produce higher order (d={:.2f}, p={:.2e})".format(d, p))
            status = 'validated'
        else:
            print("REFUTED: Mixed sign patterns produce HIGHER order (d={:.2f}, p={:.2e})".format(d, p))
            status = 'refuted'
    elif p < 0.01 and d < 0.5:
        print("REFUTED: Significant but effect size too small (d={:.2f}, p={:.2e})".format(d, p))
        status = 'refuted'
    else:
        print("REFUTED: No significant relationship (d={:.2f}, p={:.2e})".format(d, p))
        status = 'refuted'

    # Save results
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'results', 'sign_pattern_order', 'results.json'
    )

    # Convert numpy types for JSON
    save_results = {
        'summary': summary,
        'status': status,
        'uniformities': [float(x) for x in results['uniformities']],
        'orders': [float(x) for x in results['orders']],
        'sign_patterns': results['sign_patterns'],
    }

    with open(output_path, 'w') as f:
        json.dump(save_results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return status, summary

if __name__ == '__main__':
    status, summary = main()
