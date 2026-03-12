#!/usr/bin/env python3
"""
RES-081: Do weight sign patterns affect CPPN order?

Test whether the pattern of positive/negative weights matters,
or only the magnitudes. Compare:
1. Original CPPN weights
2. All-positive (absolute values)
3. All-negative
4. Random sign flips (preserve magnitudes)
5. Alternating signs

Related: RES-060 (sparsity doesn't matter)
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import CPPN, order_multiplicative as compute_order
from scipy import stats

def test_sign_patterns():
    """Test if weight sign patterns affect order."""
    np.random.seed(42)

    n_samples = 100
    results = {
        'original': [],
        'all_positive': [],
        'all_negative': [],
        'random_flips': [],
        'alternating': [],
    }

    for i in range(n_samples):
        # Create fresh CPPN
        cppn = CPPN()
        original_weights = cppn.get_weights().copy()

        # 1. Original
        img_orig = cppn.render(32)
        order_orig = compute_order(img_orig)
        results['original'].append(order_orig)

        # 2. All positive (absolute values)
        cppn_pos = cppn.copy()
        cppn_pos.set_weights(np.abs(original_weights))
        img_pos = cppn_pos.render(32)
        order_pos = compute_order(img_pos)
        results['all_positive'].append(order_pos)

        # 3. All negative
        cppn_neg = cppn.copy()
        cppn_neg.set_weights(-np.abs(original_weights))
        img_neg = cppn_neg.render(32)
        order_neg = compute_order(img_neg)
        results['all_negative'].append(order_neg)

        # 4. Random sign flips (preserve magnitudes)
        cppn_rand = cppn.copy()
        random_signs = np.random.choice([-1, 1], size=len(original_weights))
        cppn_rand.set_weights(np.abs(original_weights) * random_signs)
        img_rand = cppn_rand.render(32)
        order_rand = compute_order(img_rand)
        results['random_flips'].append(order_rand)

        # 5. Alternating signs
        cppn_alt = cppn.copy()
        alt_signs = np.array([(-1)**i for i in range(len(original_weights))])
        cppn_alt.set_weights(np.abs(original_weights) * alt_signs)
        img_alt = cppn_alt.render(32)
        order_alt = compute_order(img_alt)
        results['alternating'].append(order_alt)

    # Statistical analysis
    print("=" * 60)
    print("WEIGHT SIGN PATTERN EXPERIMENT (RES-081)")
    print("=" * 60)
    print(f"\nSamples: {n_samples}")
    print("\nMean Order by Sign Pattern:")
    print("-" * 40)

    for name, orders in results.items():
        mean = np.mean(orders)
        std = np.std(orders)
        print(f"  {name:15s}: {mean:.4f} ± {std:.4f}")

    # Statistical tests vs original
    print("\nPairwise Tests vs Original (Welch's t-test):")
    print("-" * 40)

    orig = np.array(results['original'])
    comparisons = ['all_positive', 'all_negative', 'random_flips', 'alternating']

    significant_diffs = 0
    for comp in comparisons:
        other = np.array(results[comp])
        t_stat, p_val = stats.ttest_ind(orig, other, equal_var=False)
        effect_size = (np.mean(orig) - np.mean(other)) / np.sqrt((np.std(orig)**2 + np.std(other)**2) / 2)

        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        if p_val < 0.01:
            significant_diffs += 1

        print(f"  {comp:15s}: t={t_stat:7.3f}, p={p_val:.4f}, d={effect_size:6.3f} {sig}")

    # Correlation between original and transformed orders
    print("\nCorrelation with Original Order:")
    print("-" * 40)

    for comp in comparisons:
        r, p = stats.pearsonr(results['original'], results[comp])
        print(f"  {comp:15s}: r={r:.3f}, p={p:.4f}")

    # ANOVA across all conditions
    f_stat, p_anova = stats.f_oneway(
        results['original'],
        results['all_positive'],
        results['all_negative'],
        results['random_flips'],
        results['alternating']
    )
    print(f"\nOne-way ANOVA: F={f_stat:.3f}, p={p_anova:.4f}")

    # Variance comparison
    print("\nVariance by Condition:")
    print("-" * 40)
    for name, orders in results.items():
        print(f"  {name:15s}: var={np.var(orders):.6f}")

    # Check if sign pattern preserves relative ordering
    print("\nRank Correlation (Spearman) with Original:")
    print("-" * 40)
    for comp in comparisons:
        rho, p = stats.spearmanr(results['original'], results[comp])
        print(f"  {comp:15s}: rho={rho:.3f}, p={p:.4f}")

    # Summary
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)

    max_diff = max(abs(np.mean(results[c]) - np.mean(results['original'])) for c in comparisons)
    mean_orig = np.mean(results['original'])

    if significant_diffs == 0 and p_anova > 0.01:
        status = "REFUTED"
        summary = f"Sign patterns do NOT significantly affect order. ANOVA p={p_anova:.4f}, max mean diff={max_diff:.4f}"
    elif significant_diffs >= 2 and p_anova < 0.01:
        status = "VALIDATED"
        summary = f"Sign patterns DO affect order. ANOVA p={p_anova:.4f}, {significant_diffs}/4 significant comparisons"
    else:
        status = "INCONCLUSIVE"
        summary = f"Mixed results. ANOVA p={p_anova:.4f}, {significant_diffs}/4 significant comparisons"

    print(f"\nStatus: {status}")
    print(f"Summary: {summary}")

    return {
        'status': status,
        'p_anova': float(p_anova),
        'f_stat': float(f_stat),
        'significant_comparisons': significant_diffs,
        'max_mean_diff': float(max_diff),
        'mean_original': float(mean_orig),
        'summary': summary
    }

if __name__ == '__main__':
    results = test_sign_patterns()
