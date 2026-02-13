#!/usr/bin/env python3
"""
RES-216: Symmetry Requirement Phase Transition Test

Hypothesis: Below order 0.5, images can be asymmetric. Above 0.5, achieving
high order REQUIRES bilateral or rotational symmetry (discrete constraint).

Tests:
1. Sample 100 CPPNs (50 low-order, 50 high-order)
2. Compute bilateral, rotational, and max symmetry scores
3. Compare distributions and prevalence
4. Test for phase transition at order threshold
"""

import os
import sys
import numpy as np
from pathlib import Path
from scipy import stats
import json

# Set working directory
os.chdir('/Users/matt/Development/monochrome_noise_converger')
sys.path.insert(0, os.getcwd())

# Import project modules
from core.thermo_sampler_v3 import CPPN, order_multiplicative_v2

def compute_symmetries(pixels):
    """
    Compute bilateral and rotational symmetry scores.
    Returns: (bilateral_sym, rotational_sym, max_sym)
    """
    # Bilateral symmetry (Y-axis mirror): correlation(img, fliplr(img))
    flipped = np.fliplr(pixels)
    bilateral = np.corrcoef(pixels.flatten(), flipped.flatten())[0, 1]
    if np.isnan(bilateral):
        bilateral = 0.0

    # Rotational symmetry (90° rotation)
    rotated = np.rot90(pixels)
    rotational = np.corrcoef(pixels.flatten(), rotated.flatten())[0, 1]
    if np.isnan(rotational):
        rotational = 0.0

    # Overall symmetry
    max_sym = max(bilateral, rotational)

    return bilateral, rotational, max_sym

def main():
    print("=" * 80)
    print("SYMMETRY REQUIREMENT PHASE TRANSITION TEST (RES-216)")
    print("=" * 80)

    # Create output directory
    output_dir = Path('results/symmetry_requirement')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load previously sampled CPPNs and their orders
    print("\n[1/5] Loading CPPN samples with order values...")
    try:
        # Try to load from RES-215 results first
        res215_path = Path('results/cppn_order_distribution/results.json')
        if res215_path.exists():
            with open(res215_path) as f:
                res215_data = json.load(f)
                cppn_seeds = res215_data.get('cppn_seeds', [])
                cppn_orders = res215_data.get('cppn_orders', [])
                print(f"   Loaded {len(cppn_seeds)} CPPNs from RES-215")
        else:
            print("   RES-215 results not found, generating fresh CPPNs...")
            cppn_seeds = []
            cppn_orders = []
    except Exception as e:
        print(f"   Error loading RES-215: {e}, generating fresh CPPNs...")
        cppn_seeds = []
        cppn_orders = []

    # If we don't have enough seeds from RES-215, generate new CPPNs
    if len(cppn_seeds) < 100:
        print("   Generating fresh CPPN samples...")
        np.random.seed(42)
        cppn_seeds = list(range(2160, 2260))
        cppn_orders = []

        for seed in cppn_seeds:
            np.random.seed(seed)
            cppn = CPPN()
            pixels = cppn.render(size=32)
            order = order_multiplicative_v2(pixels, resolution_ref=32)
            cppn_orders.append(float(order))

    # Sort by order to split low/high
    orders_array = np.array(cppn_orders)
    sorted_indices = np.argsort(orders_array)

    # Split: low-order < median, high-order >= median
    low_threshold = np.percentile(orders_array, 50)  # median split
    print(f"   Order distribution: min={orders_array.min():.3f}, max={orders_array.max():.3f}, median={low_threshold:.3f}")

    low_order_indices = np.where(orders_array < low_threshold)[0]
    high_order_indices = np.where(orders_array >= low_threshold)[0]

    print(f"   Low-order (< {low_threshold:.3f}): {len(low_order_indices)} CPPNs")
    print(f"   High-order (>= {low_threshold:.3f}): {len(high_order_indices)} CPPNs")

    # Render all CPPNs and compute symmetries
    print("\n[2/5] Rendering CPPNs and computing symmetries...")
    symmetry_data = {
        'low_order': {'bilateral': [], 'rotational': [], 'max': [], 'orders': []},
        'high_order': {'bilateral': [], 'rotational': [], 'max': [], 'orders': []}
    }

    for idx, seed in enumerate(cppn_seeds):
        if idx % 20 == 0:
            print(f"   Progress: {idx}/100")

        np.random.seed(seed)
        cppn = CPPN()
        pixels = cppn.render(size=32)
        order = cppn_orders[idx]

        bilateral, rotational, max_sym = compute_symmetries(pixels)

        if idx in low_order_indices:
            symmetry_data['low_order']['bilateral'].append(bilateral)
            symmetry_data['low_order']['rotational'].append(rotational)
            symmetry_data['low_order']['max'].append(max_sym)
            symmetry_data['low_order']['orders'].append(order)
        else:
            symmetry_data['high_order']['bilateral'].append(bilateral)
            symmetry_data['high_order']['rotational'].append(rotational)
            symmetry_data['high_order']['max'].append(max_sym)
            symmetry_data['high_order']['orders'].append(order)

    print(f"   ✓ Processed {len(cppn_seeds)} CPPNs")

    # Statistical analysis
    print("\n[3/5] Computing statistical tests...")

    low_max = np.array(symmetry_data['low_order']['max'])
    high_max = np.array(symmetry_data['high_order']['max'])

    # Descriptive statistics
    print(f"\n   Low-order symmetry:")
    print(f"     Mean: {low_max.mean():.4f} (σ={low_max.std():.4f})")
    print(f"     Median: {np.median(low_max):.4f}")
    print(f"     Min/Max: {low_max.min():.4f} / {low_max.max():.4f}")

    print(f"\n   High-order symmetry:")
    print(f"     Mean: {high_max.mean():.4f} (σ={high_max.std():.4f})")
    print(f"     Median: {np.median(high_max):.4f}")
    print(f"     Min/Max: {high_max.min():.4f} / {high_max.max():.4f}")

    # Mann-Whitney U test
    stat_mw, p_mw = stats.mannwhitneyu(low_max, high_max, alternative='two-sided')
    print(f"\n   Mann-Whitney U test:")
    print(f"     U-statistic: {stat_mw:.2f}")
    print(f"     p-value: {p_mw:.4e}")

    # Cliff's delta (effect size for non-parametric)
    n1, n2 = len(low_max), len(high_max)
    dominance = 0
    for x in low_max:
        for y in high_max:
            if x > y:
                dominance += 1
            elif x < y:
                dominance -= 1
    cliffs_delta = dominance / (n1 * n2)
    print(f"     Cliff's delta: {cliffs_delta:.4f}")

    # Prevalence analysis (symmetry > 0.7)
    sym_threshold = 0.7
    low_prevalent = np.sum(low_max > sym_threshold) / len(low_max)
    high_prevalent = np.sum(high_max > sym_threshold) / len(high_max)

    print(f"\n   Symmetry prevalence (sym > {sym_threshold}):")
    print(f"     Low-order: {low_prevalent*100:.1f}% ({int(np.sum(low_max > sym_threshold))}/{len(low_max)})")
    print(f"     High-order: {high_prevalent*100:.1f}% ({int(np.sum(high_max > sym_threshold))}/{len(high_max)})")

    # Chi-squared test for prevalence
    from scipy.stats import chi2_contingency
    contingency_table = np.array([
        [np.sum(low_max > sym_threshold), len(low_max) - np.sum(low_max > sym_threshold)],
        [np.sum(high_max > sym_threshold), len(high_max) - np.sum(high_max > sym_threshold)]
    ])
    chi2, p_chi2, dof, expected = chi2_contingency(contingency_table)
    print(f"\n   Chi-squared test (prevalence difference):")
    print(f"     Chi-squared: {chi2:.4f}")
    print(f"     p-value: {p_chi2:.4e}")

    # Spearman correlation
    all_orders = np.concatenate([np.array(symmetry_data['low_order']['orders']),
                                  np.array(symmetry_data['high_order']['orders'])])
    all_symmetries = np.concatenate([low_max, high_max])
    spearman_r, spearman_p = stats.spearmanr(all_orders, all_symmetries)
    print(f"\n   Spearman rank correlation (order vs symmetry):")
    print(f"     r: {spearman_r:.4f}")
    print(f"     p-value: {spearman_p:.4e}")

    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((n1-1)*low_max.std()**2 + (n2-1)*high_max.std()**2) / (n1 + n2 - 2))
    cohens_d = (high_max.mean() - low_max.mean()) / pooled_std if pooled_std > 0 else 0
    print(f"\n   Cohen's d (effect size): {cohens_d:.4f}")

    # Determine phase transition support
    print("\n[4/5] Evaluating phase transition hypothesis...")
    phase_transition_support = {
        'mean_diff': high_max.mean() - low_max.mean(),
        'prevalence_ratio': high_prevalent / (low_prevalent + 1e-6),  # avoid division by zero
        'statistical_significance': p_mw < 0.05,
        'strong_effect': abs(cohens_d) > 0.8,
        'prevalence_shift': high_prevalent > 0.6 and low_prevalent < 0.4
    }

    print(f"   Mean difference: {phase_transition_support['mean_diff']:.4f}")
    print(f"   Prevalence ratio (high/low): {phase_transition_support['prevalence_ratio']:.2f}x")
    print(f"   Statistically significant (p<0.05): {phase_transition_support['statistical_significance']}")
    print(f"   Strong effect (|d|>0.8): {phase_transition_support['strong_effect']}")
    print(f"   Clear prevalence shift: {phase_transition_support['prevalence_shift']}")

    # Determine verdict
    supports_hypothesis = sum(phase_transition_support.values()) >= 3
    status = "VALIDATED" if supports_hypothesis else "REFUTED"

    print(f"\n   Verdict: {status}")
    print(f"   Supporting evidence: {sum(phase_transition_support.values())}/5 criteria met")

    # Save results
    print("\n[5/5] Saving results...")
    results = {
        'mean_symmetry_low_order': float(low_max.mean()),
        'mean_symmetry_high_order': float(high_max.mean()),
        'std_symmetry_low_order': float(low_max.std()),
        'std_symmetry_high_order': float(high_max.std()),
        'p_value_mann_whitney': float(p_mw),
        'cliffs_delta': float(cliffs_delta),
        'cohens_d': float(cohens_d),
        'prevalence_low_order': float(low_prevalent),
        'prevalence_high_order': float(high_prevalent),
        'chi_squared': float(chi2),
        'chi_squared_p_value': float(p_chi2),
        'spearman_r': float(spearman_r),
        'spearman_p_value': float(spearman_p),
        'status': status,
        'num_low_order': len(low_max),
        'num_high_order': len(high_max),
        'phase_transition_support': {k: bool(v) if isinstance(v, (bool, np.bool_)) else v
                                      for k, v in phase_transition_support.items()}
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"   ✓ Results saved to {output_dir}/results.json")

    # Print final summary
    print("\n" + "=" * 80)
    print(f"RESULT: RES-216 | symmetry_requirement | {status} | d={cohens_d:.2f}")
    print("=" * 80)

    return status, cohens_d

if __name__ == '__main__':
    main()
