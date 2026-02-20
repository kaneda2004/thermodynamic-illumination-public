#!/usr/bin/env python3
"""
EXPERIMENT: Order Metric Sensitivity Analysis (RES-013)

HYPOTHESIS: Order metric local Lipschitz constant (sensitivity to perturbations)
scales positively with order level, indicating high-order configurations are
dynamically unstable.

NULL HYPOTHESIS: Lipschitz constant is independent of order level - sensitivity
is uniform across the order spectrum.

METHOD:
1. Generate CPPNs spanning the order spectrum
2. Compute local Lipschitz constant via random perturbations in weight space
3. Test correlation between order and Lipschitz constant
4. Bin by order level and compare distributions

NOVELTY: RES-007 studied static correlations between order and features.
This tests DYNAMIC stability - how much does order change under perturbations?
Strong positive correlation would indicate high-order images are "precarious"
and easily disrupted.
"""

import sys
import os
import numpy as np
from scipy.stats import kruskal, spearmanr, mannwhitneyu, pearsonr
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.thermo_sampler_v3 import CPPN, order_multiplicative


def compute_local_lipschitz(cppn: CPPN, image_size: int = 32,
                             n_perturbations: int = 30, epsilon: float = 0.01) -> tuple[float, float]:
    """
    Estimate local Lipschitz constant: max |delta_order| / |delta_weights|

    Samples random perturbation directions and measures max and mean ratio.
    Returns (max_ratio, mean_ratio)
    """
    weights = cppn.get_weights()
    base_order = order_multiplicative(cppn.render(image_size))

    ratios = []

    for _ in range(n_perturbations):
        # Random direction in weight space (unit norm)
        direction = np.random.randn(len(weights))
        direction = direction / np.linalg.norm(direction)

        # Perturb
        w_perturbed = weights + epsilon * direction
        cppn_perturbed = cppn.copy()
        cppn_perturbed.set_weights(w_perturbed)
        perturbed_order = order_multiplicative(cppn_perturbed.render(image_size))

        delta_order = abs(perturbed_order - base_order)
        ratio = delta_order / epsilon
        ratios.append(ratio)

    return max(ratios), np.mean(ratios)


def compute_bidirectional_sensitivity(cppn: CPPN, image_size: int = 32,
                                       n_perturbations: int = 20, epsilon: float = 0.01) -> dict:
    """
    Measure sensitivity in both positive and negative order directions.

    Returns dictionary with:
    - increase_sensitivity: how much can order INCREASE under perturbation
    - decrease_sensitivity: how much can order DECREASE under perturbation
    - asymmetry: ratio of increase to decrease sensitivity
    """
    weights = cppn.get_weights()
    base_order = order_multiplicative(cppn.render(image_size))

    increases = []
    decreases = []

    for _ in range(n_perturbations):
        direction = np.random.randn(len(weights))
        direction = direction / np.linalg.norm(direction)

        w_perturbed = weights + epsilon * direction
        cppn_perturbed = cppn.copy()
        cppn_perturbed.set_weights(w_perturbed)
        perturbed_order = order_multiplicative(cppn_perturbed.render(image_size))

        delta = perturbed_order - base_order
        if delta > 0:
            increases.append(delta / epsilon)
        else:
            decreases.append(abs(delta) / epsilon)

    return {
        'increase_sensitivity': np.mean(increases) if increases else 0,
        'decrease_sensitivity': np.mean(decreases) if decreases else 0,
        'n_increases': len(increases),
        'n_decreases': len(decreases)
    }


def run_experiment(n_samples: int = 500, image_size: int = 32, seed: int = 42):
    """
    Main experiment: measure order metric sensitivity across order spectrum.
    """
    np.random.seed(seed)

    print("=" * 70)
    print("EXPERIMENT: Order Metric Sensitivity Analysis")
    print("=" * 70)
    print()
    print("H0: Lipschitz constant is independent of order level")
    print("H1: Lipschitz constant scales positively with order level")
    print("    (high-order configurations are dynamically unstable)")
    print()

    # Collect data
    print(f"Generating {n_samples} CPPN samples and computing Lipschitz constants...")

    data = {
        'order': [],
        'lipschitz_max': [],
        'lipschitz_mean': [],
        'increase_sensitivity': [],
        'decrease_sensitivity': [],
        'n_increases': [],
        'n_decreases': []
    }

    for i in range(n_samples):
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{n_samples}")

        cppn = CPPN()
        img = cppn.render(image_size)
        order = order_multiplicative(img)

        lip_max, lip_mean = compute_local_lipschitz(cppn, image_size)
        bidir = compute_bidirectional_sensitivity(cppn, image_size)

        data['order'].append(order)
        data['lipschitz_max'].append(lip_max)
        data['lipschitz_mean'].append(lip_mean)
        data['increase_sensitivity'].append(bidir['increase_sensitivity'])
        data['decrease_sensitivity'].append(bidir['decrease_sensitivity'])
        data['n_increases'].append(bidir['n_increases'])
        data['n_decreases'].append(bidir['n_decreases'])

    # Convert to arrays
    for k in data:
        data[k] = np.array(data[k])

    print(f"\nData collected: {len(data['order'])} samples")
    print(f"Order range: [{data['order'].min():.4f}, {data['order'].max():.4f}]")
    print(f"Lipschitz (max) range: [{data['lipschitz_max'].min():.4f}, {data['lipschitz_max'].max():.4f}]")

    # Bin by order level
    order_bins = [0.0, 0.05, 0.10, 0.20, 0.40, 1.0]
    bin_labels = ['Very Low (0-0.05)', 'Low (0.05-0.10)', 'Medium (0.10-0.20)',
                  'High (0.20-0.40)', 'Very High (0.40-1.0)']

    binned_lipschitz = []
    bin_counts = []

    print("\n" + "-" * 60)
    print("Lipschitz Constant (Mean) by Order Bin:")
    print("-" * 60)

    for i in range(len(order_bins) - 1):
        mask = (data['order'] >= order_bins[i]) & (data['order'] < order_bins[i+1])
        lip_in_bin = data['lipschitz_mean'][mask]

        if len(lip_in_bin) > 0:
            binned_lipschitz.append(lip_in_bin)
            bin_counts.append(len(lip_in_bin))
            print(f"  {bin_labels[i]:25s}: n={len(lip_in_bin):4d}, "
                  f"mean={np.mean(lip_in_bin):.4f}, std={np.std(lip_in_bin):.4f}")
        else:
            binned_lipschitz.append(np.array([]))
            bin_counts.append(0)
            print(f"  {bin_labels[i]:25s}: n=0 (no samples)")

    # Statistical Tests
    print("\n" + "-" * 60)
    print("STATISTICAL TESTS:")
    print("-" * 60)

    results = {
        'n_samples': n_samples,
        'order_stats': {
            'min': float(data['order'].min()),
            'max': float(data['order'].max()),
            'mean': float(data['order'].mean()),
            'std': float(data['order'].std())
        },
        'lipschitz_stats': {
            'min': float(data['lipschitz_mean'].min()),
            'max': float(data['lipschitz_mean'].max()),
            'mean': float(data['lipschitz_mean'].mean()),
            'std': float(data['lipschitz_mean'].std())
        },
        'bin_counts': bin_counts
    }

    # Test 1: Spearman correlation (main test - monotonic relationship)
    r_spearman, p_spearman = spearmanr(data['order'], data['lipschitz_mean'])
    print(f"\n1. Spearman correlation (order vs mean Lipschitz):")
    print(f"   rho: {r_spearman:.4f}")
    print(f"   p-value: {p_spearman:.2e}")

    results['primary_test'] = {
        'test': 'Spearman correlation',
        'rho': float(r_spearman),
        'p_value': float(p_spearman)
    }

    if p_spearman < 0.01 and r_spearman > 0.5:
        print("   ** STRONGLY SIGNIFICANT: Positive correlation (rho > 0.5, p < 0.01) **")
    elif p_spearman < 0.01:
        print(f"   ** SIGNIFICANT (p < 0.01) but effect size {r_spearman:.2f} may be small **")
    else:
        print("   Not significant at p < 0.01")

    # Test 2: Pearson correlation (linear relationship)
    r_pearson, p_pearson = pearsonr(data['order'], data['lipschitz_mean'])
    print(f"\n2. Pearson correlation (order vs mean Lipschitz):")
    print(f"   r: {r_pearson:.4f}")
    print(f"   p-value: {p_pearson:.2e}")

    results['pearson'] = {
        'r': float(r_pearson),
        'p_value': float(p_pearson)
    }

    # Test 3: Kruskal-Wallis H-test (non-parametric ANOVA)
    non_empty_bins = [b for b in binned_lipschitz if len(b) >= 5]

    if len(non_empty_bins) >= 3:
        h_stat, p_kruskal = kruskal(*non_empty_bins)
        n_total = sum(len(b) for b in non_empty_bins)
        k = len(non_empty_bins)
        eta_squared = (h_stat - k + 1) / (n_total - k)

        print(f"\n3. Kruskal-Wallis H-test (Lipschitz differs across order bins):")
        print(f"   H-statistic: {h_stat:.4f}")
        print(f"   p-value: {p_kruskal:.2e}")
        print(f"   Effect size (eta^2): {eta_squared:.4f}")

        results['kruskal_wallis'] = {
            'H_statistic': float(h_stat),
            'p_value': float(p_kruskal),
            'eta_squared': float(eta_squared)
        }
    else:
        print("\n3. Kruskal-Wallis: Insufficient bins with data")
        results['kruskal_wallis'] = {'error': 'insufficient bins'}

    # Test 4: Low vs High order comparison
    low_order_mask = data['order'] < 0.10
    high_order_mask = data['order'] >= 0.15

    if np.sum(low_order_mask) >= 20 and np.sum(high_order_mask) >= 20:
        low_lip = data['lipschitz_mean'][low_order_mask]
        high_lip = data['lipschitz_mean'][high_order_mask]

        stat, p_mw = mannwhitneyu(high_lip, low_lip, alternative='greater')
        pooled_std = np.sqrt((np.var(low_lip) + np.var(high_lip)) / 2)
        cohens_d = (np.mean(high_lip) - np.mean(low_lip)) / pooled_std if pooled_std > 0 else 0

        print(f"\n4. Mann-Whitney U (high-order > low-order Lipschitz):")
        print(f"   U-statistic: {stat:.0f}")
        print(f"   p-value: {p_mw:.2e}")
        print(f"   Cohen's d: {cohens_d:.4f}")
        print(f"   Low-order Lipschitz mean: {np.mean(low_lip):.4f}")
        print(f"   High-order Lipschitz mean: {np.mean(high_lip):.4f}")

        results['mann_whitney'] = {
            'U_statistic': float(stat),
            'p_value': float(p_mw),
            'cohens_d': float(cohens_d),
            'low_order_mean': float(np.mean(low_lip)),
            'high_order_mean': float(np.mean(high_lip)),
            'n_low': int(np.sum(low_order_mask)),
            'n_high': int(np.sum(high_order_mask))
        }

        if p_mw < 0.01 and cohens_d > 0.5:
            print("   ** SIGNIFICANT: High-order images have higher Lipschitz constant **")
    else:
        print(f"\n4. Mann-Whitney: Insufficient samples (low: {np.sum(low_order_mask)}, high: {np.sum(high_order_mask)})")
        results['mann_whitney'] = {'error': 'insufficient samples'}

    # Test 5: Asymmetry analysis - is order easier to increase or decrease?
    mean_inc = np.mean(data['increase_sensitivity'])
    mean_dec = np.mean(data['decrease_sensitivity'])
    mean_n_inc = np.mean(data['n_increases'])
    mean_n_dec = np.mean(data['n_decreases'])

    print(f"\n5. Perturbation direction asymmetry:")
    print(f"   Mean increase sensitivity: {mean_inc:.4f}")
    print(f"   Mean decrease sensitivity: {mean_dec:.4f}")
    print(f"   Avg perturbations increasing order: {mean_n_inc:.1f}/20")
    print(f"   Avg perturbations decreasing order: {mean_n_dec:.1f}/20")

    asymmetry_ratio = mean_dec / mean_inc if mean_inc > 0 else float('inf')
    print(f"   Asymmetry ratio (decrease/increase): {asymmetry_ratio:.2f}")

    results['asymmetry'] = {
        'mean_increase_sensitivity': float(mean_inc),
        'mean_decrease_sensitivity': float(mean_dec),
        'mean_n_increases': float(mean_n_inc),
        'mean_n_decreases': float(mean_n_dec),
        'asymmetry_ratio': float(asymmetry_ratio) if asymmetry_ratio != float('inf') else None
    }

    # Summary and status determination
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("=" * 60)

    # Primary success criterion: Spearman rho > 0.5, p < 0.01
    primary_success = p_spearman < 0.01 and r_spearman > 0.5

    # Secondary criterion: Mann-Whitney with effect size
    mw_success = (results.get('mann_whitney', {}).get('p_value', 1) < 0.01 and
                  results.get('mann_whitney', {}).get('cohens_d', 0) > 0.5)

    if primary_success:
        status = 'validated'
        confidence = 'high'
        summary = (f"Strong positive correlation between order level and Lipschitz constant "
                   f"(rho={r_spearman:.2f}, p={p_spearman:.2e}). "
                   f"High-order images are dynamically unstable - small weight perturbations "
                   f"cause larger order changes. ")
        if asymmetry_ratio > 1.5:
            summary += f"Asymmetric: order is {asymmetry_ratio:.1f}x easier to decrease than increase."

        print(f"\nRESULT: VALIDATED")
        print(f"  - Strong positive correlation (rho = {r_spearman:.2f})")
        print(f"  - High-order configurations are dynamically unstable")
        if asymmetry_ratio > 1.5:
            print(f"  - Order is {asymmetry_ratio:.1f}x easier to decrease than increase")

    elif p_spearman < 0.01 or mw_success:
        status = 'inconclusive'
        confidence = 'medium'
        summary = (f"Significant but weak correlation (rho={r_spearman:.2f}, p={p_spearman:.2e}). "
                   f"Some evidence for increased sensitivity at higher order, but effect size "
                   f"does not meet threshold (rho > 0.5).")
        print(f"\nRESULT: INCONCLUSIVE")
        print(f"  - Significant but weak effect (rho = {r_spearman:.2f})")
    else:
        status = 'refuted'
        confidence = 'high'
        summary = (f"No significant relationship between order and Lipschitz constant "
                   f"(rho={r_spearman:.2f}, p={p_spearman:.2e}). "
                   f"Order metric sensitivity is uniform across the order spectrum.")
        print(f"\nRESULT: REFUTED")
        print(f"  - No significant correlation")

    results['status'] = status
    results['confidence'] = confidence
    results['summary'] = summary

    # Save results
    output_dir = Path("results/order_sensitivity")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "sensitivity_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}/sensitivity_results.json")

    return results


if __name__ == "__main__":
    results = run_experiment(n_samples=500, image_size=32, seed=42)
