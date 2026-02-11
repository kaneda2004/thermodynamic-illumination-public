#!/usr/bin/env python3
"""
RES-184: X/Y Weight Symmetry and Reflection Symmetry Experiment

HYPOTHESIS: CPPNs with equal |x-weight| and |y-weight| produce higher reflection
symmetry in output images.

RATIONALE: RES-171 showed that R-dominant CPPNs (|r| > |xy|) have higher rotational
symmetry. Analogously, if the x and y input weights are balanced, the CPPN should
treat both axes similarly, potentially producing images with higher horizontal and
vertical reflection symmetry.

NULL HYPOTHESIS: The ratio |w_x| / |w_y| has no correlation with reflection symmetry.

Method:
- Generate N=500 random CPPNs
- Extract x-weight and y-weight from connections
- Compute weight asymmetry ratio: |log(|w_x| / |w_y|)| (0 = balanced, large = asymmetric)
- Compute reflection symmetry (horizontal + vertical average)
- Test correlation between weight asymmetry and reflection symmetry
- Also test if balanced CPPNs (asymmetry < 0.3) have higher symmetry than imbalanced
"""

import sys
import os
import numpy as np
from scipy.stats import spearmanr, mannwhitneyu
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.thermo_sampler_v3 import CPPN, order_multiplicative, compute_symmetry


def compute_reflection_symmetry(img: np.ndarray) -> tuple[float, float, float]:
    """Compute horizontal, vertical, and average reflection symmetry."""
    h_sym = np.mean(img == np.fliplr(img))
    v_sym = np.mean(img == np.flipud(img))
    avg_sym = (h_sym + v_sym) / 2
    return h_sym, v_sym, avg_sym


def compute_xy_asymmetry(cppn: CPPN) -> float:
    """
    Compute the asymmetry between x-weight and y-weight.

    Returns |log(|w_x| / |w_y|)| - 0 means balanced, large means imbalanced.
    Uses a small epsilon to avoid log(0).
    """
    w_x = None
    w_y = None

    for conn in cppn.connections:
        if conn.from_id == 0:  # x input
            w_x = conn.weight
        elif conn.from_id == 1:  # y input
            w_y = conn.weight

    if w_x is None or w_y is None:
        return float('nan')

    eps = 1e-6
    ratio = (abs(w_x) + eps) / (abs(w_y) + eps)
    asymmetry = abs(np.log(ratio))

    return asymmetry


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def main():
    np.random.seed(42)

    n_samples = 500
    image_size = 32

    print("=" * 70)
    print("RES-184: X/Y WEIGHT SYMMETRY EXPERIMENT")
    print("=" * 70)
    print()
    print("HYPOTHESIS: CPPNs with equal |x-weight| and |y-weight| produce")
    print("            higher reflection symmetry in output images.")
    print()
    print("NULL (H0): Weight asymmetry ratio has no correlation with reflection symmetry.")
    print()
    print(f"Parameters: n_samples={n_samples}, image_size={image_size}")
    print()

    # Generate samples
    print("Generating samples...")

    data = []
    for i in range(n_samples):
        cppn = CPPN()
        img = cppn.render(image_size)

        asymmetry = compute_xy_asymmetry(cppn)
        h_sym, v_sym, avg_sym = compute_reflection_symmetry(img)
        order = order_multiplicative(img)

        # Also extract actual weights for analysis
        w_x = next((c.weight for c in cppn.connections if c.from_id == 0), None)
        w_y = next((c.weight for c in cppn.connections if c.from_id == 1), None)

        data.append({
            'asymmetry': asymmetry,
            'h_symmetry': h_sym,
            'v_symmetry': v_sym,
            'avg_symmetry': avg_sym,
            'order': order,
            'w_x': w_x,
            'w_y': w_y,
        })

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{n_samples} samples generated...")

    print(f"Done. Generated {n_samples} samples.")
    print()

    # Extract arrays for analysis
    asymmetries = np.array([d['asymmetry'] for d in data])
    avg_symmetries = np.array([d['avg_symmetry'] for d in data])
    h_symmetries = np.array([d['h_symmetry'] for d in data])
    v_symmetries = np.array([d['v_symmetry'] for d in data])
    orders = np.array([d['order'] for d in data])

    # Remove any NaN entries
    valid_mask = ~np.isnan(asymmetries)
    asymmetries = asymmetries[valid_mask]
    avg_symmetries = avg_symmetries[valid_mask]
    h_symmetries = h_symmetries[valid_mask]
    v_symmetries = v_symmetries[valid_mask]
    orders = orders[valid_mask]

    print(f"Valid samples (non-NaN): {len(asymmetries)}")
    print()

    # Summary statistics
    print("-" * 70)
    print("SUMMARY STATISTICS")
    print("-" * 70)
    print(f"Weight asymmetry: mean={np.mean(asymmetries):.3f}, std={np.std(asymmetries):.3f}")
    print(f"  min={np.min(asymmetries):.3f}, max={np.max(asymmetries):.3f}")
    print(f"Avg reflection symmetry: mean={np.mean(avg_symmetries):.3f}, std={np.std(avg_symmetries):.3f}")
    print(f"  min={np.min(avg_symmetries):.3f}, max={np.max(avg_symmetries):.3f}")
    print()

    # PRIMARY TEST: Correlation between asymmetry and symmetry
    print("=" * 70)
    print("PRIMARY TEST: Correlation between weight asymmetry and reflection symmetry")
    print("=" * 70)
    print()
    print("Hypothesis: Negative correlation (lower asymmetry -> higher symmetry)")
    print()

    # Spearman correlation (robust to outliers)
    rho_avg, p_avg = spearmanr(asymmetries, avg_symmetries)
    rho_h, p_h = spearmanr(asymmetries, h_symmetries)
    rho_v, p_v = spearmanr(asymmetries, v_symmetries)

    print(f"Spearman correlation (asymmetry vs symmetry):")
    print(f"  Average symmetry: rho={rho_avg:.4f}, p={p_avg:.2e}")
    print(f"  Horizontal symmetry: rho={rho_h:.4f}, p={p_h:.2e}")
    print(f"  Vertical symmetry: rho={rho_v:.4f}, p={p_v:.2e}")
    print()

    # SECONDARY TEST: Compare balanced vs imbalanced CPPNs
    print("=" * 70)
    print("SECONDARY TEST: Balanced vs Imbalanced CPPNs")
    print("=" * 70)
    print()

    # Define balanced as asymmetry < median (or some threshold)
    asymmetry_median = np.median(asymmetries)
    balanced_mask = asymmetries < asymmetry_median
    imbalanced_mask = asymmetries >= asymmetry_median

    balanced_sym = avg_symmetries[balanced_mask]
    imbalanced_sym = avg_symmetries[imbalanced_mask]

    print(f"Balanced (asymmetry < {asymmetry_median:.3f}): n={len(balanced_sym)}")
    print(f"  Mean symmetry: {np.mean(balanced_sym):.4f} +/- {np.std(balanced_sym):.4f}")
    print(f"Imbalanced (asymmetry >= {asymmetry_median:.3f}): n={len(imbalanced_sym)}")
    print(f"  Mean symmetry: {np.mean(imbalanced_sym):.4f} +/- {np.std(imbalanced_sym):.4f}")
    print()

    # Mann-Whitney U test
    U_stat, p_mw = mannwhitneyu(balanced_sym, imbalanced_sym, alternative='greater')
    d_balanced = cohens_d(balanced_sym, imbalanced_sym)

    print(f"Mann-Whitney U (balanced > imbalanced): U={U_stat:.1f}, p={p_mw:.2e}")
    print(f"Cohen's d: {d_balanced:.4f}")
    print()

    # Also test with stricter threshold
    strict_balanced_mask = asymmetries < 0.3  # More balanced
    strict_imbalanced_mask = asymmetries > 1.0  # More imbalanced

    if strict_balanced_mask.sum() > 10 and strict_imbalanced_mask.sum() > 10:
        strict_balanced_sym = avg_symmetries[strict_balanced_mask]
        strict_imbalanced_sym = avg_symmetries[strict_imbalanced_mask]

        print(f"Strict threshold comparison:")
        print(f"  Very balanced (asymmetry < 0.3): n={len(strict_balanced_sym)}, mean_sym={np.mean(strict_balanced_sym):.4f}")
        print(f"  Very imbalanced (asymmetry > 1.0): n={len(strict_imbalanced_sym)}, mean_sym={np.mean(strict_imbalanced_sym):.4f}")

        U_strict, p_strict = mannwhitneyu(strict_balanced_sym, strict_imbalanced_sym, alternative='greater')
        d_strict = cohens_d(strict_balanced_sym, strict_imbalanced_sym)
        print(f"  Mann-Whitney U: U={U_strict:.1f}, p={p_strict:.2e}")
        print(f"  Cohen's d: {d_strict:.4f}")
    else:
        U_strict, p_strict, d_strict = None, None, None
        print(f"  Not enough samples for strict comparison")

    print()

    # TERTIARY: Check if symmetry correlates with order
    print("=" * 70)
    print("CONTROL: Symmetry vs Order correlation")
    print("=" * 70)
    rho_sym_order, p_sym_order = spearmanr(avg_symmetries, orders)
    print(f"Spearman (symmetry vs order): rho={rho_sym_order:.4f}, p={p_sym_order:.2e}")
    print()

    # Determine validation status
    print("=" * 70)
    print("VALIDATION CRITERIA")
    print("=" * 70)
    print()
    print("For VALIDATED:")
    print("  - Significant negative correlation (p < 0.01)")
    print("  - Effect size |rho| > 0.2 or Cohen's d > 0.5")
    print()

    # Check criteria
    correlation_significant = p_avg < 0.01 and rho_avg < 0
    effect_large_enough = abs(rho_avg) > 0.2 or abs(d_balanced) > 0.5

    if correlation_significant and effect_large_enough:
        status = "validated"
        print("STATUS: VALIDATED")
        print(f"  Significant negative correlation: rho={rho_avg:.4f}, p={p_avg:.2e}")
    elif p_avg < 0.05 or abs(rho_avg) > 0.1:
        status = "inconclusive"
        print("STATUS: INCONCLUSIVE")
        print(f"  Weak evidence: rho={rho_avg:.4f}, p={p_avg:.2e}")
    else:
        status = "refuted"
        print("STATUS: REFUTED")
        print(f"  No significant correlation: rho={rho_avg:.4f}, p={p_avg:.2e}")

    print()

    # Prepare results
    results = {
        'experiment_id': 'RES-184',
        'domain': 'coordinate_effects',
        'hypothesis': 'CPPNs with equal |x-weight| and |y-weight| produce higher reflection symmetry',
        'parameters': {
            'n_samples': n_samples,
            'image_size': image_size,
            'seed': 42,
        },
        'primary_test': {
            'test': 'spearman_correlation',
            'rho_avg': float(rho_avg),
            'p_avg': float(p_avg),
            'rho_h': float(rho_h),
            'p_h': float(p_h),
            'rho_v': float(rho_v),
            'p_v': float(p_v),
        },
        'secondary_test': {
            'test': 'mann_whitney_u',
            'balanced_mean_sym': float(np.mean(balanced_sym)),
            'imbalanced_mean_sym': float(np.mean(imbalanced_sym)),
            'U_statistic': float(U_stat),
            'p_value': float(p_mw),
            'cohens_d': float(d_balanced),
        },
        'strict_test': {
            'U_statistic': float(U_strict) if U_strict is not None else None,
            'p_value': float(p_strict) if p_strict is not None else None,
            'cohens_d': float(d_strict) if d_strict is not None else None,
        },
        'control': {
            'symmetry_order_rho': float(rho_sym_order),
            'symmetry_order_p': float(p_sym_order),
        },
        'summary_stats': {
            'asymmetry_mean': float(np.mean(asymmetries)),
            'asymmetry_std': float(np.std(asymmetries)),
            'symmetry_mean': float(np.mean(avg_symmetries)),
            'symmetry_std': float(np.std(avg_symmetries)),
        },
        'status': status,
        'effect_size': float(rho_avg),
        'p_value': float(p_avg),
    }

    # Save results
    results_dir = Path(__file__).parent.parent / 'results' / 'xy_weight_symmetry'
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / 'results.json'

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {results_path}")
    print()

    # Final summary for research log
    print("=" * 70)
    print("SUMMARY FOR RESEARCH LOG")
    print("=" * 70)
    print(f"EXPERIMENT: RES-184")
    print(f"DOMAIN: coordinate_effects")
    print(f"STATUS: {status}")
    print()
    print(f"HYPOTHESIS: CPPNs with equal |x-weight| and |y-weight| produce higher reflection symmetry")
    print(f"RESULT: Weight asymmetry shows rho={rho_avg:.3f} correlation with symmetry (p={p_avg:.2e})")
    print()
    print(f"METRICS:")
    print(f"- effect_size: {rho_avg:.2f}")
    print(f"- p_value: {p_avg:.2e}")
    print()

    return results


if __name__ == "__main__":
    results = main()
