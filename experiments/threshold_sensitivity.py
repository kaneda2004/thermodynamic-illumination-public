#!/usr/bin/env python3
"""
RES-064: Threshold sensitivity analysis for binarization in order metric.

Hypothesis: Order metric is sensitive to binarization threshold, with optimal
threshold != 0.5 for maximizing order variance separation between CPPN and random.

The CPPN.render() method uses threshold=0.5. This experiment tests whether:
1. Order values change significantly with threshold
2. There exists an optimal threshold that maximizes CPPN-vs-random separation
3. The density gate (centered at 0.5) creates bias toward specific thresholds
"""

import numpy as np
from scipy import stats
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import (
    CPPN, order_multiplicative, compute_compressibility,
    compute_edge_density, compute_spectral_coherence
)


def render_at_threshold(cppn: CPPN, threshold: float, size: int = 32) -> np.ndarray:
    """Render CPPN at arbitrary threshold."""
    coords = np.linspace(-1, 1, size)
    x, y = np.meshgrid(coords, coords)
    continuous = cppn.activate(x, y)
    return (continuous > threshold).astype(np.uint8)


def random_image(size: int = 32, p: float = 0.5) -> np.ndarray:
    """Generate random binary image with given density."""
    return (np.random.random((size, size)) < p).astype(np.uint8)


def run_experiment():
    np.random.seed(42)

    # Parameters
    n_samples = 200
    size = 32
    thresholds = np.linspace(0.1, 0.9, 17)  # 0.1, 0.15, ..., 0.9

    print("=" * 60)
    print("RES-064: Threshold Sensitivity Analysis")
    print("=" * 60)

    # Generate CPPNs
    print(f"\nGenerating {n_samples} CPPNs...")
    cppns = [CPPN() for _ in range(n_samples)]

    # Results storage
    results = {
        'thresholds': thresholds,
        'cppn_orders': [],  # shape: (n_thresholds, n_samples)
        'random_orders': [],
        'cppn_densities': [],
        'random_densities': [],
    }

    # For each threshold, compute orders
    for thresh in thresholds:
        cppn_orders = []
        random_orders = []
        cppn_dens = []
        random_dens = []

        for i, cppn in enumerate(cppns):
            # CPPN image at this threshold
            img = render_at_threshold(cppn, thresh, size)
            cppn_orders.append(order_multiplicative(img))
            cppn_dens.append(np.mean(img))

            # Random image with matched density
            random_img = random_image(size, np.mean(img))
            random_orders.append(order_multiplicative(random_img))
            random_dens.append(np.mean(random_img))

        results['cppn_orders'].append(cppn_orders)
        results['random_orders'].append(random_orders)
        results['cppn_densities'].append(cppn_dens)
        results['random_densities'].append(random_dens)

    # Analysis
    print("\n" + "=" * 60)
    print("Results by Threshold")
    print("=" * 60)
    print(f"{'Thresh':>7} {'CPPN_mean':>10} {'Rand_mean':>10} {'Effect_d':>10} {'p-value':>12} {'CPPN_den':>10}")
    print("-" * 60)

    best_effect = -np.inf
    best_thresh = None
    effects = []
    pvalues = []

    for i, thresh in enumerate(thresholds):
        cppn_arr = np.array(results['cppn_orders'][i])
        rand_arr = np.array(results['random_orders'][i])
        cppn_den = np.mean(results['cppn_densities'][i])

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(cppn_arr) + np.var(rand_arr)) / 2)
        if pooled_std > 0:
            effect_d = (np.mean(cppn_arr) - np.mean(rand_arr)) / pooled_std
        else:
            effect_d = 0

        # Statistical test
        stat, pval = stats.mannwhitneyu(cppn_arr, rand_arr, alternative='greater')

        effects.append(effect_d)
        pvalues.append(pval)

        if effect_d > best_effect:
            best_effect = effect_d
            best_thresh = thresh

        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        print(f"{thresh:7.2f} {np.mean(cppn_arr):10.4f} {np.mean(rand_arr):10.4f} "
              f"{effect_d:10.2f} {pval:12.2e} {cppn_den:10.2f} {sig}")

    effects = np.array(effects)

    # Summary statistics
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    print(f"\nOptimal threshold: {best_thresh:.2f} (effect size d={best_effect:.2f})")
    print(f"Default threshold: 0.50 (effect size d={effects[8]:.2f})")

    # Is 0.5 optimal?
    idx_05 = np.argmin(np.abs(thresholds - 0.5))
    effect_at_05 = effects[idx_05]

    # Sensitivity: How much does effect change across thresholds?
    effect_range = np.max(effects) - np.min(effects)
    effect_std = np.std(effects)

    print(f"\nEffect size range: {np.min(effects):.2f} to {np.max(effects):.2f}")
    print(f"Effect size std across thresholds: {effect_std:.2f}")

    # Test: Is effect at optimal significantly different from 0.5?
    # We check if the optimal threshold is far from 0.5
    thresh_sensitivity = abs(best_thresh - 0.5)

    print(f"\nThreshold sensitivity: |optimal - 0.5| = {thresh_sensitivity:.2f}")

    # Additional analysis: correlation between threshold and effect
    corr, corr_p = stats.pearsonr(thresholds, effects)
    print(f"Correlation (threshold vs effect): r={corr:.3f}, p={corr_p:.3e}")

    # Determine outcome
    print("\n" + "=" * 60)
    print("Conclusion")
    print("=" * 60)

    # Criteria:
    # - VALIDATED if optimal threshold != 0.5 by meaningful margin AND effect varies significantly
    # - REFUTED if 0.5 is optimal or near-optimal AND effect is stable
    # - INCONCLUSIVE otherwise

    # Check if effect is consistently high across all thresholds (meaning insensitive)
    high_effect_threshold = 0.5
    n_high_effect = np.sum(effects > high_effect_threshold)

    if thresh_sensitivity < 0.1 and effect_std < 0.3:
        status = "refuted"
        summary = (f"Order metric is NOT sensitive to threshold. Effect size stable at "
                   f"d={np.mean(effects):.2f} +/- {effect_std:.2f} across thresholds 0.1-0.9. "
                   f"Default 0.5 is near-optimal (d={effect_at_05:.2f} vs best d={best_effect:.2f}).")
    elif thresh_sensitivity >= 0.15 and best_effect - effect_at_05 > 0.5:
        status = "validated"
        summary = (f"Order metric IS sensitive to threshold. Optimal at {best_thresh:.2f} "
                   f"(d={best_effect:.2f}) vs default 0.5 (d={effect_at_05:.2f}). "
                   f"Effect varies by {effect_range:.2f} across thresholds.")
    else:
        status = "inconclusive"
        summary = (f"Moderate sensitivity. Best threshold {best_thresh:.2f} (d={best_effect:.2f}) "
                   f"vs 0.5 (d={effect_at_05:.2f}). Effect std={effect_std:.2f}.")

    print(f"\nSTATUS: {status.upper()}")
    print(f"\nSUMMARY: {summary}")

    # Return structured result
    result = {
        'status': status,
        'optimal_threshold': float(best_thresh),
        'optimal_effect': float(best_effect),
        'default_effect': float(effect_at_05),
        'effect_std': float(effect_std),
        'effect_range': float(effect_range),
        'threshold_sensitivity': float(thresh_sensitivity),
        'summary': summary,
    }

    return result


if __name__ == "__main__":
    result = run_experiment()

    # Print final metrics for log
    print("\n" + "=" * 60)
    print("METRICS FOR LOG")
    print("=" * 60)
    print(f"effect_size: {result['optimal_effect']:.2f}")
    print(f"effect_at_default: {result['default_effect']:.2f}")
    print(f"threshold_sensitivity: {result['threshold_sensitivity']:.2f}")
    print(f"effect_std: {result['effect_std']:.2f}")
