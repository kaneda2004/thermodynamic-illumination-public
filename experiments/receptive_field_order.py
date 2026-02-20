#!/usr/bin/env python3
"""
RES-111: CPPN effective receptive field size correlates with image order score

Hypothesis: CPPNs with larger effective receptive fields (where small input
perturbations propagate further spatially) will produce images with different
order scores.

Methodology:
1. Generate CPPN images and compute order scores
2. For each CPPN, perturb a single input coordinate and measure how far the
   output change propagates spatially
3. Effective receptive field = spatial extent where |output_change| > threshold
4. Correlate receptive field size with order score
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import CPPN, order_multiplicative
from scipy import stats
from scipy.ndimage import label


def compute_effective_receptive_field(cppn, resolution=64, perturb_strength=0.01, threshold=0.01):
    """
    Measure effective receptive field by perturbing center input and measuring spread.

    For CPPNs, we perturb a region of input coordinates and measure how the
    continuous output changes across the image.
    """
    x = np.linspace(-1, 1, resolution)
    y = np.linspace(-1, 1, resolution)
    xx, yy = np.meshgrid(x, y)

    # Get baseline output (continuous, before thresholding)
    baseline = cppn.activate(xx, yy)

    # Perturb coordinates near center
    center_idx = resolution // 2
    xx_perturbed = xx.copy()
    yy_perturbed = yy.copy()

    # Perturb center region
    center_mask = (np.abs(xx) < 0.1) & (np.abs(yy) < 0.1)
    xx_perturbed[center_mask] += perturb_strength
    yy_perturbed[center_mask] += perturb_strength

    # Get perturbed output
    perturbed = cppn.activate(xx_perturbed, yy_perturbed)

    # Compute change magnitude
    change = np.abs(perturbed - baseline)

    # Measure affected area (above threshold)
    affected = change > threshold
    affected_fraction = affected.sum() / affected.size

    # Also measure maximum propagation distance from center
    if affected.sum() > 0:
        affected_coords = np.where(affected)
        distances = np.sqrt((affected_coords[0] - center_idx)**2 +
                           (affected_coords[1] - center_idx)**2)
        max_distance = distances.max() / (resolution / 2)  # Normalized 0-1
    else:
        max_distance = 0.0

    return affected_fraction, max_distance


def compute_gradient_receptive_field(cppn, resolution=64, eps=1e-4):
    """
    Measure receptive field via gradient magnitude - how sensitive is output
    to small coordinate changes at each location?

    High gradient magnitude = output changes rapidly with small coordinate shifts
    Low gradient magnitude = output is locally constant

    Returns mean and max gradient magnitudes.
    """
    x = np.linspace(-1, 1, resolution)
    y = np.linspace(-1, 1, resolution)
    xx, yy = np.meshgrid(x, y)

    # Get baseline
    baseline = cppn.activate(xx, yy)

    # Perturb in x direction
    xx_plus = xx + eps
    output_x_plus = cppn.activate(xx_plus, yy)
    grad_x = (output_x_plus - baseline) / eps

    # Perturb in y direction
    yy_plus = yy + eps
    output_y_plus = cppn.activate(xx, yy_plus)
    grad_y = (output_y_plus - baseline) / eps

    # Gradient magnitude at each point
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    return grad_mag.mean(), grad_mag.max(), grad_mag.std()


def run_experiment(n_samples=100, resolution=64):
    """Run the receptive field experiment."""
    results = []

    np.random.seed(42)

    for i in range(n_samples):
        # Generate random CPPN
        cppn = CPPN()

        # Get binary image and order score
        img = cppn.render(resolution)
        order = order_multiplicative(img)

        # Compute receptive field metrics
        affected_frac, max_dist = compute_effective_receptive_field(
            cppn, resolution, perturb_strength=0.01, threshold=0.01
        )

        # Compute gradient-based metrics
        grad_mean, grad_max, grad_std = compute_gradient_receptive_field(cppn, resolution)

        results.append({
            'order': order,
            'affected_fraction': affected_frac,
            'max_propagation': max_dist,
            'grad_mean': grad_mean,
            'grad_max': grad_max,
            'grad_std': grad_std
        })

        if (i + 1) % 20 == 0:
            print(f"Processed {i + 1}/{n_samples}")

    return results


def analyze_results(results):
    """Analyze correlation between receptive field and order."""
    orders = np.array([r['order'] for r in results])
    affected_fracs = np.array([r['affected_fraction'] for r in results])
    max_props = np.array([r['max_propagation'] for r in results])
    grad_means = np.array([r['grad_mean'] for r in results])
    grad_maxs = np.array([r['grad_max'] for r in results])
    grad_stds = np.array([r['grad_std'] for r in results])

    print("\n" + "="*60)
    print("RES-111: Receptive Field vs Order Correlation")
    print("="*60)

    # Correlation 1: Affected fraction vs order
    r1, p1 = stats.pearsonr(orders, affected_fracs)
    print(f"\n1. Affected Fraction vs Order:")
    print(f"   r = {r1:.4f}, p = {p1:.6f}")

    # Correlation 2: Max propagation distance vs order
    r2, p2 = stats.pearsonr(orders, max_props)
    print(f"\n2. Max Propagation Distance vs Order:")
    print(f"   r = {r2:.4f}, p = {p2:.6f}")

    # Correlation 3: Mean gradient magnitude vs order
    r3, p3 = stats.pearsonr(orders, grad_means)
    print(f"\n3. Mean Gradient Magnitude vs Order:")
    print(f"   r = {r3:.4f}, p = {p3:.6f}")

    # Correlation 4: Max gradient vs order
    r4, p4 = stats.pearsonr(orders, grad_maxs)
    print(f"\n4. Max Gradient Magnitude vs Order:")
    print(f"   r = {r4:.4f}, p = {p4:.6f}")

    # Correlation 5: Gradient std vs order
    r5, p5 = stats.pearsonr(orders, grad_stds)
    print(f"\n5. Gradient Std vs Order:")
    print(f"   r = {r5:.4f}, p = {p5:.6f}")

    # Summary stats
    print(f"\nOrder score: mean={orders.mean():.3f}, std={orders.std():.3f}")
    print(f"Affected fraction: mean={affected_fracs.mean():.3f}, std={affected_fracs.std():.3f}")
    print(f"Max propagation: mean={max_props.mean():.3f}, std={max_props.std():.3f}")
    print(f"Grad mean: mean={grad_means.mean():.3f}, std={grad_means.std():.3f}")
    print(f"Grad max: mean={grad_maxs.mean():.3f}, std={grad_maxs.std():.3f}")

    # Determine best metric
    correlations = [
        ('affected_fraction', r1, p1),
        ('max_propagation', r2, p2),
        ('grad_mean', r3, p3),
        ('grad_max', r4, p4),
        ('grad_std', r5, p5)
    ]

    best_metric, best_r, best_p = max(correlations, key=lambda x: abs(x[1]))

    print(f"\nBest correlation: {best_metric} (r={best_r:.4f}, p={best_p:.6f})")

    # Effect size (r^2)
    effect_size = best_r**2
    print(f"Effect size (r^2): {effect_size:.4f}")

    # Bonferroni correction for multiple comparisons
    bonferroni_p = best_p * len(correlations)
    print(f"Bonferroni-corrected p: {bonferroni_p:.6f}")

    # Validation criteria (use Bonferroni-corrected p)
    validated = bonferroni_p < 0.01 and abs(best_r) > 0.5

    print("\n" + "="*60)
    print(f"VALIDATION: {'PASSED' if validated else 'FAILED'}")
    print(f"  p_bonf < 0.01: {bonferroni_p < 0.01} (p_bonf={bonferroni_p:.6f})")
    print(f"  |r| > 0.5: {abs(best_r) > 0.5} (|r|={abs(best_r):.4f})")
    print("="*60)

    return {
        'validated': validated,
        'best_metric': best_metric,
        'best_r': best_r,
        'best_p': best_p,
        'bonferroni_p': bonferroni_p,
        'effect_size': effect_size,
        'all_correlations': correlations
    }


if __name__ == "__main__":
    print("Running RES-111: Receptive Field vs Order Experiment")
    print("-" * 60)

    results = run_experiment(n_samples=100, resolution=64)
    analysis = analyze_results(results)

    # Output for log
    print("\n\n--- LOG OUTPUT ---")
    status = "validated" if analysis['validated'] else "refuted"
    print(f"STATUS: {status}")
    print(f"EFFECT_SIZE: {analysis['effect_size']:.4f}")
    print(f"P_VALUE: {analysis['best_p']:.6f}")
    print(f"BEST_METRIC: {analysis['best_metric']}")
    print(f"CORRELATION: {analysis['best_r']:.4f}")
