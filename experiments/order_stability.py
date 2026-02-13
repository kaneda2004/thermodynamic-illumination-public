"""
RES-063: Order Stability Under Pixel Perturbations

Hypothesis: High-order images exhibit greater robustness (smaller order
degradation rate) under random pixel flips compared to medium-order images.

This tests whether the order metric is more stable for highly-structured
images (high order) vs moderately structured ones.
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import (
    CPPN, order_multiplicative, set_global_seed
)
from scipy import stats


def generate_image(cppn: CPPN, size: int = 32) -> np.ndarray:
    """Generate binary image from CPPN."""
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(x, y)
    img = cppn.activate(xx, yy)
    return (img > 0.5).astype(np.uint8)


def perturb_image(img: np.ndarray, flip_fraction: float, seed: int = None) -> np.ndarray:
    """Flip a fraction of pixels randomly."""
    if seed is not None:
        np.random.seed(seed)
    perturbed = img.copy()
    n_pixels = img.size
    n_flip = int(n_pixels * flip_fraction)

    # Random pixel indices to flip
    flat = perturbed.flatten()
    flip_indices = np.random.choice(n_pixels, size=n_flip, replace=False)
    flat[flip_indices] = 1 - flat[flip_indices]

    return flat.reshape(img.shape)


def measure_stability(img: np.ndarray, flip_fractions: list, n_trials: int = 10) -> dict:
    """Measure how order degrades with increasing perturbation."""
    original_order = order_multiplicative(img)

    results = {'original_order': original_order, 'fractions': [], 'mean_orders': [], 'std_orders': []}

    for frac in flip_fractions:
        orders = []
        for trial in range(n_trials):
            perturbed = perturb_image(img, frac, seed=trial * 1000 + int(frac * 1000))
            orders.append(order_multiplicative(perturbed))
        results['fractions'].append(frac)
        results['mean_orders'].append(np.mean(orders))
        results['std_orders'].append(np.std(orders))

    return results


def compute_degradation_rate(results: dict) -> float:
    """Compute slope of order degradation vs flip fraction."""
    fractions = np.array(results['fractions'])
    orders = np.array(results['mean_orders'])

    # Linear regression: order = a + b * fraction
    # Degradation rate is -b (negative slope)
    slope, intercept, r_value, p_value, std_err = stats.linregress(fractions, orders)

    return {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_value ** 2,
        'degradation_rate': -slope,  # positive = order decreases
        'std_err': std_err
    }


def main():
    print("=" * 60)
    print("RES-063: Order Stability Under Pixel Perturbations")
    print("=" * 60)

    set_global_seed(42)

    # Parameters
    n_cppns = 100  # Generate many CPPNs to get a range of orders
    img_size = 32
    flip_fractions = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]
    n_trials_per_fraction = 10

    # Collect images and their initial orders
    print(f"\nGenerating {n_cppns} CPPN images...")
    images = []
    initial_orders = []

    for i in range(n_cppns):
        set_global_seed(i * 7 + 13)  # Different seed for each CPPN
        cppn = CPPN()
        img = generate_image(cppn, img_size)
        order = order_multiplicative(img)
        images.append(img)
        initial_orders.append(order)

    initial_orders = np.array(initial_orders)
    print(f"Order distribution: min={initial_orders.min():.3f}, max={initial_orders.max():.3f}, "
          f"mean={initial_orders.mean():.3f}, std={initial_orders.std():.3f}")

    # Split into high-order (top 25%) and medium-order (25-75 percentile) groups
    p75 = np.percentile(initial_orders, 75)
    p25 = np.percentile(initial_orders, 25)

    high_order_mask = initial_orders >= p75
    medium_order_mask = (initial_orders >= p25) & (initial_orders < p75)

    high_order_images = [img for img, mask in zip(images, high_order_mask) if mask]
    medium_order_images = [img for img, mask in zip(images, medium_order_mask) if mask]

    print(f"\nHigh-order group (order >= {p75:.3f}): {len(high_order_images)} images")
    print(f"Medium-order group ({p25:.3f} <= order < {p75:.3f}): {len(medium_order_images)} images")

    # Measure stability for each group
    print("\nMeasuring stability for high-order images...")
    high_order_degradation_rates = []
    high_order_initial = []

    for img in high_order_images:
        stability = measure_stability(img, flip_fractions, n_trials_per_fraction)
        degradation = compute_degradation_rate(stability)
        high_order_degradation_rates.append(degradation['degradation_rate'])
        high_order_initial.append(stability['original_order'])

    print("Measuring stability for medium-order images...")
    medium_order_degradation_rates = []
    medium_order_initial = []

    for img in medium_order_images:
        stability = measure_stability(img, flip_fractions, n_trials_per_fraction)
        degradation = compute_degradation_rate(stability)
        medium_order_degradation_rates.append(degradation['degradation_rate'])
        medium_order_initial.append(stability['original_order'])

    high_rates = np.array(high_order_degradation_rates)
    medium_rates = np.array(medium_order_degradation_rates)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\nHigh-order group (n={len(high_rates)}):")
    print(f"  Initial order: {np.mean(high_order_initial):.3f} +/- {np.std(high_order_initial):.3f}")
    print(f"  Degradation rate: {high_rates.mean():.4f} +/- {high_rates.std():.4f}")

    print(f"\nMedium-order group (n={len(medium_rates)}):")
    print(f"  Initial order: {np.mean(medium_order_initial):.3f} +/- {np.std(medium_order_initial):.3f}")
    print(f"  Degradation rate: {medium_rates.mean():.4f} +/- {medium_rates.std():.4f}")

    # Statistical test: Mann-Whitney U (non-parametric)
    u_stat, p_value = stats.mannwhitneyu(high_rates, medium_rates, alternative='less')

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((high_rates.std()**2 + medium_rates.std()**2) / 2)
    cohens_d = (high_rates.mean() - medium_rates.mean()) / pooled_std if pooled_std > 0 else 0

    print(f"\nStatistical Comparison:")
    print(f"  Mann-Whitney U: {u_stat:.1f}")
    print(f"  P-value (high < medium): {p_value:.6f}")
    print(f"  Cohen's d: {cohens_d:.3f}")

    # Also compute correlation between initial order and degradation rate
    all_orders = np.array(high_order_initial + medium_order_initial)
    all_rates = np.concatenate([high_rates, medium_rates])

    corr, corr_p = stats.pearsonr(all_orders, all_rates)
    spearman_corr, spearman_p = stats.spearmanr(all_orders, all_rates)

    print(f"\nCorrelation (all images):")
    print(f"  Pearson r (order vs degradation): {corr:.3f} (p={corr_p:.6f})")
    print(f"  Spearman rho: {spearman_corr:.3f} (p={spearman_p:.6f})")

    # Determine outcome
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)

    # Hypothesis: high-order images have LOWER degradation rates (more stable)
    if p_value < 0.01 and cohens_d < -0.5:
        status = "VALIDATED"
        conclusion = f"High-order images are more stable (lower degradation rate) with effect size d={cohens_d:.2f}"
    elif p_value < 0.01 and cohens_d > 0.5:
        status = "REFUTED"
        conclusion = f"Opposite effect: high-order images LESS stable with effect size d={cohens_d:.2f}"
    elif corr_p < 0.01 and abs(corr) > 0.3:
        if corr < 0:
            status = "VALIDATED"
            conclusion = f"Significant negative correlation (r={corr:.3f}): higher order -> lower degradation"
        else:
            status = "REFUTED"
            conclusion = f"Opposite correlation (r={corr:.3f}): higher order -> higher degradation"
    else:
        status = "INCONCLUSIVE"
        conclusion = f"No significant relationship found (p={p_value:.3f}, d={cohens_d:.2f})"

    print(f"Status: {status}")
    print(f"Summary: {conclusion}")

    return {
        'status': status.lower(),
        'high_order_mean_degradation': float(high_rates.mean()),
        'medium_order_mean_degradation': float(medium_rates.mean()),
        'effect_size': float(cohens_d),
        'p_value': float(p_value),
        'correlation': float(corr),
        'correlation_p': float(corr_p),
        'spearman_rho': float(spearman_corr),
        'spearman_p': float(spearman_p),
        'conclusion': conclusion
    }


if __name__ == "__main__":
    results = main()
