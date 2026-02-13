"""
RES-154: Test if order metric gradient magnitude varies across image regions.

Hypothesis: gradient is larger in mid-order regions than extremes
"""

import numpy as np
from scipy import stats
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import CPPN, compute_order


def compute_gradient_magnitude(image, pixel_idx, eps=0.01):
    """Compute gradient magnitude via finite differences on a pixel."""
    y, x = np.unravel_index(pixel_idx, image.shape)

    # Forward difference
    image_plus = image.copy()
    image_plus[y, x] += eps

    # Backward difference
    image_minus = image.copy()
    image_minus[y, x] -= eps

    order_plus = compute_order(image_plus)
    order_minus = compute_order(image_minus)

    grad = (order_plus - order_minus) / (2 * eps)
    return abs(grad)


def main():
    np.random.seed(42)

    n_samples = 80
    resolution = 32

    coords = np.linspace(-1, 1, resolution)
    coords_x, coords_y = np.meshgrid(coords, coords)

    low_order_grads = []
    mid_order_grads = []
    high_order_grads = []

    print("Generating images and computing gradients...")
    for i in range(n_samples):
        cppn = CPPN()
        img = cppn.activate(coords_x, coords_y)
        order = compute_order(img)

        # Compute gradient at a random pixel
        pixel_idx = np.random.randint(0, resolution * resolution)
        grad_mag = compute_gradient_magnitude(img, pixel_idx)

        if order < 0.3:
            low_order_grads.append(grad_mag)
        elif order < 0.6:
            mid_order_grads.append(grad_mag)
        else:
            high_order_grads.append(grad_mag)

    low_order_grads = np.array(low_order_grads)
    mid_order_grads = np.array(mid_order_grads)
    high_order_grads = np.array(high_order_grads)

    # ANOVA
    f_stat, p_anova = stats.f_oneway(low_order_grads, mid_order_grads, high_order_grads)

    # Effect size (Cohen's d) between low and high
    if len(low_order_grads) > 0 and len(high_order_grads) > 0:
        pooled_std = np.sqrt((np.std(low_order_grads)**2 + np.std(high_order_grads)**2) / 2)
        effect_size = (np.mean(mid_order_grads) - np.mean(low_order_grads)) / (pooled_std + 1e-10)
    else:
        effect_size = 0.0

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Low-order gradient: {np.mean(low_order_grads):.4f} +/- {np.std(low_order_grads):.4f}")
    print(f"Mid-order gradient: {np.mean(mid_order_grads):.4f} +/- {np.std(mid_order_grads):.4f}")
    print(f"High-order gradient: {np.mean(high_order_grads):.4f} +/- {np.std(high_order_grads):.4f}")
    print(f"ANOVA F-statistic: {f_stat:.2f}, p={p_anova:.2e}")
    print(f"Effect size (Cohen's d): {effect_size:.2f}")

    validated = p_anova < 0.01 and np.mean(mid_order_grads) > np.mean(low_order_grads)
    status = 'validated' if validated else 'refuted'
    print(f"\nSTATUS: {status}")

    return validated, effect_size, p_anova


if __name__ == '__main__':
    validated, effect_size, p_value = main()
    print(f"\nFinal: d={effect_size:.2f}, p={p_value:.2e}")
