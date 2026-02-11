"""
RES-150: Test if high-order CPPN images exhibit scale invariance.

Hypothesis: structured images have similar structure at multiple zoom levels
"""

import numpy as np
from scipy import stats
from scipy import ndimage
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import CPPN, nested_sampling, compute_order


def compute_scale_invariance(image):
    """Compute correlation between image at different scales."""
    # Downsample and compare with original downsampled
    orig_downsampled = ndimage.zoom(image, 0.25, order=1)

    # Upsample back to original size
    upsampled = ndimage.zoom(orig_downsampled, 4, order=1)

    # Clip to match size
    upsampled = upsampled[:image.shape[0], :image.shape[1]]

    # Compute correlation
    corr = np.corrcoef(image.flatten(), upsampled.flatten())[0, 1]

    return corr


def compute_fractal_dimension(image, threshold=0.5):
    """Estimate fractal dimension via box-counting."""
    binary = (image > threshold).astype(int)

    box_counts = []
    for scale in [1, 2, 4, 8, 16]:
        boxes = binary.shape[0] // scale
        if boxes == 0:
            continue
        count = 0
        for i in range(0, binary.shape[0], scale):
            for j in range(0, binary.shape[1], scale):
                region = binary[i:i+scale, j:j+scale]
                if np.any(region):
                    count += 1
        box_counts.append((scale, count))

    if len(box_counts) > 1:
        scales = np.array([b[0] for b in box_counts])
        counts = np.array([b[1] for b in box_counts])
        dim, _ = np.polyfit(np.log(scales), np.log(counts), 1)
        return -dim
    return 0.0


def main():
    np.random.seed(42)

    n_samples = 100
    resolution = 64

    coords = np.linspace(-1, 1, resolution)
    coords_x, coords_y = np.meshgrid(coords, coords)

    # Generate high-order CPPNs
    print("Generating high-order CPPNs...")
    high_scale_inv = []
    high_fractal = []
    high_orders = []

    for i in range(n_samples):
        cppn, order = nested_sampling(max_iterations=100, n_live=20)
        img = cppn.activate(coords_x, coords_y)
        scale_inv = compute_scale_invariance(img)
        fractal = compute_fractal_dimension(img)
        high_scale_inv.append(scale_inv)
        high_fractal.append(fractal)
        high_orders.append(order)

    # Generate low-order CPPNs
    print("Generating low-order CPPNs...")
    low_scale_inv = []
    low_fractal = []
    low_orders = []

    for i in range(n_samples):
        cppn = CPPN()
        img = cppn.activate(coords_x, coords_y)
        order = compute_order(img)
        scale_inv = compute_scale_invariance(img)
        fractal = compute_fractal_dimension(img)
        low_scale_inv.append(scale_inv)
        low_fractal.append(fractal)
        low_orders.append(order)

    high_scale_inv = np.array(high_scale_inv)
    low_scale_inv = np.array(low_scale_inv)

    # Statistical test
    t_stat, p_value = stats.ttest_ind(high_scale_inv, low_scale_inv)

    # Effect size
    pooled_std = np.sqrt((np.std(high_scale_inv)**2 + np.std(low_scale_inv)**2) / 2)
    effect_size = (np.mean(high_scale_inv) - np.mean(low_scale_inv)) / (pooled_std + 1e-10)

    # Correlation with order
    all_orders = np.concatenate([high_orders, low_orders])
    all_scales = np.concatenate([high_scale_inv, low_scale_inv])
    all_fractals = np.concatenate([high_fractal, low_fractal])

    corr_scale, p_scale = stats.pearsonr(all_orders, all_scales)
    corr_fractal, p_fractal = stats.pearsonr(all_orders, all_fractals)

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"High-order scale invariance: {np.mean(high_scale_inv):.4f}")
    print(f"Low-order scale invariance: {np.mean(low_scale_inv):.4f}")
    print(f"Effect size (Cohen's d): {effect_size:.2f}")
    print(f"p-value: {p_value:.2e}")
    print(f"Scale invariance correlation with order: r={corr_scale:.3f}, p={p_scale:.2e}")
    print(f"Fractal dimension correlation: r={corr_fractal:.3f}, p={p_fractal:.2e}")

    validated = effect_size > 0.5 and p_scale < 0.01
    status = 'validated' if validated else 'refuted'
    print(f"\nSTATUS: {status}")

    return validated, effect_size, p_value


if __name__ == '__main__':
    validated, effect_size, p_value = main()
    print(f"\nFinal: d={effect_size:.2f}, p={p_value:.2e}")
