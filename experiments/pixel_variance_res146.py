"""
RES-146: Test if CPPN images have lower local pixel variance than random images.

Hypothesis: smooth CPPN functions produce spatially coherent patches
"""

import numpy as np
from scipy import stats
from scipy.ndimage import uniform_filter
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import CPPN, nested_sampling, compute_order


def compute_local_variance(image, neighborhood_size=3):
    """Compute mean local pixel variance in neighborhoods."""
    # Use uniform_filter to compute local statistics
    mean_filter = uniform_filter(image, size=neighborhood_size, mode='reflect')
    sq_filter = uniform_filter(image**2, size=neighborhood_size, mode='reflect')
    local_var = sq_filter - mean_filter**2

    return np.mean(local_var)


def main():
    np.random.seed(42)

    n_samples = 100
    resolution = 64

    coords = np.linspace(-1, 1, resolution)
    coords_x, coords_y = np.meshgrid(coords, coords)

    # Generate CPPN images
    print("Generating CPPN images...")
    cppn_variances = []
    for i in range(n_samples):
        cppn, order = nested_sampling(max_iterations=100, n_live=20)
        img = cppn.activate(coords_x, coords_y)
        var = compute_local_variance(img)
        cppn_variances.append(var)

    # Generate random images
    print("Generating random images...")
    random_variances = []
    for i in range(n_samples):
        img = np.random.random((resolution, resolution))
        var = compute_local_variance(img)
        random_variances.append(var)

    cppn_variances = np.array(cppn_variances)
    random_variances = np.array(random_variances)

    # Statistical test
    t_stat, p_value = stats.ttest_ind(cppn_variances, random_variances)

    # Effect size
    pooled_std = np.sqrt((np.std(cppn_variances)**2 + np.std(random_variances)**2) / 2)
    effect_size = (np.mean(random_variances) - np.mean(cppn_variances)) / pooled_std

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"CPPN local variance: {np.mean(cppn_variances):.6f}")
    print(f"Random local variance: {np.mean(random_variances):.6f}")
    print(f"Ratio: {np.mean(random_variances) / (np.mean(cppn_variances) + 1e-10):.1f}x")
    print(f"Effect size (Cohen's d): {effect_size:.2f}")
    print(f"p-value: {p_value:.2e}")

    validated = effect_size > 0.5 and p_value < 0.01
    status = 'validated' if validated else 'refuted'
    print(f"\nSTATUS: {status}")

    return validated, effect_size, p_value


if __name__ == '__main__':
    validated, effect_size, p_value = main()
    print(f"\nFinal: d={effect_size:.2f}, p={p_value:.2e}")
