"""
RES-151: Test if CPPN images have lower variance in order across thresholds.

Hypothesis: CPPN sigmoid outputs concentrate near 0/1, making order stable
"""

import numpy as np
from scipy import stats
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import CPPN, nested_sampling, compute_order


def compute_order_at_thresholds(image, thresholds=None):
    """Compute order at multiple thresholds."""
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 9)

    orders = []
    for thresh in thresholds:
        order = compute_order(image, threshold=thresh)
        orders.append(order)

    return np.array(orders)


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
        orders = compute_order_at_thresholds(img)
        variance = np.var(orders)
        cppn_variances.append(variance)

    # Generate random images
    print("Generating random images...")
    random_variances = []
    for i in range(n_samples):
        img = np.random.random((resolution, resolution))
        orders = compute_order_at_thresholds(img)
        variance = np.var(orders)
        random_variances.append(variance)

    cppn_variances = np.array(cppn_variances)
    random_variances = np.array(random_variances)

    # Statistical test
    t_stat, p_value = stats.ttest_ind(cppn_variances, random_variances)

    # Effect size
    pooled_std = np.sqrt((np.std(cppn_variances)**2 + np.std(random_variances)**2) / 2)
    effect_size = (np.mean(random_variances) - np.mean(cppn_variances)) / (pooled_std + 1e-10)

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"CPPN order variance across thresholds: {np.mean(cppn_variances):.6f}")
    print(f"Random order variance across thresholds: {np.mean(random_variances):.6f}")
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
