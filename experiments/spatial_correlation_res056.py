"""
RES-056: Test if high-order CPPN images have longer spatial correlation lengths.

Hypothesis: smooth functions produce larger coherent regions
"""

import numpy as np
from scipy import stats
from scipy import ndimage
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import CPPN, nested_sampling, compute_order


def compute_correlation_length(image, threshold=0.5):
    """Estimate spatial correlation length."""
    binary = (image > threshold).astype(int)

    # Autocorrelation
    mean = np.mean(binary)
    centered = binary - mean

    # Compute autocorrelation function
    autocorr = np.zeros(binary.shape[0] // 2)
    for lag in range(len(autocorr)):
        if lag == 0:
            autocorr[lag] = np.mean(centered**2)
        else:
            shifted = np.roll(centered, lag, axis=0)
            autocorr[lag] = np.mean(centered * shifted)

    # Normalize
    autocorr = autocorr / (autocorr[0] + 1e-10)

    # Find where autocorr drops to 1/e
    xi = 0
    for i in range(len(autocorr)):
        if autocorr[i] < np.exp(-1):
            xi = i
            break

    return max(xi, 1.0)


def main():
    np.random.seed(42)

    n_samples = 100
    resolution = 64

    coords = np.linspace(-1, 1, resolution)
    coords_x, coords_y = np.meshgrid(coords, coords)

    # High-order CPPNs
    print("Generating high-order CPPNs...")
    high_corr_lengths = []
    high_orders = []

    for i in range(n_samples):
        cppn, order = nested_sampling(max_iterations=100, n_live=20)
        img = cppn.activate(coords_x, coords_y)

        corr_len = compute_correlation_length(img)
        high_corr_lengths.append(corr_len)
        high_orders.append(order)

    # Low-order CPPNs
    print("Generating low-order CPPNs...")
    low_corr_lengths = []
    low_orders = []

    for i in range(n_samples):
        cppn = CPPN()
        img = cppn.activate(coords_x, coords_y)
        order = compute_order(img)

        corr_len = compute_correlation_length(img)
        low_corr_lengths.append(corr_len)
        low_orders.append(order)

    high_corr_lengths = np.array(high_corr_lengths)
    low_corr_lengths = np.array(low_corr_lengths)
    high_orders = np.array(high_orders)
    low_orders = np.array(low_orders)

    # Statistical test
    t_stat, p_value = stats.ttest_ind(high_corr_lengths, low_corr_lengths)

    # Effect size
    pooled_std = np.sqrt((np.std(high_corr_lengths)**2 + np.std(low_corr_lengths)**2) / 2)
    effect_size = (np.mean(high_corr_lengths) - np.mean(low_corr_lengths)) / (pooled_std + 1e-10)

    # Correlation with order
    all_orders = np.concatenate([high_orders, low_orders])
    all_lengths = np.concatenate([high_corr_lengths, low_corr_lengths])

    corr, p_corr = stats.pearsonr(all_orders, all_lengths)

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"High-order correlation length: {np.mean(high_corr_lengths):.2f}")
    print(f"Low-order correlation length: {np.mean(low_corr_lengths):.2f}")
    print(f"Ratio: {np.mean(high_corr_lengths) / (np.mean(low_corr_lengths) + 1e-10):.2f}x")
    print(f"Effect size (Cohen's d): {effect_size:.2f}")
    print(f"p-value: {p_value:.2e}")
    print(f"Correlation with order: r={corr:.3f}, p={p_corr:.2e}")

    validated = effect_size > 0.5 and p_value < 0.01
    status = 'validated' if validated else 'refuted'
    print(f"\nSTATUS: {status}")

    return validated, effect_size, p_value


if __name__ == '__main__':
    validated, effect_size, p_value = main()
    print(f"\nFinal: d={effect_size:.2f}, p={p_value:.2e}")
