"""
RES-117: Test if CPPN image order correlates with wavelet energy decay.

Hypothesis: structured images have rapid energy decay from coarse to fine scales
"""

import numpy as np
from scipy import stats
from scipy import signal
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import CPPN, nested_sampling, compute_order


def compute_wavelet_decay(image):
    """Compute wavelet scale decay rate."""
    # Simple multi-scale analysis via Gaussian blur
    energies = [np.sum(image**2)]

    for scale in range(1, 5):
        sigma = scale
        blurred = signal.convolve2d(image, np.ones((3,3))/9, mode='same')
        energies.append(np.sum(blurred**2))

    energies = np.array(energies)

    # Fit exponential decay: E(scale) ~ exp(-rate * scale)
    scales = np.arange(len(energies))
    log_energies = np.log(energies + 1e-10)

    slope, _ = np.polyfit(scales, log_energies, 1)
    return -slope  # Positive = rapid decay


def main():
    np.random.seed(42)

    n_samples = 100
    resolution = 64

    coords = np.linspace(-1, 1, resolution)
    coords_x, coords_y = np.meshgrid(coords, coords)

    # Generate CPPN images with varying order
    print("Generating CPPN images...")
    cppn_images = []
    orders = []
    decay_rates = []

    for i in range(n_samples):
        cppn, order = nested_sampling(max_iterations=100, n_live=20)
        img = cppn.activate(coords_x, coords_y)
        cppn_images.append(img)
        orders.append(order)
        decay = compute_wavelet_decay(img)
        decay_rates.append(decay)

    orders = np.array(orders)
    decay_rates = np.array(decay_rates)

    # Correlation analysis
    corr, p_corr = stats.pearsonr(orders, decay_rates)

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Correlation between order and wavelet decay: r={corr:.3f}")
    print(f"p-value: {p_corr:.2e}")
    print(f"Mean decay rate: {np.mean(decay_rates):.4f}")

    validated = corr > 0.5 and p_corr < 0.01
    status = 'validated' if validated else 'refuted'
    print(f"\nSTATUS: {status}")

    return validated, corr, p_corr


if __name__ == '__main__':
    validated, effect_size, p_value = main()
    print(f"\nFinal: r={effect_size:.3f}, p={p_value:.2e}")
