"""
RES-054: Test if high-order CPPN images concentrate energy in low frequencies.

Hypothesis: structured images have strong DC components and weak high-frequency content
"""

import numpy as np
from scipy import stats
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import CPPN, nested_sampling, compute_order


def compute_frequency_energy(image, k_threshold=8):
    """Compute energy in low vs high frequency bands."""
    # FFT
    fft = np.fft.fft2(image)
    power = np.abs(fft)**2

    # Create frequency grid
    freqs_y = np.fft.fftfreq(image.shape[0])
    freqs_x = np.fft.fftfreq(image.shape[1])
    fy, fx = np.meshgrid(freqs_y, freqs_x, indexing='ij')

    # Frequency magnitude
    freq_mag = np.sqrt(fx**2 + fy**2)

    # Low frequencies (k < k_threshold / image.shape[0])
    threshold = k_threshold / image.shape[0]
    low_freq_energy = np.sum(power[freq_mag < threshold])
    total_energy = np.sum(power)

    return low_freq_energy / (total_energy + 1e-10)


def compute_spectral_slope(image):
    """Compute power-law slope in frequency domain."""
    fft = np.fft.fft2(image)
    power = np.abs(fft)**2

    # Radial average
    y, x = np.ogrid[-1:1:image.shape[0]*1j, -1:1:image.shape[1]*1j]
    r = np.sqrt(x**2 + y**2)

    slopes = []
    for scale in range(1, 5):
        annulus_power = power[(r >= scale * 0.2) & (r < (scale + 1) * 0.2)]
        if len(annulus_power) > 0:
            slopes.append(np.mean(annulus_power))

    if len(slopes) > 1:
        log_scales = np.log(np.arange(1, len(slopes) + 1))
        log_powers = np.log(slopes + 1e-10)
        slope, _ = np.polyfit(log_scales, log_powers, 1)
        return -slope  # Positive = decay
    return 0.0


def main():
    np.random.seed(42)

    n_samples = 100
    resolution = 64

    coords = np.linspace(-1, 1, resolution)
    coords_x, coords_y = np.meshgrid(coords, coords)

    # High-order images
    print("Generating high-order CPPNs...")
    high_low_freq = []
    high_slopes = []
    high_orders = []

    for i in range(n_samples):
        cppn, order = nested_sampling(max_iterations=100, n_live=20)
        img = cppn.activate(coords_x, coords_y)

        low_freq = compute_frequency_energy(img)
        slope = compute_spectral_slope(img)

        high_low_freq.append(low_freq)
        high_slopes.append(slope)
        high_orders.append(order)

    # Low-order images
    print("Generating low-order CPPNs...")
    low_low_freq = []
    low_slopes = []
    low_orders = []

    for i in range(n_samples):
        cppn = CPPN()
        img = cppn.activate(coords_x, coords_y)
        order = compute_order(img)

        low_freq = compute_frequency_energy(img)
        slope = compute_spectral_slope(img)

        low_low_freq.append(low_freq)
        low_slopes.append(slope)
        low_orders.append(order)

    high_low_freq = np.array(high_low_freq)
    low_low_freq = np.array(low_low_freq)
    high_slopes = np.array(high_slopes)
    low_slopes = np.array(low_slopes)

    # Statistical tests
    t_freq, p_freq = stats.ttest_ind(high_low_freq, low_low_freq)
    t_slope, p_slope = stats.ttest_ind(high_slopes, low_slopes)

    # Effect sizes
    pooled_std_freq = np.sqrt((np.std(high_low_freq)**2 + np.std(low_low_freq)**2) / 2)
    d_freq = (np.mean(high_low_freq) - np.mean(low_low_freq)) / (pooled_std_freq + 1e-10)

    pooled_std_slope = np.sqrt((np.std(high_slopes)**2 + np.std(low_slopes)**2) / 2)
    d_slope = (np.mean(high_slopes) - np.mean(low_slopes)) / (pooled_std_slope + 1e-10)

    # Correlation with order
    all_orders = np.concatenate([high_orders, low_orders])
    all_slopes = np.concatenate([high_slopes, low_slopes])

    corr_slope, p_corr = stats.pearsonr(all_orders, all_slopes)

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"High-order low-frequency energy: {np.mean(high_low_freq):.1%}")
    print(f"Low-order low-frequency energy: {np.mean(low_low_freq):.1%}")
    print(f"Effect size (Cohen's d): {d_freq:.2f}")
    print(f"p-value: {p_freq:.2e}")
    print(f"\nSpectral slope correlation with order: r={corr_slope:.3f}, p={p_corr:.2e}")

    validated = d_freq > 0.5 and p_freq < 0.01
    status = 'validated' if validated else 'refuted'
    print(f"\nSTATUS: {status}")

    return validated, d_freq, p_freq


if __name__ == '__main__':
    validated, effect_size, p_value = main()
    print(f"\nFinal: d={effect_size:.2f}, p={p_value:.2e}")
