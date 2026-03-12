"""
RES-054: Frequency Band Concentration Experiment

Hypothesis: High-order CPPN images concentrate energy in a characteristic
frequency band (DC to k=8), while low-order images distribute energy uniformly
across all frequencies.

Tests if band-specific energy ratio predicts order better than overall spectral slope.
"""

import numpy as np
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.thermo_sampler_v3 import CPPN, order_multiplicative as compute_order
from scipy import stats

def compute_fft_bands(image: np.ndarray, band_edges: list = [0, 4, 8, 16]):
    """Compute energy in different frequency bands using 2D FFT."""
    fft = np.fft.fft2(image.astype(float))
    fft_shifted = np.fft.fftshift(fft)
    power = np.abs(fft_shifted) ** 2

    h, w = image.shape
    cy, cx = h // 2, w // 2

    # Create radial distance map
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((y - cy)**2 + (x - cx)**2)

    band_energies = []
    for i in range(len(band_edges) - 1):
        mask = (r >= band_edges[i]) & (r < band_edges[i+1])
        band_energies.append(np.sum(power[mask]))

    # Also get high frequency band (beyond last edge)
    mask_high = r >= band_edges[-1]
    band_energies.append(np.sum(power[mask_high]))

    total_energy = np.sum(power)
    band_ratios = [e / total_energy if total_energy > 0 else 0 for e in band_energies]

    return {
        'band_energies': band_energies,
        'band_ratios': band_ratios,
        'total_energy': total_energy,
        'low_freq_ratio': band_ratios[0] + band_ratios[1],  # DC to k=8
        'high_freq_ratio': band_ratios[2] + band_ratios[3] if len(band_ratios) > 3 else band_ratios[2]  # k>8
    }

def compute_spectral_slope(image: np.ndarray):
    """Compute power spectral slope (beta in P(k) ~ k^beta)."""
    fft = np.fft.fft2(image.astype(float))
    fft_shifted = np.fft.fftshift(fft)
    power = np.abs(fft_shifted) ** 2

    h, w = image.shape
    cy, cx = h // 2, w // 2

    y, x = np.ogrid[:h, :w]
    r = np.sqrt((y - cy)**2 + (x - cx)**2)

    # Radially average
    max_r = min(cy, cx)
    radii = np.arange(1, max_r)
    radial_power = []

    for rad in radii:
        mask = (r >= rad - 0.5) & (r < rad + 0.5)
        if np.any(mask):
            radial_power.append(np.mean(power[mask]))
        else:
            radial_power.append(0)

    radial_power = np.array(radial_power)
    valid = radial_power > 0

    if np.sum(valid) > 2:
        log_k = np.log(radii[valid])
        log_p = np.log(radial_power[valid])
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_k, log_p)
        return slope
    return 0

def generate_sample(size=32):
    """Generate a single CPPN image with its order."""
    cppn = CPPN()
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)

    continuous = cppn.activate(X, Y)
    binary = (continuous > 0.5).astype(int)
    order = compute_order(binary)

    return binary, order

def run_experiment(n_samples=500, image_size=32, seed=42):
    """Run the frequency band experiment."""
    np.random.seed(seed)

    results = []
    for i in range(n_samples):
        img, order = generate_sample(image_size)
        band_data = compute_fft_bands(img)
        slope = compute_spectral_slope(img)

        results.append({
            'order': order,
            'low_freq_ratio': band_data['low_freq_ratio'],
            'high_freq_ratio': band_data['high_freq_ratio'],
            'band_ratios': band_data['band_ratios'],
            'spectral_slope': slope
        })

        if (i + 1) % 100 == 0:
            print(f"Generated {i+1}/{n_samples} samples")

    # Analysis
    orders = np.array([r['order'] for r in results])
    low_freq_ratios = np.array([r['low_freq_ratio'] for r in results])
    high_freq_ratios = np.array([r['high_freq_ratio'] for r in results])
    slopes = np.array([r['spectral_slope'] for r in results])

    # Correlation: order vs low_freq_ratio
    rho_low, p_low = stats.spearmanr(orders, low_freq_ratios)

    # Correlation: order vs spectral slope
    rho_slope, p_slope = stats.spearmanr(orders, slopes)

    # Compare predictive power: multiple regression
    from scipy.stats import pearsonr
    r_low, _ = pearsonr(orders, low_freq_ratios)
    r_slope, _ = pearsonr(orders, slopes)

    # Group comparison: high vs low order
    median_order = np.median(orders)
    high_order_mask = orders > median_order
    low_order_mask = orders <= median_order

    low_freq_high_order = low_freq_ratios[high_order_mask]
    low_freq_low_order = low_freq_ratios[low_order_mask]

    stat_u, p_u = stats.mannwhitneyu(low_freq_high_order, low_freq_low_order, alternative='two-sided')

    # Effect size (Cohen's d)
    cohens_d = (np.mean(low_freq_high_order) - np.mean(low_freq_low_order)) / np.sqrt(
        (np.std(low_freq_high_order)**2 + np.std(low_freq_low_order)**2) / 2
    )

    # Which predicts better?
    r2_low = r_low ** 2
    r2_slope = r_slope ** 2

    # Quartile analysis
    quartiles = np.percentile(orders, [25, 50, 75])
    q_labels = ['Q1 (lowest)', 'Q2', 'Q3', 'Q4 (highest)']
    q_masks = [
        orders <= quartiles[0],
        (orders > quartiles[0]) & (orders <= quartiles[1]),
        (orders > quartiles[1]) & (orders <= quartiles[2]),
        orders > quartiles[2]
    ]

    quartile_means = {
        'low_freq_ratio': [float(np.mean(low_freq_ratios[m])) for m in q_masks],
        'high_freq_ratio': [float(np.mean(high_freq_ratios[m])) for m in q_masks],
        'spectral_slope': [float(np.mean(slopes[m])) for m in q_masks]
    }

    output = {
        'n_samples': n_samples,
        'correlations': {
            'order_vs_low_freq_ratio': {'rho': float(rho_low), 'p': float(p_low)},
            'order_vs_spectral_slope': {'rho': float(rho_slope), 'p': float(p_slope)}
        },
        'r_squared': {
            'low_freq_ratio': float(r2_low),
            'spectral_slope': float(r2_slope),
            'low_freq_better': bool(r2_low > r2_slope)
        },
        'group_comparison': {
            'high_order_low_freq_mean': float(np.mean(low_freq_high_order)),
            'low_order_low_freq_mean': float(np.mean(low_freq_low_order)),
            'mann_whitney_U': float(stat_u),
            'p_value': float(p_u),
            'cohens_d': float(cohens_d)
        },
        'quartile_analysis': quartile_means,
        'summary_stats': {
            'mean_order': float(np.mean(orders)),
            'mean_low_freq_ratio': float(np.mean(low_freq_ratios)),
            'mean_high_freq_ratio': float(np.mean(high_freq_ratios)),
            'mean_spectral_slope': float(np.mean(slopes))
        }
    }

    return output

if __name__ == '__main__':
    print("Running RES-054: Frequency Band Concentration Experiment")
    print("=" * 60)

    results = run_experiment(n_samples=500, image_size=32, seed=42)

    # Save results
    os.makedirs('results/frequency_band', exist_ok=True)
    with open('results/frequency_band/frequency_band_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n=== RESULTS ===")
    print(f"\nCorrelation (order vs low_freq_ratio): rho={results['correlations']['order_vs_low_freq_ratio']['rho']:.3f}, p={results['correlations']['order_vs_low_freq_ratio']['p']:.2e}")
    print(f"Correlation (order vs spectral_slope): rho={results['correlations']['order_vs_spectral_slope']['rho']:.3f}, p={results['correlations']['order_vs_spectral_slope']['p']:.2e}")

    print(f"\nR^2 (low_freq_ratio): {results['r_squared']['low_freq_ratio']:.3f}")
    print(f"R^2 (spectral_slope): {results['r_squared']['spectral_slope']:.3f}")
    print(f"Low freq ratio predicts better: {results['r_squared']['low_freq_better']}")

    print(f"\nGroup comparison (high vs low order):")
    print(f"  High-order low_freq_mean: {results['group_comparison']['high_order_low_freq_mean']:.3f}")
    print(f"  Low-order low_freq_mean: {results['group_comparison']['low_order_low_freq_mean']:.3f}")
    print(f"  Mann-Whitney U: {results['group_comparison']['mann_whitney_U']:.1f}")
    print(f"  p-value: {results['group_comparison']['p_value']:.2e}")
    print(f"  Cohen's d: {results['group_comparison']['cohens_d']:.2f}")

    print("\nQuartile progression (low_freq_ratio):")
    for i, val in enumerate(results['quartile_analysis']['low_freq_ratio']):
        print(f"  Q{i+1}: {val:.3f}")

    print("\n=== VERDICT ===")
    d = abs(results['group_comparison']['cohens_d'])
    p = results['group_comparison']['p_value']

    if p < 0.01 and d > 0.5:
        print("VALIDATED: Strong effect with significance p<0.01 and |d|>0.5")
    elif p < 0.05:
        print("INCONCLUSIVE: Significant but weak effect")
    else:
        print("REFUTED: No significant effect")

    print(f"\nResults saved to: results/frequency_band/frequency_band_results.json")
