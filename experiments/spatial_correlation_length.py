"""
RES-056: Spatial Correlation Length in CPPN Images

Hypothesis: High-order CPPN images have longer spatial correlation lengths than
low-order images, with correlation length scaling with order parameter.

Method: Compute radial autocorrelation function for CPPN images binned by order,
fit exponential decay to extract correlation length xi.
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import CPPN, order_multiplicative
from scipy import stats
from scipy.optimize import curve_fit


def compute_radial_autocorrelation(img: np.ndarray, max_lag: int = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute radially-averaged autocorrelation function.

    Returns:
        lags: Array of lag distances
        acf: Autocorrelation values at each lag
    """
    h, w = img.shape
    if max_lag is None:
        max_lag = min(h, w) // 2

    # Center the image
    img_centered = img.astype(float) - img.mean()

    # Compute 2D autocorrelation via FFT
    f = np.fft.fft2(img_centered)
    acf_2d = np.real(np.fft.ifft2(f * np.conj(f)))
    acf_2d = np.fft.fftshift(acf_2d)

    # Normalize by variance at zero lag
    acf_2d = acf_2d / acf_2d[h//2, w//2]

    # Radially average
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx)**2 + (y - cy)**2)

    lags = np.arange(1, max_lag + 1)
    acf = np.zeros(max_lag)

    for i, lag in enumerate(lags):
        mask = (r >= lag - 0.5) & (r < lag + 0.5)
        if np.any(mask):
            acf[i] = np.mean(acf_2d[mask])

    return lags, acf


def fit_correlation_length(lags: np.ndarray, acf: np.ndarray) -> float:
    """
    Fit exponential decay exp(-r/xi) to extract correlation length xi.
    """
    # Only fit positive ACF values
    valid = acf > 0.01
    if np.sum(valid) < 3:
        return 0.0

    lags_valid = lags[valid]
    acf_valid = acf[valid]

    try:
        # Fit log(acf) = -r/xi (linear in log space)
        slope, intercept, r_val, p_val, std_err = stats.linregress(
            lags_valid, np.log(acf_valid)
        )
        if slope >= 0:  # Should be negative
            return 0.0
        xi = -1.0 / slope
        return xi
    except:
        return 0.0


def main():
    np.random.seed(42)

    n_samples = 500
    size = 64  # Larger size for better correlation estimates

    print("RES-056: Spatial Correlation Length Analysis")
    print("=" * 60)
    print(f"Generating {n_samples} CPPN images at {size}x{size}...")

    orders = []
    correlation_lengths = []

    for i in range(n_samples):
        cppn = CPPN()
        img = cppn.render(size)
        order = order_multiplicative(img)

        lags, acf = compute_radial_autocorrelation(img)
        xi = fit_correlation_length(lags, acf)

        orders.append(order)
        correlation_lengths.append(xi)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{n_samples}")

    orders = np.array(orders)
    correlation_lengths = np.array(correlation_lengths)

    # Remove any failed fits
    valid = correlation_lengths > 0
    orders_valid = orders[valid]
    xi_valid = correlation_lengths[valid]

    print(f"\nValid fits: {np.sum(valid)}/{n_samples}")

    # Bin by order quintiles
    quintiles = np.percentile(orders_valid, [0, 20, 40, 60, 80, 100])

    print("\nCorrelation Length by Order Quintile:")
    print("-" * 50)

    quintile_data = []
    for i in range(5):
        mask = (orders_valid >= quintiles[i]) & (orders_valid < quintiles[i+1] + 1e-10)
        xi_q = xi_valid[mask]
        order_q = orders_valid[mask]

        mean_xi = np.mean(xi_q)
        std_xi = np.std(xi_q)
        mean_order = np.mean(order_q)

        print(f"Q{i+1} (order {quintiles[i]:.3f}-{quintiles[i+1]:.3f}): "
              f"xi = {mean_xi:.2f} +/- {std_xi:.2f}, n={len(xi_q)}")

        quintile_data.append((mean_order, mean_xi, std_xi, len(xi_q)))

    # Correlation between order and correlation length
    r, p = stats.pearsonr(orders_valid, xi_valid)

    # Spearman for robustness
    rho, p_spearman = stats.spearmanr(orders_valid, xi_valid)

    print("\n" + "=" * 60)
    print("STATISTICAL RESULTS:")
    print(f"  Pearson correlation (order vs xi): r = {r:.4f}, p = {p:.2e}")
    print(f"  Spearman correlation: rho = {rho:.4f}, p = {p_spearman:.2e}")

    # Compare high vs low order (top/bottom 20%)
    low_mask = orders_valid <= np.percentile(orders_valid, 20)
    high_mask = orders_valid >= np.percentile(orders_valid, 80)

    xi_low = xi_valid[low_mask]
    xi_high = xi_valid[high_mask]

    t_stat, p_ttest = stats.ttest_ind(xi_high, xi_low)
    effect_size = (np.mean(xi_high) - np.mean(xi_low)) / np.sqrt(
        (np.std(xi_high)**2 + np.std(xi_low)**2) / 2
    )

    print(f"\nHigh vs Low Order Comparison:")
    print(f"  Low order (Q1): xi = {np.mean(xi_low):.2f} +/- {np.std(xi_low):.2f}")
    print(f"  High order (Q5): xi = {np.mean(xi_high):.2f} +/- {np.std(xi_high):.2f}")
    print(f"  t-statistic: {t_stat:.2f}")
    print(f"  p-value: {p_ttest:.2e}")
    print(f"  Cohen's d: {effect_size:.2f}")

    # Determine status
    if p < 0.01 and abs(effect_size) > 0.5:
        status = "VALIDATED" if r > 0 else "REFUTED (negative correlation)"
    elif p < 0.05:
        status = "INCONCLUSIVE (weak evidence)"
    else:
        status = "REFUTED (no significant correlation)"

    print(f"\n{'=' * 60}")
    print(f"STATUS: {status}")
    print(f"{'=' * 60}")

    # Output for log_manager
    print(f"\n## SUMMARY FOR LOG ##")
    print(f"effect_size: {effect_size:.3f}")
    print(f"p_value: {p:.2e}")
    print(f"correlation_r: {r:.4f}")
    print(f"mean_xi_low: {np.mean(xi_low):.2f}")
    print(f"mean_xi_high: {np.mean(xi_high):.2f}")

    return {
        'status': status,
        'effect_size': effect_size,
        'p_value': p,
        'r': r,
        'xi_low': np.mean(xi_low),
        'xi_high': np.mean(xi_high)
    }


if __name__ == '__main__':
    main()
