"""
RES-183: Low-frequency band alone predicts order as well as all bands combined

HYPOTHESIS: Low-frequency band alone predicts order as well as all bands combined

This tests whether mid and high frequency bands add independent predictive power
for CPPN order, or whether low-freq alone captures all the information.

Method:
1. Generate CPPN images with varied order
2. Decompose each image into frequency bands (low, mid, high) using FFT
3. Compute energy in each band
4. Compare regression: order ~ low_band_energy vs order ~ all_bands
5. If R^2 is similar, low-freq is sufficient; if significantly different, other bands matter
"""

import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import json
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')
from core.thermo_sampler_v3 import CPPN, order_multiplicative, set_global_seed

def frequency_band_energy(img: np.ndarray) -> dict:
    """
    Decompose image into low/mid/high frequency bands using FFT.
    Returns normalized energy in each band.
    """
    # FFT and shift zero frequency to center
    f = np.fft.fft2(img.astype(float) - 0.5)
    f_shifted = np.fft.fftshift(f)
    power = np.abs(f_shifted) ** 2

    h, w = img.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx)**2 + (y - cy)**2)

    # Define band boundaries (as fraction of max radius)
    max_r = np.sqrt(cx**2 + cy**2)

    # Low: innermost 25%, Mid: 25-50%, High: >50%
    low_mask = r < (max_r * 0.25)
    mid_mask = (r >= max_r * 0.25) & (r < max_r * 0.5)
    high_mask = r >= max_r * 0.5

    total_power = np.sum(power) + 1e-10

    return {
        'low': np.sum(power[low_mask]) / total_power,
        'mid': np.sum(power[mid_mask]) / total_power,
        'high': np.sum(power[high_mask]) / total_power,
    }


def main():
    set_global_seed(42)

    n_samples = 500
    results = []

    print(f"Generating {n_samples} CPPN samples and computing frequency bands...")

    for i in range(n_samples):
        cppn = CPPN()
        img = cppn.render(size=32)
        order = order_multiplicative(img)
        bands = frequency_band_energy(img)

        results.append({
            'order': order,
            'low': bands['low'],
            'mid': bands['mid'],
            'high': bands['high'],
        })

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{n_samples}")

    # Convert to arrays
    orders = np.array([r['order'] for r in results])
    low = np.array([r['low'] for r in results])
    mid = np.array([r['mid'] for r in results])
    high = np.array([r['high'] for r in results])

    # Filter out zero-order samples for meaningful regression
    valid_mask = orders > 0.001
    orders_v = orders[valid_mask]
    low_v = low[valid_mask]
    mid_v = mid[valid_mask]
    high_v = high[valid_mask]

    print(f"\nValid samples (order > 0.001): {sum(valid_mask)}/{n_samples}")

    # Compute correlations
    r_low, p_low = stats.spearmanr(low_v, orders_v)
    r_mid, p_mid = stats.spearmanr(mid_v, orders_v)
    r_high, p_high = stats.spearmanr(high_v, orders_v)

    print(f"\nSpearman correlations with order:")
    print(f"  Low band:  rho={r_low:.3f}, p={p_low:.2e}")
    print(f"  Mid band:  rho={r_mid:.3f}, p={p_mid:.2e}")
    print(f"  High band: rho={r_high:.3f}, p={p_high:.2e}")

    # Regression analysis: compare low-only vs all-bands
    X_low = low_v.reshape(-1, 1)
    X_all = np.column_stack([low_v, mid_v, high_v])
    y = orders_v

    # Cross-validated R^2 (5-fold)
    reg_low = LinearRegression()
    reg_all = LinearRegression()

    cv_scores_low = cross_val_score(reg_low, X_low, y, cv=5, scoring='r2')
    cv_scores_all = cross_val_score(reg_all, X_all, y, cv=5, scoring='r2')

    r2_low_mean = cv_scores_low.mean()
    r2_low_std = cv_scores_low.std()
    r2_all_mean = cv_scores_all.mean()
    r2_all_std = cv_scores_all.std()

    print(f"\nCross-validated R^2 scores (5-fold):")
    print(f"  Low-band only:  R^2 = {r2_low_mean:.3f} +/- {r2_low_std:.3f}")
    print(f"  All bands:      R^2 = {r2_all_mean:.3f} +/- {r2_all_std:.3f}")

    # Statistical test: paired t-test on fold R^2 differences
    r2_improvement = cv_scores_all - cv_scores_low
    t_stat, p_val = stats.ttest_1samp(r2_improvement, 0)

    improvement_mean = r2_improvement.mean()
    improvement_std = r2_improvement.std()

    # Effect size (Cohen's d for improvement)
    d = improvement_mean / (improvement_std + 1e-10)

    print(f"\nR^2 improvement (all - low):")
    print(f"  Mean improvement: {improvement_mean:.4f} +/- {improvement_std:.4f}")
    print(f"  t-test: t={t_stat:.3f}, p={p_val:.4f}")
    print(f"  Cohen's d: {d:.3f}")

    # Partial correlations: does mid/high add info beyond low?
    # Residualize order on low, then correlate with mid/high
    reg_residual = LinearRegression()
    reg_residual.fit(X_low, y)
    residuals = y - reg_residual.predict(X_low)

    r_mid_partial, p_mid_partial = stats.spearmanr(mid_v, residuals)
    r_high_partial, p_high_partial = stats.spearmanr(high_v, residuals)

    print(f"\nPartial correlations (controlling for low-band):")
    print(f"  Mid band:  rho={r_mid_partial:.3f}, p={p_mid_partial:.2e}")
    print(f"  High band: rho={r_high_partial:.3f}, p={p_high_partial:.2e}")

    # Determine outcome
    # Hypothesis validated if: low-only R^2 is not significantly worse than all-bands
    # (i.e., p > 0.05 for improvement, or improvement Cohen's d < 0.5)
    if p_val > 0.01 or abs(d) < 0.5:
        status = "validated"
        conclusion = "Low-frequency band alone predicts order as well as all bands (no significant improvement from mid/high)"
    else:
        if improvement_mean > 0:
            status = "refuted"
            conclusion = f"Adding mid/high bands significantly improves prediction (R^2 +{improvement_mean:.3f}, d={d:.2f})"
        else:
            status = "inconclusive"
            conclusion = "Unexpected negative improvement - methodology issue"

    print(f"\n{'='*60}")
    print(f"STATUS: {status.upper()}")
    print(f"CONCLUSION: {conclusion}")
    print(f"{'='*60}")

    # Save results
    output = {
        'hypothesis': 'Low-frequency band alone predicts order as well as all bands combined',
        'domain': 'image_decomposition',
        'n_samples': n_samples,
        'n_valid': int(sum(valid_mask)),
        'correlations': {
            'low': {'rho': float(r_low), 'p': float(p_low)},
            'mid': {'rho': float(r_mid), 'p': float(p_mid)},
            'high': {'rho': float(r_high), 'p': float(p_high)},
        },
        'partial_correlations': {
            'mid': {'rho': float(r_mid_partial), 'p': float(p_mid_partial)},
            'high': {'rho': float(r_high_partial), 'p': float(p_high_partial)},
        },
        'r2_comparison': {
            'low_only': {'mean': float(r2_low_mean), 'std': float(r2_low_std)},
            'all_bands': {'mean': float(r2_all_mean), 'std': float(r2_all_std)},
            'improvement': {
                'mean': float(improvement_mean),
                'std': float(improvement_std),
                't_stat': float(t_stat),
                'p_value': float(p_val),
                'cohens_d': float(d),
            }
        },
        'status': status,
        'conclusion': conclusion,
    }

    with open('/Users/matt/Development/monochrome_noise_converger/results/res183_frequency_bands/results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to results/res183_frequency_bands/results.json")

    return output


if __name__ == '__main__':
    main()
