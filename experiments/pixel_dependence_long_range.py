"""
RES-108: Long-range pixel correlation in CPPN vs random images

Hypothesis: CPPN images exhibit higher long-range pixel correlation
(correlation between pixels at distance > half image width) compared to random images.

Domain: pixel_dependence
"""

import numpy as np
from scipy import stats
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import CPPN


def sample_cppn_image(size=64):
    """Generate a random CPPN image."""
    cppn = CPPN()
    weights = np.random.randn(len(cppn.get_weights())) * 2.0
    cppn.set_weights(weights)
    return cppn.render(size=size)


def compute_long_range_correlation(image, min_distance_ratio=0.5):
    """
    Compute correlation between pixels separated by at least min_distance_ratio * image_width.
    Returns mean correlation coefficient.
    """
    h, w = image.shape
    min_dist = int(min_distance_ratio * w)

    # Sample pixel pairs at long range
    n_samples = 5000
    correlations = []

    flat_img = image.flatten()
    coords = [(i, j) for i in range(h) for j in range(w)]

    np.random.seed(42)
    for _ in range(n_samples):
        # Pick two random pixels
        idx1, idx2 = np.random.choice(len(coords), 2, replace=False)
        y1, x1 = coords[idx1]
        y2, x2 = coords[idx2]

        # Check distance
        dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if dist >= min_dist:
            correlations.append((flat_img[idx1], flat_img[idx2]))

    if len(correlations) < 100:
        return np.nan

    correlations = np.array(correlations)
    # Pearson correlation between paired distant pixels
    r, _ = stats.pearsonr(correlations[:, 0], correlations[:, 1])
    return r


def compute_spatial_autocorr_decay(image, max_lag=None):
    """
    Compute how autocorrelation decays with distance (horizontal direction).
    Returns correlation at various lag distances.
    """
    h, w = image.shape
    if max_lag is None:
        max_lag = w // 2

    autocorrs = []
    for lag in range(1, max_lag + 1):
        # Compute correlation between image[:, :-lag] and image[:, lag:]
        left = image[:, :-lag].flatten()
        right = image[:, lag:].flatten()
        r, _ = stats.pearsonr(left, right)
        autocorrs.append(r)

    return np.array(autocorrs)


def main():
    print("RES-108: Long-range pixel correlation analysis")
    print("=" * 60)

    n_samples = 100

    # Generate CPPN images
    print(f"\nGenerating {n_samples} CPPN images...")
    cppn_images = []
    for i in range(n_samples):
        img = sample_cppn_image(size=64)
        cppn_images.append(img)

    # Generate random images
    print(f"Generating {n_samples} random images...")
    np.random.seed(123)
    random_images = [np.random.rand(64, 64) for _ in range(n_samples)]

    # Compute long-range correlations
    print("\nComputing long-range correlations...")
    cppn_lr_corrs = []
    random_lr_corrs = []

    for i, (cppn_img, rand_img) in enumerate(zip(cppn_images, random_images)):
        cppn_lr = compute_long_range_correlation(cppn_img)
        rand_lr = compute_long_range_correlation(rand_img)
        if not np.isnan(cppn_lr):
            cppn_lr_corrs.append(abs(cppn_lr))  # Use absolute correlation
        if not np.isnan(rand_lr):
            random_lr_corrs.append(abs(rand_lr))

    cppn_lr_corrs = np.array(cppn_lr_corrs)
    random_lr_corrs = np.array(random_lr_corrs)

    print(f"\nCPPN long-range |correlation|: mean={np.mean(cppn_lr_corrs):.4f}, std={np.std(cppn_lr_corrs):.4f}")
    print(f"Random long-range |correlation|: mean={np.mean(random_lr_corrs):.4f}, std={np.std(random_lr_corrs):.4f}")

    # Statistical test
    t_stat, p_value = stats.ttest_ind(cppn_lr_corrs, random_lr_corrs)

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.var(cppn_lr_corrs) + np.var(random_lr_corrs)) / 2)
    cohens_d = (np.mean(cppn_lr_corrs) - np.mean(random_lr_corrs)) / pooled_std

    print(f"\nStatistical Analysis:")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.2e}")
    print(f"  Cohen's d: {cohens_d:.4f}")

    # Autocorrelation decay analysis
    print("\n" + "=" * 60)
    print("Autocorrelation decay analysis")
    print("=" * 60)

    # Sample 30 images for decay analysis
    cppn_decays = []
    random_decays = []

    for i in range(30):
        cppn_decay = compute_spatial_autocorr_decay(cppn_images[i])
        rand_decay = compute_spatial_autocorr_decay(random_images[i])
        cppn_decays.append(cppn_decay)
        random_decays.append(rand_decay)

    cppn_decays = np.array(cppn_decays)
    random_decays = np.array(random_decays)

    # Compare decay rates at far distances (lag > 20)
    far_lags = slice(20, 32)
    cppn_far_autocorr = np.mean(np.abs(cppn_decays[:, far_lags]), axis=1)
    random_far_autocorr = np.mean(np.abs(random_decays[:, far_lags]), axis=1)

    print(f"\nMean |autocorr| at far distances (lag 20-32):")
    print(f"  CPPN: {np.mean(cppn_far_autocorr):.4f} +/- {np.std(cppn_far_autocorr):.4f}")
    print(f"  Random: {np.mean(random_far_autocorr):.4f} +/- {np.std(random_far_autocorr):.4f}")

    t_decay, p_decay = stats.ttest_ind(cppn_far_autocorr, random_far_autocorr)
    pooled_std_decay = np.sqrt((np.var(cppn_far_autocorr) + np.var(random_far_autocorr)) / 2)
    d_decay = (np.mean(cppn_far_autocorr) - np.mean(random_far_autocorr)) / pooled_std_decay

    print(f"\nDecay analysis statistics:")
    print(f"  t-statistic: {t_decay:.4f}")
    print(f"  p-value: {p_decay:.2e}")
    print(f"  Cohen's d: {d_decay:.4f}")

    # Final verdict
    print("\n" + "=" * 60)
    print("FINAL VERDICT")
    print("=" * 60)

    validated = p_value < 0.01 and cohens_d > 0.5

    if validated:
        print("STATUS: VALIDATED")
    elif p_value < 0.05:
        print("STATUS: INCONCLUSIVE (p < 0.05 but criteria not fully met)")
    else:
        print("STATUS: REFUTED")

    print(f"\nPrimary result: p={p_value:.2e}, Cohen's d={cohens_d:.2f}")

    return {
        'status': 'validated' if validated else ('inconclusive' if p_value < 0.05 else 'refuted'),
        'p_value': p_value,
        'effect_size': cohens_d,
        'cppn_mean': float(np.mean(cppn_lr_corrs)),
        'random_mean': float(np.mean(random_lr_corrs))
    }


if __name__ == '__main__':
    main()
