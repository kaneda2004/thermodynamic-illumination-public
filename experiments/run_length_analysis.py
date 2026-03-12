"""
RES-096: Run-length encoding analysis of high-order vs random images.

Hypothesis: High-order CPPN images exhibit significantly longer run-lengths
(consecutive same-value pixels) in horizontal/vertical directions compared to
random images, with mean run-length positively correlated with order score.

Run-length encoding captures local structure: structured images have
longer runs (contiguous regions) while random images have ~2 pixel average runs.
"""

import numpy as np
from scipy import stats
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')
from core.thermo_sampler_v3 import CPPN, order_multiplicative, set_global_seed


def compute_run_lengths(img: np.ndarray, direction: str = 'horizontal') -> np.ndarray:
    """
    Compute run lengths for binary image.

    Args:
        img: Binary image (0s and 1s)
        direction: 'horizontal', 'vertical', or 'diagonal'

    Returns:
        Array of all run lengths found
    """
    runs = []

    if direction == 'horizontal':
        for row in img:
            run_len = 1
            for i in range(1, len(row)):
                if row[i] == row[i-1]:
                    run_len += 1
                else:
                    runs.append(run_len)
                    run_len = 1
            runs.append(run_len)

    elif direction == 'vertical':
        for col_idx in range(img.shape[1]):
            col = img[:, col_idx]
            run_len = 1
            for i in range(1, len(col)):
                if col[i] == col[i-1]:
                    run_len += 1
                else:
                    runs.append(run_len)
                    run_len = 1
            runs.append(run_len)

    elif direction == 'diagonal':
        # Main diagonals
        h, w = img.shape
        for offset in range(-(h-1), w):
            diag = np.diagonal(img, offset=offset)
            if len(diag) < 2:
                continue
            run_len = 1
            for i in range(1, len(diag)):
                if diag[i] == diag[i-1]:
                    run_len += 1
                else:
                    runs.append(run_len)
                    run_len = 1
            runs.append(run_len)

    return np.array(runs)


def analyze_image(img: np.ndarray) -> dict:
    """Compute run-length statistics for an image."""
    h_runs = compute_run_lengths(img, 'horizontal')
    v_runs = compute_run_lengths(img, 'vertical')
    d_runs = compute_run_lengths(img, 'diagonal')
    all_runs = np.concatenate([h_runs, v_runs])

    return {
        'h_mean': np.mean(h_runs),
        'v_mean': np.mean(v_runs),
        'd_mean': np.mean(d_runs),
        'combined_mean': np.mean(all_runs),
        'h_max': np.max(h_runs),
        'v_max': np.max(v_runs),
        'combined_max': np.max(all_runs),
        # Long run fraction: runs >= 4 pixels
        'long_run_frac': np.mean(all_runs >= 4),
    }


def generate_cppn_samples(n_samples: int, size: int = 32, seed: int = 42) -> list:
    """Generate CPPN images with order scores."""
    set_global_seed(seed)
    samples = []
    for i in range(n_samples):
        cppn = CPPN()
        img = cppn.render(size)
        order = order_multiplicative(img)
        samples.append((img, order))
    return samples


def generate_random_samples(n_samples: int, size: int = 32, seed: int = 42) -> list:
    """Generate random binary images."""
    np.random.seed(seed + 1000)
    samples = []
    for _ in range(n_samples):
        img = (np.random.rand(size, size) > 0.5).astype(np.uint8)
        order = order_multiplicative(img)
        samples.append((img, order))
    return samples


def main():
    print("=" * 60)
    print("RES-096: Run-Length Encoding Analysis")
    print("=" * 60)

    N_SAMPLES = 500
    SIZE = 32

    print(f"\nGenerating {N_SAMPLES} CPPN samples...")
    cppn_samples = generate_cppn_samples(N_SAMPLES, SIZE)

    print(f"Generating {N_SAMPLES} random samples...")
    random_samples = generate_random_samples(N_SAMPLES, SIZE)

    # Analyze all samples
    cppn_stats = [analyze_image(img) for img, _ in cppn_samples]
    random_stats = [analyze_image(img) for img, _ in random_samples]

    cppn_orders = [o for _, o in cppn_samples]
    random_orders = [o for _, o in random_samples]

    # Extract metrics
    cppn_combined_means = [s['combined_mean'] for s in cppn_stats]
    random_combined_means = [s['combined_mean'] for s in random_stats]

    cppn_long_frac = [s['long_run_frac'] for s in cppn_stats]
    random_long_frac = [s['long_run_frac'] for s in random_stats]

    # Statistical tests
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    # Test 1: CPPN vs Random mean run length
    t_stat, p_value = stats.ttest_ind(cppn_combined_means, random_combined_means)
    effect_size = (np.mean(cppn_combined_means) - np.mean(random_combined_means)) / \
                  np.sqrt((np.var(cppn_combined_means) + np.var(random_combined_means)) / 2)

    print(f"\n1. Mean Run Length Comparison:")
    print(f"   CPPN mean:   {np.mean(cppn_combined_means):.3f} +/- {np.std(cppn_combined_means):.3f}")
    print(f"   Random mean: {np.mean(random_combined_means):.3f} +/- {np.std(random_combined_means):.3f}")
    print(f"   t-statistic: {t_stat:.3f}")
    print(f"   p-value:     {p_value:.2e}")
    print(f"   Effect size (Cohen's d): {effect_size:.3f}")

    # Test 2: Long run fraction comparison
    t_stat2, p_value2 = stats.ttest_ind(cppn_long_frac, random_long_frac)
    effect_size2 = (np.mean(cppn_long_frac) - np.mean(random_long_frac)) / \
                   np.sqrt((np.var(cppn_long_frac) + np.var(random_long_frac)) / 2)

    print(f"\n2. Long Run Fraction (runs >= 4 pixels):")
    print(f"   CPPN mean:   {np.mean(cppn_long_frac):.3f} +/- {np.std(cppn_long_frac):.3f}")
    print(f"   Random mean: {np.mean(random_long_frac):.3f} +/- {np.std(random_long_frac):.3f}")
    print(f"   t-statistic: {t_stat2:.3f}")
    print(f"   p-value:     {p_value2:.2e}")
    print(f"   Effect size (Cohen's d): {effect_size2:.3f}")

    # Test 3: Correlation with order score (within CPPN samples)
    # Filter to high-order CPPN samples for correlation
    high_order_mask = np.array(cppn_orders) > 0.1
    if np.sum(high_order_mask) > 10:
        high_order_orders = np.array(cppn_orders)[high_order_mask]
        high_order_runs = np.array(cppn_combined_means)[high_order_mask]
        corr, corr_p = stats.pearsonr(high_order_orders, high_order_runs)
        print(f"\n3. Correlation: Order vs Mean Run Length (high-order CPPN, n={np.sum(high_order_mask)}):")
        print(f"   Pearson r:   {corr:.3f}")
        print(f"   p-value:     {corr_p:.2e}")
    else:
        corr, corr_p = 0, 1
        print(f"\n3. Insufficient high-order samples for correlation analysis")

    # Test 4: Direction-specific analysis
    print(f"\n4. Direction-Specific Run Lengths (CPPN):")
    h_means = [s['h_mean'] for s in cppn_stats]
    v_means = [s['v_mean'] for s in cppn_stats]
    d_means = [s['d_mean'] for s in cppn_stats]
    print(f"   Horizontal: {np.mean(h_means):.3f} +/- {np.std(h_means):.3f}")
    print(f"   Vertical:   {np.mean(v_means):.3f} +/- {np.std(v_means):.3f}")
    print(f"   Diagonal:   {np.mean(d_means):.3f} +/- {np.std(d_means):.3f}")

    print(f"\n   Direction-Specific Run Lengths (Random):")
    h_means_r = [s['h_mean'] for s in random_stats]
    v_means_r = [s['v_mean'] for s in random_stats]
    d_means_r = [s['d_mean'] for s in random_stats]
    print(f"   Horizontal: {np.mean(h_means_r):.3f} +/- {np.std(h_means_r):.3f}")
    print(f"   Vertical:   {np.mean(v_means_r):.3f} +/- {np.std(v_means_r):.3f}")
    print(f"   Diagonal:   {np.mean(d_means_r):.3f} +/- {np.std(d_means_r):.3f}")

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Primary validation criteria
    validated = (
        p_value < 0.01 and
        effect_size > 0.5 and
        np.mean(cppn_combined_means) > np.mean(random_combined_means)
    )

    print(f"\nPrimary test (mean run length):")
    print(f"  - p < 0.01: {'PASS' if p_value < 0.01 else 'FAIL'} (p = {p_value:.2e})")
    print(f"  - effect size > 0.5: {'PASS' if effect_size > 0.5 else 'FAIL'} (d = {effect_size:.3f})")
    print(f"  - CPPN > Random: {'PASS' if np.mean(cppn_combined_means) > np.mean(random_combined_means) else 'FAIL'}")

    status = "validated" if validated else "refuted"
    print(f"\nHYPOTHESIS STATUS: {status.upper()}")

    # Return summary for logging
    return {
        'status': status,
        'effect_size': effect_size,
        'p_value': p_value,
        'cppn_mean_run': np.mean(cppn_combined_means),
        'random_mean_run': np.mean(random_combined_means),
        'long_run_effect': effect_size2,
        'correlation': corr if 'corr' in dir() else 0,
    }


if __name__ == '__main__':
    results = main()
