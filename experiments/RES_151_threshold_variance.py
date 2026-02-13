"""
RES-151: CPPN images have higher variance in order across thresholds than random images

Hypothesis: When sweeping binarization threshold from 0 to 1, CPPN continuous outputs
produce higher variance in the resulting order scores compared to random continuous images.

Rationale: CPPN outputs have structured gradients that create threshold-dependent patterns,
while random noise has similar statistics regardless of threshold.
"""

import numpy as np
from scipy import stats
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')
from core.thermo_sampler_v3 import CPPN, order_multiplicative

def render_at_threshold(continuous_img: np.ndarray, threshold: float) -> np.ndarray:
    """Binarize continuous image at given threshold."""
    return (continuous_img > threshold).astype(np.uint8)

def compute_order_variance_across_thresholds(continuous_img: np.ndarray, thresholds: np.ndarray) -> tuple:
    """Compute order at each threshold and return variance and orders."""
    orders = []
    for t in thresholds:
        binary = render_at_threshold(continuous_img, t)
        orders.append(order_multiplicative(binary))
    return np.var(orders), np.array(orders)

def main():
    np.random.seed(42)

    n_samples = 200
    thresholds = np.linspace(0.05, 0.95, 19)  # 19 thresholds from 0.05 to 0.95
    size = 32

    # Generate CPPN continuous outputs
    cppn_variances = []
    cppn_mean_orders = []
    for _ in range(n_samples):
        cppn = CPPN()
        coords = np.linspace(-1, 1, size)
        x, y = np.meshgrid(coords, coords)
        continuous = cppn.activate(x, y)  # Continuous output before thresholding
        var, orders = compute_order_variance_across_thresholds(continuous, thresholds)
        cppn_variances.append(var)
        cppn_mean_orders.append(np.mean(orders))

    # Generate random continuous images (uniform noise in [0,1])
    random_variances = []
    random_mean_orders = []
    for _ in range(n_samples):
        continuous = np.random.rand(size, size)
        var, orders = compute_order_variance_across_thresholds(continuous, thresholds)
        random_variances.append(var)
        random_mean_orders.append(np.mean(orders))

    cppn_variances = np.array(cppn_variances)
    random_variances = np.array(random_variances)

    # Statistical tests
    t_stat, p_value = stats.ttest_ind(cppn_variances, random_variances)
    cohens_d = (np.mean(cppn_variances) - np.mean(random_variances)) / np.sqrt(
        (np.std(cppn_variances)**2 + np.std(random_variances)**2) / 2
    )

    # Mann-Whitney U test (non-parametric)
    u_stat, p_mw = stats.mannwhitneyu(cppn_variances, random_variances, alternative='two-sided')

    print("="*60)
    print("RES-151: Threshold Variance Comparison")
    print("="*60)
    print(f"\nCPPN threshold variance: {np.mean(cppn_variances):.6f} +/- {np.std(cppn_variances):.6f}")
    print(f"Random threshold variance: {np.mean(random_variances):.6f} +/- {np.std(random_variances):.6f}")
    print(f"\nCPPN mean order: {np.mean(cppn_mean_orders):.4f}")
    print(f"Random mean order: {np.mean(random_mean_orders):.4f}")
    print(f"\nt-statistic: {t_stat:.4f}")
    print(f"p-value (t-test): {p_value:.2e}")
    print(f"Cohen's d: {cohens_d:.4f}")
    print(f"Mann-Whitney U: {u_stat:.1f}, p={p_mw:.2e}")

    # Interpret
    print("\n" + "="*60)
    if p_value < 0.01 and abs(cohens_d) > 0.5:
        if cohens_d > 0:
            print("VALIDATED: CPPNs show significantly HIGHER threshold variance")
        else:
            print("REFUTED: CPPNs show significantly LOWER threshold variance")
    else:
        print("INCONCLUSIVE: Effect size or significance insufficient")
    print(f"Effect size threshold: |d| > 0.5, actual |d| = {abs(cohens_d):.4f}")
    print(f"Significance threshold: p < 0.01, actual p = {p_value:.2e}")
    print("="*60)

    return cohens_d, p_value

if __name__ == "__main__":
    main()
