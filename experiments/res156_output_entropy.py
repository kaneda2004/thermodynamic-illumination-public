"""
RES-156: CPPN grayscale output entropy negatively correlates with order

Hypothesis: CPPN grayscale output entropy (before thresholding) negatively
correlates with order metric (more peaked distribution = higher order).

Background:
- RES-094: High-order CPPNs have bimodal pixel histograms (BC=0.584)
- RES-048: Output activation variance correlates positively with order (r=0.52)
- RES-151: CPPN sigmoid outputs concentrate values near 0/1

Logic: High-order CPPNs should have lower entropy (peaked at 0/1) while
low-order CPPNs should have higher entropy (spread across [0,1]).
"""

import numpy as np
from scipy import stats
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')
from core.thermo_sampler_v3 import CPPN, order_multiplicative

def compute_pixel_entropy(grayscale_img: np.ndarray, n_bins: int = 50) -> float:
    """Compute Shannon entropy of grayscale pixel distribution."""
    hist, _ = np.histogram(grayscale_img.flatten(), bins=n_bins, range=(0, 1), density=True)
    # Normalize to probability distribution
    hist = hist / (hist.sum() + 1e-10)
    # Shannon entropy
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    return entropy

def get_grayscale_output(cppn: CPPN, size: int = 32) -> np.ndarray:
    """Get raw sigmoid output (grayscale) before thresholding."""
    coords = np.linspace(-1, 1, size)
    x, y = np.meshgrid(coords, coords)
    return cppn.activate(x, y)

def run_experiment(n_samples: int = 500, seed: int = 42):
    """Test if grayscale output entropy correlates with order."""
    np.random.seed(seed)

    entropies = []
    orders = []

    for _ in range(n_samples):
        cppn = CPPN()

        # Get grayscale output
        grayscale = get_grayscale_output(cppn)

        # Binary image for order
        binary = (grayscale > 0.5).astype(np.uint8)

        # Compute metrics
        entropy = compute_pixel_entropy(grayscale)
        order = order_multiplicative(binary)

        entropies.append(entropy)
        orders.append(order)

    entropies = np.array(entropies)
    orders = np.array(orders)

    # Correlation analysis
    r_pearson, p_pearson = stats.pearsonr(entropies, orders)
    r_spearman, p_spearman = stats.spearmanr(entropies, orders)

    # Effect size (Cohen's d for high vs low entropy groups)
    median_entropy = np.median(entropies)
    low_entropy_orders = orders[entropies < median_entropy]
    high_entropy_orders = orders[entropies >= median_entropy]

    pooled_std = np.sqrt((np.std(low_entropy_orders)**2 + np.std(high_entropy_orders)**2) / 2)
    cohens_d = (np.mean(low_entropy_orders) - np.mean(high_entropy_orders)) / (pooled_std + 1e-10)

    # T-test
    t_stat, t_pval = stats.ttest_ind(low_entropy_orders, high_entropy_orders)

    print("="*60)
    print("RES-156: CPPN Output Entropy vs Order")
    print("="*60)
    print(f"\nSamples: {n_samples}")
    print(f"\nEntropy stats: mean={np.mean(entropies):.3f}, std={np.std(entropies):.3f}")
    print(f"Order stats: mean={np.mean(orders):.3f}, std={np.std(orders):.3f}")

    print(f"\n--- Correlation Analysis ---")
    print(f"Pearson r = {r_pearson:.3f} (p = {p_pearson:.2e})")
    print(f"Spearman rho = {r_spearman:.3f} (p = {p_spearman:.2e})")

    print(f"\n--- Group Comparison (low vs high entropy) ---")
    print(f"Low entropy group: mean order = {np.mean(low_entropy_orders):.3f}")
    print(f"High entropy group: mean order = {np.mean(high_entropy_orders):.3f}")
    print(f"Cohen's d = {cohens_d:.3f}")
    print(f"T-test: t = {t_stat:.2f}, p = {t_pval:.2e}")

    # Validation criteria
    significant = p_pearson < 0.01
    large_effect = abs(cohens_d) >= 0.5
    negative_correlation = r_pearson < 0

    print(f"\n--- Validation ---")
    print(f"Significant (p < 0.01): {significant}")
    print(f"Large effect (|d| >= 0.5): {large_effect}")
    print(f"Negative correlation: {negative_correlation}")

    if significant and negative_correlation:
        status = "validated" if large_effect else "inconclusive"
    else:
        status = "refuted"

    print(f"\nSTATUS: {status.upper()}")

    return {
        'r_pearson': r_pearson,
        'p_pearson': p_pearson,
        'r_spearman': r_spearman,
        'cohens_d': cohens_d,
        'status': status,
        'mean_entropy': np.mean(entropies),
        'mean_order': np.mean(orders)
    }

if __name__ == "__main__":
    results = run_experiment()
