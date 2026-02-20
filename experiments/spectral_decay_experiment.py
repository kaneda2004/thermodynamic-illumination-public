"""
Spectral Decay Experiment (RES-017)

Hypothesis: High-order CPPN images exhibit steeper power spectral decay
(more negative beta in P(k) ~ k^beta), converging toward natural image
statistics (beta ~ -2).

Null hypothesis: Power spectral exponent beta is independent of order level.

Novelty: RES-007 measured spectral COHERENCE (low-freq/total ratio).
This measures spectral SLOPE - characterizing scale-invariance of structure.
Natural images have beta ~ -2 (1/f^2 noise). Does high order approach this?

Method:
1. Generate N CPPN images with varying order
2. Compute 2D FFT, radially average to get P(k)
3. Fit power law P(k) ~ k^beta using log-log linear regression
4. Correlate beta with order using Spearman rho
5. Compare low-order vs high-order beta using Mann-Whitney U
"""

import numpy as np
import sys
from pathlib import Path
from scipy import stats
from typing import Tuple
import json

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.thermo_sampler_v3 import CPPN, order_multiplicative, set_global_seed


def compute_radial_power_spectrum(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute radially averaged power spectrum.

    Returns:
        k: spatial frequencies (excluding DC component)
        P_k: radially averaged power at each k
    """
    # Center the image (remove DC component bias)
    img_centered = img.astype(float) - np.mean(img)

    # 2D FFT
    f = np.fft.fft2(img_centered)
    f_shifted = np.fft.fftshift(f)
    power = np.abs(f_shifted) ** 2

    h, w = img.shape
    cy, cx = h // 2, w // 2

    # Create radial coordinate
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx)**2 + (y - cy)**2)

    # Bin by integer radius
    max_r = int(min(h, w) / 2)
    k_values = np.arange(1, max_r)  # Skip k=0 (DC)
    P_k = np.zeros(len(k_values))

    for i, k in enumerate(k_values):
        mask = (r >= k - 0.5) & (r < k + 0.5)
        if np.any(mask):
            P_k[i] = np.mean(power[mask])

    return k_values, P_k


def fit_power_law(k: np.ndarray, P_k: np.ndarray) -> Tuple[float, float, float]:
    """
    Fit power law P(k) ~ k^beta using log-log linear regression.

    Returns:
        beta: power law exponent
        intercept: log-space intercept
        r_squared: goodness of fit
    """
    # Filter out zero or negative values
    valid = (k > 0) & (P_k > 0)
    k_valid = k[valid]
    P_valid = P_k[valid]

    if len(k_valid) < 3:
        return 0.0, 0.0, 0.0

    log_k = np.log(k_valid)
    log_P = np.log(P_valid)

    slope, intercept, r_value, p_value, std_err = stats.linregress(log_k, log_P)

    return slope, intercept, r_value ** 2


def run_experiment(n_samples: int = 500, image_size: int = 32, seed: int = 42):
    """
    Run the spectral decay experiment.
    """
    set_global_seed(seed)

    print(f"=== Spectral Decay Experiment ===")
    print(f"N samples: {n_samples}")
    print(f"Image size: {image_size}")
    print()

    # Collect data
    orders = []
    betas = []
    r_squareds = []

    print("Generating CPPN samples and computing spectral decay...")
    for i in range(n_samples):
        cppn = CPPN()
        img = cppn.render(image_size)
        order = order_multiplicative(img)

        k, P_k = compute_radial_power_spectrum(img)
        beta, intercept, r_sq = fit_power_law(k, P_k)

        orders.append(order)
        betas.append(beta)
        r_squareds.append(r_sq)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{n_samples}")

    orders = np.array(orders)
    betas = np.array(betas)
    r_squareds = np.array(r_squareds)

    print(f"\nOrder range: [{orders.min():.4f}, {orders.max():.4f}]")
    print(f"Beta range: [{betas.min():.2f}, {betas.max():.2f}]")
    print(f"Mean R^2 of power law fits: {r_squareds.mean():.3f}")

    # Primary analysis: Spearman correlation between order and beta
    rho, p_spearman = stats.spearmanr(orders, betas)
    print(f"\n--- Primary Analysis: Order vs Beta Correlation ---")
    print(f"Spearman rho: {rho:.4f}")
    print(f"Spearman p-value: {p_spearman:.2e}")

    # Pearson for comparison
    r_pearson, p_pearson = stats.pearsonr(orders, betas)
    print(f"Pearson r: {r_pearson:.4f}")
    print(f"Pearson p-value: {p_pearson:.2e}")

    # Secondary analysis: Low vs High order comparison
    # Split at median order
    median_order = np.median(orders)
    low_mask = orders < median_order
    high_mask = orders >= median_order

    beta_low = betas[low_mask]
    beta_high = betas[high_mask]

    print(f"\n--- Secondary Analysis: Low vs High Order ---")
    print(f"Low order (n={len(beta_low)}): beta = {beta_low.mean():.3f} +/- {beta_low.std():.3f}")
    print(f"High order (n={len(beta_high)}): beta = {beta_high.mean():.3f} +/- {beta_high.std():.3f}")

    # Mann-Whitney U test
    U, p_mann_whitney = stats.mannwhitneyu(beta_low, beta_high, alternative='two-sided')
    print(f"Mann-Whitney U: {U:.1f}")
    print(f"Mann-Whitney p-value: {p_mann_whitney:.2e}")

    # Cohen's d effect size
    pooled_std = np.sqrt((beta_low.std()**2 + beta_high.std()**2) / 2)
    cohens_d = (beta_high.mean() - beta_low.mean()) / pooled_std if pooled_std > 0 else 0
    print(f"Cohen's d: {cohens_d:.3f}")

    # Tertiary analysis: Distance from natural image statistics (beta ~ -2)
    natural_beta = -2.0
    dist_low = np.abs(beta_low.mean() - natural_beta)
    dist_high = np.abs(beta_high.mean() - natural_beta)

    print(f"\n--- Tertiary Analysis: Distance from Natural Image Statistics ---")
    print(f"Natural image beta (target): {natural_beta}")
    print(f"Low order distance from target: {dist_low:.3f}")
    print(f"High order distance from target: {dist_high:.3f}")
    print(f"High order closer to natural? {dist_high < dist_low}")

    # Quartile analysis for gradient
    q1, q2, q3 = np.percentile(orders, [25, 50, 75])
    quartile_betas = []
    for low, high, name in [(0, q1, "Q1 (lowest)"), (q1, q2, "Q2"), (q2, q3, "Q3"), (q3, 1.1, "Q4 (highest)")]:
        mask = (orders >= low) & (orders < high)
        if np.sum(mask) > 0:
            qb = betas[mask].mean()
            quartile_betas.append(qb)
            print(f"  {name}: beta = {qb:.3f} (n={np.sum(mask)})")

    # Kruskal-Wallis for monotonic trend
    q_masks = [
        (orders >= 0) & (orders < q1),
        (orders >= q1) & (orders < q2),
        (orders >= q2) & (orders < q3),
        (orders >= q3)
    ]
    q_groups = [betas[m] for m in q_masks if np.sum(m) > 0]
    H, p_kruskal = stats.kruskal(*q_groups)
    print(f"\nKruskal-Wallis H: {H:.2f}")
    print(f"Kruskal-Wallis p: {p_kruskal:.2e}")

    # Determine status
    # Criteria: p < 0.01, |effect size| > 0.5
    is_significant = p_spearman < 0.01 and np.abs(cohens_d) > 0.5

    # Check if high order is MORE NEGATIVE (steeper decay)
    direction = "more negative (steeper)" if cohens_d < 0 else "less negative (flatter)"

    print(f"\n=== CONCLUSION ===")
    if is_significant:
        if cohens_d < 0:  # High order has more negative beta
            print(f"VALIDATED: High-order images have steeper spectral decay.")
            print(f"High order beta is {direction} than low order.")
            status = "validated"
        else:
            print(f"REFUTED (direction): High-order images have FLATTER spectral decay.")
            print(f"Opposite of hypothesized direction.")
            status = "refuted"
    else:
        print(f"INCONCLUSIVE: Effect not significant (p={p_spearman:.2e}, d={cohens_d:.3f})")
        status = "inconclusive"

    # Save results
    results = {
        'n_samples': n_samples,
        'image_size': image_size,
        'seed': seed,
        'order_range': [float(orders.min()), float(orders.max())],
        'beta_range': [float(betas.min()), float(betas.max())],
        'mean_r_squared': float(r_squareds.mean()),
        'spearman_rho': float(rho),
        'spearman_p': float(p_spearman),
        'pearson_r': float(r_pearson),
        'pearson_p': float(p_pearson),
        'beta_low_mean': float(beta_low.mean()),
        'beta_low_std': float(beta_low.std()),
        'beta_high_mean': float(beta_high.mean()),
        'beta_high_std': float(beta_high.std()),
        'mann_whitney_U': float(U),
        'mann_whitney_p': float(p_mann_whitney),
        'cohens_d': float(cohens_d),
        'natural_beta': natural_beta,
        'dist_low_from_natural': float(dist_low),
        'dist_high_from_natural': float(dist_high),
        'high_order_closer_to_natural': bool(dist_high < dist_low),
        'kruskal_H': float(H),
        'kruskal_p': float(p_kruskal),
        'quartile_betas': quartile_betas,
        'status': status
    }

    # Save to results directory
    results_dir = Path(__file__).parent.parent / "results" / "spectral_decay"
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "spectral_decay_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_dir / 'spectral_decay_results.json'}")

    return results


if __name__ == "__main__":
    results = run_experiment()
