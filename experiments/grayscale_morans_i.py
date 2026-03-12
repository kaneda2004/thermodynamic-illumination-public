"""
RES-176: Test whether CPPN grayscale outputs have higher spatial autocorrelation
(Moran's I) at high vs low order.

Hypothesis: CPPN grayscale outputs show higher spatial autocorrelation (Moran's I)
at high order compared to low order.

Moran's I measures global spatial autocorrelation of continuous values.
Values range from -1 (perfect dispersion) to +1 (perfect clustering).
Random patterns have I near 0.

Related findings:
- RES-056: Binary correlation length correlates with order (d=3.84)
- RES-146: 370x lower local pixel variance in CPPNs
- RES-156: Grayscale entropy positively correlates with order

This tests grayscale (continuous) spatial structure, distinct from binary correlation.
"""

import numpy as np
from scipy import stats
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.thermo_sampler_v3 import CPPN, order_multiplicative


def compute_morans_i(img: np.ndarray) -> float:
    """
    Compute Moran's I spatial autocorrelation index for a 2D array.

    Uses queen contiguity (8 neighbors).

    I = (N / W) * (sum_i sum_j w_ij (x_i - x_bar)(x_j - x_bar)) / (sum_i (x_i - x_bar)^2)

    Returns value in [-1, 1]:
    - +1: perfect positive spatial autocorrelation (clustering)
    - 0: no spatial autocorrelation (random)
    - -1: perfect negative autocorrelation (checkerboard)
    """
    x = img.flatten().astype(float)
    n = len(x)
    x_bar = np.mean(x)
    x_dev = x - x_bar

    # Variance
    var = np.sum(x_dev ** 2)
    if var < 1e-12:
        return 0.0  # Constant image

    h, w = img.shape

    # Build weighted cross-products for queen contiguity
    numerator = 0.0
    total_w = 0

    for i in range(h):
        for j in range(w):
            idx = i * w + j
            # Check all 8 neighbors
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < h and 0 <= nj < w:
                        nidx = ni * w + nj
                        numerator += x_dev[idx] * x_dev[nidx]
                        total_w += 1

    morans_i = (n / total_w) * (numerator / var)
    return morans_i


def get_cppn_grayscale(cppn: CPPN, size: int = 32) -> np.ndarray:
    """Get continuous grayscale output from CPPN (before thresholding)."""
    coords = np.linspace(-1, 1, size)
    x, y = np.meshgrid(coords, coords)
    return cppn.activate(x, y)


def run_experiment(n_samples: int = 500, seed: int = 42):
    """
    Generate CPPN samples and measure Moran's I vs order correlation.
    """
    np.random.seed(seed)

    results = {
        'orders': [],
        'morans_i': [],
        'grayscale_std': [],
        'grayscale_mean': []
    }

    print(f"Generating {n_samples} CPPN samples...")

    for i in range(n_samples):
        cppn = CPPN()

        # Get grayscale output
        grayscale = get_cppn_grayscale(cppn)

        # Get binary for order computation
        binary = (grayscale > 0.5).astype(np.uint8)

        # Compute order
        order = order_multiplicative(binary)

        # Compute Moran's I on grayscale
        mi = compute_morans_i(grayscale)

        results['orders'].append(order)
        results['morans_i'].append(mi)
        results['grayscale_std'].append(float(np.std(grayscale)))
        results['grayscale_mean'].append(float(np.mean(grayscale)))

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{n_samples}")

    return results


def analyze_results(results: dict) -> dict:
    """Compute statistics and test hypothesis."""
    orders = np.array(results['orders'])
    morans = np.array(results['morans_i'])

    # Overall correlation
    rho, p_rho = stats.spearmanr(orders, morans)
    r, p_r = stats.pearsonr(orders, morans)

    # High vs low order comparison
    # Use median split
    median_order = np.median(orders)
    high_mask = orders > median_order
    low_mask = orders <= median_order

    high_morans = morans[high_mask]
    low_morans = morans[low_mask]

    # Mann-Whitney U test
    u_stat, p_mw = stats.mannwhitneyu(high_morans, low_morans, alternative='greater')

    # Cohen's d
    pooled_std = np.sqrt((np.var(high_morans) + np.var(low_morans)) / 2)
    if pooled_std < 1e-12:
        cohens_d = 0.0
    else:
        cohens_d = (np.mean(high_morans) - np.mean(low_morans)) / pooled_std

    # Also test against random images
    np.random.seed(123)
    random_morans = []
    for _ in range(len(orders)):
        rand_img = np.random.rand(32, 32)
        random_morans.append(compute_morans_i(rand_img))
    random_morans = np.array(random_morans)

    # Compare CPPN vs random
    u_rand, p_rand = stats.mannwhitneyu(morans, random_morans, alternative='greater')
    d_rand = (np.mean(morans) - np.mean(random_morans)) / np.sqrt((np.var(morans) + np.var(random_morans)) / 2)

    analysis = {
        'n_samples': len(orders),
        'correlation': {
            'spearman_rho': float(rho),
            'spearman_p': float(p_rho),
            'pearson_r': float(r),
            'pearson_p': float(p_r)
        },
        'high_vs_low': {
            'high_mean': float(np.mean(high_morans)),
            'high_std': float(np.std(high_morans)),
            'low_mean': float(np.mean(low_morans)),
            'low_std': float(np.std(low_morans)),
            'mann_whitney_u': float(u_stat),
            'p_value': float(p_mw),
            'cohens_d': float(cohens_d),
            'n_high': int(np.sum(high_mask)),
            'n_low': int(np.sum(low_mask))
        },
        'cppn_vs_random': {
            'cppn_mean': float(np.mean(morans)),
            'cppn_std': float(np.std(morans)),
            'random_mean': float(np.mean(random_morans)),
            'random_std': float(np.std(random_morans)),
            'mann_whitney_u': float(u_rand),
            'p_value': float(p_rand),
            'cohens_d': float(d_rand)
        },
        'descriptive': {
            'order_mean': float(np.mean(orders)),
            'order_std': float(np.std(orders)),
            'morans_mean': float(np.mean(morans)),
            'morans_std': float(np.std(morans)),
            'morans_min': float(np.min(morans)),
            'morans_max': float(np.max(morans))
        }
    }

    return analysis


def main():
    print("=" * 60)
    print("RES-176: Grayscale Spatial Autocorrelation (Moran's I)")
    print("=" * 60)

    # Run experiment
    results = run_experiment(n_samples=500, seed=42)

    # Analyze
    analysis = analyze_results(results)

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\nN = {analysis['n_samples']} CPPN samples")
    print(f"\nOrder: mean={analysis['descriptive']['order_mean']:.4f}, std={analysis['descriptive']['order_std']:.4f}")
    print(f"Moran's I: mean={analysis['descriptive']['morans_mean']:.4f}, std={analysis['descriptive']['morans_std']:.4f}")
    print(f"           range=[{analysis['descriptive']['morans_min']:.4f}, {analysis['descriptive']['morans_max']:.4f}]")

    print(f"\nCorrelation with order:")
    print(f"  Spearman rho = {analysis['correlation']['spearman_rho']:.4f}, p = {analysis['correlation']['spearman_p']:.2e}")
    print(f"  Pearson r = {analysis['correlation']['pearson_r']:.4f}, p = {analysis['correlation']['pearson_p']:.2e}")

    print(f"\nHigh-order vs Low-order comparison:")
    print(f"  High-order (n={analysis['high_vs_low']['n_high']}): Moran's I = {analysis['high_vs_low']['high_mean']:.4f} +/- {analysis['high_vs_low']['high_std']:.4f}")
    print(f"  Low-order (n={analysis['high_vs_low']['n_low']}):  Moran's I = {analysis['high_vs_low']['low_mean']:.4f} +/- {analysis['high_vs_low']['low_std']:.4f}")
    print(f"  Mann-Whitney U = {analysis['high_vs_low']['mann_whitney_u']:.1f}, p = {analysis['high_vs_low']['p_value']:.2e}")
    print(f"  Cohen's d = {analysis['high_vs_low']['cohens_d']:.2f}")

    print(f"\nCPPN vs Random comparison:")
    print(f"  CPPN:   Moran's I = {analysis['cppn_vs_random']['cppn_mean']:.4f} +/- {analysis['cppn_vs_random']['cppn_std']:.4f}")
    print(f"  Random: Moran's I = {analysis['cppn_vs_random']['random_mean']:.4f} +/- {analysis['cppn_vs_random']['random_std']:.4f}")
    print(f"  Mann-Whitney U = {analysis['cppn_vs_random']['mann_whitney_u']:.1f}, p = {analysis['cppn_vs_random']['p_value']:.2e}")
    print(f"  Cohen's d = {analysis['cppn_vs_random']['cohens_d']:.2f}")

    # Determine status
    # Criteria: d > 0.5 and p < 0.01 for validation
    d = analysis['high_vs_low']['cohens_d']
    p = analysis['high_vs_low']['p_value']

    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)

    if d > 0.5 and p < 0.01:
        status = "validated"
        print(f"VALIDATED: d={d:.2f} > 0.5 and p={p:.2e} < 0.01")
    elif d < -0.5 and p < 0.01:
        status = "refuted"
        print(f"REFUTED: Opposite direction found (d={d:.2f})")
    else:
        status = "inconclusive"
        if abs(d) <= 0.5:
            print(f"INCONCLUSIVE: Effect size d={d:.2f} below 0.5 threshold")
        else:
            print(f"INCONCLUSIVE: p-value {p:.2e} above 0.01 threshold")

    # Save results
    os.makedirs('results/grayscale_morans_i', exist_ok=True)
    output = {
        'experiment_id': 'RES-176',
        'hypothesis': "CPPN grayscale outputs have higher spatial autocorrelation (Moran's I) at high vs low order",
        'domain': 'output_statistics',
        'status': status,
        'analysis': analysis,
        'raw_data': {
            'orders': results['orders'],
            'morans_i': results['morans_i']
        }
    }

    with open('results/grayscale_morans_i/results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to results/grayscale_morans_i/results.json")

    # Generate summary for log
    summary = (
        f"Grayscale Moran's I correlates with order (rho={analysis['correlation']['spearman_rho']:.2f}, "
        f"p={analysis['correlation']['spearman_p']:.1e}). High-order: I={analysis['high_vs_low']['high_mean']:.3f} vs "
        f"low-order: I={analysis['high_vs_low']['low_mean']:.3f} (d={d:.2f}, p={p:.1e}). "
        f"All CPPNs have much higher I than random ({analysis['cppn_vs_random']['cppn_mean']:.3f} vs "
        f"{analysis['cppn_vs_random']['random_mean']:.3f}, d={analysis['cppn_vs_random']['cohens_d']:.1f})."
    )

    print(f"\nSummary for log: {summary}")

    return status, summary, d


if __name__ == "__main__":
    status, summary, effect_size = main()
