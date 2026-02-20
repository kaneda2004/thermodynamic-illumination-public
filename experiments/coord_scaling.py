"""
RES-079: Test if input coordinate range scaling affects CPPN order distribution.

Hypothesis: Wider coordinate ranges ([-2,2] vs [-1,1]) produce different order
statistics due to nonlinear activation function behavior at different input scales.

Design:
- Generate 1000 CPPNs at different coordinate scales: [-0.5,0.5], [-1,1], [-2,2], [-4,4]
- Measure order statistics (mean, std, distribution) for each scale
- Statistical test: Kruskal-Wallis H-test across groups + pairwise Mann-Whitney U

Expected: If activation functions saturate at larger ranges, order distributions shift.
"""

import numpy as np
from scipy import stats
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import CPPN, order_multiplicative


def render_with_range(cppn: CPPN, size: int, coord_range: float) -> np.ndarray:
    """Render CPPN with custom coordinate range [-coord_range, coord_range]."""
    coords = np.linspace(-coord_range, coord_range, size)
    x, y = np.meshgrid(coords, coords)
    return (cppn.activate(x, y) > 0.5).astype(np.uint8)


def main():
    np.random.seed(42)

    n_samples = 1000
    size = 32
    coord_ranges = [0.5, 1.0, 2.0, 4.0]

    print("RES-079: Coordinate Range Scaling Experiment")
    print("=" * 60)
    print(f"Samples per range: {n_samples}")
    print(f"Image size: {size}x{size}")
    print(f"Coordinate ranges: {coord_ranges}")
    print()

    # Generate CPPNs once, render at different scales
    cppns = [CPPN() for _ in range(n_samples)]

    results = {}
    for cr in coord_ranges:
        orders = []
        for cppn in cppns:
            img = render_with_range(cppn, size, cr)
            o = order_multiplicative(img)
            orders.append(o)
        results[cr] = np.array(orders)
        print(f"Range [-{cr}, {cr}]: mean={np.mean(orders):.4f}, std={np.std(orders):.4f}, "
              f"median={np.median(orders):.4f}")

    print()

    # Kruskal-Wallis H-test (non-parametric ANOVA)
    groups = [results[cr] for cr in coord_ranges]
    h_stat, p_kw = stats.kruskal(*groups)
    print(f"Kruskal-Wallis H-test: H={h_stat:.2f}, p={p_kw:.2e}")

    # Pairwise Mann-Whitney U tests with Bonferroni correction
    print("\nPairwise Mann-Whitney U (Bonferroni-corrected):")
    n_comparisons = len(coord_ranges) * (len(coord_ranges) - 1) // 2
    alpha_corrected = 0.01 / n_comparisons

    significant_pairs = []
    for i, cr1 in enumerate(coord_ranges):
        for cr2 in coord_ranges[i+1:]:
            u_stat, p_val = stats.mannwhitneyu(results[cr1], results[cr2], alternative='two-sided')
            effect_size = (np.mean(results[cr1]) - np.mean(results[cr2])) / np.sqrt(
                (np.std(results[cr1])**2 + np.std(results[cr2])**2) / 2
            )
            sig = "***" if p_val < alpha_corrected else ""
            print(f"  [{-cr1},{cr1}] vs [{-cr2},{cr2}]: U={u_stat:.0f}, p={p_val:.2e}, "
                  f"Cohen's d={effect_size:.3f} {sig}")
            if p_val < alpha_corrected:
                significant_pairs.append((cr1, cr2, effect_size))

    print()

    # Effect of scale on activation saturation
    print("Activation Analysis (fraction of saturated outputs):")
    for cr in coord_ranges:
        saturated_frac = []
        for cppn in cppns[:100]:  # Sample for speed
            coords = np.linspace(-cr, cr, size)
            x, y = np.meshgrid(coords, coords)
            raw_output = cppn.activate(x, y)
            # Saturated = very close to 0 or 1 (threshold interpretation)
            near_threshold = np.mean(np.abs(raw_output - 0.5) > 0.4)
            saturated_frac.append(near_threshold)
        print(f"  Range [-{cr}, {cr}]: {np.mean(saturated_frac)*100:.1f}% outputs far from threshold")

    print()

    # Summary
    print("=" * 60)
    if p_kw < 0.01 and len(significant_pairs) > 0:
        max_effect = max(abs(d) for _, _, d in significant_pairs)
        if max_effect > 0.5:
            print("RESULT: VALIDATED - Coordinate range significantly affects order distribution")
            print(f"  Max effect size: {max_effect:.3f}")
            status = "VALIDATED"
        else:
            print("RESULT: INCONCLUSIVE - Significant but small effect")
            status = "INCONCLUSIVE"
    else:
        print("RESULT: REFUTED - No significant effect of coordinate range on order")
        status = "REFUTED"

    # Detailed metrics for research log
    print()
    print("METRICS:")
    print(f"  kruskal_h: {h_stat:.2f}")
    print(f"  kruskal_p: {p_kw:.2e}")
    print(f"  n_significant_pairs: {len(significant_pairs)}")
    if significant_pairs:
        print(f"  max_cohen_d: {max(abs(d) for _, _, d in significant_pairs):.3f}")

    return status, h_stat, p_kw, significant_pairs


if __name__ == "__main__":
    main()
