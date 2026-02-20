"""
RES-199: CPPNs with weights that create integer-pi phase alignment produce higher order outputs

Hypothesis: The sin/cos activation functions (sin(pi*x), cos(pi*x)) produce the most
structured outputs when the weighted inputs create integer multiples of pi within
the input domain [-1, 1] or [0, sqrt(2)] for r. Weights that create "phase alignment"
(completing integer cycles) should produce more coherent patterns.

Test: Measure "phase alignment score" as how close the weight*input_range products
are to integer multiples of pi, then correlate with order.
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')
from core.thermo_sampler_v3 import CPPN, order_multiplicative
from scipy import stats
import json

np.random.seed(42)

def compute_phase_alignment(cppn):
    """
    Measure how close weights create integer-pi phase alignment.

    For sin(pi*x) and cos(pi*x) activations, the output completes one full
    cycle when the input spans 2 units. For input x in [-1, 1] (range=2),
    weight w produces w cycles. Integer cycles mean the pattern tiles cleanly.

    Phase alignment score = average of (1 - min_distance_to_integer) across weights.
    """
    weights = cppn.get_weights()[:4]  # First 4 are input weights (x, y, r, bias)

    # Input ranges: x,y in [-1,1] (range=2), r in [0,sqrt(2)] (range~1.41)
    input_ranges = [2.0, 2.0, np.sqrt(2), 0.0]  # bias has no range effect

    alignment_scores = []
    for i, (w, r) in enumerate(zip(weights, input_ranges)):
        if r == 0:  # skip bias
            continue
        # Number of cycles = |w| * range (since sin(pi * w * x))
        cycles = abs(w * r)
        # Distance to nearest integer
        dist_to_int = abs(cycles - round(cycles))
        # Convert to alignment score (1 = perfect alignment, 0 = half cycle)
        alignment = 1 - 2 * dist_to_int  # ranges from 0 (at 0.5) to 1 (at integer)
        alignment_scores.append(max(0, alignment))

    return np.mean(alignment_scores) if alignment_scores else 0


def compute_weighted_phase_alignment(cppn):
    """
    Weight-magnitude-weighted phase alignment.
    Larger weights have more influence on output pattern.
    """
    weights = cppn.get_weights()[:4]
    input_ranges = [2.0, 2.0, np.sqrt(2), 0.0]

    weighted_sum = 0
    total_weight = 0

    for i, (w, r) in enumerate(zip(weights, input_ranges)):
        if r == 0:
            continue
        cycles = abs(w * r)
        dist_to_int = abs(cycles - round(cycles))
        alignment = 1 - 2 * dist_to_int

        # Weight by absolute weight magnitude
        mag = abs(w)
        weighted_sum += max(0, alignment) * mag
        total_weight += mag

    return weighted_sum / (total_weight + 1e-10)


def main():
    n_samples = 1000

    orders = []
    phase_alignments = []
    weighted_alignments = []

    print(f"Generating {n_samples} random CPPNs...")

    for i in range(n_samples):
        cppn = CPPN()  # Random weights from prior
        img = cppn.render(32)
        order = order_multiplicative(img)

        pa = compute_phase_alignment(cppn)
        wpa = compute_weighted_phase_alignment(cppn)

        orders.append(order)
        phase_alignments.append(pa)
        weighted_alignments.append(wpa)

        if (i + 1) % 200 == 0:
            print(f"  Processed {i+1}/{n_samples}")

    orders = np.array(orders)
    phase_alignments = np.array(phase_alignments)
    weighted_alignments = np.array(weighted_alignments)

    # Correlation analysis
    r_pa, p_pa = stats.pearsonr(phase_alignments, orders)
    rho_pa, prho_pa = stats.spearmanr(phase_alignments, orders)

    r_wpa, p_wpa = stats.pearsonr(weighted_alignments, orders)
    rho_wpa, prho_wpa = stats.spearmanr(weighted_alignments, orders)

    print(f"\nPhase Alignment vs Order:")
    print(f"  Pearson r = {r_pa:.4f}, p = {p_pa:.2e}")
    print(f"  Spearman rho = {rho_pa:.4f}, p = {prho_pa:.2e}")

    print(f"\nWeighted Phase Alignment vs Order:")
    print(f"  Pearson r = {r_wpa:.4f}, p = {p_wpa:.2e}")
    print(f"  Spearman rho = {rho_wpa:.4f}, p = {prho_wpa:.2e}")

    # Quartile analysis
    high_order_mask = orders > np.percentile(orders, 75)
    low_order_mask = orders < np.percentile(orders, 25)

    high_pa = phase_alignments[high_order_mask]
    low_pa = phase_alignments[low_order_mask]

    high_wpa = weighted_alignments[high_order_mask]
    low_wpa = weighted_alignments[low_order_mask]

    # Effect sizes
    d_pa = (np.mean(high_pa) - np.mean(low_pa)) / np.sqrt(
        (np.std(high_pa)**2 + np.std(low_pa)**2) / 2
    )
    d_wpa = (np.mean(high_wpa) - np.mean(low_wpa)) / np.sqrt(
        (np.std(high_wpa)**2 + np.std(low_wpa)**2) / 2
    )

    # T-tests
    t_pa, tp_pa = stats.ttest_ind(high_pa, low_pa)
    t_wpa, tp_wpa = stats.ttest_ind(high_wpa, low_wpa)

    print(f"\nQuartile Analysis (Phase Alignment):")
    print(f"  High-order mean PA: {np.mean(high_pa):.4f}")
    print(f"  Low-order mean PA: {np.mean(low_pa):.4f}")
    print(f"  Cohen's d = {d_pa:.3f}")
    print(f"  t-test p = {tp_pa:.2e}")

    print(f"\nQuartile Analysis (Weighted Phase Alignment):")
    print(f"  High-order mean WPA: {np.mean(high_wpa):.4f}")
    print(f"  Low-order mean WPA: {np.mean(low_wpa):.4f}")
    print(f"  Cohen's d = {d_wpa:.3f}")
    print(f"  t-test p = {tp_wpa:.2e}")

    # Validation criteria
    validated = (abs(r_wpa) > 0.1 or abs(d_wpa) > 0.5) and tp_wpa < 0.01

    results = {
        'n_samples': n_samples,
        'phase_alignment': {
            'pearson_r': float(r_pa),
            'pearson_p': float(p_pa),
            'spearman_rho': float(rho_pa),
            'spearman_p': float(prho_pa),
            'high_order_mean': float(np.mean(high_pa)),
            'low_order_mean': float(np.mean(low_pa)),
            'cohens_d': float(d_pa),
            'ttest_p': float(tp_pa)
        },
        'weighted_phase_alignment': {
            'pearson_r': float(r_wpa),
            'pearson_p': float(p_wpa),
            'spearman_rho': float(rho_wpa),
            'spearman_p': float(prho_wpa),
            'high_order_mean': float(np.mean(high_wpa)),
            'low_order_mean': float(np.mean(low_wpa)),
            'cohens_d': float(d_wpa),
            'ttest_p': float(tp_wpa)
        },
        'validated': bool(validated),
        'primary_metric': 'weighted_phase_alignment',
        'primary_effect_size': float(d_wpa),
        'primary_p_value': float(tp_wpa)
    }

    import os
    os.makedirs('/Users/matt/Development/monochrome_noise_converger/results', exist_ok=True)
    with open('/Users/matt/Development/monochrome_noise_converger/results/phase_alignment_order.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"CONCLUSION: {'VALIDATED' if validated else 'REFUTED'}")
    print(f"Phase alignment correlation r={r_wpa:.3f}, d={d_wpa:.3f}")
    print(f"{'='*60}")

    return results


if __name__ == '__main__':
    main()
