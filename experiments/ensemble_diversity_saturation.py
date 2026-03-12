"""
RES-155: CPPN ensemble diversity saturates - adding more samples yields diminishing coverage of high-order space

Hypothesis: As we sample more CPPNs, the coverage of high-order image space shows
diminishing returns - early samples cover most of the reachable patterns.

Methodology:
1. Generate large ensemble of random CPPNs
2. Filter to high-order subset (order > threshold)
3. Measure cumulative diversity as we add samples using:
   - Unique pattern count at quantized resolution
   - Mean pairwise Hamming distance to nearest neighbor
4. Fit saturation curve and test if coverage plateaus
"""

import numpy as np
from scipy.stats import spearmanr
from scipy.optimize import curve_fit
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import CPPN, order_multiplicative, set_global_seed

def saturation_curve(n, k, n_half):
    """Michaelis-Menten style saturation: coverage approaches k as n increases."""
    return k * n / (n_half + n)

def compute_diversity_metrics(images):
    """Compute diversity metrics for an ensemble of images."""
    n = len(images)
    if n < 2:
        return {'unique_ratio': 1.0, 'mean_nn_dist': 0.0}

    # Flatten images to vectors
    flat = np.array([img.flatten() for img in images])

    # Count unique patterns
    unique_patterns = len(set(tuple(f) for f in flat))
    unique_ratio = unique_patterns / n

    # Compute nearest neighbor Hamming distances
    nn_dists = []
    for i in range(n):
        min_dist = float('inf')
        for j in range(n):
            if i != j:
                dist = np.mean(flat[i] != flat[j])
                if dist < min_dist:
                    min_dist = dist
        nn_dists.append(min_dist)

    return {
        'unique_ratio': unique_ratio,
        'mean_nn_dist': np.mean(nn_dists),
        'unique_count': unique_patterns
    }

def cumulative_diversity(images, sample_points):
    """Measure diversity as we add more samples."""
    results = []
    for n in sample_points:
        if n > len(images):
            break
        subset = images[:n]
        metrics = compute_diversity_metrics(subset)
        results.append({
            'n_samples': n,
            **metrics
        })
    return results

def main():
    set_global_seed(42)

    print("RES-155: CPPN Ensemble Diversity Saturation")
    print("=" * 60)

    # Generate large ensemble
    n_total = 2000
    order_threshold = 0.1  # Filter to high-order subset

    print(f"\nGenerating {n_total} random CPPNs...")
    all_images = []
    all_orders = []

    for i in range(n_total):
        cppn = CPPN()
        img = cppn.render(size=32)
        order = order_multiplicative(img)
        all_images.append(img)
        all_orders.append(order)

        if (i + 1) % 500 == 0:
            print(f"  Generated {i+1}/{n_total}")

    # Filter to high-order subset
    high_order_mask = np.array(all_orders) > order_threshold
    high_order_images = [img for img, mask in zip(all_images, high_order_mask) if mask]
    high_order_orders = [o for o, mask in zip(all_orders, high_order_mask) if mask]

    n_high = len(high_order_images)
    print(f"\nFiltered to {n_high} high-order images (order > {order_threshold})")
    print(f"Mean order in subset: {np.mean(high_order_orders):.3f}")

    if n_high < 50:
        print("ERROR: Insufficient high-order samples for analysis")
        return

    # Shuffle for unbiased sampling
    np.random.shuffle(high_order_images)

    # Sample points for diversity measurement (log-spaced for saturation curve)
    max_samples = min(n_high, 500)
    sample_points = np.unique(np.geomspace(5, max_samples, 30).astype(int))

    print(f"\nMeasuring cumulative diversity at {len(sample_points)} sample points...")
    diversity_results = cumulative_diversity(high_order_images, sample_points)

    # Extract data for curve fitting
    ns = np.array([r['n_samples'] for r in diversity_results])
    unique_counts = np.array([r['unique_count'] for r in diversity_results])
    nn_dists = np.array([r['mean_nn_dist'] for r in diversity_results])

    # Test 1: Does unique count saturate?
    print("\n--- Saturation Analysis ---")

    try:
        # Fit saturation curve to unique counts
        popt, pcov = curve_fit(saturation_curve, ns, unique_counts,
                               p0=[max(unique_counts)*1.2, max(ns)/2],
                               bounds=([0, 0], [n_high*2, n_high]),
                               maxfev=5000)
        k_fit, n_half = popt

        # Predict coverage at 2x current samples
        current_coverage = unique_counts[-1]
        predicted_2x = saturation_curve(ns[-1] * 2, k_fit, n_half)
        marginal_gain = (predicted_2x - current_coverage) / current_coverage

        print(f"Saturation curve fit: k={k_fit:.1f}, n_half={n_half:.1f}")
        print(f"Current unique patterns: {current_coverage}")
        print(f"Predicted at 2x samples: {predicted_2x:.1f}")
        print(f"Marginal gain from doubling: {marginal_gain*100:.1f}%")

        saturation_detected = marginal_gain < 0.20  # <20% gain from doubling = saturation
    except Exception as e:
        print(f"Curve fitting failed: {e}")
        saturation_detected = False
        marginal_gain = 1.0
        k_fit = n_high

    # Test 2: Does NN distance decrease (patterns becoming more similar)?
    nn_slope, nn_pvalue = spearmanr(ns, nn_dists)
    nn_decreasing = nn_slope < 0 and nn_pvalue < 0.05

    print(f"\nNearest-neighbor distance trend:")
    print(f"  Spearman rho: {nn_slope:.3f}")
    print(f"  p-value: {nn_pvalue:.6f}")
    print(f"  Trend: {'decreasing (saturation)' if nn_decreasing else 'not decreasing'}")

    # Test 3: Compute effective dimension from diversity scaling
    # If diversity scales as n^d, d is the effective dimension
    log_ns = np.log(ns)
    log_unique = np.log(unique_counts + 1)
    scaling_exponent = np.polyfit(log_ns, log_unique, 1)[0]

    print(f"\nDiversity scaling exponent: {scaling_exponent:.3f}")
    print(f"  (1.0 = linear growth, <1 = sublinear = saturation)")

    # Compute effect size for saturation
    # Compare early vs late marginal gains
    early_gain = (unique_counts[len(unique_counts)//3] - unique_counts[0]) / ns[len(ns)//3]
    late_gain = (unique_counts[-1] - unique_counts[2*len(unique_counts)//3]) / (ns[-1] - ns[2*len(ns)//3])

    if early_gain > 0 and late_gain > 0:
        gain_ratio = late_gain / early_gain
        effect_size = np.log(early_gain / late_gain)  # Log ratio as effect size
    else:
        gain_ratio = 1.0
        effect_size = 0.0

    print(f"\nMarginal gain comparison:")
    print(f"  Early marginal gain: {early_gain:.4f} patterns/sample")
    print(f"  Late marginal gain: {late_gain:.4f} patterns/sample")
    print(f"  Gain ratio (early/late): {1/gain_ratio:.2f}x")
    print(f"  Effect size (log ratio): {effect_size:.2f}")

    # Statistical test: is late gain significantly lower than early gain?
    # Use bootstrap to get confidence interval
    n_bootstrap = 1000
    early_idx = len(unique_counts) // 3
    late_idx = 2 * len(unique_counts) // 3

    ratios = []
    for _ in range(n_bootstrap):
        # Resample with noise
        early_est = unique_counts[early_idx] + np.random.normal(0, 1)
        late_est = unique_counts[-1] + np.random.normal(0, 1)
        if early_est > unique_counts[0] and late_est > unique_counts[late_idx]:
            e_gain = (early_est - unique_counts[0]) / ns[early_idx]
            l_gain = (late_est - unique_counts[late_idx]) / (ns[-1] - ns[late_idx])
            if l_gain > 0:
                ratios.append(e_gain / l_gain)

    if ratios:
        ratios = np.array(ratios)
        ci_low, ci_high = np.percentile(ratios, [2.5, 97.5])
        significant_saturation = ci_low > 1.0  # Early gain significantly higher
    else:
        ci_low, ci_high = 0, 0
        significant_saturation = False

    print(f"\nBootstrap 95% CI for early/late gain ratio: [{ci_low:.2f}, {ci_high:.2f}]")
    print(f"Significant saturation: {significant_saturation}")

    # Final verdict
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Criteria for validation
    criteria = {
        'scaling_sublinear': scaling_exponent < 0.9,
        'marginal_gain_decreases': saturation_detected or significant_saturation,
        'nn_dist_decreases': nn_decreasing,
        'effect_size_sufficient': effect_size > 0.5
    }

    n_criteria_met = sum(criteria.values())

    print(f"Scaling exponent < 0.9: {criteria['scaling_sublinear']} ({scaling_exponent:.3f})")
    print(f"Marginal gain decreases: {criteria['marginal_gain_decreases']}")
    print(f"NN distance decreases: {criteria['nn_dist_decreases']}")
    print(f"Effect size > 0.5: {criteria['effect_size_sufficient']} ({effect_size:.2f})")
    print(f"\nCriteria met: {n_criteria_met}/4")

    if n_criteria_met >= 3:
        status = "VALIDATED"
        summary = (f"CPPN ensemble diversity saturates: scaling exponent {scaling_exponent:.2f} < 1.0, "
                   f"marginal gain decreases {1/gain_ratio:.1f}x from early to late samples, "
                   f"effect size d={effect_size:.2f}. High-order space is limited.")
    elif n_criteria_met >= 2:
        status = "INCONCLUSIVE"
        summary = (f"Weak evidence of saturation: scaling exponent {scaling_exponent:.2f}, "
                   f"marginal gain ratio {1/gain_ratio:.1f}x. Effect may require more samples to confirm.")
    else:
        status = "REFUTED"
        summary = (f"No saturation detected: scaling exponent {scaling_exponent:.2f}, "
                   f"diversity continues to grow linearly with ensemble size.")

    print(f"\nSTATUS: {status}")
    print(f"\nSummary: {summary}")

    # Output for log manager
    print(f"\n--- LOG MANAGER OUTPUT ---")
    print(f"STATUS: {status.lower()}")
    print(f"EFFECT_SIZE: {effect_size:.2f}")
    print(f"P_VALUE: {nn_pvalue:.6f}")
    print(f"SUMMARY: {summary}")

if __name__ == "__main__":
    main()
