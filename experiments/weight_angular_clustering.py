"""
RES-144: High-order CPPN weight vectors cluster in narrow angular cones in weight space

Hypothesis: High-order weight vectors are concentrated in specific angular directions
(spherical coordinates), not just regions. This would explain why random sampling
rarely hits high-order configurations.

Methodology:
1. Generate many random CPPNs, compute order
2. Normalize weight vectors to unit sphere (remove magnitude)
3. Compare angular dispersion of high-order vs low-order weight vectors
4. Use angular concentration metrics: mean pairwise cosine similarity, cone angle

If validated: High-order configurations form narrow angular cones, explaining rarity.
If refuted: High-order regions are scattered across angular space.
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import CPPN, order_multiplicative, set_global_seed
from scipy import stats

def normalize_to_sphere(w):
    """Project weight vector to unit sphere."""
    norm = np.linalg.norm(w)
    if norm < 1e-10:
        return w
    return w / norm

def mean_pairwise_cosine(vectors):
    """Compute mean pairwise cosine similarity for a set of unit vectors."""
    n = len(vectors)
    if n < 2:
        return 0.0

    # Stack vectors into matrix
    V = np.array(vectors)  # (n, d)

    # Compute all pairwise cosines: V @ V.T gives (n, n) matrix of dot products
    cosines = V @ V.T

    # Get upper triangular (exclude diagonal)
    upper = cosines[np.triu_indices(n, k=1)]
    return np.mean(upper)

def angular_concentration(vectors):
    """Estimate angular concentration using von Mises-Fisher-like dispersion."""
    if len(vectors) < 2:
        return 0.0

    V = np.array(vectors)
    # Mean direction
    mean_dir = np.mean(V, axis=0)
    mean_dir_norm = np.linalg.norm(mean_dir)

    if mean_dir_norm < 1e-10:
        return 0.0

    mean_dir = mean_dir / mean_dir_norm

    # R-bar: resultant length (1 = all same direction, 0 = uniform sphere)
    R_bar = mean_dir_norm

    return R_bar

def cone_angle_containing_fraction(vectors, fraction=0.5):
    """Compute cone angle containing given fraction of vectors around their mean."""
    if len(vectors) < 2:
        return np.pi

    V = np.array(vectors)
    mean_dir = np.mean(V, axis=0)
    mean_dir_norm = np.linalg.norm(mean_dir)

    if mean_dir_norm < 1e-10:
        return np.pi

    mean_dir = mean_dir / mean_dir_norm

    # Cosines from mean direction
    cosines = V @ mean_dir
    cosines = np.clip(cosines, -1, 1)

    # Sort by angle (ascending = closest to mean first)
    angles = np.arccos(cosines)
    angles_sorted = np.sort(angles)

    # Angle containing fraction of vectors
    idx = int(fraction * len(angles_sorted))
    if idx >= len(angles_sorted):
        idx = len(angles_sorted) - 1

    return angles_sorted[idx]

def main():
    set_global_seed(42)

    n_samples = 1000
    print(f"Generating {n_samples} random CPPNs...")

    weights = []
    orders = []

    for i in range(n_samples):
        cppn = CPPN()
        w = cppn.get_weights()
        img = cppn.render(32)
        order = order_multiplicative(img)

        weights.append(w)
        orders.append(order)

    weights = np.array(weights)
    orders = np.array(orders)

    # Normalize to unit sphere
    unit_weights = np.array([normalize_to_sphere(w) for w in weights])

    # Split into high-order and low-order
    threshold_high = np.percentile(orders, 75)
    threshold_low = np.percentile(orders, 25)

    high_mask = orders >= threshold_high
    low_mask = orders <= threshold_low

    high_weights = unit_weights[high_mask]
    low_weights = unit_weights[low_mask]

    print(f"\nOrder statistics:")
    print(f"  Mean: {np.mean(orders):.4f}")
    print(f"  Std: {np.std(orders):.4f}")
    print(f"  High threshold (75th): {threshold_high:.4f}")
    print(f"  Low threshold (25th): {threshold_low:.4f}")
    print(f"  N high-order: {len(high_weights)}")
    print(f"  N low-order: {len(low_weights)}")

    # Metric 1: Mean pairwise cosine similarity
    cos_high = mean_pairwise_cosine(high_weights)
    cos_low = mean_pairwise_cosine(low_weights)

    print(f"\nMean pairwise cosine similarity:")
    print(f"  High-order: {cos_high:.4f}")
    print(f"  Low-order: {cos_low:.4f}")

    # Metric 2: Angular concentration (R-bar)
    rbar_high = angular_concentration(high_weights)
    rbar_low = angular_concentration(low_weights)

    print(f"\nAngular concentration (R-bar):")
    print(f"  High-order: {rbar_high:.4f}")
    print(f"  Low-order: {rbar_low:.4f}")

    # Metric 3: Cone angle containing 50% of vectors
    cone_high = cone_angle_containing_fraction(high_weights, 0.5)
    cone_low = cone_angle_containing_fraction(low_weights, 0.5)

    print(f"\nCone angle containing 50% (radians):")
    print(f"  High-order: {cone_high:.4f} ({np.degrees(cone_high):.1f} degrees)")
    print(f"  Low-order: {cone_low:.4f} ({np.degrees(cone_low):.1f} degrees)")

    # Bootstrap for statistical testing
    print("\n--- Statistical Tests ---")

    # Test 1: Permutation test for cosine similarity difference
    n_perms = 10000
    observed_diff_cos = cos_high - cos_low

    combined = np.vstack([high_weights, low_weights])
    n_high = len(high_weights)

    perm_diffs = []
    for _ in range(n_perms):
        perm_idx = np.random.permutation(len(combined))
        perm_high = combined[perm_idx[:n_high]]
        perm_low = combined[perm_idx[n_high:]]
        perm_diff = mean_pairwise_cosine(perm_high) - mean_pairwise_cosine(perm_low)
        perm_diffs.append(perm_diff)

    perm_diffs = np.array(perm_diffs)
    p_value_cos = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff_cos))

    # Effect size (standardized)
    effect_size_cos = observed_diff_cos / (np.std(perm_diffs) + 1e-10)

    print(f"\nCosine similarity difference:")
    print(f"  Observed diff: {observed_diff_cos:.4f}")
    print(f"  Permutation std: {np.std(perm_diffs):.4f}")
    print(f"  Effect size (d): {effect_size_cos:.2f}")
    print(f"  p-value: {p_value_cos:.4f}")

    # Test 2: Direct comparison of within-group angular spread
    # Using bootstrap for cone angle comparison
    bootstrap_cone_diff = []
    for _ in range(1000):
        high_sample = high_weights[np.random.choice(len(high_weights), len(high_weights), replace=True)]
        low_sample = low_weights[np.random.choice(len(low_weights), len(low_weights), replace=True)]
        cone_h = cone_angle_containing_fraction(high_sample, 0.5)
        cone_l = cone_angle_containing_fraction(low_sample, 0.5)
        bootstrap_cone_diff.append(cone_h - cone_l)

    bootstrap_cone_diff = np.array(bootstrap_cone_diff)
    cone_diff_mean = np.mean(bootstrap_cone_diff)
    cone_diff_ci = np.percentile(bootstrap_cone_diff, [2.5, 97.5])

    # Effect size for cone angle
    effect_size_cone = cone_diff_mean / (np.std(bootstrap_cone_diff) + 1e-10)

    print(f"\nCone angle difference (high - low):")
    print(f"  Mean diff: {cone_diff_mean:.4f} rad ({np.degrees(cone_diff_mean):.1f} deg)")
    print(f"  95% CI: [{cone_diff_ci[0]:.4f}, {cone_diff_ci[1]:.4f}]")
    print(f"  Effect size (d): {effect_size_cone:.2f}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    validated = (p_value_cos < 0.01 and abs(effect_size_cos) > 0.5 and observed_diff_cos > 0)

    if validated:
        print("VALIDATED: High-order weight vectors show higher angular clustering")
        print(f"  Cosine similarity: {cos_high:.4f} vs {cos_low:.4f} (d={effect_size_cos:.2f}, p={p_value_cos:.4f})")
    else:
        if observed_diff_cos <= 0:
            print("REFUTED: High-order vectors are NOT more clustered (opposite direction)")
        elif p_value_cos >= 0.01:
            print("REFUTED: Difference not significant (p >= 0.01)")
        else:
            print("INCONCLUSIVE: Effect size below threshold")
        print(f"  Cosine similarity: {cos_high:.4f} vs {cos_low:.4f} (d={effect_size_cos:.2f}, p={p_value_cos:.4f})")

    # Return key metrics for logging
    return {
        'cos_high': cos_high,
        'cos_low': cos_low,
        'effect_size': effect_size_cos,
        'p_value': p_value_cos,
        'cone_high_deg': np.degrees(cone_high),
        'cone_low_deg': np.degrees(cone_low),
        'validated': validated
    }

if __name__ == '__main__':
    main()
