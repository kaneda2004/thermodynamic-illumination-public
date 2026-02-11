"""
RES-143: Order gradient at high-low boundary points toward nearest high-order peak not global max

Hypothesis: At the boundary between high and low order regions in weight space,
the order gradient points toward the nearest local high-order peak rather than
the global maximum. This would explain why gradient descent from random init
gets stuck at local optima.

Method:
1. Find multiple high-order optima via nested sampling
2. Identify boundary points (order between 0.1-0.3)
3. Compute order gradient at boundary points
4. Check if gradient direction points toward nearest vs farthest high-order peak
5. Measure alignment with nearest peak direction
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import CPPN, order_multiplicative
from scipy.stats import pearsonr, ttest_ind
from scipy.spatial.distance import cdist

def compute_order_gradient(weights, cppn, epsilon=0.01):
    """Compute numerical gradient of order w.r.t. weights."""
    grad = np.zeros_like(weights)
    cppn.set_weights(weights)
    base_order = order_multiplicative(cppn.render(32))

    for i in range(len(weights)):
        w_plus = weights.copy()
        w_plus[i] += epsilon
        cppn.set_weights(w_plus)
        order_plus = order_multiplicative(cppn.render(32))
        grad[i] = (order_plus - base_order) / epsilon

    cppn.set_weights(weights)  # Reset
    return grad

def find_high_order_peaks(n_peaks=20, n_samples=500, threshold=0.3):
    """Find high-order weight configurations via random search."""
    peaks = []
    orders = []

    for _ in range(n_samples):
        cppn = CPPN()
        weights = np.random.randn(5) * 1.0  # 4 weights + 1 bias
        cppn.set_weights(weights)
        order = order_multiplicative(cppn.render(32))

        if order > threshold:
            peaks.append(weights.copy())
            orders.append(order)

    # Return top n_peaks
    if len(peaks) == 0:
        # Lower threshold if needed
        return find_high_order_peaks(n_peaks, n_samples * 2, threshold * 0.5)

    sorted_idx = np.argsort(orders)[::-1][:n_peaks]
    return [peaks[i] for i in sorted_idx], [orders[i] for i in sorted_idx]

def find_boundary_points(n_points=100, order_range=(0.1, 0.3)):
    """Find points at the boundary between high and low order."""
    boundary_points = []
    boundary_orders = []

    attempts = 0
    while len(boundary_points) < n_points and attempts < n_points * 50:
        cppn = CPPN()
        weights = np.random.randn(5) * 1.5
        cppn.set_weights(weights)
        order = order_multiplicative(cppn.render(32))

        if order_range[0] <= order <= order_range[1]:
            boundary_points.append(weights.copy())
            boundary_orders.append(order)
        attempts += 1

    return boundary_points, boundary_orders

def main():
    np.random.seed(42)

    print("Finding high-order peaks...")
    peaks, peak_orders = find_high_order_peaks(n_peaks=30, n_samples=1000, threshold=0.2)
    print(f"Found {len(peaks)} peaks with orders: {np.mean(peak_orders):.3f} +/- {np.std(peak_orders):.3f}")

    print("\nFinding boundary points...")
    boundary_pts, boundary_orders = find_boundary_points(n_points=100, order_range=(0.05, 0.2))
    print(f"Found {len(boundary_pts)} boundary points, order range: [{min(boundary_orders):.3f}, {max(boundary_orders):.3f}]")

    # Convert to arrays for distance computation
    peaks_arr = np.array(peaks)
    boundary_arr = np.array(boundary_pts)

    # For each boundary point, compute gradient and check alignment
    nearest_alignments = []
    farthest_alignments = []
    random_alignments = []

    print("\nComputing gradients and alignments...")
    cppn = CPPN()

    for i, (bpt, b_order) in enumerate(zip(boundary_pts, boundary_orders)):
        if i % 20 == 0:
            print(f"  Processing {i}/{len(boundary_pts)}...")

        # Compute distances to all peaks
        dists = cdist([bpt], peaks_arr, metric='euclidean')[0]
        nearest_idx = np.argmin(dists)
        farthest_idx = np.argmax(dists)

        # Direction vectors
        dir_nearest = peaks_arr[nearest_idx] - bpt
        dir_farthest = peaks_arr[farthest_idx] - bpt
        dir_random = peaks_arr[np.random.randint(len(peaks))] - bpt

        # Normalize
        dir_nearest = dir_nearest / (np.linalg.norm(dir_nearest) + 1e-10)
        dir_farthest = dir_farthest / (np.linalg.norm(dir_farthest) + 1e-10)
        dir_random = dir_random / (np.linalg.norm(dir_random) + 1e-10)

        # Compute order gradient
        grad = compute_order_gradient(bpt, cppn)
        grad_norm = np.linalg.norm(grad)

        if grad_norm < 1e-10:
            continue

        grad_unit = grad / grad_norm

        # Alignment (cosine similarity)
        align_nearest = np.dot(grad_unit, dir_nearest)
        align_farthest = np.dot(grad_unit, dir_farthest)
        align_random = np.dot(grad_unit, dir_random)

        nearest_alignments.append(align_nearest)
        farthest_alignments.append(align_farthest)
        random_alignments.append(align_random)

    nearest_alignments = np.array(nearest_alignments)
    farthest_alignments = np.array(farthest_alignments)
    random_alignments = np.array(random_alignments)

    print(f"\n=== RESULTS (n={len(nearest_alignments)}) ===")
    print(f"Alignment with NEAREST peak:  {np.mean(nearest_alignments):.4f} +/- {np.std(nearest_alignments):.4f}")
    print(f"Alignment with FARTHEST peak: {np.mean(farthest_alignments):.4f} +/- {np.std(farthest_alignments):.4f}")
    print(f"Alignment with RANDOM peak:   {np.mean(random_alignments):.4f} +/- {np.std(random_alignments):.4f}")

    # Statistical tests
    # Test if nearest alignment is significantly greater than farthest
    t_stat, p_nearest_vs_farthest = ttest_ind(nearest_alignments, farthest_alignments)

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.var(nearest_alignments) + np.var(farthest_alignments)) / 2)
    d_nearest_vs_farthest = (np.mean(nearest_alignments) - np.mean(farthest_alignments)) / pooled_std

    # Test if nearest > random
    t_stat_rand, p_nearest_vs_random = ttest_ind(nearest_alignments, random_alignments)
    pooled_std_rand = np.sqrt((np.var(nearest_alignments) + np.var(random_alignments)) / 2)
    d_nearest_vs_random = (np.mean(nearest_alignments) - np.mean(random_alignments)) / pooled_std_rand

    print(f"\n=== STATISTICAL TESTS ===")
    print(f"Nearest vs Farthest: t={t_stat:.3f}, p={p_nearest_vs_farthest:.6f}, d={d_nearest_vs_farthest:.3f}")
    print(f"Nearest vs Random:   t={t_stat_rand:.3f}, p={p_nearest_vs_random:.6f}, d={d_nearest_vs_random:.3f}")

    # Additional: What fraction of gradient points more toward nearest than farthest?
    fraction_nearest = np.mean(nearest_alignments > farthest_alignments)
    print(f"\nFraction where gradient points more toward nearest: {fraction_nearest:.3f}")

    # Check if mean alignment is positive (gradient points uphill toward ANY peak)
    t_pos, p_positive = ttest_ind(nearest_alignments, np.zeros_like(nearest_alignments))
    print(f"Is mean nearest alignment > 0? mean={np.mean(nearest_alignments):.4f}, p={p_positive:.6f}")

    # Verdict
    print("\n=== VERDICT ===")
    validated = (p_nearest_vs_farthest < 0.01 and d_nearest_vs_farthest > 0.5 and
                 np.mean(nearest_alignments) > np.mean(farthest_alignments))

    if validated:
        print("VALIDATED: Gradient at boundary points toward nearest peak (d={:.2f}, p={:.2e})".format(
            d_nearest_vs_farthest, p_nearest_vs_farthest))
    else:
        # Check if gradient is actually NEGATIVE (pointing away from peaks)
        if np.mean(nearest_alignments) < 0:
            print("REFUTED: Gradient points AWAY from peaks at boundary")
        elif abs(d_nearest_vs_farthest) < 0.5:
            print("INCONCLUSIVE: Effect size too small (d={:.2f})".format(d_nearest_vs_farthest))
        else:
            print("REFUTED: Gradient does not preferentially point toward nearest peak")

    return {
        'n': len(nearest_alignments),
        'mean_nearest': float(np.mean(nearest_alignments)),
        'mean_farthest': float(np.mean(farthest_alignments)),
        'mean_random': float(np.mean(random_alignments)),
        'd_nearest_vs_farthest': float(d_nearest_vs_farthest),
        'p_nearest_vs_farthest': float(p_nearest_vs_farthest),
        'fraction_nearest': float(fraction_nearest)
    }

if __name__ == "__main__":
    results = main()
