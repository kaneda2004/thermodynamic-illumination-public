"""
RES-202: High-order weight regions exhibit higher box-counting fractal dimension

Hypothesis: High-order weight configurations occupy a fractal structure with
higher box-counting dimension than low-order configurations.

Building on:
- RES-172: High-order basins 2.3x smaller volume
- RES-147: Local minima form connected submanifolds
- RES-167: No Euclidean clustering of high-order weights

If high-order configurations are rare (small volume) but connected (RES-147),
they may form complex, filamentary structures with higher fractal dimension
than the more dispersed low-order regions.
"""

import numpy as np
from scipy import stats
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import CPPN, order_multiplicative as compute_order_metric

def box_counting_dimension(points, n_scales=10):
    """
    Estimate box-counting fractal dimension of point cloud.

    For each scale epsilon, count boxes needed to cover all points.
    D = -lim log(N)/log(epsilon) as epsilon -> 0
    """
    if len(points) < 10:
        return np.nan

    # Normalize points to [0, 1]^d
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1  # Avoid division by zero
    normalized = (points - mins) / ranges

    # Generate scales (powers of 2)
    epsilons = 2.0 ** np.arange(0, -n_scales, -1)  # 1, 0.5, 0.25, ...
    counts = []

    for eps in epsilons:
        # Discretize points to grid
        grid_coords = (normalized / eps).astype(int)
        # Count unique occupied boxes
        unique_boxes = len(set(map(tuple, grid_coords)))
        counts.append(unique_boxes)

    # Linear regression: log(N) vs log(1/epsilon)
    log_eps = np.log(1 / epsilons)
    log_counts = np.log(np.array(counts))

    # Use middle scales (avoid boundary effects)
    mid_start = n_scales // 4
    mid_end = 3 * n_scales // 4

    slope, _, r, p, _ = stats.linregress(
        log_eps[mid_start:mid_end],
        log_counts[mid_start:mid_end]
    )

    return slope  # This is the fractal dimension estimate


def correlation_dimension(points, k_range=(5, 20)):
    """
    Estimate correlation dimension using pairwise distance distribution.
    Alternative to box-counting, may be more stable for sparse points.
    """
    if len(points) < 50:
        return np.nan

    # Compute pairwise distances
    n = len(points)
    distances = []
    for i in range(min(n, 500)):  # Subsample for efficiency
        for j in range(i + 1, min(n, 500)):
            d = np.linalg.norm(points[i] - points[j])
            if d > 0:
                distances.append(d)

    if len(distances) < 100:
        return np.nan

    distances = np.array(distances)

    # Count pairs within radius r for multiple r values
    r_values = np.percentile(distances, np.linspace(10, 90, 20))
    C_r = []
    for r in r_values:
        C_r.append(np.mean(distances < r))

    # log(C(r)) vs log(r) slope is correlation dimension
    log_r = np.log(r_values)
    log_C = np.log(np.maximum(C_r, 1e-10))

    # Linear regression on middle portion
    slope, _, r_val, p, _ = stats.linregress(log_r[5:-5], log_C[5:-5])

    return slope


def run_experiment(n_samples=2000, n_bootstrap=50):
    """Compare fractal dimension of high vs low order weight configurations."""

    print("RES-202: Box-counting fractal dimension of weight configurations")
    print("=" * 60)

    # Generate samples
    print(f"\nGenerating {n_samples} CPPN samples...")
    weights = []
    orders = []

    for i in range(n_samples):
        cppn = CPPN()
        w = cppn.get_weights()
        img = cppn.render(size=32)
        order = compute_order_metric(img)
        weights.append(w)
        orders.append(order)

        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{n_samples}")

    weights = np.array(weights)
    orders = np.array(orders)

    print(f"\nOrder statistics: mean={np.mean(orders):.3f}, std={np.std(orders):.3f}")
    print(f"Weight space dimension: {weights.shape[1]}")

    # Split into high and low order quartiles
    q75 = np.percentile(orders, 75)
    q25 = np.percentile(orders, 25)

    high_idx = orders >= q75
    low_idx = orders <= q25

    high_weights = weights[high_idx]
    low_weights = weights[low_idx]

    print(f"\nHigh-order (>={q75:.3f}): {np.sum(high_idx)} samples")
    print(f"Low-order (<={q25:.3f}): {np.sum(low_idx)} samples")

    # Compute fractal dimensions with bootstrap
    print("\nComputing box-counting fractal dimensions...")

    high_dims = []
    low_dims = []

    for b in range(n_bootstrap):
        # Bootstrap sample
        high_boot = high_weights[np.random.choice(len(high_weights), len(high_weights), replace=True)]
        low_boot = low_weights[np.random.choice(len(low_weights), len(low_weights), replace=True)]

        high_dims.append(box_counting_dimension(high_boot))
        low_dims.append(box_counting_dimension(low_boot))

    high_dims = np.array([d for d in high_dims if not np.isnan(d)])
    low_dims = np.array([d for d in low_dims if not np.isnan(d)])

    print(f"\nBox-counting dimension:")
    print(f"  High-order: {np.mean(high_dims):.3f} +/- {np.std(high_dims):.3f}")
    print(f"  Low-order:  {np.mean(low_dims):.3f} +/- {np.std(low_dims):.3f}")

    # Statistical test
    t_stat, p_value = stats.ttest_ind(high_dims, low_dims)
    cohens_d = (np.mean(high_dims) - np.mean(low_dims)) / np.sqrt((np.var(high_dims) + np.var(low_dims)) / 2)

    print(f"\n  t-statistic: {t_stat:.3f}")
    print(f"  p-value: {p_value:.2e}")
    print(f"  Cohen's d: {cohens_d:.3f}")

    # Also compute correlation dimension as alternative measure
    print("\n\nComputing correlation dimension...")

    high_corr_dims = []
    low_corr_dims = []

    for b in range(min(n_bootstrap, 20)):  # Fewer bootstrap for correlation dim
        high_boot = high_weights[np.random.choice(len(high_weights), min(500, len(high_weights)), replace=False)]
        low_boot = low_weights[np.random.choice(len(low_weights), min(500, len(low_weights)), replace=False)]

        high_corr_dims.append(correlation_dimension(high_boot))
        low_corr_dims.append(correlation_dimension(low_boot))

    high_corr_dims = np.array([d for d in high_corr_dims if not np.isnan(d)])
    low_corr_dims = np.array([d for d in low_corr_dims if not np.isnan(d)])

    if len(high_corr_dims) > 0 and len(low_corr_dims) > 0:
        print(f"\nCorrelation dimension:")
        print(f"  High-order: {np.mean(high_corr_dims):.3f} +/- {np.std(high_corr_dims):.3f}")
        print(f"  Low-order:  {np.mean(low_corr_dims):.3f} +/- {np.std(low_corr_dims):.3f}")

        t_stat2, p_value2 = stats.ttest_ind(high_corr_dims, low_corr_dims)
        cohens_d2 = (np.mean(high_corr_dims) - np.mean(low_corr_dims)) / np.sqrt((np.var(high_corr_dims) + np.var(low_corr_dims)) / 2)

        print(f"\n  t-statistic: {t_stat2:.3f}")
        print(f"  p-value: {p_value2:.2e}")
        print(f"  Cohen's d: {cohens_d2:.3f}")

    # Check if dimension correlates with order across full sample
    print("\n\nCorrelation analysis: Local dimension vs order")

    # Compute local dimension for individual samples using k-NN
    from sklearn.neighbors import NearestNeighbors

    knn = NearestNeighbors(n_neighbors=20)
    knn.fit(weights)
    distances, _ = knn.kneighbors(weights)

    # Local dimension estimate from k-NN distances (MLE estimator)
    # D_local ≈ k / sum(log(r_k/r_i) for i in 1..k-1)
    local_dims = []
    for i in range(len(weights)):
        r = distances[i, 1:]  # Exclude self
        r_k = r[-1]
        if r_k > 0 and np.all(r[:-1] > 0):
            # MLE formula
            d = (len(r) - 1) / np.sum(np.log(r_k / r[:-1]))
            local_dims.append(d)
        else:
            local_dims.append(np.nan)

    local_dims = np.array(local_dims)
    valid = ~np.isnan(local_dims)

    rho, p_rho = stats.spearmanr(orders[valid], local_dims[valid])
    r_pearson, p_pearson = stats.pearsonr(orders[valid], local_dims[valid])

    print(f"\n  Spearman rho: {rho:.3f}, p={p_rho:.2e}")
    print(f"  Pearson r: {r_pearson:.3f}, p={p_pearson:.2e}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if p_value < 0.01 and abs(cohens_d) > 0.5:
        if cohens_d > 0:
            status = "VALIDATED"
            summary = f"High-order weight regions have HIGHER box-counting dimension ({np.mean(high_dims):.2f} vs {np.mean(low_dims):.2f}, d={cohens_d:.2f})"
        else:
            status = "REFUTED"
            summary = f"High-order weight regions have LOWER box-counting dimension ({np.mean(high_dims):.2f} vs {np.mean(low_dims):.2f}, d={cohens_d:.2f})"
    else:
        status = "REFUTED"
        summary = f"No significant difference in fractal dimension (high={np.mean(high_dims):.2f}, low={np.mean(low_dims):.2f}, d={cohens_d:.2f}, p={p_value:.2e})"

    print(f"\nStatus: {status}")
    print(f"Result: {summary}")
    print(f"\nEffect size (Cohen's d): {cohens_d:.2f}")
    print(f"P-value: {p_value:.2e}")

    return {
        'status': status,
        'high_dim_mean': float(np.mean(high_dims)),
        'low_dim_mean': float(np.mean(low_dims)),
        'cohens_d': float(cohens_d),
        'p_value': float(p_value),
        'spearman_rho': float(rho),
        'summary': summary
    }


if __name__ == '__main__':
    results = run_experiment()
