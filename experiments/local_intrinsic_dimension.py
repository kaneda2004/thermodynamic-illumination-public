"""
RES-187: Local intrinsic dimension (MLE at k-NN) negatively correlates with CPPN order

Hypothesis: High-order CPPN weight vectors reside on a lower local-dimensional
manifold than low-order ones - the manifold "narrows" near structured outputs.

Method:
1. Generate N CPPNs, extract weight vectors and compute order for each
2. For each sample, estimate LOCAL intrinsic dimension using MLE (Levina & Bickel 2004)
   based on distances to its k nearest neighbors in weight space
3. Test correlation between local dimension and order

The MLE estimator: d_hat = 1 / mean(log(r_k / r_j) for j=1..k-1)
where r_j is the distance to j-th nearest neighbor.
"""

import numpy as np
import json
import os
from pathlib import Path
from scipy import stats
from sklearn.neighbors import NearestNeighbors

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.thermo_sampler_v3 import CPPN, order_multiplicative, set_global_seed


def mle_local_dimension(distances: np.ndarray, k: int = 10) -> float:
    """
    Estimate local intrinsic dimension using MLE (Levina & Bickel 2004).

    Args:
        distances: sorted distances to k nearest neighbors (excluding self)
        k: number of neighbors to use

    Returns:
        Estimated local dimension
    """
    # Use distances to first k neighbors
    dists = distances[1:k+1]  # Skip self (distance 0)

    # Avoid numerical issues
    dists = np.maximum(dists, 1e-10)

    if len(dists) < 2 or dists[-1] < 1e-9:
        return np.nan

    # MLE: d = (k-1) / sum_{j=1}^{k-1} log(r_k / r_j)
    r_k = dists[-1]  # Distance to k-th neighbor
    log_ratios = np.log(r_k / dists[:-1])  # log(r_k / r_j) for j=1..k-1

    if np.sum(log_ratios) < 1e-10:
        return np.nan

    d_hat = (len(log_ratios)) / np.sum(log_ratios)
    return d_hat


def run_experiment(n_samples=500, image_size=32, k_values=[5, 10, 20], seed=42):
    """Run the local intrinsic dimension experiment."""
    set_global_seed(seed)

    print(f"Generating {n_samples} CPPN samples...")

    # Generate samples
    weight_vectors = []
    orders = []

    for i in range(n_samples):
        cppn = CPPN()
        img = cppn.render(image_size)
        order = order_multiplicative(img)

        weight_vectors.append(cppn.get_weights())
        orders.append(order)

        if (i+1) % 100 == 0:
            print(f"  Generated {i+1}/{n_samples}")

    weight_vectors = np.array(weight_vectors)
    orders = np.array(orders)

    print(f"\nWeight space dimension: {weight_vectors.shape[1]}")
    print(f"Order stats: mean={orders.mean():.4f}, std={orders.std():.4f}")
    print(f"Order range: [{orders.min():.4f}, {orders.max():.4f}]")

    # Compute local intrinsic dimension for each sample using k-NN
    results = {}

    for k in k_values:
        print(f"\nComputing local dimension with k={k}...")

        # Fit k-NN
        nn = NearestNeighbors(n_neighbors=k+1)  # +1 to include self
        nn.fit(weight_vectors)
        distances, _ = nn.kneighbors(weight_vectors)

        # Compute local dimension for each sample
        local_dims = []
        for i in range(n_samples):
            d = mle_local_dimension(distances[i], k=k)
            local_dims.append(d)

        local_dims = np.array(local_dims)
        valid_mask = ~np.isnan(local_dims)

        print(f"  Valid samples: {np.sum(valid_mask)}/{n_samples}")
        print(f"  Local dim stats: mean={np.nanmean(local_dims):.2f}, std={np.nanstd(local_dims):.2f}")

        # Correlation with order
        valid_dims = local_dims[valid_mask]
        valid_orders = orders[valid_mask]

        r, p = stats.pearsonr(valid_dims, valid_orders)
        rho, p_spearman = stats.spearmanr(valid_dims, valid_orders)

        print(f"  Pearson r={r:.4f}, p={p:.2e}")
        print(f"  Spearman rho={rho:.4f}, p={p_spearman:.2e}")

        # Stratify by order quartiles
        quartiles = np.percentile(valid_orders, [25, 50, 75])
        low_mask = valid_orders <= quartiles[0]
        high_mask = valid_orders >= quartiles[2]

        low_dim = valid_dims[low_mask].mean()
        high_dim = valid_dims[high_mask].mean()

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((valid_dims[low_mask].std()**2 + valid_dims[high_mask].std()**2) / 2)
        cohens_d = (high_dim - low_dim) / (pooled_std + 1e-10)

        # T-test
        t_stat, t_p = stats.ttest_ind(valid_dims[high_mask], valid_dims[low_mask])

        print(f"  High-order quartile dim: {high_dim:.2f}")
        print(f"  Low-order quartile dim: {low_dim:.2f}")
        print(f"  Cohen's d (high-low): {cohens_d:.3f}")
        print(f"  T-test p-value: {t_p:.2e}")

        results[f'k={k}'] = {
            'pearson_r': float(r),
            'pearson_p': float(p),
            'spearman_rho': float(rho),
            'spearman_p': float(p_spearman),
            'mean_local_dim': float(np.nanmean(local_dims)),
            'std_local_dim': float(np.nanstd(local_dims)),
            'high_order_dim': float(high_dim),
            'low_order_dim': float(low_dim),
            'cohens_d': float(cohens_d),
            't_test_p': float(t_p),
            'valid_samples': int(np.sum(valid_mask))
        }

    # Primary result (k=10 is standard)
    primary_k = 'k=10'
    primary = results[primary_k]

    # Determine validation status
    # Hypothesis: negative correlation (high order -> lower local dim)
    validated = (
        primary['pearson_p'] < 0.01 and
        abs(primary['cohens_d']) > 0.5 and
        primary['pearson_r'] < 0  # Expecting negative correlation
    )

    refuted = (
        primary['pearson_p'] < 0.01 and
        abs(primary['cohens_d']) > 0.5 and
        primary['pearson_r'] > 0  # Opposite direction
    )

    if validated:
        status = 'validated'
    elif refuted:
        status = 'refuted'
    else:
        status = 'inconclusive'

    print(f"\n{'='*60}")
    print(f"PRIMARY RESULT (k=10):")
    print(f"  Correlation: r={primary['pearson_r']:.4f}, p={primary['pearson_p']:.2e}")
    print(f"  Effect size: d={primary['cohens_d']:.3f}")
    print(f"  Status: {status.upper()}")
    print(f"{'='*60}")

    # Save results
    output_dir = Path(__file__).parent.parent / 'results' / 'local_intrinsic_dimension'
    output_dir.mkdir(parents=True, exist_ok=True)

    full_results = {
        'experiment_id': 'RES-187',
        'hypothesis': 'Local intrinsic dimension (MLE at k-NN) negatively correlates with CPPN order',
        'domain': 'manifold_structure',
        'status': status,
        'parameters': {
            'n_samples': n_samples,
            'image_size': image_size,
            'k_values': k_values,
            'seed': seed,
            'weight_dim': int(weight_vectors.shape[1])
        },
        'order_stats': {
            'mean': float(orders.mean()),
            'std': float(orders.std()),
            'min': float(orders.min()),
            'max': float(orders.max())
        },
        'results_by_k': results,
        'primary_k': primary_k,
        'primary_result': {
            'pearson_r': primary['pearson_r'],
            'pearson_p': primary['pearson_p'],
            'cohens_d': primary['cohens_d'],
            'high_order_dim': primary['high_order_dim'],
            'low_order_dim': primary['low_order_dim']
        }
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(full_results, f, indent=2)

    print(f"\nResults saved to {output_dir / 'results.json'}")

    return full_results


if __name__ == '__main__':
    results = run_experiment()
