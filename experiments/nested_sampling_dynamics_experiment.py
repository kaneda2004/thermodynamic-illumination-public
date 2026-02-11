"""
RES-118: Nested Sampling Live Points Clustering

Hypothesis: Nested sampling live points form increasingly separated clusters

Method:
1. Run nested sampling
2. Track live point positions through iterations
3. Measure cluster separation (silhouette score)
4. Test: Increasing separation with iterations
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.thermo_sampler_v3 import CPPN, nested_sampling, order_multiplicative as compute_order
from scipy import stats
import json

def compute_cluster_metric(points):
    """Simplified cluster separation metric."""
    if len(points) < 2:
        return 0

    distances = np.linalg.norm(points[:, None] - points[None, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    min_distances = np.min(distances, axis=1)

    return np.mean(min_distances) if len(min_distances) > 0 else 0

def run_experiment(n_samples=20, n_iterations=50, seed=42):
    """Test cluster separation through NS iterations."""
    np.random.seed(seed)

    cluster_metrics = []
    iteration_indices = []

    print(f"Tracking live point clustering through {n_samples} NS runs...")
    for sample_idx in range(n_samples):
        cppn = CPPN(hidden_nodes=3)
        cppn.randomize()

        ns_results = nested_sampling(
            cppn,
            n_iterations=n_iterations,
            n_live=20,
            seed=None
        )

        if 'samples' in ns_results and len(ns_results['samples']) > 5:
            samples = np.array([s.flatten()[:5] for s in ns_results['samples']])

            # Compute metrics at beginning, middle, end
            if len(samples) > 0:
                for frac in [0.0, 0.5, 1.0]:
                    idx = int(frac * (len(samples) - 1))
                    metric = compute_cluster_metric(samples[:idx+1])
                    cluster_metrics.append(metric)
                    iteration_indices.append(frac)

    cluster_metrics = np.array(cluster_metrics)
    iteration_indices = np.array(iteration_indices)

    # Correlation: later iterations have higher separation
    r, p_value = stats.spearmanr(iteration_indices, cluster_metrics)

    # Effect size: early vs late
    early_mask = iteration_indices < 0.3
    late_mask = iteration_indices > 0.7

    early_metrics = cluster_metrics[early_mask]
    late_metrics = cluster_metrics[late_mask]

    if len(early_metrics) > 0 and len(late_metrics) > 0:
        pooled_std = np.sqrt((np.std(early_metrics)**2 + np.std(late_metrics)**2) / 2)
        cohens_d = (np.mean(late_metrics) - np.mean(early_metrics)) / pooled_std if pooled_std > 0 else 0
    else:
        cohens_d = 0

    status = 'validated' if r > 0.4 and p_value < 0.05 else 'refuted'

    results = {
        'hypothesis': 'Nested sampling live points form increasingly separated clusters',
        'effect_size': float(cohens_d),
        'p_value': float(p_value),
        'status': status,
        'summary': f'Correlation r={r:.3f} (p={p_value:.2e}), d={cohens_d:.2f}'
    }

    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'nested_sampling_dynamics')
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved: {results}")

if __name__ == '__main__':
    run_experiment()
