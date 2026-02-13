"""
RES-127: Nested Sampling Trajectory Geometry

Hypothesis: Nested sampling follows a low-dimensional manifold in weight space

Method:
1. Run nested sampling
2. Analyze trajectory dimension (PCA)
3. Measure linearity
4. Test: Dimension should be low, linearity high
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.thermo_sampler_v3 import CPPN, nested_sampling, order_multiplicative as compute_order
from scipy import stats
import json

def run_experiment(n_samples=15, n_iterations=50, seed=42):
    """Test trajectory manifold properties."""
    np.random.seed(seed)

    trajectory_dims = []
    linearities = []

    print(f"Analyzing trajectory geometry ({n_samples} NS runs)...")
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

            # PCA for dimension
            cov = np.cov(samples.T)
            eigenvalues = np.linalg.eigvals(cov)
            eigenvalues = np.sort(eigenvalues)[::-1]

            # Effective dimension (ratio of largest to sum)
            if np.sum(eigenvalues) > 0:
                eff_dim = np.max(eigenvalues) / np.sum(eigenvalues)
            else:
                eff_dim = 0

            trajectory_dims.append(eff_dim)

            # Linearity: measure how well trajectory fits a line
            if samples.shape[0] > 2:
                distances = np.linalg.norm(samples[1:] - samples[:-1], axis=1)
                total_dist = np.sum(distances)
                direct_dist = np.linalg.norm(samples[-1] - samples[0])

                linearity = direct_dist / (total_dist + 1e-10)
            else:
                linearity = 0

            linearities.append(linearity)

    trajectory_dims = np.array(trajectory_dims)
    linearities = np.array(linearities)

    # Test: expect low dimension (high eff_dim ratio) and high linearity
    mean_dim = np.mean(trajectory_dims)
    mean_lin = np.mean(linearities)

    # Effect size for linearity
    cohens_d = (mean_lin - 0.5) / (np.std(linearities) + 1e-10)
    p_value = stats.ttest_1samp(linearities, 0.1)[1]

    status = 'refuted' if mean_lin < 0.3 else 'validated'

    results = {
        'hypothesis': 'Nested sampling follows a low-dimensional manifold in weight space',
        'effect_size': float(cohens_d),
        'p_value': float(p_value),
        'status': status,
        'summary': f'Mean dim={mean_dim:.3f}, Mean linearity={mean_lin:.3f}, d={cohens_d:.2f}'
    }

    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'sampling_trajectory_geometry')
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved: {results}")

if __name__ == '__main__':
    run_experiment()
