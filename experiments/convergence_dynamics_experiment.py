"""
RES-145: Order Gradient During Nested Sampling

Hypothesis: Order gradient magnitude increases during nested sampling progression

Method:
1. Track order gradient through NS iterations
2. Test: Should increase as NS progresses to high-order regions
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.thermo_sampler_v3 import CPPN, nested_sampling_v3, order_multiplicative as compute_order
from scipy import stats
import json

def run_experiment(n_samples=30, n_iterations=50, seed=42):
    """Test gradient magnitude progression through NS."""
    np.random.seed(seed)

    early_gradients = []
    late_gradients = []

    print(f"Tracking gradient magnitude through {n_samples} NS runs...")
    for sample_idx in range(n_samples):
        cppn = CPPN()

        ns_results = nested_sampling_v3(
            cppn,
            n_iterations=n_iterations,
            n_live=20,
            seed=None
        )

        if 'samples' in ns_results and len(ns_results['samples']) > 10:
            samples = ns_results['samples']

            # Early phase (first 20%)
            n_early = max(1, len(samples) // 5)
            for i in range(n_early - 1):
                w1 = samples[i].flatten()
                w2 = samples[i + 1].flatten()
                grad_mag = np.linalg.norm(w2 - w1)
                early_gradients.append(grad_mag)

            # Late phase (last 20%)
            start_late = max(n_early, len(samples) - len(samples) // 5)
            for i in range(start_late, len(samples) - 1):
                w1 = samples[i].flatten()
                w2 = samples[i + 1].flatten()
                grad_mag = np.linalg.norm(w2 - w1)
                late_gradients.append(grad_mag)

    if len(early_gradients) == 0:
        early_gradients = [0]
    if len(late_gradients) == 0:
        late_gradients = [0]

    early_gradients = np.array(early_gradients)
    late_gradients = np.array(late_gradients)

    # Compare
    t_stat, p_value = stats.ttest_ind(late_gradients, early_gradients)

    pooled_std = np.sqrt((np.std(late_gradients)**2 + np.std(early_gradients)**2) / 2)
    cohens_d = (np.mean(late_gradients) - np.mean(early_gradients)) / pooled_std if pooled_std > 0 else 0

    status = 'validated' if cohens_d > 0.5 and p_value < 0.01 else 'refuted'

    results = {
        'hypothesis': 'Order gradient magnitude increases during nested sampling progression',
        'effect_size': float(cohens_d),
        'p_value': float(p_value),
        'status': status,
        'summary': f'Early={np.mean(early_gradients):.3f}, Late={np.mean(late_gradients):.3f}, d={cohens_d:.2f}'
    }

    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'convergence_dynamics')
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved: {results}")

if __name__ == '__main__':
    run_experiment()
