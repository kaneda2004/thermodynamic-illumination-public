"""
RES-096: Run-Length Encoding and Order

Hypothesis: High-order CPPN images have longer run-lengths than random images

Method:
1. Generate CPPN and random images
2. Compute run-length statistics
3. Compare distributions
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.thermo_sampler_v3 import CPPN, order_multiplicative as compute_order
from scipy import stats
import json

def compute_run_lengths(img):
    """Compute mean run length (horizontal consecutive pixels)."""
    runs = []
    for row in img:
        binary = (row > 0.5).astype(int)
        changes = np.diff(np.concatenate([[0], binary, [0]]))
        run_starts = np.where(changes == 1)[0]
        run_ends = np.where(changes == -1)[0]
        run_lengths = run_ends - run_starts
        runs.extend(run_lengths)

    return np.mean(runs) if len(runs) > 0 else 0

def run_experiment(n_samples=100, seed=42):
    """Test run-length in CPPN vs random images."""
    np.random.seed(seed)

    cppn_runs = []
    random_runs = []

    print(f"Measuring run-lengths in {n_samples} image pairs...")
    for i in range(n_samples):
        # CPPN image
        cppn = CPPN(hidden_nodes=3)
        cppn.randomize()
        img_cppn = cppn.render(64)
        run_cppn = compute_run_lengths(img_cppn)
        cppn_runs.append(run_cppn)

        # Random image
        img_rand = np.random.rand(64, 64)
        run_rand = compute_run_lengths(img_rand)
        random_runs.append(run_rand)

    cppn_runs = np.array(cppn_runs)
    random_runs = np.array(random_runs)

    # Compare
    t_stat, p_value = stats.ttest_ind(cppn_runs, random_runs)

    pooled_std = np.sqrt((np.std(cppn_runs)**2 + np.std(random_runs)**2) / 2)
    cohens_d = (np.mean(cppn_runs) - np.mean(random_runs)) / pooled_std if pooled_std > 0 else 0

    status = 'validated' if cohens_d > 0.5 and p_value < 0.001 else 'refuted'

    results = {
        'hypothesis': 'High-order CPPN images have longer run-lengths than random images',
        'effect_size': float(cohens_d),
        'p_value': float(p_value),
        'status': status,
        'summary': f'CPPN runs={np.mean(cppn_runs):.2f}, Random runs={np.mean(random_runs):.2f}, d={cohens_d:.2f}'
    }

    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'run_length_encoding')
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved: {results}")

if __name__ == '__main__':
    run_experiment()
