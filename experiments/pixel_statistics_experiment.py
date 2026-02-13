"""
RES-146: Pixel Variance in Neighborhoods

Hypothesis: CPPN images have lower local pixel variance in neighborhoods than random images

Method:
1. Compute local neighborhood variance
2. Compare CPPN vs random images
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.thermo_sampler_v3 import CPPN, order_multiplicative as compute_order
from scipy import stats
import json

def compute_local_variance(img, kernel_size=3):
    """Compute mean local pixel variance in neighborhoods."""
    pad_size = kernel_size // 2
    padded = np.pad(img, pad_size, mode='edge')

    local_vars = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            patch = padded[i:i+kernel_size, j:j+kernel_size]
            local_vars.append(np.var(patch))

    return np.mean(local_vars) if len(local_vars) > 0 else 0

def run_experiment(n_samples=100, seed=42):
    """Test local pixel variance."""
    np.random.seed(seed)

    cppn_variances = []
    random_variances = []

    print(f"Computing local pixel variance for {n_samples} image pairs...")
    for i in range(n_samples):
        # CPPN
        cppn = CPPN(hidden_nodes=3)
        cppn.randomize()
        img_cppn = cppn.render(64)
        var_cppn = compute_local_variance(img_cppn)
        cppn_variances.append(var_cppn)

        # Random
        img_rand = np.random.rand(64, 64)
        var_rand = compute_local_variance(img_rand)
        random_variances.append(var_rand)

    cppn_variances = np.array(cppn_variances)
    random_variances = np.array(random_variances)

    # Compare
    t_stat, p_value = stats.ttest_ind(cppn_variances, random_variances)

    pooled_std = np.sqrt((np.std(cppn_variances)**2 + np.std(random_variances)**2) / 2)
    cohens_d = (np.mean(cppn_variances) - np.mean(random_variances)) / pooled_std if pooled_std > 0 else 0

    status = 'validated' if cohens_d < -1.0 and p_value < 0.001 else 'refuted'

    results = {
        'hypothesis': 'CPPN images have lower local pixel variance in neighborhoods than random images',
        'effect_size': float(cohens_d),
        'p_value': float(p_value),
        'status': status,
        'summary': f'CPPN var={np.mean(cppn_variances):.6f}, Random var={np.mean(random_variances):.6f}, d={cohens_d:.2f}'
    }

    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'pixel_statistics')
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved: {results}")

if __name__ == '__main__':
    run_experiment()
