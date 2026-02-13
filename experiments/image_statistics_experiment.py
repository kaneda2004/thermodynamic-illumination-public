"""
RES-142: Local Gradient Coherence

Hypothesis: CPPN images have higher local gradient coherence than random images

Method:
1. Generate CPPN and random images
2. Compute local gradient structure tensor coherence
3. Compare distributions
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.thermo_sampler_v3 import CPPN, order_multiplicative as compute_order
from scipy import stats
import json

def compute_gradient_coherence(img):
    """Compute local gradient coherence using structure tensor."""
    # Compute gradients
    gy, gx = np.gradient(img)

    # Structure tensor components
    ixx = gx * gx
    iyy = gy * gy
    ixy = gx * gy

    # Local sums (simplified: just use pixel variance)
    coherence = np.mean(np.abs(ixy)) / (np.mean(ixx) + np.mean(iyy) + 1e-10)

    return coherence

def run_experiment(n_samples=100, seed=42):
    """Test gradient coherence in CPPN vs random."""
    np.random.seed(seed)

    cppn_coherences = []
    random_coherences = []

    print(f"Computing gradient coherence for {n_samples} image pairs...")
    for i in range(n_samples):
        # CPPN
        cppn = CPPN(hidden_nodes=3)
        cppn.randomize()
        img_cppn = cppn.render(64)
        coh_cppn = compute_gradient_coherence(img_cppn)
        cppn_coherences.append(coh_cppn)

        # Random
        img_rand = np.random.rand(64, 64)
        coh_rand = compute_gradient_coherence(img_rand)
        random_coherences.append(coh_rand)

    cppn_coherences = np.array(cppn_coherences)
    random_coherences = np.array(random_coherences)

    # Compare
    t_stat, p_value = stats.ttest_ind(cppn_coherences, random_coherences)

    pooled_std = np.sqrt((np.std(cppn_coherences)**2 + np.std(random_coherences)**2) / 2)
    cohens_d = (np.mean(cppn_coherences) - np.mean(random_coherences)) / pooled_std if pooled_std > 0 else 0

    status = 'validated' if cohens_d > 0.5 and p_value < 0.001 else 'refuted'

    results = {
        'hypothesis': 'CPPN images have higher local gradient coherence than random images',
        'effect_size': float(cohens_d),
        'p_value': float(p_value),
        'status': status,
        'summary': f'CPPN coh={np.mean(cppn_coherences):.4f}, Random coh={np.mean(random_coherences):.4f}, d={cohens_d:.2f}'
    }

    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'image_statistics')
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved: {results}")

if __name__ == '__main__':
    run_experiment()
