"""
RES-068: Symmetry Distribution in High-Order CPPN Images

Hypothesis: High-order CPPN images have bilateral symmetry over rotational

Method:
1. Generate CPPN samples
2. Measure bilateral (vertical/horizontal) and rotational symmetry
3. Test chi-square for symmetry type distribution
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.thermo_sampler_v3 import CPPN, order_multiplicative as compute_order
from scipy import stats
import json

def measure_symmetries(img):
    """Measure bilateral and rotational symmetries."""
    # Vertical flip
    v_flip = np.flip(img, axis=0)
    v_sym = np.mean(np.abs(img - v_flip))

    # Horizontal flip
    h_flip = np.flip(img, axis=1)
    h_sym = np.mean(np.abs(img - h_flip))

    # 90-degree rotation
    rot90 = np.rot90(img)
    r90_sym = np.mean(np.abs(img - rot90))

    return v_sym, h_sym, r90_sym

def run_experiment(n_samples=100, seed=42):
    """Measure symmetry types in CPPN images."""
    np.random.seed(seed)

    v_sims = []
    h_sims = []
    r_sims = []

    print(f"Sampling {n_samples} CPPNs and measuring symmetries...")
    for i in range(n_samples):
        cppn = CPPN(hidden_nodes=3)
        cppn.randomize()
        img = cppn.render(64)

        v, h, r = measure_symmetries(img)
        v_sims.append(v)
        h_sims.append(h)
        r_sims.append(r)

    v_sims = np.array(v_sims)
    h_sims = np.array(h_sims)
    r_sims = np.array(r_sims)

    # Categorize each image by strongest symmetry
    bilateral_count = 0
    rot_count = 0

    for v, h, r in zip(v_sims, h_sims, r_sims):
        bilateral_error = min(v, h)
        if bilateral_error < r:
            bilateral_count += 1
        else:
            rot_count += 1

    # Chi-square test
    chi2, p_value = stats.chisquare([bilateral_count, rot_count], f_exp=[n_samples/2, n_samples/2])

    # Effect size
    proportions = [bilateral_count / n_samples, rot_count / n_samples]
    cohens_w = np.sqrt(np.sum((proportions - 0.5)**2 / 0.5))

    status = 'validated' if p_value < 0.01 else 'refuted'

    results = {
        'hypothesis': 'High-order CPPN images have bilateral symmetry over rotational',
        'effect_size': float(cohens_w),
        'p_value': float(p_value),
        'status': status,
        'summary': f'Bilateral: {bilateral_count/n_samples*100:.1f}%, Rotational: {rot_count/n_samples*100:.1f}%, chi2={chi2:.2f}'
    }

    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'symmetry_analysis')
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved: {results}")

if __name__ == '__main__':
    run_experiment()
