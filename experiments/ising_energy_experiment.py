"""
RES-105: Ising Model Energy of CPPN Images

Hypothesis: CPPN images have lower Ising model energy than random images

Method:
1. Generate CPPN and random images
2. Compute Ising energy (neighbor interaction cost)
3. Compare distributions
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.thermo_sampler_v3 import CPPN, order_multiplicative as compute_order
from scipy import stats
import json

def compute_ising_energy(img, J=1.0):
    """Compute Ising model energy (coupling between neighboring pixels)."""
    binary = (img > 0.5).astype(float)

    # Horizontal neighbors
    h_diff = np.sum(np.abs(np.diff(binary, axis=1)))

    # Vertical neighbors
    v_diff = np.sum(np.abs(np.diff(binary, axis=0)))

    # Total energy normalized
    energy = (h_diff + v_diff) / (binary.size * 2)
    return energy

def run_experiment(n_samples=100, seed=42):
    """Test Ising energy in CPPN vs random images."""
    np.random.seed(seed)

    cppn_energies = []
    random_energies = []

    print(f"Computing Ising energy for {n_samples} image pairs...")
    for i in range(n_samples):
        # CPPN image
        cppn = CPPN(hidden_nodes=3)
        cppn.randomize()
        img_cppn = cppn.render(64)
        energy_cppn = compute_ising_energy(img_cppn)
        cppn_energies.append(energy_cppn)

        # Random image
        img_rand = np.random.rand(64, 64)
        energy_rand = compute_ising_energy(img_rand)
        random_energies.append(energy_rand)

    cppn_energies = np.array(cppn_energies)
    random_energies = np.array(random_energies)

    # Compare
    t_stat, p_value = stats.ttest_ind(cppn_energies, random_energies)

    pooled_std = np.sqrt((np.std(cppn_energies)**2 + np.std(random_energies)**2) / 2)
    cohens_d = (np.mean(cppn_energies) - np.mean(random_energies)) / pooled_std if pooled_std > 0 else 0

    status = 'validated' if cohens_d < -2.0 and p_value < 0.001 else 'refuted'

    results = {
        'hypothesis': 'CPPN images have lower Ising model energy than random images',
        'effect_size': float(cohens_d),
        'p_value': float(p_value),
        'status': status,
        'summary': f'CPPN energy={np.mean(cppn_energies):.4f}, Random energy={np.mean(random_energies):.4f}, d={cohens_d:.2f}'
    }

    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'ising_energy')
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved: {results}")

if __name__ == '__main__':
    run_experiment()
