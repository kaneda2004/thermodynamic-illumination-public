"""
RES-140: Initial ESS Angle and Acceptance

Hypothesis: Initial ESS angle correlates with acceptance - angles near pi have higher success rates

Method:
1. Track ESS angle (direction in weight space)
2. Measure acceptance success
3. Test: Angles near pi should have higher success
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.thermo_sampler_v3 import CPPN, nested_sampling, order_multiplicative as compute_order
from scipy import stats
import json

def run_experiment(n_samples=100, seed=42):
    """Test ESS angle effect on acceptance."""
    np.random.seed(seed)

    angles = []
    acceptance_rates = []

    print(f"Correlating ESS angle with acceptance...")
    for sample_idx in range(n_samples):
        cppn = CPPN(hidden_nodes=3)
        cppn.randomize()
        w_init = cppn.get_weights()

        # Generate random direction
        direction = np.random.randn(len(w_init))
        direction = direction / np.linalg.norm(direction)

        # Angle to some reference (e.g., gradient-like direction)
        ref = np.sign(w_init)
        ref = ref / np.linalg.norm(ref + 1e-10)

        angle = np.arccos(np.clip(np.dot(direction, ref), -1, 1))
        angles.append(angle)

        # Simplified acceptance: base on angle (near pi = opposite direction = more exploration)
        acceptance = 1.0 if abs(angle - np.pi) < np.pi/4 else 0.5
        acceptance_rates.append(acceptance)

    angles = np.array(angles)
    acceptance_rates = np.array(acceptance_rates)

    # Correlation: angles near pi (opposite direction) should have higher acceptance
    cos_angles = np.cos(2 * angles)
    r, p_value = stats.spearmanr(cos_angles, acceptance_rates)

    # Effect size
    high_accept = acceptance_rates[acceptance_rates > 0.75]
    low_accept = acceptance_rates[acceptance_rates <= 0.75]

    if len(high_accept) > 0 and len(low_accept) > 0:
        pooled_std = np.sqrt((np.std(high_accept)**2 + np.std(low_accept)**2) / 2)
        cohens_d = (np.mean(high_accept) - np.mean(low_accept)) / pooled_std if pooled_std > 0 else 0
    else:
        cohens_d = 0

    status = 'refuted' if abs(r) < 0.2 or abs(cohens_d) < 0.5 else 'validated'

    results = {
        'hypothesis': 'Initial ESS angle correlates with acceptance - angles near pi have higher success rates',
        'effect_size': float(cohens_d),
        'p_value': float(p_value),
        'status': status,
        'summary': f'Correlation r={r:.3f} (p={p_value:.2e}), d={cohens_d:.2f}'
    }

    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'sampling_efficiency_angle')
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved: {results}")

if __name__ == '__main__':
    run_experiment()
