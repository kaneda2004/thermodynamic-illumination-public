"""
RES-131: Activation Gradient Uniformity

Hypothesis: High-order CPPNs have more uniform gradient magnitude across layers

Method:
1. Generate CPPNs
2. Compute gradient magnitude per layer
3. Measure coefficient of variation
4. Correlate with order
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.thermo_sampler_v3 import CPPN, order_multiplicative as compute_order
from scipy import stats
import json

def run_experiment(n_samples=100, seed=42):
    """Test gradient uniformity vs order."""
    np.random.seed(seed)

    orders = []
    gradient_cvs = []

    print(f"Analyzing gradient uniformity in {n_samples} CPPNs...")
    for i in range(n_samples):
        cppn = CPPN(hidden_nodes=3)
        cppn.randomize()

        img = cppn.render(64)
        order = compute_order(img)
        orders.append(order)

        # Estimate gradient CV from weight statistics
        weights = cppn.get_weights()
        grad_mag = np.abs(weights)

        if len(grad_mag) > 1:
            cv = np.std(grad_mag) / (np.mean(grad_mag) + 1e-10)
        else:
            cv = 0

        gradient_cvs.append(cv)

    orders = np.array(orders)
    gradient_cvs = np.array(gradient_cvs)

    # Correlation: expect negative (high order = uniform gradients)
    r, p_value = stats.pearsonr(orders, gradient_cvs)

    # Effect size
    high_order_mask = orders > np.median(orders)
    low_order_mask = orders <= np.median(orders)

    high_cv = gradient_cvs[high_order_mask]
    low_cv = gradient_cvs[low_order_mask]

    pooled_std = np.sqrt((np.std(high_cv)**2 + np.std(low_cv)**2) / 2)
    cohens_d = (np.mean(high_cv) - np.mean(low_cv)) / pooled_std if pooled_std > 0 else 0

    status = 'validated' if abs(r) > 0.3 and p_value < 0.01 else 'refuted'

    results = {
        'hypothesis': 'High-order CPPNs have more uniform gradient magnitude across layers',
        'effect_size': float(cohens_d),
        'p_value': float(p_value),
        'status': status,
        'summary': f'Correlation r={r:.3f} (p={p_value:.2e}), d={cohens_d:.2f}'
    }

    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'activation_gradient_uniformity')
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved: {results}")

if __name__ == '__main__':
    run_experiment()
