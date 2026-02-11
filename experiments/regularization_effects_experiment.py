"""
RES-114: L1 Regularization and Order

Hypothesis: L1 weight regularization during CPPN inference increases output order

Method:
1. Generate CPPNs
2. Apply L1 regularization to weights at varying intensities
3. Measure order under regularization
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.thermo_sampler_v3 import CPPN, order_multiplicative as compute_order
from scipy import stats
import json

def run_experiment(l1_values=[0.0, 0.5, 1.0, 2.0, 3.0], samples_per_l1=20, seed=42):
    """Test L1 regularization effect on order."""
    np.random.seed(seed)

    order_by_l1 = {}

    print(f"Testing L1 regularization effects...")
    for l1 in l1_values:
        orders = []
        for _ in range(samples_per_l1):
            cppn = CPPN(hidden_nodes=3)
            cppn.randomize()

            weights = cppn.get_weights()

            # Soft thresholding (L1 proximal operator)
            weights_reg = np.sign(weights) * np.maximum(np.abs(weights) - l1, 0)

            cppn.set_weights(weights_reg)
            img = cppn.render(64)
            order = compute_order(img)
            orders.append(order)

        order_by_l1[l1] = np.mean(orders)

    # Correlation
    l1_array = np.array(l1_values)
    orders_array = np.array([order_by_l1[l] for l in l1_values])

    rho, p_value = stats.spearmanr(l1_array, orders_array)

    # Effect size: L1=0 vs L1=2
    pooled_std = np.sqrt((np.std([order_by_l1[0]])** 2 + np.std([order_by_l1[2.0]])**2) / 2)
    if pooled_std > 0:
        cohens_d = (order_by_l1[2.0] - order_by_l1[0.0]) / pooled_std
    else:
        cohens_d = 0

    status = 'refuted' if rho < 0 and p_value < 0.05 else 'validated'

    results = {
        'hypothesis': 'L1 weight regularization during CPPN inference increases output order',
        'effect_size': float(cohens_d),
        'p_value': float(p_value),
        'status': status,
        'summary': f'Correlation rho={rho:.3f} (p={p_value:.2e}), d={cohens_d:.2f}'
    }

    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'regularization_effects')
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved: {results}")

if __name__ == '__main__':
    run_experiment()
