"""
RES-135: Zero-Bias CPPN and Order

Hypothesis: Zero-bias CPPNs produce higher order by keeping activations at steep gradients

Method:
1. Generate CPPNs with and without bias
2. Compare order and activation statistics
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.thermo_sampler_v3 import CPPN, order_multiplicative as compute_order
from scipy import stats
import json

def run_experiment(n_samples=100, seed=42):
    """Test bias removal effect on order."""
    np.random.seed(seed)

    with_bias_orders = []
    no_bias_orders = []

    print(f"Testing bias removal effect on {n_samples} CPPNs...")
    for i in range(n_samples):
        # With bias
        cppn_bias = CPPN(hidden_nodes=3)
        cppn_bias.randomize()
        img_bias = cppn_bias.render(64)
        order_bias = compute_order(img_bias)
        with_bias_orders.append(order_bias)

        # Without bias (remove last weight which acts as bias)
        cppn_no_bias = CPPN(hidden_nodes=3)
        cppn_no_bias.randomize()
        weights = cppn_no_bias.get_weights()

        # Simplified: set bias-like weights to zero
        if len(weights) > 0:
            weights[-1] = 0

        cppn_no_bias.set_weights(weights)
        img_no_bias = cppn_no_bias.render(64)
        order_no_bias = compute_order(img_no_bias)
        no_bias_orders.append(order_no_bias)

    with_bias_orders = np.array(with_bias_orders)
    no_bias_orders = np.array(no_bias_orders)

    # Compare
    t_stat, p_value = stats.ttest_ind(no_bias_orders, with_bias_orders)

    pooled_std = np.sqrt((np.std(no_bias_orders)**2 + np.std(with_bias_orders)**2) / 2)
    cohens_d = (np.mean(no_bias_orders) - np.mean(with_bias_orders)) / pooled_std if pooled_std > 0 else 0

    status = 'validated' if cohens_d > 0.1 and p_value < 0.01 else 'refuted'

    results = {
        'hypothesis': 'Zero-bias CPPNs produce higher order by keeping activations at steep gradients',
        'effect_size': float(cohens_d),
        'p_value': float(p_value),
        'status': status,
        'summary': f'With bias={np.mean(with_bias_orders):.3f}, No bias={np.mean(no_bias_orders):.3f}, d={cohens_d:.2f}'
    }

    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'bias_removal_benefit')
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved: {results}")

if __name__ == '__main__':
    run_experiment()
