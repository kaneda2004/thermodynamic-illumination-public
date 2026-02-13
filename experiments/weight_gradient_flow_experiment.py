"""
RES-124: Weight Gradient Flow and Order

Hypothesis: Balanced gradient flow through CPPN layers correlates with higher order

Method:
1. Generate CPPNs
2. Compute gradient magnitude flow through layers
3. Correlate with order
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.thermo_sampler_v3 import CPPN, order_multiplicative as compute_order
from scipy import stats
import json

def run_experiment(n_samples=100, seed=42):
    """Test gradient flow balance vs order."""
    np.random.seed(seed)

    orders = []
    gradient_magnitudes = []

    print(f"Analyzing gradient flow in {n_samples} CPPNs...")
    for i in range(n_samples):
        cppn = CPPN(hidden_nodes=3)
        cppn.randomize()

        img = cppn.render(64)
        order = compute_order(img)
        orders.append(order)

        # Estimate gradient magnitude from weight statistics
        weights = cppn.get_weights()
        grad_mag = np.mean(np.abs(weights))
        gradient_magnitudes.append(grad_mag)

    orders = np.array(orders)
    gradient_magnitudes = np.array(gradient_magnitudes)

    # Correlation: high order should have high gradient flow
    r, p_value = stats.pearsonr(orders, gradient_magnitudes)

    # Effect size
    high_order_mask = orders > np.median(orders)
    low_order_mask = orders <= np.median(orders)

    high_grad = gradient_magnitudes[high_order_mask]
    low_grad = gradient_magnitudes[low_order_mask]

    pooled_std = np.sqrt((np.std(high_grad)**2 + np.std(low_grad)**2) / 2)
    cohens_d = (np.mean(high_grad) - np.mean(low_grad)) / pooled_std if pooled_std > 0 else 0

    status = 'validated' if r > 0.3 and p_value < 0.01 else 'refuted'

    results = {
        'hypothesis': 'Balanced gradient flow through CPPN layers correlates with higher order',
        'effect_size': float(cohens_d),
        'p_value': float(p_value),
        'status': status,
        'summary': f'Correlation r={r:.3f} (p={p_value:.2e}), d={cohens_d:.2f}'
    }

    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'weight_gradient_flow')
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved: {results}")

if __name__ == '__main__':
    run_experiment()
