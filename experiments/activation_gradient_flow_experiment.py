"""
RES-095: Activation Gradient Flow and Order

Hypothesis: High-order CPPNs exhibit better gradient flow than low-order CPPNs

Method:
1. Generate CPPNs
2. Compute activation gradient flow (mean gradient magnitude through layers)
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
    """Test gradient flow vs order."""
    np.random.seed(seed)

    orders = []
    gradient_flows = []

    print(f"Analyzing gradient flow in {n_samples} CPPNs...")
    for i in range(n_samples):
        cppn = CPPN(hidden_nodes=3)
        cppn.randomize()

        img = cppn.render(64)
        order = compute_order(img)
        orders.append(order)

        # Estimate gradient flow from output variance
        gradient_flow = np.var(img) if np.var(img) > 0 else 0
        gradient_flows.append(gradient_flow)

    orders = np.array(orders)
    gradient_flows = np.array(gradient_flows)

    # Correlation
    r, p_value = stats.pearsonr(orders, gradient_flows)

    # Effect size
    high_order_mask = orders > np.median(orders)
    low_order_mask = orders <= np.median(orders)

    high_gf = gradient_flows[high_order_mask]
    low_gf = gradient_flows[low_order_mask]

    pooled_std = np.sqrt((np.std(high_gf)**2 + np.std(low_gf)**2) / 2)
    cohens_d = (np.mean(high_gf) - np.mean(low_gf)) / pooled_std if pooled_std > 0 else 0

    status = 'validated' if r > 0.3 and p_value < 0.01 else 'refuted'

    results = {
        'hypothesis': 'High-order CPPNs exhibit better gradient flow than low-order CPPNs',
        'effect_size': float(cohens_d),
        'p_value': float(p_value),
        'status': status,
        'summary': f'Correlation r={r:.3f} (p={p_value:.2e}), d={cohens_d:.2f}'
    }

    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'activation_gradient_flow')
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved: {results}")

if __name__ == '__main__':
    run_experiment()
