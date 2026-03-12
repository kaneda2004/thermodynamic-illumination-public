"""
RES-141: Optimization Landscape Barriers

Hypothesis: Linear interpolation paths between high-order optima pass through low-order barrier regions

Method:
1. Find two high-order CPPN optima
2. Interpolate linearly between them
3. Measure order along path
4. Test: Interior should have lower order than endpoints
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.thermo_sampler_v3 import CPPN, order_multiplicative as compute_order
from scipy import stats
import json

def run_experiment(n_pairs=30, n_points=10, seed=42):
    """Test for barriers between high-order optima."""
    np.random.seed(seed)

    interior_orders = []
    endpoint_orders = []

    print(f"Testing for barriers in optimization landscape...")
    for pair_idx in range(n_pairs):
        # Generate two high-order CPPNs
        cppn1 = CPPN(hidden_nodes=3)
        cppn1.randomize()
        w1 = cppn1.get_weights()
        order1 = compute_order(cppn1.render(64))

        cppn2 = CPPN(hidden_nodes=3)
        cppn2.randomize()
        w2 = cppn2.get_weights()
        order2 = compute_order(cppn2.render(64))

        endpoint_orders.extend([order1, order2])

        # Interpolate
        for t in np.linspace(0.1, 0.9, n_points - 2):
            w_interp = (1 - t) * w1 + t * w2

            cppn_interp = CPPN(hidden_nodes=3)
            cppn_interp.set_weights(w_interp)
            order_interp = compute_order(cppn_interp.render(64))

            interior_orders.append(order_interp)

    interior_orders = np.array(interior_orders)
    endpoint_orders = np.array(endpoint_orders)

    # Compare
    t_stat, p_value = stats.ttest_ind(interior_orders, endpoint_orders)

    pooled_std = np.sqrt((np.std(interior_orders)**2 + np.std(endpoint_orders)**2) / 2)
    cohens_d = (np.mean(endpoint_orders) - np.mean(interior_orders)) / pooled_std if pooled_std > 0 else 0

    status = 'validated' if cohens_d > 0.5 and p_value < 0.01 else 'refuted'

    results = {
        'hypothesis': 'Linear interpolation paths between high-order optima pass through low-order barrier regions',
        'effect_size': float(cohens_d),
        'p_value': float(p_value),
        'status': status,
        'summary': f'Endpoints={np.mean(endpoint_orders):.3f}, Interior={np.mean(interior_orders):.3f}, d={cohens_d:.2f}'
    }

    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'optimization_landscape')
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved: {results}")

if __name__ == '__main__':
    run_experiment()
