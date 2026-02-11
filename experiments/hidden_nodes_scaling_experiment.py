"""
RES-074: Hidden Node Count and Order

Hypothesis: Optimal hidden node count exists with diminishing returns

Method:
1. Test CPPNs with varying hidden node counts (1, 2, 4, 8, 16, 32)
2. Measure mean order for each configuration
3. Test: Order scales log-linearly with hidden nodes
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.thermo_sampler_v3 import CPPN, order_multiplicative as compute_order
from scipy import stats
import json

def run_experiment(hidden_node_configs=[1, 2, 4, 8, 16, 32], samples_per_config=20, seed=42):
    """Test hidden node count effect on order."""
    np.random.seed(seed)

    order_by_nodes = {}

    print(f"Testing hidden node configurations...")
    for n_nodes in hidden_node_configs:
        orders = []
        for _ in range(samples_per_config):
            cppn = CPPN(hidden_nodes=n_nodes)
            cppn.randomize()
            img = cppn.render(64)
            order = compute_order(img)
            orders.append(order)

        order_by_nodes[n_nodes] = np.mean(orders)

    # Fit log-linear relationship
    nodes_array = np.array(hidden_node_configs, dtype=float)
    orders_array = np.array([order_by_nodes[n] for n in hidden_node_configs])

    log_nodes = np.log(nodes_array + 1)
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_nodes, orders_array)

    # Effect size: 1 node vs 32 nodes
    cohens_d = (order_by_nodes[32] - order_by_nodes[1]) / (np.std(list(order_by_nodes.values())) + 1e-10)

    status = 'validated' if r_value > 0.7 and p_value < 0.01 else 'refuted'

    results = {
        'hypothesis': 'Optimal hidden node count exists with diminishing returns',
        'effect_size': float(cohens_d),
        'p_value': float(p_value),
        'status': status,
        'summary': f'Log-linear fit R^2={r_value**2:.3f}, slope={slope:.3f}'
    }

    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'hidden_nodes_scaling')
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved: {results}")

if __name__ == '__main__':
    run_experiment()
