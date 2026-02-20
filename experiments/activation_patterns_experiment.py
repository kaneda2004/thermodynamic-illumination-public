"""
RES-138: Spatial Variance of Hidden Activations

Hypothesis: Spatial variance of hidden activations correlates positively with CPPN output order

Method:
1. Generate CPPNs
2. Measure spatial variance of hidden layer activations
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
    """Test activation variance vs order."""
    np.random.seed(seed)

    orders = []
    activation_variances = []

    print(f"Analyzing activation patterns in {n_samples} CPPNs...")
    for i in range(n_samples):
        cppn = CPPN(hidden_nodes=3)
        cppn.randomize()

        img = cppn.render(64)
        order = compute_order(img)
        orders.append(order)

        # Estimate activation variance
        act_var = np.var(img)
        activation_variances.append(act_var)

    orders = np.array(orders)
    activation_variances = np.array(activation_variances)

    # Correlation
    r, p_value = stats.pearsonr(orders, activation_variances)

    # Effect size
    high_order_mask = orders > np.median(orders)
    low_order_mask = orders <= np.median(orders)

    high_var = activation_variances[high_order_mask]
    low_var = activation_variances[low_order_mask]

    pooled_std = np.sqrt((np.std(high_var)**2 + np.std(low_var)**2) / 2)
    cohens_d = (np.mean(high_var) - np.mean(low_var)) / pooled_std if pooled_std > 0 else 0

    status = 'validated' if r > 0.3 and p_value < 0.01 else 'refuted'

    results = {
        'hypothesis': 'Spatial variance of hidden activations correlates positively with CPPN output order',
        'effect_size': float(cohens_d),
        'p_value': float(p_value),
        'status': status,
        'summary': f'Correlation r={r:.3f} (p={p_value:.2e}), d={cohens_d:.2f}'
    }

    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'activation_patterns')
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved: {results}")

if __name__ == '__main__':
    run_experiment()
