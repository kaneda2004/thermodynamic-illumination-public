"""
RES-129: Weight Space Curvature and Order

Hypothesis: High-order regions have higher curvature indicating sharper peaks

Method:
1. Generate CPPNs
2. Compute Hessian eigenvalues (curvature)
3. Correlate with order
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.thermo_sampler_v3 import CPPN, order_multiplicative as compute_order
from scipy import stats
import json

def estimate_curvature(cppn, eps=0.01):
    """Estimate Hessian norm (curvature) of order surface."""
    weights = cppn.get_weights()
    base_order = compute_order(cppn.render(32))

    hessian_norm = 0
    for i in range(min(3, len(weights))):
        w_plus = weights.copy()
        w_plus[i] += eps
        cppn_p = CPPN(hidden_nodes=3)
        cppn_p.set_weights(w_plus)
        order_plus = compute_order(cppn_p.render(32))

        w_minus = weights.copy()
        w_minus[i] -= eps
        cppn_m = CPPN(hidden_nodes=3)
        cppn_m.set_weights(w_minus)
        order_minus = compute_order(cppn_m.render(32))

        second_deriv = (order_plus - 2*base_order + order_minus) / (eps**2)
        hessian_norm += second_deriv**2

    return np.sqrt(hessian_norm)

def run_experiment(n_samples=80, seed=42):
    """Test curvature vs order."""
    np.random.seed(seed)

    orders = []
    curvatures = []

    print(f"Computing curvature at {n_samples} points...")
    for i in range(n_samples):
        cppn = CPPN(hidden_nodes=3)
        cppn.randomize()

        img = cppn.render(64)
        order = compute_order(img)
        orders.append(order)

        curv = estimate_curvature(cppn)
        curvatures.append(curv)

    orders = np.array(orders)
    curvatures = np.array(curvatures)

    # Correlation
    r, p_value = stats.pearsonr(orders, curvatures)

    # Effect size
    high_order_mask = orders > np.median(orders)
    low_order_mask = orders <= np.median(orders)

    high_curv = curvatures[high_order_mask]
    low_curv = curvatures[low_order_mask]

    pooled_std = np.sqrt((np.std(high_curv)**2 + np.std(low_curv)**2) / 2)
    cohens_d = (np.mean(high_curv) - np.mean(low_curv)) / pooled_std if pooled_std > 0 else 0

    status = 'validated' if r > 0.6 and p_value < 0.01 else 'refuted'

    results = {
        'hypothesis': 'High-order regions have higher curvature indicating sharper peaks',
        'effect_size': float(cohens_d),
        'p_value': float(p_value),
        'status': status,
        'summary': f'Correlation r={r:.3f} (p={p_value:.2e}), d={cohens_d:.2f}'
    }

    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'weight_space_curvature')
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved: {results}")

if __name__ == '__main__':
    run_experiment()
