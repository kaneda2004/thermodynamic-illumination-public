"""
RES-069: Resolution Effect on Order Metrics

Hypothesis: Same CPPN produces higher order at higher resolutions

Method:
1. Generate CPPNs and render at multiple resolutions (32, 64, 128, 256)
2. Compute order at each resolution
3. Test: Order should increase with resolution
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.thermo_sampler_v3 import CPPN, order_multiplicative as compute_order
from scipy import stats
import json

def run_experiment(n_samples=50, seed=42):
    """Test resolution effect on order."""
    np.random.seed(seed)

    resolutions = [32, 64, 128, 256]
    order_by_res = {r: [] for r in resolutions}

    print(f"Sampling {n_samples} CPPNs at multiple resolutions...")
    for i in range(n_samples):
        cppn = CPPN(hidden_nodes=3)
        cppn.randomize()

        for res in resolutions:
            img = cppn.render(res)
            order = compute_order(img)
            order_by_res[res].append(order)

    # Correlation between resolution and order
    res_array = np.array(resolutions)
    mean_orders = np.array([np.mean(order_by_res[r]) for r in resolutions])

    rho, p_value = stats.spearmanr(res_array, mean_orders)

    # Effect size: compare 32px vs 256px
    orders_32 = np.array(order_by_res[32])
    orders_256 = np.array(order_by_res[256])

    pooled_std = np.sqrt((np.std(orders_32)**2 + np.std(orders_256)**2) / 2)
    cohens_d = (np.mean(orders_256) - np.mean(orders_32)) / pooled_std if pooled_std > 0 else 0

    # Status: expect positive correlation
    status = 'validated' if rho > 0.3 and p_value < 0.01 else 'refuted'

    results = {
        'hypothesis': 'Same CPPN produces higher order at higher resolutions',
        'effect_size': float(cohens_d),
        'p_value': float(p_value),
        'status': status,
        'summary': f'Correlation rho={rho:.3f} (p={p_value:.2e}), d={cohens_d:.2f}'
    }

    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'resolution_effect')
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved: {results}")

if __name__ == '__main__':
    run_experiment()
