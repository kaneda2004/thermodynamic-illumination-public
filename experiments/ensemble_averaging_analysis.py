"""
RES-090: Weight-space vs Output-space Averaging

Hypothesis: Weight-space averaging preserves order better than output-space averaging

Method:
1. Generate multiple CPPNs
2. Average in weight space vs output space
3. Compare order of averaged results
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.thermo_sampler_v3 import CPPN, order_multiplicative as compute_order
from scipy import stats
import json

def run_experiment(n_trials=50, ensemble_size=5, seed=42):
    """Test weight-space vs output-space averaging."""
    np.random.seed(seed)

    weight_space_orders = []
    output_space_orders = []

    print(f"Testing ensemble averaging ({n_trials} trials)...")
    for trial in range(n_trials):
        # Generate ensemble
        cppns = [CPPN(hidden_nodes=3) for _ in range(ensemble_size)]
        for cppn in cppns:
            cppn.randomize()

        # Weight-space averaging
        avg_weights = np.mean([cppn.get_weights() for cppn in cppns], axis=0)
        cppn_wavg = CPPN(hidden_nodes=3)
        cppn_wavg.set_weights(avg_weights)
        img_wavg = cppn_wavg.render(64)
        order_wavg = compute_order(img_wavg)
        weight_space_orders.append(order_wavg)

        # Output-space averaging
        imgs = [cppn.render(64) for cppn in cppns]
        img_oavg = np.mean(imgs, axis=0)
        order_oavg = compute_order(img_oavg)
        output_space_orders.append(order_oavg)

    weight_space_orders = np.array(weight_space_orders)
    output_space_orders = np.array(output_space_orders)

    # Compare
    t_stat, p_value = stats.ttest_ind(weight_space_orders, output_space_orders)

    pooled_std = np.sqrt((np.std(weight_space_orders)**2 + np.std(output_space_orders)**2) / 2)
    cohens_d = (np.mean(weight_space_orders) - np.mean(output_space_orders)) / pooled_std if pooled_std > 0 else 0

    status = 'refuted' if abs(cohens_d) < 0.5 or p_value > 0.01 else 'validated'

    results = {
        'hypothesis': 'Weight-space averaging preserves order better than output-space averaging',
        'effect_size': float(cohens_d),
        'p_value': float(p_value),
        'status': status,
        'summary': f'Weight-space={np.mean(weight_space_orders):.3f}, Output-space={np.mean(output_space_orders):.3f}, d={cohens_d:.2f}'
    }

    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'ensemble_averaging')
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved: {results}")

if __name__ == '__main__':
    run_experiment()
