"""
RES-137: Dropout During CPPN Inference

Hypothesis: Dropout during CPPN inference increases order by forcing redundant representations

Method:
1. Apply dropout at various rates
2. Measure order under dropout
3. Test: Higher dropout should increase order
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.thermo_sampler_v3 import CPPN, order_multiplicative as compute_order
from scipy import stats
import json

def run_experiment(dropout_rates=[0.0, 0.1, 0.2, 0.3, 0.4], samples_per_rate=20, seed=42):
    """Test dropout effect on order."""
    np.random.seed(seed)

    order_by_dropout = {}

    print(f"Testing dropout effects...")
    for dropout in dropout_rates:
        orders = []
        for _ in range(samples_per_rate):
            cppn = CPPN(hidden_nodes=3)
            cppn.randomize()
            weights = cppn.get_weights()

            # Apply dropout (zero out weights randomly)
            if dropout > 0:
                dropout_mask = np.random.rand(len(weights)) > dropout
                weights_drop = weights * dropout_mask
            else:
                weights_drop = weights

            cppn.set_weights(weights_drop)
            img = cppn.render(64)
            order = compute_order(img)
            orders.append(order)

        order_by_dropout[dropout] = np.mean(orders)

    # Correlation: dropout should increase order (hypothesis expects positive)
    dropout_array = np.array(dropout_rates)
    orders_array = np.array([order_by_dropout[d] for d in dropout_rates])

    r, p_value = stats.spearmanr(dropout_array, orders_array)

    # Effect size: 0% vs 30% dropout
    pooled_std = np.sqrt((np.std([order_by_dropout[0.0]])**2 + np.std([order_by_dropout[0.3]])**2) / 2)
    if pooled_std > 0:
        cohens_d = (order_by_dropout[0.3] - order_by_dropout[0.0]) / pooled_std
    else:
        cohens_d = 0

    status = 'refuted' if r < 0 else 'validated'

    results = {
        'hypothesis': 'Dropout during CPPN inference increases order by forcing redundant representations',
        'effect_size': float(cohens_d),
        'p_value': float(p_value),
        'status': status,
        'summary': f'Correlation r={r:.3f} (p={p_value:.2e}), d={cohens_d:.2f}'
    }

    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'dropout_effect')
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved: {results}")

if __name__ == '__main__':
    run_experiment()
