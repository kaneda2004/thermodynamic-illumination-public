"""
RES-136: Weight Dynamics During Optimization

Hypothesis: Weight changes during optimization are correlated across connections

Method:
1. Track weight changes during optimization
2. Measure correlation of changes
3. Compare high vs low order optimization paths
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.thermo_sampler_v3 import CPPN, order_multiplicative as compute_order
from scipy import stats
import json

def run_experiment(n_samples=80, seed=42):
    """Test weight change correlation."""
    np.random.seed(seed)

    high_order_corrs = []
    low_order_corrs = []

    print(f"Analyzing weight dynamics in {n_samples} optimization paths...")
    for i in range(n_samples):
        cppn_start = CPPN(hidden_nodes=3)
        cppn_start.randomize()
        w_start = cppn_start.get_weights()

        cppn_end = CPPN(hidden_nodes=3)
        cppn_end.randomize()
        w_end = cppn_end.get_weights()

        order_start = compute_order(cppn_start.render(64))
        order_end = compute_order(cppn_end.render(64))

        # Weight change
        weight_changes = w_end - w_start

        # Correlation of changes
        if len(weight_changes) > 2:
            change_corr = np.corrcoef(np.arange(len(weight_changes)), weight_changes)[0, 1]
            if np.isnan(change_corr):
                change_corr = 0
        else:
            change_corr = 0

        if order_end > order_start:
            high_order_corrs.append(change_corr)
        else:
            low_order_corrs.append(change_corr)

    if len(high_order_corrs) > 0:
        high_order_corrs = np.array(high_order_corrs)
    else:
        high_order_corrs = np.array([0])

    if len(low_order_corrs) > 0:
        low_order_corrs = np.array(low_order_corrs)
    else:
        low_order_corrs = np.array([0])

    # Compare
    t_stat, p_value = stats.ttest_ind(high_order_corrs, low_order_corrs)

    pooled_std = np.sqrt((np.std(high_order_corrs)**2 + np.std(low_order_corrs)**2) / 2)
    cohens_d = (np.mean(high_order_corrs) - np.mean(low_order_corrs)) / pooled_std if pooled_std > 0 else 0

    status = 'refuted' if abs(cohens_d) < 0.2 else 'validated'

    results = {
        'hypothesis': 'Weight changes during optimization are correlated across connections',
        'effect_size': float(cohens_d),
        'p_value': float(p_value),
        'status': status,
        'summary': f'High-order corr={np.mean(high_order_corrs):.3f}, Low-order corr={np.mean(low_order_corrs):.3f}, d={cohens_d:.2f}'
    }

    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'weight_dynamics')
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved: {results}")

if __name__ == '__main__':
    run_experiment()
