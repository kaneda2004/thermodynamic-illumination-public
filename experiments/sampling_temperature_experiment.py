"""
RES-067: Temperature Schedule Steepness and Order Ceiling

Hypothesis: Temperature schedule steepness determines order ceiling in nested sampling

Method:
1. Run nested sampling with varying schedule steepness (n_live parameter)
2. Measure max order reached with same sample budget
3. Test: Steeper schedules (larger n_live) should reach higher order
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.thermo_sampler_v3 import CPPN, nested_sampling, order_multiplicative as compute_order
from scipy import stats
import json

def run_experiment(n_live_vals=[20, 50, 100, 200, 500], samples_per_val=10, seed=42):
    """Test temperature schedule steepness via n_live parameter."""
    np.random.seed(seed)

    results_by_nlive = {}

    for n_live in n_live_vals:
        max_orders = []
        sample_budget = 2000

        print(f"Testing n_live={n_live}...")
        for _ in range(samples_per_val):
            cppn = CPPN(hidden_nodes=3)
            cppn.randomize()

            ns_results = nested_sampling(
                cppn,
                n_iterations=max(5, sample_budget // n_live),
                n_live=n_live,
                seed=None
            )

            if 'samples' in ns_results:
                orders = [compute_order(cppn.render(64, weights=w)) for w in ns_results['samples']]
                max_orders.append(np.max(orders))

        results_by_nlive[n_live] = max_orders

    # Compare fastest schedule (small n_live) vs slowest (large n_live)
    fast_schedule = np.array(results_by_nlive[20])
    slow_schedule = np.array(results_by_nlive[500])

    pooled_std = np.sqrt((np.std(fast_schedule)**2 + np.std(slow_schedule)**2) / 2)
    cohens_d = (np.mean(fast_schedule) - np.mean(slow_schedule)) / pooled_std if pooled_std > 0 else 0
    p_value = stats.ttest_ind(fast_schedule, slow_schedule)[1]

    # Status: faster schedule should reach higher order
    status = 'validated' if cohens_d > 0.5 and p_value < 0.01 else 'refuted'

    results = {
        'hypothesis': 'Temperature schedule steepness determines order ceiling',
        'effect_size': float(cohens_d),
        'p_value': float(p_value),
        'status': status,
        'summary': f'Fast (n_live=20) order={np.mean(fast_schedule):.3f} vs Slow (n_live=500) order={np.mean(slow_schedule):.3f}, d={cohens_d:.2f}'
    }

    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'sampling_temperature')
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved: {results}")

if __name__ == '__main__':
    run_experiment()
