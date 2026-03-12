"""
RES-081: Weight Sign Patterns and CPPN Order

Hypothesis: Weight sign patterns affect CPPN order; random flipping preserves it

Method:
1. Generate CPPN, measure baseline order
2. Flip all weights to same sign (all positive/negative)
3. Randomly flip individual weights
4. Measure order in each case
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.thermo_sampler_v3 import CPPN, order_multiplicative as compute_order
from scipy import stats
import json

def run_experiment(n_samples=50, seed=42):
    """Test weight sign pattern effects."""
    np.random.seed(seed)

    baseline_orders = []
    uniform_positive_orders = []
    uniform_negative_orders = []
    random_flip_orders = []
    alternating_orders = []

    print(f"Testing weight sign patterns on {n_samples} CPPNs...")
    for i in range(n_samples):
        cppn = CPPN(hidden_nodes=3)
        cppn.randomize()
        weights = cppn.get_weights()

        # Baseline
        img = cppn.render(64)
        baseline_orders.append(compute_order(img))

        # All positive
        cppn_pos = CPPN(hidden_nodes=3)
        cppn_pos.set_weights(np.abs(weights))
        img_pos = cppn_pos.render(64)
        uniform_positive_orders.append(compute_order(img_pos))

        # All negative
        cppn_neg = CPPN(hidden_nodes=3)
        cppn_neg.set_weights(-np.abs(weights))
        img_neg = cppn_neg.render(64)
        uniform_negative_orders.append(compute_order(img_neg))

        # Random flips (50% of weights)
        cppn_rand = CPPN(hidden_nodes=3)
        w_rand = weights.copy()
        flip_mask = np.random.rand(len(w_rand)) < 0.5
        w_rand[flip_mask] *= -1
        cppn_rand.set_weights(w_rand)
        img_rand = cppn_rand.render(64)
        random_flip_orders.append(compute_order(img_rand))

        # Alternating signs
        cppn_alt = CPPN(hidden_nodes=3)
        w_alt = np.abs(weights)
        w_alt[::2] *= -1
        cppn_alt.set_weights(w_alt)
        img_alt = cppn_alt.render(64)
        alternating_orders.append(compute_order(img_alt))

    baseline_orders = np.array(baseline_orders)
    uniform_positive_orders = np.array(uniform_positive_orders)
    uniform_negative_orders = np.array(uniform_negative_orders)
    random_flip_orders = np.array(random_flip_orders)
    alternating_orders = np.array(alternating_orders)

    # Compare uniform signs vs baseline
    pooled_std = np.sqrt((np.std(baseline_orders)**2 + np.std(uniform_positive_orders)**2) / 2)
    cohens_d = (np.mean(baseline_orders) - np.mean(uniform_positive_orders)) / pooled_std if pooled_std > 0 else 0
    p_uniform = stats.ttest_ind(baseline_orders, uniform_positive_orders)[1]

    # Random flips should preserve order
    p_random = stats.ttest_ind(baseline_orders, random_flip_orders)[1]

    status = 'validated' if abs(cohens_d) > 0.5 and p_uniform < 0.01 else 'refuted'

    results = {
        'hypothesis': 'Weight sign patterns affect CPPN order; random flipping preserves it',
        'effect_size': float(cohens_d),
        'p_value': float(p_uniform),
        'status': status,
        'summary': f'Baseline={np.mean(baseline_orders):.3f}, Uniform={np.mean(uniform_positive_orders):.3f}, Random={np.mean(random_flip_orders):.3f}'
    }

    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'weight_signs')
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved: {results}")

if __name__ == '__main__':
    run_experiment()
