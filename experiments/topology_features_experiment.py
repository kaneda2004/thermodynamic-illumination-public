"""
RES-113: Topology Features and Order

Hypothesis: High-order CPPN images have simpler topology than low-order images

Method:
1. Generate CPPNs
2. Compute connected components and topology metrics
3. Test: High-order should have simpler topology
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.thermo_sampler_v3 import CPPN, order_multiplicative as compute_order
from scipy import stats
import json
from scipy.ndimage import label

def count_components(img):
    """Count connected components in binary image."""
    binary = (img > 0.5).astype(int)
    labeled, num = label(binary)
    return num

def run_experiment(n_samples=80, seed=42):
    """Test topology complexity vs order."""
    np.random.seed(seed)

    orders = []
    components = []

    print(f"Analyzing topology in {n_samples} CPPNs...")
    for i in range(n_samples):
        cppn = CPPN(hidden_nodes=3)
        cppn.randomize()

        img = cppn.render(64)
        order = compute_order(img)
        orders.append(order)

        comp = count_components(img)
        components.append(comp)

    orders = np.array(orders)
    components = np.array(components)

    # Correlation
    r, p_value = stats.spearmanr(orders, components)

    # Effect size
    high_order_mask = orders > np.median(orders)
    low_order_mask = orders <= np.median(orders)

    high_comp = components[high_order_mask]
    low_comp = components[low_order_mask]

    pooled_std = np.sqrt((np.std(high_comp)**2 + np.std(low_comp)**2) / 2)
    cohens_d = (np.mean(high_comp) - np.mean(low_comp)) / pooled_std if pooled_std > 0 else 0

    # Hypothesis expects simpler topology (fewer components) at high order
    status = 'validated' if cohens_d < -0.5 and p_value < 0.01 else 'refuted'

    results = {
        'hypothesis': 'High-order CPPN images have simpler topology than low-order images',
        'effect_size': float(cohens_d),
        'p_value': float(p_value),
        'status': status,
        'summary': f'Correlation r={r:.3f} (p={p_value:.2e}), d={cohens_d:.2f}'
    }

    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'topology_features')
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved: {results}")

if __name__ == '__main__':
    run_experiment()
