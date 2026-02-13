"""
RES-138: Test if spatial variance of hidden activations correlates with CPPN output order.

Hypothesis: high-order CPPNs have more diverse activations across the image
"""

import numpy as np
from scipy import stats
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import CPPN, nested_sampling, compute_order, ACTIVATIONS


def compute_activation_variance(cppn, coords_x, coords_y):
    """Compute spatial variance of hidden activations."""
    r = np.sqrt(coords_x**2 + coords_y**2)
    bias = np.ones_like(coords_x)
    values = {0: coords_x, 1: coords_y, 2: r, 3: bias}

    variances = []
    eval_order = cppn._get_eval_order()

    for nid in eval_order[:-1]:  # Exclude output node
        node = next(n for n in cppn.nodes if n.id == nid)
        total = np.zeros_like(coords_x) + node.bias
        for conn in cppn.connections:
            if conn.to_id == nid and conn.enabled and conn.from_id in values:
                total += values[conn.from_id] * conn.weight

        values[nid] = ACTIVATIONS[node.activation](total)
        variances.append(np.var(values[nid]))

    return np.mean(variances) if variances else 0.0


def main():
    np.random.seed(42)

    n_samples = 100
    resolution = 64

    coords = np.linspace(-1, 1, resolution)
    coords_x, coords_y = np.meshgrid(coords, coords)

    # Generate high-order CPPNs
    print("Generating high-order CPPNs...")
    high_cppns = []
    high_orders = []
    high_variances = []

    for i in range(n_samples):
        cppn, order = nested_sampling(max_iterations=100, n_live=20)
        high_cppns.append(cppn)
        high_orders.append(order)
        var = compute_activation_variance(cppn, coords_x, coords_y)
        high_variances.append(var)

    # Generate low-order CPPNs
    print("Generating low-order CPPNs...")
    low_cppns = []
    low_orders = []
    low_variances = []

    for i in range(n_samples):
        cppn = CPPN()
        order = compute_order(cppn.activate(coords_x, coords_y))
        low_cppns.append(cppn)
        low_orders.append(order)
        var = compute_activation_variance(cppn, coords_x, coords_y)
        low_variances.append(var)

    high_variances = np.array(high_variances)
    low_variances = np.array(low_variances)

    # Statistical analysis
    t_stat, p_ttest = stats.ttest_ind(high_variances, low_variances)

    pooled_std = np.sqrt((np.std(high_variances)**2 + np.std(low_variances)**2) / 2)
    effect_size = (np.mean(high_variances) - np.mean(low_variances)) / pooled_std

    # Correlation with order
    all_orders = np.concatenate([high_orders, low_orders])
    all_variances = np.concatenate([high_variances, low_variances])
    corr, p_corr = stats.pearsonr(all_orders, all_variances)

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"High-order activation variance: {np.mean(high_variances):.4f}")
    print(f"Low-order activation variance: {np.mean(low_variances):.4f}")
    print(f"Effect size (Cohen's d): {effect_size:.2f}")
    print(f"p-value: {p_ttest:.2e}")
    print(f"Correlation with order: r={corr:.3f}, p={p_corr:.2e}")

    validated = effect_size > 0.5 and p_corr < 0.01
    status = 'validated' if validated else 'refuted'
    print(f"\nSTATUS: {status}")

    return validated, effect_size, p_ttest


if __name__ == '__main__':
    validated, effect_size, p_value = main()
    print(f"\nFinal: d={effect_size:.2f}, p={p_value:.2e}")
