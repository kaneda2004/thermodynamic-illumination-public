"""
RES-135: Test if zero-bias CPPNs produce higher order.

Hypothesis: bias shifts activations away from steep gradient regions
"""

import numpy as np
from scipy import stats
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import CPPN, nested_sampling, compute_order, Node, Connection, ACTIVATIONS


def create_zero_bias_cppn(template_cppn):
    """Create CPPN copy with zero biases."""
    nodes_zero = []
    for node in template_cppn.nodes:
        if node.id not in template_cppn.input_ids:
            nodes_zero.append(Node(node.id, node.activation, 0.0))
        else:
            nodes_zero.append(Node(node.id, node.activation, 0.0))

    cppn_zero = CPPN(
        nodes=nodes_zero,
        connections=[Connection(c.from_id, c.to_id, c.weight, c.enabled) for c in template_cppn.connections],
        input_ids=template_cppn.input_ids,
        output_id=template_cppn.output_id
    )

    return cppn_zero


def compute_activation_gradients(cppn, coords_x, coords_y):
    """Compute mean activation gradients."""
    r = np.sqrt(coords_x**2 + coords_y**2)
    bias = np.ones_like(coords_x)
    values = {0: coords_x, 1: coords_y, 2: r, 3: bias}

    gradients = []

    for nid in cppn._get_eval_order():
        node = next(n for n in cppn.nodes if n.id == nid)
        total = np.zeros_like(coords_x) + node.bias

        for conn in cppn.connections:
            if conn.to_id == nid and conn.enabled and conn.from_id in values:
                total += values[conn.from_id] * conn.weight

        values[nid] = ACTIVATIONS[node.activation](total)

        # Gradient of activation (numerical)
        eps = 1e-4
        total_plus = total + eps
        total_minus = total - eps

        grad_plus = ACTIVATIONS[node.activation](total_plus)
        grad_minus = ACTIVATIONS[node.activation](total_minus)

        grad = np.mean(np.abs(grad_plus - grad_minus) / (2 * eps))
        gradients.append(grad)

    return np.mean(gradients) if gradients else 0.0


def main():
    np.random.seed(42)

    n_samples = 100
    resolution = 64

    coords = np.linspace(-1, 1, resolution)
    coords_x, coords_y = np.meshgrid(coords, coords)

    with_bias_orders = []
    zero_bias_orders = []
    with_bias_grads = []
    zero_bias_grads = []

    print("Generating CPPNs...")
    for i in range(n_samples):
        cppn_with_bias, order_with = nested_sampling(max_iterations=100, n_live=20)
        with_bias_orders.append(order_with)

        grad_with = compute_activation_gradients(cppn_with_bias, coords_x, coords_y)
        with_bias_grads.append(grad_with)

        cppn_zero = create_zero_bias_cppn(cppn_with_bias)
        order_zero = compute_order(cppn_zero.activate(coords_x, coords_y))
        zero_bias_orders.append(order_zero)

        grad_zero = compute_activation_gradients(cppn_zero, coords_x, coords_y)
        zero_bias_grads.append(grad_zero)

    with_bias_orders = np.array(with_bias_orders)
    zero_bias_orders = np.array(zero_bias_orders)
    with_bias_grads = np.array(with_bias_grads)
    zero_bias_grads = np.array(zero_bias_grads)

    # Statistical test
    t_order, p_order = stats.ttest_rel(zero_bias_orders, with_bias_orders)
    t_grad, p_grad = stats.ttest_rel(zero_bias_grads, with_bias_grads)

    # Effect size
    d_order = np.mean(zero_bias_orders - with_bias_orders) / np.std(zero_bias_orders - with_bias_orders + 1e-10)
    d_grad = np.mean(zero_bias_grads - with_bias_grads) / np.std(zero_bias_grads - with_bias_grads + 1e-10)

    # Correlation
    corr_grad, p_corr = stats.pearsonr(zero_bias_grads, zero_bias_orders)

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"With bias order: {np.mean(with_bias_orders):.4f}")
    print(f"Zero bias order: {np.mean(zero_bias_orders):.4f}")
    print(f"Order effect size (Cohen's d): {d_order:.3f}")
    print(f"p-value: {p_order:.2e}")
    print(f"\nGradient (with bias): {np.mean(with_bias_grads):.4f}")
    print(f"Gradient (zero bias): {np.mean(zero_bias_grads):.4f}")
    print(f"Gradient effect size: {d_grad:.3f}")
    print(f"Gradient-order correlation: r={corr_grad:.3f}, p={p_corr:.2e}")

    validated = d_order > 0.1 and p_order < 0.01
    status = 'validated' if validated else 'refuted'
    print(f"\nSTATUS: {status}")

    return validated, d_order, p_order


if __name__ == '__main__':
    validated, effect_size, p_value = main()
    print(f"\nFinal: d={effect_size:.3f}, p={p_value:.2e}")
