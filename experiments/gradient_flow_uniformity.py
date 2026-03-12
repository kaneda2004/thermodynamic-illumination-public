"""
RES-131: Gradient Flow Uniformity in CPPNs

Hypothesis: High-order CPPN images have more uniform gradient magnitude distribution
across layers (lower coefficient of variation in layer-wise gradient norms) than
low-order images.

Building on:
- RES-124: Mean gradient correlates with order (r=0.49), but balanced flow does not
- RES-130: Saturation NEGATIVELY correlates with order (r=-0.24), high-order uses moderate activations
- RES-111: Effective receptive field (mean gradient) correlates with order (r=0.76)

Key insight: RES-124 found balanced flow (ratio of layer gradients) doesn't matter,
but we haven't tested UNIFORMITY (coefficient of variation). High-order CPPNs may have
consistent gradient magnitudes layer-to-layer, not just balanced first/last ratio.
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')
from core.thermo_sampler_v3 import CPPN, ACTIVATIONS, order_multiplicative
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def create_deep_cppn(depth=3, width=4):
    """Create a CPPN with multiple hidden layers to measure gradient flow."""
    cppn = CPPN()

    # Clear default connections
    cppn.connections.clear()

    # Create hidden layers
    hidden_nodes = []
    node_id = 5
    activations = ['tanh', 'sin', 'gauss', 'sigmoid']

    for layer in range(depth):
        layer_nodes = []
        for i in range(width):
            act = activations[(layer + i) % len(activations)]
            from core.thermo_sampler_v3 import Node
            cppn.nodes.append(Node(node_id, act, np.random.randn()))
            layer_nodes.append(node_id)
            node_id += 1
        hidden_nodes.append(layer_nodes)

    # Connect inputs to first hidden layer
    from core.thermo_sampler_v3 import Connection
    if depth > 0:
        for inp in cppn.input_ids:
            for hid in hidden_nodes[0]:
                cppn.connections.append(Connection(inp, hid, np.random.randn()))

    # Connect hidden layers
    for layer in range(depth - 1):
        for src in hidden_nodes[layer]:
            for dst in hidden_nodes[layer + 1]:
                cppn.connections.append(Connection(src, dst, np.random.randn()))

    # Connect last hidden to output
    if depth > 0:
        for hid in hidden_nodes[-1]:
            cppn.connections.append(Connection(hid, cppn.output_id, np.random.randn()))
    else:
        for inp in cppn.input_ids:
            cppn.connections.append(Connection(inp, cppn.output_id, np.random.randn()))

    return cppn, hidden_nodes


def compute_layerwise_gradients(cppn, hidden_nodes, size=32, eps=1e-5):
    """Compute gradient magnitude at each hidden layer via finite differences."""
    coords = np.linspace(-1, 1, size)
    x, y = np.meshgrid(coords, coords)
    r = np.sqrt(x**2 + y**2)
    bias = np.ones_like(x)

    # Get baseline activations for each layer
    def forward_pass(cppn_local):
        values = {0: x, 1: y, 2: r, 3: bias}
        layer_activations = []

        for layer_nodes in hidden_nodes:
            layer_vals = []
            for nid in layer_nodes:
                node = next(n for n in cppn_local.nodes if n.id == nid)
                total = np.zeros_like(x) + node.bias
                for conn in cppn_local.connections:
                    if conn.to_id == nid and conn.enabled and conn.from_id in values:
                        total += values[conn.from_id] * conn.weight
                values[nid] = ACTIVATIONS[node.activation](total)
                layer_vals.append(np.mean(np.abs(values[nid])))
            layer_activations.append(np.mean(layer_vals))

        # Output
        output_node = next(n for n in cppn_local.nodes if n.id == cppn_local.output_id)
        total = np.zeros_like(x) + output_node.bias
        for conn in cppn_local.connections:
            if conn.to_id == cppn_local.output_id and conn.enabled and conn.from_id in values:
                total += values[conn.from_id] * conn.weight
        output = ACTIVATIONS[output_node.activation](total)
        layer_activations.append(np.mean(np.abs(output)))

        return layer_activations

    # Compute gradients by perturbing weights
    baseline = forward_pass(cppn)
    layer_gradients = []

    # Group connections by target layer
    for layer_idx, layer_nodes in enumerate(hidden_nodes):
        grads_this_layer = []
        for nid in layer_nodes:
            # Perturb bias
            for node in cppn.nodes:
                if node.id == nid:
                    orig = node.bias
                    node.bias = orig + eps
                    plus = forward_pass(cppn)
                    node.bias = orig - eps
                    minus = forward_pass(cppn)
                    node.bias = orig
                    grad = (plus[layer_idx] - minus[layer_idx]) / (2 * eps)
                    grads_this_layer.append(abs(grad))

            # Perturb incoming weights
            for conn in cppn.connections:
                if conn.to_id == nid and conn.enabled:
                    orig = conn.weight
                    conn.weight = orig + eps
                    plus = forward_pass(cppn)
                    conn.weight = orig - eps
                    minus = forward_pass(cppn)
                    conn.weight = orig
                    grad = (plus[layer_idx] - minus[layer_idx]) / (2 * eps)
                    grads_this_layer.append(abs(grad))

        if grads_this_layer:
            layer_gradients.append(np.mean(grads_this_layer))

    return layer_gradients


def gradient_cv(layer_gradients):
    """Coefficient of variation of gradient magnitudes across layers."""
    if len(layer_gradients) < 2:
        return np.nan
    mean_grad = np.mean(layer_gradients)
    if mean_grad < 1e-10:
        return np.nan
    return np.std(layer_gradients) / mean_grad


def run_experiment(n_samples=300, depth=3, width=4, seed=42):
    """Run experiment: sample CPPNs, measure order and gradient flow uniformity."""
    np.random.seed(seed)

    orders = []
    gradient_cvs = []
    mean_grads = []

    print(f"Sampling {n_samples} CPPNs (depth={depth}, width={width})...")

    for i in range(n_samples):
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{n_samples}")

        cppn, hidden_nodes = create_deep_cppn(depth=depth, width=width)

        # Compute order
        img = cppn.render(32)
        order = order_multiplicative(img)

        # Compute gradient flow
        layer_grads = compute_layerwise_gradients(cppn, hidden_nodes)
        cv = gradient_cv(layer_grads)

        if not np.isnan(cv) and not np.isnan(order):
            orders.append(order)
            gradient_cvs.append(cv)
            mean_grads.append(np.mean(layer_grads))

    orders = np.array(orders)
    gradient_cvs = np.array(gradient_cvs)
    mean_grads = np.array(mean_grads)

    print(f"\nValid samples: {len(orders)}")
    print(f"Order range: [{orders.min():.3f}, {orders.max():.3f}]")
    print(f"Gradient CV range: [{gradient_cvs.min():.3f}, {gradient_cvs.max():.3f}]")

    # Hypothesis: High order -> LOW CV (more uniform gradients)
    # So we expect NEGATIVE correlation
    r_cv, p_cv = stats.pearsonr(orders, gradient_cvs)

    # Replicate RES-124 / RES-111: mean gradient vs order
    r_mean, p_mean = stats.pearsonr(orders, mean_grads)

    # Spearman for robustness
    rho_cv, p_rho_cv = stats.spearmanr(orders, gradient_cvs)

    # Effect size: split into high/low order groups
    median_order = np.median(orders)
    high_order = gradient_cvs[orders > median_order]
    low_order = gradient_cvs[orders <= median_order]

    pooled_std = np.sqrt((np.var(high_order) + np.var(low_order)) / 2)
    cohens_d = (np.mean(high_order) - np.mean(low_order)) / pooled_std if pooled_std > 0 else 0

    print(f"\n=== RESULTS ===")
    print(f"Gradient CV vs Order: r={r_cv:.3f}, p={p_cv:.2e}")
    print(f"Spearman rho: {rho_cv:.3f}, p={p_rho_cv:.2e}")
    print(f"Mean gradient vs Order: r={r_mean:.3f} (cf. RES-111: r=0.76)")
    print(f"Cohen's d (high vs low order): {cohens_d:.3f}")
    print(f"High-order mean CV: {np.mean(high_order):.3f}")
    print(f"Low-order mean CV: {np.mean(low_order):.3f}")

    # Validation criteria: p < 0.01, |d| > 0.5
    significant = p_cv < 0.01
    large_effect = abs(cohens_d) > 0.5
    correct_direction = r_cv < 0  # Hypothesis predicts negative correlation

    if significant and large_effect and correct_direction:
        status = "validated"
    elif significant and correct_direction:
        status = "inconclusive"  # Significant but small effect
    else:
        status = "refuted"

    print(f"\nSTATUS: {status.upper()}")
    print(f"  - Significant (p<0.01): {significant}")
    print(f"  - Large effect (|d|>0.5): {large_effect}")
    print(f"  - Correct direction (r<0): {correct_direction}")

    return {
        'r_cv': r_cv,
        'p_cv': p_cv,
        'rho_cv': rho_cv,
        'cohens_d': cohens_d,
        'r_mean_grad': r_mean,
        'status': status,
        'n_samples': len(orders)
    }


if __name__ == "__main__":
    results = run_experiment(n_samples=300, depth=3, width=4, seed=42)

    # Additional robustness check with different depths
    print("\n=== ROBUSTNESS: Different Depths ===")
    for d in [2, 4]:
        r = run_experiment(n_samples=100, depth=d, width=4, seed=d*10)
        print(f"Depth {d}: r={r['r_cv']:.3f}, d={r['cohens_d']:.3f}")
