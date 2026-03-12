#!/usr/bin/env python3
"""
RES-196: Test whether layer-wise activation sparsity gradient predicts CPPN order.

Hypothesis: High-order CPPNs show steeper decrease in activation sparsity through layers
(i.e., activations become less sparse/more distributed as we progress through layers).

Background:
- RES-138: Spatial variance correlates with order (r=0.43)
- RES-130: Total saturation NEGATIVELY correlates with order
- RES-048: Activation variance correlates POSITIVELY (r=0.52)
- RES-158, RES-163: Velocity/curvature don't predict order
- RES-175: Spatial entropy slope shows NO correlation

This tests a NEW angle: sparsity patterns (fraction near-zero) through layers.
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import (
    CPPN, Node, Connection, ACTIVATIONS,
    compute_compressibility, compute_edge_density, compute_spectral_coherence,
    order_multiplicative
)
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def compute_order(img: np.ndarray) -> float:
    """Combined order metric."""
    return order_multiplicative(img)


def create_cppn_with_hidden(n_hidden=4, depth=2, seed=None):
    """Create CPPN with hidden layers."""
    if seed is not None:
        np.random.seed(seed)

    nodes = [
        Node(0, 'identity', 0.0), Node(1, 'identity', 0.0),
        Node(2, 'identity', 0.0), Node(3, 'identity', 0.0),
    ]
    connections = []

    node_id = 4
    prev_layer = [0, 1, 2, 3]  # input ids
    layer_node_ids = []  # Track node IDs by layer

    for layer in range(depth):
        current_layer = []
        for _ in range(n_hidden):
            act = np.random.choice(['tanh', 'sin', 'sigmoid', 'gauss'])
            bias = np.random.randn()
            nodes.append(Node(node_id, act, bias))
            for prev_id in prev_layer:
                connections.append(Connection(prev_id, node_id, np.random.randn()))
            current_layer.append(node_id)
            node_id += 1
        prev_layer = current_layer
        layer_node_ids.append(current_layer)

    # Output node
    nodes.append(Node(node_id, 'sigmoid', np.random.randn()))
    for prev_id in prev_layer:
        connections.append(Connection(prev_id, node_id, np.random.randn()))

    cppn = CPPN(nodes=nodes, connections=connections, input_ids=[0,1,2,3], output_id=node_id)
    cppn._layer_node_ids = layer_node_ids  # Store for analysis
    return cppn


def compute_hidden_activations(cppn, size=32):
    """Get activations at each hidden node for all pixels."""
    coords = np.linspace(-1, 1, size)
    x, y = np.meshgrid(coords, coords)
    r = np.sqrt(x**2 + y**2)
    bias = np.ones_like(x)

    values = {0: x, 1: y, 2: r, 3: bias}
    hidden_activations = {}

    eval_order = cppn._get_eval_order()

    for nid in eval_order:
        node = next(n for n in cppn.nodes if n.id == nid)
        total = np.zeros_like(x) + node.bias
        for conn in cppn.connections:
            if conn.to_id == nid and conn.enabled and conn.from_id in values:
                total += values[conn.from_id] * conn.weight
        values[nid] = ACTIVATIONS[node.activation](total)

        if nid != cppn.output_id and nid not in cppn.input_ids:
            hidden_activations[nid] = values[nid]

    return hidden_activations, values[cppn.output_id]


def compute_sparsity(activations, threshold=0.1):
    """Compute fraction of activations near zero."""
    return np.mean(np.abs(activations) < threshold)


def compute_layer_sparsity(cppn, hidden_activations):
    """Compute sparsity for each hidden layer."""
    layer_sparsities = []

    if hasattr(cppn, '_layer_node_ids'):
        for layer_ids in cppn._layer_node_ids:
            layer_acts = [hidden_activations[nid] for nid in layer_ids if nid in hidden_activations]
            if layer_acts:
                combined = np.stack(layer_acts, axis=-1)
                sparsity = compute_sparsity(combined)
                layer_sparsities.append(sparsity)

    return layer_sparsities


def main():
    np.random.seed(42)

    n_samples = 500
    img_size = 32
    n_hidden = 4
    depth = 3  # 3 hidden layers for better gradient analysis

    print("RES-196: Layer-wise activation sparsity gradient analysis")
    print("=" * 60)
    print(f"N samples: {n_samples}")
    print(f"Hidden layers: {depth}, nodes per layer: {n_hidden}")
    print()

    # Generate samples
    orders = []
    all_layer_sparsities = []
    sparsity_gradients = []
    input_sparsities = []
    output_sparsities = []

    for i in range(n_samples):
        cppn = create_cppn_with_hidden(n_hidden=n_hidden, depth=depth, seed=i*1000)
        img = cppn.render(img_size)
        order = compute_order(img)
        orders.append(order)

        hidden_acts, output_act = compute_hidden_activations(cppn, img_size)
        layer_sparsities = compute_layer_sparsity(cppn, hidden_acts)
        all_layer_sparsities.append(layer_sparsities)

        # Compute input sparsity (x, y, r coordinates)
        coords = np.linspace(-1, 1, img_size)
        x, y = np.meshgrid(coords, coords)
        r = np.sqrt(x**2 + y**2)
        input_sparsity = compute_sparsity(np.stack([x, y, r], axis=-1))
        input_sparsities.append(input_sparsity)

        # Output sparsity
        output_sparsity = compute_sparsity(output_act)
        output_sparsities.append(output_sparsity)

        # Sparsity gradient (slope from input to output)
        if len(layer_sparsities) > 1:
            # Use linear regression slope across layers
            x_layers = np.arange(len(layer_sparsities) + 2)  # input + hidden + output
            y_sparsity = [input_sparsity] + layer_sparsities + [output_sparsity]
            slope, _, _, _, _ = stats.linregress(x_layers, y_sparsity)
            sparsity_gradients.append(slope)
        else:
            sparsity_gradients.append(0)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{n_samples}")

    orders = np.array(orders)
    sparsity_gradients = np.array(sparsity_gradients)
    input_sparsities = np.array(input_sparsities)
    output_sparsities = np.array(output_sparsities)

    # Layer-wise sparsity arrays
    n_layers = len(all_layer_sparsities[0])
    layer_arrays = [np.array([s[l] for s in all_layer_sparsities]) for l in range(n_layers)]

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    # Basic statistics
    print("\n1. SPARSITY BY LAYER (mean +/- std):")
    print(f"   Input:  {input_sparsities.mean():.3f} +/- {input_sparsities.std():.3f}")
    for l in range(n_layers):
        print(f"   Layer {l+1}: {layer_arrays[l].mean():.3f} +/- {layer_arrays[l].std():.3f}")
    print(f"   Output: {output_sparsities.mean():.3f} +/- {output_sparsities.std():.3f}")

    print("\n2. SPARSITY GRADIENT (slope across layers):")
    print(f"   Mean: {sparsity_gradients.mean():.4f} +/- {sparsity_gradients.std():.4f}")
    print(f"   Range: [{sparsity_gradients.min():.4f}, {sparsity_gradients.max():.4f}]")

    # Correlation with order
    print("\n3. CORRELATIONS WITH ORDER:")

    r_grad, p_grad = stats.pearsonr(sparsity_gradients, orders)
    rho_grad, p_rho = stats.spearmanr(sparsity_gradients, orders)

    print(f"   Sparsity gradient (Pearson):  r={r_grad:+.3f}, p={p_grad:.2e}")
    print(f"   Sparsity gradient (Spearman): rho={rho_grad:+.3f}, p={p_rho:.2e}")

    # Layer-by-layer correlations
    print("\n   Layer-wise sparsity correlations:")
    for l in range(n_layers):
        r_l, p_l = stats.pearsonr(layer_arrays[l], orders)
        print(f"     Layer {l+1}: r={r_l:+.3f}, p={p_l:.2e}")

    r_out, p_out = stats.pearsonr(output_sparsities, orders)
    print(f"     Output:   r={r_out:+.3f}, p={p_out:.2e}")

    # Split analysis: high vs low order
    print("\n4. HIGH vs LOW ORDER COMPARISON:")
    high_mask = orders >= np.percentile(orders, 75)
    low_mask = orders <= np.percentile(orders, 25)

    high_gradient = sparsity_gradients[high_mask]
    low_gradient = sparsity_gradients[low_mask]

    print(f"   High-order (Q4) sparsity gradient: {high_gradient.mean():.4f} +/- {high_gradient.std():.4f}")
    print(f"   Low-order  (Q1) sparsity gradient: {low_gradient.mean():.4f} +/- {low_gradient.std():.4f}")

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((high_gradient.std()**2 + low_gradient.std()**2) / 2)
    cohens_d = (high_gradient.mean() - low_gradient.mean()) / pooled_std if pooled_std > 0 else 0

    # Mann-Whitney U test
    mw_stat, mw_p = stats.mannwhitneyu(high_gradient, low_gradient, alternative='two-sided')

    print(f"\n   Cohen's d: {cohens_d:+.3f}")
    print(f"   Mann-Whitney U: stat={mw_stat:.1f}, p={mw_p:.2e}")

    # Check layer-by-layer for high vs low
    print("\n5. LAYER-BY-LAYER HIGH vs LOW ORDER:")
    for l in range(n_layers):
        high_l = layer_arrays[l][high_mask]
        low_l = layer_arrays[l][low_mask]
        d_l = (high_l.mean() - low_l.mean()) / np.sqrt((high_l.std()**2 + low_l.std()**2) / 2)
        print(f"   Layer {l+1}: high={high_l.mean():.3f}, low={low_l.mean():.3f}, d={d_l:+.3f}")

    # Final verdict
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)

    # Hypothesis: high-order has steeper decrease (more negative gradient)
    # A steeper negative gradient means sparsity decreases faster through layers

    if abs(cohens_d) > 0.5 and mw_p < 0.01:
        if cohens_d < 0:
            status = "VALIDATED"
            summary = "High-order CPPNs show steeper sparsity decrease through layers"
        else:
            status = "REFUTED"
            summary = "High-order CPPNs show LESS steep sparsity decrease (opposite direction)"
    elif abs(r_grad) > 0.2 and p_grad < 0.01:
        status = "INCONCLUSIVE"
        summary = "Weak correlation found but effect size below threshold"
    else:
        status = "REFUTED"
        summary = "No significant correlation between sparsity gradient and order"

    print(f"\nPrimary metric: Sparsity gradient (slope across layers)")
    print(f"  Pearson r={r_grad:+.3f}, p={p_grad:.2e}")
    print(f"  Cohen's d={cohens_d:+.3f}")
    print(f"  Mann-Whitney p={mw_p:.2e}")
    print(f"\nSTATUS: {status}")
    print(f"SUMMARY: {summary}")

    # Effect size for reporting
    effect_size = abs(cohens_d)
    p_value = mw_p

    print(f"\nFINAL METRICS:")
    print(f"  effect_size: {effect_size:.2f}")
    print(f"  p_value: {p_value:.2e}")
    print(f"  r: {r_grad:+.3f}")
    print(f"  d: {cohens_d:+.3f}")

    return {
        'status': status,
        'effect_size': effect_size,
        'p_value': p_value,
        'r': r_grad,
        'cohens_d': cohens_d,
        'high_mean': high_gradient.mean(),
        'low_mean': low_gradient.mean(),
        'summary': summary,
    }


if __name__ == '__main__':
    main()
