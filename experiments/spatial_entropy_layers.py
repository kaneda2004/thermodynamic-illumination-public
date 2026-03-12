"""
RES-175: Spatial activation entropy increases through layers for high-order but not low-order CPPNs

Hypothesis: High-order CPPNs progressively differentiate spatial signals (entropy increases through layers)
while low-order CPPNs maintain uniform spatial patterns (entropy stays flat).

This tests information FLOW through layers via spatial entropy, distinct from:
- RES-138: Spatial variance (not entropy) of activations
- RES-148: MI compression rate (global, not spatial)
- RES-087: Inter-layer MI (correlation, not spatial entropy)
"""

import numpy as np
import json
from scipy import stats
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.thermo_sampler_v3 import CPPN, Node, Connection, order_multiplicative as compute_order, ACTIVATIONS, PRIOR_SIGMA, set_global_seed


def compute_spatial_entropy(activation_map: np.ndarray, n_bins: int = 32) -> float:
    """Compute entropy of activation values across spatial locations."""
    # Discretize activations into bins
    flat = activation_map.flatten()
    # Normalize to [0, 1] for consistent binning
    flat_norm = (flat - flat.min()) / (flat.max() - flat.min() + 1e-10)
    hist, _ = np.histogram(flat_norm, bins=n_bins, range=(0, 1))
    # Compute entropy
    probs = hist / hist.sum()
    probs = probs[probs > 0]  # Avoid log(0)
    entropy = -np.sum(probs * np.log2(probs))
    return entropy


def create_layered_cppn(n_hidden: int = 5, depth: int = 3) -> CPPN:
    """Create a CPPN with explicit layer structure for analysis."""
    # Nodes: 0-3 inputs, then hidden layers, then output
    nodes = [
        Node(0, 'identity', 0.0),
        Node(1, 'identity', 0.0),
        Node(2, 'identity', 0.0),
        Node(3, 'identity', 0.0),
    ]

    # Hidden layers
    hidden_acts = ['sin', 'gauss', 'tanh', 'cos', 'sigmoid']
    node_id = 5  # Start after output node ID
    layers = []

    for layer_idx in range(depth):
        layer = []
        for i in range(n_hidden):
            act = hidden_acts[i % len(hidden_acts)]
            bias = np.random.randn() * PRIOR_SIGMA
            nodes.append(Node(node_id, act, bias))
            layer.append(node_id)
            node_id += 1
        layers.append(layer)

    # Output node
    output_id = 4
    nodes.append(Node(output_id, 'sigmoid', np.random.randn() * PRIOR_SIGMA))

    # Connections
    connections = []

    # Input -> first hidden layer
    for inp in [0, 1, 2, 3]:
        for hid in layers[0]:
            connections.append(Connection(inp, hid, np.random.randn() * PRIOR_SIGMA))

    # Hidden layer -> hidden layer
    for i in range(len(layers) - 1):
        for from_node in layers[i]:
            for to_node in layers[i + 1]:
                connections.append(Connection(from_node, to_node, np.random.randn() * PRIOR_SIGMA))

    # Last hidden -> output
    for hid in layers[-1]:
        connections.append(Connection(hid, output_id, np.random.randn() * PRIOR_SIGMA))

    cppn = CPPN(
        nodes=nodes,
        connections=connections,
        input_ids=[0, 1, 2, 3],
        output_id=output_id
    )
    return cppn, layers


def get_layer_activations(cppn: CPPN, layers: list, size: int = 32) -> dict:
    """Get activation maps for each layer."""
    coords = np.linspace(-1, 1, size)
    x, y = np.meshgrid(coords, coords)
    r = np.sqrt(x**2 + y**2)
    bias = np.ones_like(x)

    values = {0: x, 1: y, 2: r, 3: bias}
    layer_activations = []

    # Process each hidden layer
    for layer_nodes in layers:
        layer_acts = []
        for nid in layer_nodes:
            node = next(n for n in cppn.nodes if n.id == nid)
            total = np.zeros_like(x) + node.bias
            for conn in cppn.connections:
                if conn.to_id == nid and conn.enabled and conn.from_id in values:
                    total += values[conn.from_id] * conn.weight
            act_value = ACTIVATIONS[node.activation](total)
            values[nid] = act_value
            layer_acts.append(act_value)
        layer_activations.append(layer_acts)

    # Output layer
    output_id = cppn.output_id
    node = next(n for n in cppn.nodes if n.id == output_id)
    total = np.zeros_like(x) + node.bias
    for conn in cppn.connections:
        if conn.to_id == output_id and conn.enabled and conn.from_id in values:
            total += values[conn.from_id] * conn.weight
    output_act = ACTIVATIONS[node.activation](total)

    return layer_activations, output_act


def compute_layer_entropy_profile(cppn: CPPN, layers: list, size: int = 32) -> list:
    """Compute average spatial entropy for each layer."""
    layer_acts, _ = get_layer_activations(cppn, layers, size)

    entropies = []
    for layer_act_list in layer_acts:
        # Average entropy across nodes in layer
        layer_entropy = np.mean([compute_spatial_entropy(act) for act in layer_act_list])
        entropies.append(layer_entropy)

    return entropies


def main():
    set_global_seed(42)

    n_samples = 500
    n_hidden = 5
    depth = 4  # 4 hidden layers

    print(f"Generating {n_samples} layered CPPNs (depth={depth}, hidden={n_hidden})...")

    results = []
    for i in range(n_samples):
        cppn, layers = create_layered_cppn(n_hidden=n_hidden, depth=depth)

        # Get entropy profile
        entropy_profile = compute_layer_entropy_profile(cppn, layers)

        # Compute order
        img = (get_layer_activations(cppn, layers)[1] > 0.5).astype(np.uint8)
        order = compute_order(img)

        # Compute entropy slope (linear trend)
        x = np.arange(len(entropy_profile))
        slope, intercept, r_val, p_val, std_err = stats.linregress(x, entropy_profile)

        results.append({
            'order': order,
            'entropy_profile': entropy_profile,
            'entropy_slope': slope,
            'entropy_r': r_val,
            'first_layer_entropy': entropy_profile[0],
            'last_layer_entropy': entropy_profile[-1],
            'entropy_change': entropy_profile[-1] - entropy_profile[0]
        })

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{n_samples}")

    # Convert to arrays
    orders = np.array([r['order'] for r in results])
    slopes = np.array([r['entropy_slope'] for r in results])
    entropy_changes = np.array([r['entropy_change'] for r in results])

    # Split into high/low order groups (tertiles)
    order_thresh_high = np.percentile(orders, 67)
    order_thresh_low = np.percentile(orders, 33)

    high_order_mask = orders >= order_thresh_high
    low_order_mask = orders <= order_thresh_low

    high_slopes = slopes[high_order_mask]
    low_slopes = slopes[low_order_mask]

    print(f"\nOrder distribution: mean={orders.mean():.4f}, std={orders.std():.4f}")
    print(f"High-order threshold: {order_thresh_high:.4f} (n={sum(high_order_mask)})")
    print(f"Low-order threshold: {order_thresh_low:.4f} (n={sum(low_order_mask)})")

    # Test 1: Correlation between order and entropy slope
    corr, p_corr = stats.spearmanr(orders, slopes)
    print(f"\nCorrelation (order vs slope): rho={corr:.4f}, p={p_corr:.2e}")

    # Test 2: Compare slopes between high and low order groups
    t_stat, p_ttest = stats.ttest_ind(high_slopes, low_slopes)

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((high_slopes.std()**2 + low_slopes.std()**2) / 2)
    cohens_d = (high_slopes.mean() - low_slopes.mean()) / pooled_std

    print(f"\nHigh-order entropy slope: mean={high_slopes.mean():.4f}, std={high_slopes.std():.4f}")
    print(f"Low-order entropy slope: mean={low_slopes.mean():.4f}, std={low_slopes.std():.4f}")
    print(f"t-test: t={t_stat:.4f}, p={p_ttest:.2e}")
    print(f"Cohen's d: {cohens_d:.4f}")

    # Test 3: Does entropy increase in high-order but not low-order?
    high_increases = np.mean(high_slopes > 0)
    low_increases = np.mean(low_slopes > 0)
    print(f"\nFraction with increasing entropy: high-order={high_increases:.2%}, low-order={low_increases:.2%}")

    # Test 4: Mean entropy profiles for each group
    high_profiles = np.array([r['entropy_profile'] for r, m in zip(results, high_order_mask) if m])
    low_profiles = np.array([r['entropy_profile'] for r, m in zip(results, low_order_mask) if m])

    print(f"\nMean entropy by layer:")
    print(f"  High-order: {high_profiles.mean(axis=0).round(3)}")
    print(f"  Low-order: {low_profiles.mean(axis=0).round(3)}")

    # Determine status
    if abs(cohens_d) > 0.5 and p_ttest < 0.01:
        if cohens_d > 0 and high_slopes.mean() > 0 and low_slopes.mean() <= 0:
            status = "validated"
        else:
            status = "refuted"
    else:
        status = "inconclusive" if p_ttest >= 0.01 else "refuted"

    # Save results
    output = {
        'experiment': 'RES-175',
        'hypothesis': 'Spatial activation entropy increases through layers for high-order but not low-order CPPNs',
        'n_samples': n_samples,
        'depth': depth,
        'n_hidden': n_hidden,
        'correlation': {'rho': corr, 'p': p_corr},
        'high_order': {
            'mean_slope': float(high_slopes.mean()),
            'std_slope': float(high_slopes.std()),
            'frac_increasing': float(high_increases),
            'mean_profile': high_profiles.mean(axis=0).tolist()
        },
        'low_order': {
            'mean_slope': float(low_slopes.mean()),
            'std_slope': float(low_slopes.std()),
            'frac_increasing': float(low_increases),
            'mean_profile': low_profiles.mean(axis=0).tolist()
        },
        't_stat': float(t_stat),
        'p_value': float(p_ttest),
        'cohens_d': float(cohens_d),
        'status': status
    }

    out_path = Path(__file__).parent.parent / 'results' / 'spatial_entropy_layers' / 'results.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {out_path}")
    print(f"\nSTATUS: {status}")
    print(f"Effect size d={cohens_d:.4f}, p={p_ttest:.2e}")

    # Summary for log
    summary = (
        f"Spatial entropy slope correlation with order: rho={corr:.3f}, p={p_corr:.2e}. "
        f"High-order CPPNs have entropy slope={high_slopes.mean():.3f} vs low-order={low_slopes.mean():.3f} "
        f"(d={cohens_d:.2f}, p={p_ttest:.2e}). "
    )
    if status == 'validated':
        summary += "High-order CPPNs show increasing spatial entropy through layers while low-order stay flat."
    elif status == 'refuted':
        summary += f"Both groups show similar entropy trends ({high_increases:.0%} vs {low_increases:.0%} increasing). Entropy slope does not differentiate high/low order."
    else:
        summary += "Effect below threshold."

    print(f"\nSummary: {summary}")

    return status, cohens_d, p_ttest, summary


if __name__ == '__main__':
    status, d, p, summary = main()
