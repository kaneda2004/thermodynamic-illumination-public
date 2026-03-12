"""
RES-052: Layer Contribution Analysis
Hypothesis: Final layers contribute disproportionately to CPPN output order

Method: Use weight perturbation sensitivity to measure how much each layer's
weights affect output order. Layers with higher sensitivity contribute more.
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import (
    CPPN, Node, Connection, order_multiplicative, ACTIVATIONS
)
from scipy import stats

np.random.seed(42)

def create_deep_cppn(num_hidden_layers: int = 4, nodes_per_layer: int = 3) -> tuple:
    """Create a CPPN with explicit layer structure for analysis."""
    nodes = [
        Node(0, 'identity', 0.0), Node(1, 'identity', 0.0),
        Node(2, 'identity', 0.0), Node(3, 'identity', 0.0),
    ]
    connections = []

    input_ids = [0, 1, 2, 3]
    next_id = 5  # Output is 4

    # Track which connections belong to which layer
    layer_connections = []  # layer_connections[i] = indices of connections TO layer i

    # Build hidden layers
    layer_node_ids = [input_ids]  # Layer 0 = inputs

    for layer_idx in range(num_hidden_layers):
        current_layer_ids = []
        current_conn_indices = []
        for _ in range(nodes_per_layer):
            activation = np.random.choice(['sin', 'tanh', 'gauss'])
            nodes.append(Node(next_id, activation, np.random.randn() * 1.0))
            current_layer_ids.append(next_id)
            # Connect from previous layer
            for prev_id in layer_node_ids[-1]:
                current_conn_indices.append(len(connections))
                connections.append(Connection(prev_id, next_id, np.random.randn() * 1.0))
            next_id += 1
        layer_node_ids.append(current_layer_ids)
        layer_connections.append(current_conn_indices)

    # Output node
    nodes.append(Node(4, 'sigmoid', np.random.randn() * 1.0))
    output_conn_indices = []
    for prev_id in layer_node_ids[-1]:
        output_conn_indices.append(len(connections))
        connections.append(Connection(prev_id, 4, np.random.randn() * 1.0))
    layer_connections.append(output_conn_indices)

    cppn = CPPN(nodes=nodes, connections=connections, input_ids=input_ids, output_id=4)
    return cppn, layer_connections

def measure_layer_sensitivity(cppn: CPPN, layer_conn_indices: list, epsilon: float = 0.1, n_samples: int = 10) -> float:
    """
    Measure layer sensitivity via weight perturbation.
    Returns average absolute change in order per unit weight perturbation.
    """
    original_order = order_multiplicative(cppn.render(32))

    sensitivities = []
    for conn_idx in layer_conn_indices:
        for _ in range(n_samples):
            perturbed = cppn.copy()
            delta = np.random.choice([-epsilon, epsilon])
            perturbed.connections[conn_idx].weight += delta
            new_order = order_multiplicative(perturbed.render(32))
            sensitivity = abs(new_order - original_order) / abs(delta)
            sensitivities.append(sensitivity)

    return np.mean(sensitivities) if sensitivities else 0.0

def run_experiment(n_networks=150, n_layers=4):
    """Run layer contribution analysis across many networks."""

    print(f"Testing {n_networks} networks with {n_layers} hidden layers each")
    print("=" * 60)

    # Collect sensitivities by layer position
    # Position 0 = first hidden, ..., Position n_layers = output
    sensitivities = {i: [] for i in range(n_layers + 1)}

    valid_networks = 0
    attempts = 0
    max_attempts = n_networks * 10

    while valid_networks < n_networks and attempts < max_attempts:
        attempts += 1
        cppn, layer_conns = create_deep_cppn(num_hidden_layers=n_layers)
        base_order = order_multiplicative(cppn.render(32))

        # Skip networks with too low or too high order
        if base_order < 0.05 or base_order > 0.95:
            continue

        valid_networks += 1

        # Measure sensitivity of each layer
        for layer_idx in range(n_layers + 1):
            sens = measure_layer_sensitivity(cppn, layer_conns[layer_idx])
            sensitivities[layer_idx].append(sens)

        if valid_networks % 25 == 0:
            print(f"  Processed {valid_networks}/{n_networks} networks...")

    print(f"\nAnalyzed {valid_networks} valid networks\n")

    # Statistical analysis
    layer_names = [f"Hidden {i+1}" for i in range(n_layers)] + ["Output"]
    means = []
    stds = []

    print("Layer Sensitivities (order change per unit weight):")
    print("-" * 60)
    for idx, name in enumerate(layer_names):
        data = np.array(sensitivities[idx])
        mean = np.mean(data)
        std = np.std(data)
        means.append(mean)
        stds.append(std)
        print(f"  {name:12s}: mean={mean:.4f} ± {std:.4f}")

    # Test hypothesis: Final layers contribute more
    # Compare last hidden + output vs first two hidden layers
    early_layers = np.concatenate([sensitivities[0], sensitivities[1]])
    late_layers = np.concatenate([sensitivities[n_layers-1], sensitivities[n_layers]])

    # Mann-Whitney U test (non-parametric)
    statistic, p_value = stats.mannwhitneyu(late_layers, early_layers, alternative='greater')

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.var(early_layers) + np.var(late_layers)) / 2)
    effect_size = (np.mean(late_layers) - np.mean(early_layers)) / pooled_std if pooled_std > 0 else 0

    print("\n" + "=" * 60)
    print("HYPOTHESIS TEST: Late layers > Early layers sensitivity")
    print("-" * 60)
    print(f"  Early layers mean: {np.mean(early_layers):.4f}")
    print(f"  Late layers mean:  {np.mean(late_layers):.4f}")
    print(f"  Mann-Whitney U p-value: {p_value:.6f}")
    print(f"  Effect size (Cohen's d): {effect_size:.3f}")

    # Determine result
    if p_value < 0.01 and effect_size > 0.5:
        status = "VALIDATED"
    elif p_value > 0.05 or effect_size < 0.2:
        status = "REFUTED"
    else:
        status = "INCONCLUSIVE"

    print(f"\n  RESULT: {status}")

    # Additional analysis: Correlation with layer depth
    positions = []
    all_sens = []
    for idx in range(n_layers + 1):
        for s in sensitivities[idx]:
            positions.append(idx)
            all_sens.append(s)

    corr, corr_p = stats.spearmanr(positions, all_sens)
    print(f"\n  Spearman correlation (depth vs sensitivity): r={corr:.3f}, p={corr_p:.6f}")

    return {
        'status': status,
        'p_value': float(p_value),
        'effect_size': float(effect_size),
        'early_mean': float(np.mean(early_layers)),
        'late_mean': float(np.mean(late_layers)),
        'depth_correlation': float(corr),
        'depth_corr_p': float(corr_p),
        'layer_means': [float(m) for m in means]
    }

if __name__ == "__main__":
    results = run_experiment(n_networks=150, n_layers=4)
    print("\n" + "=" * 60)
    print("SUMMARY FOR LOG:")
    print(f"  Status: {results['status']}")
    print(f"  Effect size: {results['effect_size']:.3f}")
    print(f"  P-value: {results['p_value']:.6f}")
