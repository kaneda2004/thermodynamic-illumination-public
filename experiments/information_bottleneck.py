#!/usr/bin/env python3
"""
RES-122: Test Information Bottleneck hypothesis for CPPN layers.

Hypothesis: CPPN layers form an information bottleneck - mutual information
between input coordinates and hidden layer activations decreases in middle
layers then increases toward output.

Method: Build CPPNs with multiple hidden layers and measure MI(inputs, layer_k)
for each layer using histogram-based MI estimation.
"""

import numpy as np
from scipy import stats
from dataclasses import dataclass, field
from typing import Callable
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import ACTIVATIONS, PRIOR_SIGMA


@dataclass
class Node:
    id: int
    activation: str
    bias: float


@dataclass
class Connection:
    from_id: int
    to_id: int
    weight: float
    enabled: bool = True


@dataclass
class MultiLayerCPPN:
    """CPPN with configurable hidden layers for information bottleneck analysis."""
    n_hidden_layers: int = 3
    nodes_per_layer: int = 4
    nodes: list = field(default_factory=list)
    connections: list = field(default_factory=list)
    input_ids: list = field(default_factory=lambda: [0, 1, 2, 3])  # x, y, r, bias
    output_id: int = -1
    layer_ids: list = field(default_factory=list)  # IDs for each layer's nodes

    def __post_init__(self):
        if not self.nodes:
            self._build_network()

    def _build_network(self):
        """Build a feedforward CPPN with hidden layers."""
        node_id = 4  # Start after input nodes (0-3)

        # Input nodes
        self.nodes = [
            Node(0, 'identity', 0.0),  # x
            Node(1, 'identity', 0.0),  # y
            Node(2, 'identity', 0.0),  # r
            Node(3, 'identity', 0.0),  # bias
        ]
        self.layer_ids = [self.input_ids.copy()]

        activations = ['sin', 'tanh', 'gauss', 'sigmoid']

        prev_layer_ids = self.input_ids.copy()

        # Hidden layers
        for layer_idx in range(self.n_hidden_layers):
            current_layer_ids = []
            for _ in range(self.nodes_per_layer):
                act = activations[node_id % len(activations)]
                self.nodes.append(Node(node_id, act, np.random.randn() * PRIOR_SIGMA))

                # Connect from all previous layer nodes
                for from_id in prev_layer_ids:
                    self.connections.append(
                        Connection(from_id, node_id, np.random.randn() * PRIOR_SIGMA)
                    )

                current_layer_ids.append(node_id)
                node_id += 1

            self.layer_ids.append(current_layer_ids)
            prev_layer_ids = current_layer_ids

        # Output node
        self.output_id = node_id
        self.nodes.append(Node(self.output_id, 'sigmoid', np.random.randn() * PRIOR_SIGMA))
        for from_id in prev_layer_ids:
            self.connections.append(
                Connection(from_id, self.output_id, np.random.randn() * PRIOR_SIGMA)
            )
        self.layer_ids.append([self.output_id])

    def activate_with_intermediates(self, x: np.ndarray, y: np.ndarray) -> dict:
        """
        Forward pass returning activations for all layers.
        Returns dict: layer_idx -> (n_nodes, n_pixels) array of activations
        """
        r = np.sqrt(x**2 + y**2)
        bias = np.ones_like(x)

        values = {0: x.flatten(), 1: y.flatten(), 2: r.flatten(), 3: bias.flatten()}
        layer_activations = {0: np.stack([x.flatten(), y.flatten(), r.flatten(), bias.flatten()])}

        for layer_idx, layer_node_ids in enumerate(self.layer_ids[1:], start=1):
            layer_vals = []
            for nid in layer_node_ids:
                node = next(n for n in self.nodes if n.id == nid)
                total = np.zeros_like(x.flatten()) + node.bias

                for conn in self.connections:
                    if conn.to_id == nid and conn.enabled and conn.from_id in values:
                        total += values[conn.from_id] * conn.weight

                values[nid] = ACTIVATIONS[node.activation](total)
                layer_vals.append(values[nid])

            layer_activations[layer_idx] = np.stack(layer_vals)

        return layer_activations

    def render(self, size: int = 32) -> np.ndarray:
        coords = np.linspace(-1, 1, size)
        x, y = np.meshgrid(coords, coords)
        activations = self.activate_with_intermediates(x, y)
        output_layer_idx = len(self.layer_ids) - 1
        return (activations[output_layer_idx][0].reshape(size, size) > 0.5).astype(np.uint8)


def estimate_mi_histogram(X: np.ndarray, Y: np.ndarray, n_bins: int = 20) -> float:
    """
    Estimate mutual information between X and Y using histogram method.

    MI(X;Y) = H(X) + H(Y) - H(X,Y)

    Args:
        X: 1D array of values
        Y: 1D array of values
        n_bins: number of bins for histogram

    Returns:
        Estimated MI in bits
    """
    # Discretize
    x_bins = np.linspace(X.min() - 1e-8, X.max() + 1e-8, n_bins + 1)
    y_bins = np.linspace(Y.min() - 1e-8, Y.max() + 1e-8, n_bins + 1)

    x_disc = np.digitize(X, x_bins) - 1
    y_disc = np.digitize(Y, y_bins) - 1

    # Joint histogram
    joint_hist, _, _ = np.histogram2d(x_disc, y_disc, bins=[n_bins, n_bins])
    joint_hist = joint_hist / joint_hist.sum()

    # Marginals
    px = joint_hist.sum(axis=1)
    py = joint_hist.sum(axis=0)

    # Entropies
    def entropy(p):
        p = p[p > 0]
        return -np.sum(p * np.log2(p))

    H_X = entropy(px)
    H_Y = entropy(py)
    H_XY = entropy(joint_hist.flatten())

    return H_X + H_Y - H_XY


def compute_layer_mi(layer_activations: dict, input_coords: np.ndarray) -> dict:
    """
    Compute MI between input coordinates and each layer's activations.

    Args:
        layer_activations: dict from layer_idx to (n_nodes, n_pixels) activations
        input_coords: (2, n_pixels) array of x, y coordinates

    Returns:
        dict: layer_idx -> MI value (averaged across nodes and input dims)
    """
    x_flat = input_coords[0]
    y_flat = input_coords[1]

    layer_mi = {}

    for layer_idx, activations in layer_activations.items():
        if layer_idx == 0:
            # Skip input layer - MI would be trivial
            continue

        # Average MI across all nodes in layer and both input dimensions
        mi_values = []
        for node_idx in range(activations.shape[0]):
            node_act = activations[node_idx]
            mi_x = estimate_mi_histogram(x_flat, node_act)
            mi_y = estimate_mi_histogram(y_flat, node_act)
            mi_values.append((mi_x + mi_y) / 2)

        layer_mi[layer_idx] = np.mean(mi_values)

    return layer_mi


def test_information_bottleneck(n_networks: int = 50, n_hidden_layers: int = 5,
                                 nodes_per_layer: int = 4, image_size: int = 32) -> dict:
    """
    Test the information bottleneck hypothesis across multiple CPPNs.

    Returns:
        dict with MI curves and statistical analysis
    """
    print(f"Testing information bottleneck with {n_networks} networks, "
          f"{n_hidden_layers} hidden layers, {nodes_per_layer} nodes/layer")

    # Generate coordinate grid
    coords = np.linspace(-1, 1, image_size)
    x, y = np.meshgrid(coords, coords)
    input_coords = np.stack([x.flatten(), y.flatten()])

    all_mi_curves = []

    for i in range(n_networks):
        if (i + 1) % 10 == 0:
            print(f"  Processing network {i+1}/{n_networks}")

        # Create network with random weights
        cppn = MultiLayerCPPN(n_hidden_layers=n_hidden_layers,
                               nodes_per_layer=nodes_per_layer)

        # Get activations for all layers
        layer_activations = cppn.activate_with_intermediates(x, y)

        # Compute MI for each layer
        layer_mi = compute_layer_mi(layer_activations, input_coords)

        # Store as curve: [layer1_mi, layer2_mi, ..., output_mi]
        mi_curve = [layer_mi.get(j, np.nan) for j in range(1, n_hidden_layers + 2)]
        all_mi_curves.append(mi_curve)

    all_mi_curves = np.array(all_mi_curves)

    # Analyze the curves
    mean_curve = np.nanmean(all_mi_curves, axis=0)
    std_curve = np.nanstd(all_mi_curves, axis=0)

    # Check for bottleneck pattern: decrease then increase
    n_layers = len(mean_curve)

    # Find minimum MI layer
    min_idx = np.argmin(mean_curve)

    # Test 1: Is there a clear minimum in the middle layers?
    is_middle_min = 0 < min_idx < n_layers - 1

    # Test 2: Does MI decrease then increase? (Compare first half to second half)
    first_half = mean_curve[:n_layers//2 + 1]
    second_half = mean_curve[n_layers//2:]

    # For bottleneck: first half should generally decrease, second half should increase
    first_half_trend = np.polyfit(np.arange(len(first_half)), first_half, 1)[0]
    second_half_trend = np.polyfit(np.arange(len(second_half)), second_half, 1)[0]

    has_bottleneck_shape = first_half_trend < 0 and second_half_trend > 0

    # Test 3: Statistical test - Is the minimum significantly lower than endpoints?
    min_mi = mean_curve[min_idx]
    endpoint_mi = (mean_curve[0] + mean_curve[-1]) / 2

    # Use paired t-test comparing min layer MI to endpoint average across networks
    min_layer_values = all_mi_curves[:, min_idx]
    endpoint_values = (all_mi_curves[:, 0] + all_mi_curves[:, -1]) / 2

    t_stat, p_value = stats.ttest_rel(min_layer_values, endpoint_values)

    # Effect size (Cohen's d for paired samples)
    diff = endpoint_values - min_layer_values
    effect_size = np.mean(diff) / np.std(diff)

    results = {
        'mean_mi_curve': mean_curve.tolist(),
        'std_mi_curve': std_curve.tolist(),
        'min_layer_idx': int(min_idx),
        'is_middle_min': bool(is_middle_min),
        'first_half_trend': float(first_half_trend),
        'second_half_trend': float(second_half_trend),
        'has_bottleneck_shape': bool(has_bottleneck_shape),
        'min_mi': float(min_mi),
        'endpoint_mi': float(endpoint_mi),
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'effect_size': float(effect_size),
        'n_networks': n_networks,
        'n_hidden_layers': n_hidden_layers,
    }

    return results


def main():
    print("=" * 70)
    print("RES-122: Information Bottleneck in CPPN Layers")
    print("=" * 70)

    # Run with different network depths to see if bottleneck is consistent
    results_by_depth = {}

    for n_layers in [3, 5, 7]:
        print(f"\n--- Testing {n_layers} hidden layers ---")
        results = test_information_bottleneck(
            n_networks=100,
            n_hidden_layers=n_layers,
            nodes_per_layer=4,
            image_size=32
        )
        results_by_depth[n_layers] = results

        print(f"\nResults for {n_layers} hidden layers:")
        print(f"  Mean MI curve: {[f'{v:.3f}' for v in results['mean_mi_curve']]}")
        print(f"  Min layer index: {results['min_layer_idx']} (0=first hidden, {n_layers}=output)")
        print(f"  Is middle minimum: {results['is_middle_min']}")
        print(f"  First half trend: {results['first_half_trend']:.4f}")
        print(f"  Second half trend: {results['second_half_trend']:.4f}")
        print(f"  Has bottleneck shape: {results['has_bottleneck_shape']}")
        print(f"  Effect size (d): {results['effect_size']:.3f}")
        print(f"  p-value: {results['p_value']:.6f}")

    # Overall assessment
    print("\n" + "=" * 70)
    print("OVERALL ASSESSMENT")
    print("=" * 70)

    bottleneck_count = sum(1 for r in results_by_depth.values() if r['has_bottleneck_shape'])
    significant_count = sum(1 for r in results_by_depth.values() if r['p_value'] < 0.01)
    large_effect_count = sum(1 for r in results_by_depth.values() if abs(r['effect_size']) > 0.5)

    print(f"Networks with bottleneck shape: {bottleneck_count}/3")
    print(f"Networks with p < 0.01: {significant_count}/3")
    print(f"Networks with |d| > 0.5: {large_effect_count}/3")

    # Determine status
    if bottleneck_count >= 2 and significant_count >= 2 and large_effect_count >= 2:
        status = "validated"
        summary = "CPPN layers form information bottleneck with MI decreasing then increasing"
    elif bottleneck_count == 0:
        status = "refuted"
        summary = "No bottleneck pattern found - MI does not decrease then increase through layers"
    else:
        status = "inconclusive"
        summary = f"Mixed results: {bottleneck_count}/3 bottleneck, {significant_count}/3 significant"

    print(f"\nSTATUS: {status.upper()}")
    print(f"SUMMARY: {summary}")

    # Save detailed results
    import json
    output_path = '/Users/matt/Development/monochrome_noise_converger/results/res_122_information_bottleneck.json'
    with open(output_path, 'w') as f:
        json.dump({
            'experiment_id': 'RES-122',
            'status': status,
            'summary': summary,
            'results_by_depth': results_by_depth
        }, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return status, results_by_depth


if __name__ == '__main__':
    main()
