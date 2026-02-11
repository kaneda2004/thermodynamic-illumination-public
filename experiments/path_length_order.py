"""
RES-089: Path Length vs Maximum Order Analysis

Hypothesis: Effective path length (number of layers between input and output)
in CPPN architecture correlates positively with maximum achievable order score.

Theory: More nonlinear transformations (deeper networks) can represent more
complex compositional patterns, enabling higher-order structure.
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import (
    CPPN, Node, Connection, ACTIVATIONS,
    order_multiplicative, PRIOR_SIGMA
)
from scipy import stats

# Activation functions to use for hidden nodes
HIDDEN_ACTIVATIONS = ['sin', 'cos', 'gauss', 'tanh', 'abs']

def create_cppn_with_depth(n_hidden_layers: int, nodes_per_layer: int = 2) -> CPPN:
    """
    Create a CPPN with specified depth (number of hidden layers).

    Architecture:
    - Input: 4 nodes (x, y, r, bias) with IDs 0-3
    - Hidden layers: Each with nodes_per_layer nodes, fully connected to previous
    - Output: 1 node (ID 4) with sigmoid activation

    Path length = n_hidden_layers + 1 (input->hidden(s)->output)
    """
    # Start with input nodes (IDs 0-3, identity activation, zero bias)
    nodes = [
        Node(0, 'identity', 0.0),
        Node(1, 'identity', 0.0),
        Node(2, 'identity', 0.0),
        Node(3, 'identity', 0.0),
    ]
    input_ids = [0, 1, 2, 3]

    connections = []
    next_id = 5  # Output is always 4, hidden starts at 5

    prev_layer_ids = input_ids

    # Add hidden layers
    for layer in range(n_hidden_layers):
        layer_ids = []
        for _ in range(nodes_per_layer):
            act = np.random.choice(HIDDEN_ACTIVATIONS)
            bias = np.random.randn() * PRIOR_SIGMA
            nodes.append(Node(next_id, act, bias))
            layer_ids.append(next_id)

            # Connect from all nodes in previous layer
            for from_id in prev_layer_ids:
                weight = np.random.randn() * PRIOR_SIGMA
                connections.append(Connection(from_id, next_id, weight))

            next_id += 1

        prev_layer_ids = layer_ids

    # Output node (ID 4, sigmoid activation)
    output_id = 4
    nodes.append(Node(output_id, 'sigmoid', np.random.randn() * PRIOR_SIGMA))

    # Connect final hidden layer (or inputs if no hidden) to output
    for from_id in prev_layer_ids:
        weight = np.random.randn() * PRIOR_SIGMA
        connections.append(Connection(from_id, output_id, weight))

    return CPPN(nodes=nodes, connections=connections, input_ids=input_ids, output_id=output_id)


def get_effective_path_length(cppn: CPPN) -> int:
    """
    Compute effective path length from input to output.
    This is the longest path through the network.
    """
    # Build adjacency list
    adj = {}
    for conn in cppn.connections:
        if conn.enabled:
            if conn.from_id not in adj:
                adj[conn.from_id] = []
            adj[conn.from_id].append(conn.to_id)

    # BFS/DFS to find longest path from any input to output
    def longest_path_from(node_id, visited):
        if node_id == cppn.output_id:
            return 0
        if node_id in visited:
            return -float('inf')  # Cycle
        if node_id not in adj:
            return -float('inf')  # Dead end

        visited.add(node_id)
        max_len = -float('inf')
        for next_id in adj[node_id]:
            path_len = longest_path_from(next_id, visited.copy())
            max_len = max(max_len, 1 + path_len)
        return max_len

    # Find longest path from any input
    max_path = 0
    for inp_id in cppn.input_ids:
        path_len = longest_path_from(inp_id, set())
        if path_len > 0:
            max_path = max(max_path, path_len)

    return max_path


def sample_max_order_for_depth(n_hidden_layers: int, n_samples: int = 100,
                               nodes_per_layer: int = 2, image_size: int = 32) -> list:
    """Sample multiple CPPNs with given depth and record their order scores."""
    orders = []
    for _ in range(n_samples):
        cppn = create_cppn_with_depth(n_hidden_layers, nodes_per_layer)
        img = cppn.render(image_size)
        order = order_multiplicative(img)
        orders.append(order)
    return orders


def main():
    print("="*70)
    print("RES-089: Path Length vs Maximum Order Analysis")
    print("="*70)
    print()

    np.random.seed(42)

    # Test depths from 0 (direct input->output) to 6 hidden layers
    depths = [0, 1, 2, 3, 4, 5, 6]
    n_samples = 300  # More samples for better statistics
    nodes_per_layer = 2
    image_size = 32

    print(f"Configuration:")
    print(f"  Depths tested: {depths}")
    print(f"  Samples per depth: {n_samples}")
    print(f"  Nodes per hidden layer: {nodes_per_layer}")
    print(f"  Image size: {image_size}x{image_size}")
    print()

    results = {}

    for depth in depths:
        print(f"Testing depth {depth}...", end=" ", flush=True)
        orders = sample_max_order_for_depth(depth, n_samples, nodes_per_layer, image_size)

        results[depth] = {
            'orders': orders,
            'mean': np.mean(orders),
            'std': np.std(orders),
            'max': np.max(orders),
            'p90': np.percentile(orders, 90),
            'p95': np.percentile(orders, 95),
        }

        print(f"mean={results[depth]['mean']:.4f}, max={results[depth]['max']:.4f}, p95={results[depth]['p95']:.4f}")

    print()

    # Statistical analysis
    print("="*70)
    print("Statistical Analysis")
    print("="*70)

    # Correlation between depth and metrics
    all_depths = []
    all_means = []
    all_maxes = []
    all_p95s = []

    for depth in depths:
        all_depths.append(depth)
        all_means.append(results[depth]['mean'])
        all_maxes.append(results[depth]['max'])
        all_p95s.append(results[depth]['p95'])

    # Spearman correlation (handles non-linear relationships)
    corr_mean, p_mean = stats.spearmanr(all_depths, all_means)
    corr_max, p_max = stats.spearmanr(all_depths, all_maxes)
    corr_p95, p_p95 = stats.spearmanr(all_depths, all_p95s)

    print(f"\nSpearman Correlations (depth vs metric):")
    print(f"  Mean order:  r={corr_mean:.4f}, p={p_mean:.2e}")
    print(f"  Max order:   r={corr_max:.4f}, p={p_max:.2e}")
    print(f"  P95 order:   r={corr_p95:.4f}, p={p_p95:.2e}")

    # Linear regression for effect size
    slope, intercept, r_value, p_value, std_err = stats.linregress(all_depths, all_means)
    print(f"\nLinear Regression (depth vs mean order):")
    print(f"  Slope: {slope:.4f} (order increase per layer)")
    print(f"  R-squared: {r_value**2:.4f}")
    print(f"  p-value: {p_value:.2e}")

    # Compare depth 0 vs depth with highest P95
    best_depth = max(depths, key=lambda d: results[d]['p95'])
    orders_shallow = results[0]['orders']
    orders_best = results[best_depth]['orders']

    print(f"\nBest depth by P95: {best_depth} (P95={results[best_depth]['p95']:.4f})")

    # Compare depth 0 vs best depth
    orders_deep = orders_best

    # Mann-Whitney U test (non-parametric)
    stat, p_mw = stats.mannwhitneyu(orders_deep, orders_shallow, alternative='greater')

    # Cohen's d effect size
    pooled_std = np.sqrt((np.var(orders_shallow) + np.var(orders_deep)) / 2)
    cohens_d = (np.mean(orders_deep) - np.mean(orders_shallow)) / pooled_std if pooled_std > 0 else 0

    print(f"\nDepth 0 vs Best Depth ({best_depth}) Comparison:")
    print(f"  Depth 0: mean={np.mean(orders_shallow):.4f}, std={np.std(orders_shallow):.4f}")
    print(f"  Depth {best_depth}: mean={np.mean(orders_deep):.4f}, std={np.std(orders_deep):.4f}")
    print(f"  Mann-Whitney U p-value: {p_mw:.2e}")
    print(f"  Cohen's d: {cohens_d:.4f}")

    # Test for optimal depth (non-monotonic relationship)
    print(f"\n--- Testing for Optimal Depth (non-monotonic) ---")

    # Find if there's a peak
    p95_values = [results[d]['p95'] for d in depths]
    peak_idx = np.argmax(p95_values)
    peak_depth = depths[peak_idx]

    # Compare peak vs extremes
    if peak_depth > 0 and peak_depth < max(depths):
        # Peak is in the middle - test if it's significantly higher
        orders_peak = results[peak_depth]['orders']
        orders_first = results[depths[0]]['orders']
        orders_last = results[depths[-1]]['orders']

        _, p_vs_first = stats.mannwhitneyu(orders_peak, orders_first, alternative='greater')
        _, p_vs_last = stats.mannwhitneyu(orders_peak, orders_last, alternative='greater')

        print(f"  Peak depth: {peak_depth} (P95={results[peak_depth]['p95']:.4f})")
        print(f"  Peak vs Depth 0: p={p_vs_first:.2e}")
        print(f"  Peak vs Depth {depths[-1]}: p={p_vs_last:.2e}")

        # Effect sizes
        d_vs_first = (np.mean(orders_peak) - np.mean(orders_first)) / np.sqrt((np.var(orders_peak) + np.var(orders_first)) / 2)
        d_vs_last = (np.mean(orders_peak) - np.mean(orders_last)) / np.sqrt((np.var(orders_peak) + np.var(orders_last)) / 2)
        print(f"  Cohen's d (peak vs depth 0): {d_vs_first:.4f}")
        print(f"  Cohen's d (peak vs depth {depths[-1]}): {d_vs_last:.4f}")

        # Optimal depth is validated if peak is significantly better than both extremes
        optimal_validated = p_vs_first < 0.01 and p_vs_last < 0.01
        print(f"  Optimal depth validated: {optimal_validated}")
    else:
        optimal_validated = False
        print(f"  No clear optimal depth (peak at extreme: {peak_depth})")

    # Verify path lengths are as expected
    print(f"\nPath Length Verification:")
    for depth in [0, 2, 5]:
        cppn = create_cppn_with_depth(depth, nodes_per_layer)
        actual_path = get_effective_path_length(cppn)
        expected_path = depth + 1  # input -> hidden(s) -> output
        print(f"  Depth {depth}: expected path={expected_path}, actual path={actual_path}")

    # Summary
    print()
    print("="*70)
    print("SUMMARY")
    print("="*70)

    # Determine outcome for ORIGINAL HYPOTHESIS:
    # "Path length correlates positively with maximum achievable order"
    #
    # Criteria for validation:
    # - Significant positive correlation (p < 0.01, r > 0.5)
    # - Depth 5 significantly better than depth 0

    monotonic_positive = corr_p95 > 0.5 and p_p95 < 0.01

    # Check if deeper is actually better
    deeper_better = p_mw < 0.01  # depth 0 vs best depth

    # The hypothesis is about MONOTONIC POSITIVE correlation
    # If we see a peak in the middle, the hypothesis is REFUTED
    has_interior_peak = peak_depth > 0 and peak_depth < max(depths)

    if monotonic_positive and deeper_better:
        status = "VALIDATED"
    elif has_interior_peak or corr_p95 < 0.3:
        # Refuted: Either there's a peak (non-monotonic) or no correlation
        status = "REFUTED"
    else:
        status = "INCONCLUSIVE"

    print(f"\nStatus: {status}")
    print(f"\nKey Metrics:")
    print(f"  Correlation (depth vs P95 order): r={corr_p95:.4f}, p={p_p95:.2e}")
    print(f"  Effect size (Cohen's d, depth 0 vs best): {cohens_d:.4f}")
    print(f"  Peak depth: {peak_depth}")
    print(f"  Monotonic positive relationship: {monotonic_positive}")

    if status == "VALIDATED":
        print(f"\nConclusion: Deeper CPPN architectures achieve significantly higher")
        print(f"order scores. Each additional hidden layer adds ~{slope:.3f} to mean order.")
    elif status == "REFUTED":
        print(f"\nConclusion: Path length does NOT positively correlate with order.")
        print(f"Instead, there appears to be an OPTIMAL depth around {peak_depth} layers.")
        print(f"Deeper networks show diminishing returns, possibly due to:")
        print(f"  - Vanishing/exploding signal magnitudes")
        print(f"  - Over-parameterization leading to less structured outputs")
        print(f"  - Harder optimization landscape")
    else:
        print(f"\nConclusion: Results are inconclusive. More samples or")
        print(f"different architectures may be needed.")

    return {
        'status': status,
        'correlation_p95': float(corr_p95),
        'p_correlation': float(p_p95),
        'p_value_mw': float(p_mw),
        'cohens_d': float(cohens_d),
        'slope_per_layer': float(slope),
        'peak_depth': int(peak_depth),
        'peak_p95': float(results[peak_depth]['p95']),
    }


if __name__ == "__main__":
    results = main()
