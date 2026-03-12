"""
RES-074: Test if optimal hidden node count exists with diminishing returns.

Hypothesis: order scales log-linearly with hidden node count
"""

import numpy as np
from scipy import stats
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import CPPN, compute_order, Node, Connection, ACTIVATIONS


def create_cppn_with_hidden_nodes(n_hidden):
    """Create CPPN with specified number of hidden nodes."""
    nodes = [
        Node(0, 'identity', 0.0), Node(1, 'identity', 0.0),
        Node(2, 'identity', 0.0), Node(3, 'identity', 0.0),
    ]

    # Add hidden nodes
    for i in range(n_hidden):
        nodes.append(Node(5 + i, 'tanh', np.random.randn()))

    # Output node
    nodes.append(Node(4, 'sigmoid', np.random.randn()))

    connections = []

    # Input to hidden
    for inp in [0, 1, 2, 3]:
        for h in range(n_hidden):
            connections.append(Connection(inp, 5 + h, np.random.randn()))

    # Hidden to output
    for h in range(n_hidden):
        connections.append(Connection(5 + h, 4, np.random.randn()))

    return CPPN(nodes=nodes, connections=connections, input_ids=[0, 1, 2, 3], output_id=4)


def main():
    np.random.seed(42)

    hidden_counts = [0, 1, 2, 4, 8, 16, 32]
    n_samples_per_count = 30
    resolution = 64

    coords = np.linspace(-1, 1, resolution)
    coords_x, coords_y = np.meshgrid(coords, coords)

    results = {}

    print("Testing different hidden node counts...")
    for n_hidden in hidden_counts:
        print(f"  {n_hidden} hidden nodes...")
        orders = []

        for sample in range(n_samples_per_count):
            cppn = create_cppn_with_hidden_nodes(n_hidden)
            img = cppn.activate(coords_x, coords_y)
            order = compute_order(img)
            orders.append(order)

        results[n_hidden] = np.array(orders)

    # Extract statistics
    hidden_array = np.array(hidden_counts)
    mean_orders = np.array([np.mean(results[h]) for h in hidden_counts])
    std_orders = np.array([np.std(results[h]) for h in hidden_counts])

    # Test log-linear relationship
    nonzero_hidden = hidden_array[hidden_array > 0]
    nonzero_orders = mean_orders[hidden_array > 0]

    if len(nonzero_hidden) > 2:
        # Fit: order ~ log(n+1)
        log_hidden = np.log(nonzero_hidden + 1)
        slope, intercept = np.polyfit(log_hidden, nonzero_orders, 1)

        fitted = slope * log_hidden + intercept
        r_squared = 1 - np.sum((nonzero_orders - fitted)**2) / np.sum((nonzero_orders - np.mean(nonzero_orders))**2)

        # ANOVA across counts
        f_stat, p_anova = stats.f_oneway(*[results[h] for h in hidden_counts if len(results[h]) > 0])
    else:
        slope, r_squared, f_stat, p_anova = 0, 0, 0, 1

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    for h in hidden_counts:
        print(f"{h:2d} nodes: {np.mean(results[h]):.4f} +/- {np.std(results[h]):.4f}")

    print(f"\nLog-linear fit:")
    print(f"  Slope: {slope:.4f}")
    print(f"  R²: {r_squared:.3f}")
    print(f"ANOVA F={f_stat:.2f}, p={p_anova:.2e}")

    effect_size = abs(slope)
    validated = r_squared > 0.7 and p_anova < 0.01
    status = 'validated' if validated else 'refuted'
    print(f"\nSTATUS: {status}")

    return validated, effect_size, p_anova


if __name__ == '__main__':
    validated, effect_size, p_value = main()
    print(f"\nFinal: slope={effect_size:.4f}, p={p_value:.2e}")
