"""
RES-070: Test if recurrent CPPN connections improve order via iterative refinement.

Hypothesis: feedback loops enable refinement to structured patterns
"""

import numpy as np
from scipy import stats
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import CPPN, nested_sampling, compute_order, Connection, Node, ACTIVATIONS


def create_recurrent_cppn(iterations=2):
    """Create CPPN with recurrent feedback."""
    nodes = [
        Node(0, 'identity', 0.0), Node(1, 'identity', 0.0),
        Node(2, 'identity', 0.0), Node(3, 'identity', 0.0),
        Node(4, 'sigmoid', np.random.randn()),  # Output node
    ]

    connections = []
    for inp in [0, 1, 2, 3]:
        connections.append(Connection(inp, 4, np.random.randn()))

    # Add recurrent connection (output to itself)
    for _ in range(iterations):
        connections.append(Connection(4, 4, np.random.randn() * 0.1))

    return CPPN(nodes=nodes, connections=connections, input_ids=[0, 1, 2, 3], output_id=4)


def activate_recurrent(cppn, coords_x, coords_y, iterations=2):
    """Activate CPPN with recurrent iterations."""
    r = np.sqrt(coords_x**2 + coords_y**2)
    bias = np.ones_like(coords_x)
    values = {0: coords_x, 1: coords_y, 2: r, 3: bias}

    # Initialize output
    output = np.zeros_like(coords_x)
    values[4] = output

    # Iterate
    for it in range(iterations):
        for conn in cppn.connections:
            if conn.from_id == 4 and conn.to_id == 4:
                # Recurrent: add to output
                values[4] = values[4] + conn.weight * output
            elif conn.to_id == 4:
                # Normal input connection
                values[4] = values[4] + conn.weight * values[conn.from_id]

        node = next(n for n in cppn.nodes if n.id == 4)
        output = ACTIVATIONS[node.activation](values[4] + node.bias)
        values[4] = output

    return output


def main():
    np.random.seed(42)

    n_samples = 100
    resolution = 64

    coords = np.linspace(-1, 1, resolution)
    coords_x, coords_y = np.meshgrid(coords, coords)

    # Test feedforward (no recurrence)
    print("Testing feedforward CPPNs...")
    feedforward_orders = []
    for i in range(n_samples):
        cppn = CPPN()
        img = cppn.activate(coords_x, coords_y)
        order = compute_order(img)
        feedforward_orders.append(order)

    # Test recurrent variants
    recurrent_orders = {1: [], 2: [], 3: []}

    for iter_count in [1, 2, 3]:
        print(f"Testing {iter_count}-iteration recurrent CPPNs...")
        for i in range(n_samples):
            cppn = create_recurrent_cppn(iterations=iter_count)
            img = activate_recurrent(cppn, coords_x, coords_y, iterations=iter_count)
            order = compute_order(img)
            recurrent_orders[iter_count].append(order)

    feedforward_orders = np.array(feedforward_orders)

    # Compare feedforward vs best recurrent
    best_recurrent = max(
        (np.mean(recurrent_orders[k]), k) for k in recurrent_orders.keys()
    )
    best_iter, best_orders = best_recurrent[1], recurrent_orders[best_recurrent[1]]
    best_orders = np.array(best_orders)

    t_stat, p_value = stats.ttest_ind(best_orders, feedforward_orders)

    pooled_std = np.sqrt((np.std(best_orders)**2 + np.std(feedforward_orders)**2) / 2)
    effect_size = (np.mean(best_orders) - np.mean(feedforward_orders)) / (pooled_std + 1e-10)

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Feedforward order: {np.mean(feedforward_orders):.4f}")
    for iter_count in [1, 2, 3]:
        print(f"{iter_count}-iteration recurrent order: {np.mean(recurrent_orders[iter_count]):.4f}")
    print(f"Best recurrent: {best_iter} iterations")
    print(f"Effect size (Cohen's d): {effect_size:.2f}")
    print(f"p-value: {p_value:.2e}")

    validated = effect_size > 0.5 and p_value < 0.01
    status = 'validated' if validated else 'refuted'
    print(f"\nSTATUS: {status}")

    return validated, effect_size, p_value


if __name__ == '__main__':
    validated, effect_size, p_value = main()
    print(f"\nFinal: d={effect_size:.2f}, p={p_value:.2e}")
