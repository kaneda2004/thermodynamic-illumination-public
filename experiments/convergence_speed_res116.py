"""
RES-116: Test if deeper CPPNs require fewer iterations to reach high order.

Hypothesis: network depth enables faster convergence to structured patterns
"""

import numpy as np
from scipy import stats
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import CPPN, nested_sampling, compute_order, Node, Connection


def create_cppn_with_depth(depth):
    """Create CPPN with specific hidden layer depth."""
    nodes = [
        Node(0, 'identity', 0.0), Node(1, 'identity', 0.0),
        Node(2, 'identity', 0.0), Node(3, 'identity', 0.0),
    ]

    connections = []
    input_ids = [0, 1, 2, 3]
    hidden_id = 5

    # Create depth-1, depth-2 layers
    for d in range(depth):
        nodes.append(Node(hidden_id + d, 'tanh', np.random.randn()))
        if d == 0:
            for inp in input_ids:
                connections.append(Connection(inp, hidden_id + d, np.random.randn()))
        else:
            connections.append(Connection(hidden_id + d - 1, hidden_id + d, np.random.randn()))

    # Output from last hidden or input
    output_id = 4
    nodes.append(Node(output_id, 'sigmoid', np.random.randn()))
    if depth > 0:
        connections.append(Connection(hidden_id + depth - 1, output_id, np.random.randn()))
    else:
        for inp in input_ids:
            connections.append(Connection(inp, output_id, np.random.randn()))

    cppn = CPPN(nodes=nodes, connections=connections, input_ids=input_ids, output_id=output_id)
    return cppn


def measure_convergence_iterations(cppn, target_order=0.4, max_iter=100):
    """Measure iterations needed to reach target order."""
    coords = np.linspace(-1, 1, 32)
    coords_x, coords_y = np.meshgrid(coords, coords)

    for iteration in range(max_iter):
        order = compute_order(cppn.activate(coords_x, coords_y))
        if order >= target_order:
            return iteration

    return max_iter


def main():
    np.random.seed(42)

    depths = [0, 1, 2, 3, 4]
    n_samples = 20

    results = {}

    for depth in depths:
        print(f"Testing depth={depth}...")
        iterations_needed = []

        for sample in range(n_samples):
            cppn = create_cppn_with_depth(depth)
            iters = measure_convergence_iterations(cppn)
            iterations_needed.append(iters)

        results[depth] = iterations_needed

    # Extract data
    depths_array = np.array(depths).repeat(n_samples)
    iters_array = np.concatenate([np.array(results[d]) for d in depths])

    # ANOVA
    f_stat, p_anova = stats.f_oneway(*[np.array(results[d]) for d in depths])

    # Correlation: depth vs iterations
    corr, p_corr = stats.pearsonr(depths_array, iters_array)

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    for depth in depths:
        print(f"Depth {depth}: {np.mean(results[depth]):.1f} +/- {np.std(results[depth]):.1f} iterations")

    print(f"\nANOVA F-statistic: {f_stat:.2f}, p={p_anova:.2e}")
    print(f"Correlation (depth vs iterations): r={corr:.3f}, p={p_corr:.2e}")

    effect_size = abs(corr)
    validated = effect_size > 0.5 and p_corr < 0.01
    status = 'validated' if validated else 'refuted'
    print(f"\nSTATUS: {status}")

    return validated, effect_size, p_anova


if __name__ == '__main__':
    validated, effect_size, p_value = main()
    print(f"\nFinal: r={effect_size:.3f}, p={p_value:.2e}")
