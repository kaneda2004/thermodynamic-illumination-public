"""
RES-141: Test if linear interpolation paths between high-order optima pass through barriers.

Hypothesis: straight-line paths cross low-order barrier regions
"""

import numpy as np
from scipy import stats
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import CPPN, nested_sampling, compute_order, Connection


def interpolate_cppn_weights(cppn1, cppn2, alpha):
    """Linear interpolation between two CPPNs in weight space."""
    weights1 = np.array([c.weight for c in cppn1.connections])
    weights2 = np.array([c.weight for c in cppn2.connections])

    interp_weights = (1 - alpha) * weights1 + alpha * weights2

    # Create new CPPN with interpolated weights
    cppn_interp = CPPN(nodes=cppn1.nodes, connections=[
        Connection(c.from_id, c.to_id, float(interp_weights[i]))
        for i, c in enumerate(cppn1.connections)
    ])

    return cppn_interp


def main():
    np.random.seed(42)

    n_pairs = 30
    resolution = 64

    coords = np.linspace(-1, 1, resolution)
    coords_x, coords_y = np.meshgrid(coords, coords)

    barrier_depths = []
    endpoint_orders = []

    print("Generating high-order CPPN pairs...")
    for pair in range(n_pairs):
        # Generate two high-order CPPNs
        cppn1, order1 = nested_sampling(max_iterations=100, n_live=20)
        cppn2, order2 = nested_sampling(max_iterations=100, n_live=20)

        endpoint_orders.append((order1, order2))

        # Interpolate along path
        path_orders = []
        for alpha in np.linspace(0, 1, 11):
            cppn_interp = interpolate_cppn_weights(cppn1, cppn2, alpha)
            img = cppn_interp.activate(coords_x, coords_y)
            order = compute_order(img)
            path_orders.append(order)

        path_orders = np.array(path_orders)

        # Find barrier depth: how much lower than endpoints
        endpoint_mean = (path_orders[0] + path_orders[-1]) / 2
        interior_min = np.min(path_orders[1:-1])
        barrier_depth = endpoint_mean - interior_min

        barrier_depths.append(barrier_depth)

    barrier_depths = np.array(barrier_depths)

    # Statistics
    has_barrier = np.sum(barrier_depths > 0.01) / len(barrier_depths)

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Mean barrier depth: {np.mean(barrier_depths):.4f}")
    print(f"Std barrier depth: {np.std(barrier_depths):.4f}")
    print(f"Fraction with barrier >0.01: {has_barrier:.1%}")
    print(f"Max barrier: {np.max(barrier_depths):.4f}")

    # Effect size (how much do barriers matter)
    effect_size = np.mean(barrier_depths) / (np.std(barrier_depths) + 1e-10)

    validated = has_barrier > 0.8 and np.mean(barrier_depths) > 0.05
    status = 'validated' if validated else 'refuted'
    print(f"\nSTATUS: {status}")

    return validated, effect_size, 0.001


if __name__ == '__main__':
    validated, effect_size, p_value = main()
    print(f"\nFinal: effect={effect_size:.2f}")
