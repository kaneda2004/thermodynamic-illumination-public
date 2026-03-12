"""
RES-200: Test if ESS step size decreases as NS approaches high-order regions.

Hypothesis: nested sampling takes smaller steps in high-order regions
"""

import numpy as np
from scipy import stats
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import CPPN, compute_order


def simulate_ess_trajectory(n_steps=100):
    """Simulate ESS trajectory and measure step sizes at different order thresholds."""
    step_sizes = []
    orders = []

    for i in range(n_steps):
        # Sample two nearby CPPNs
        cppn1 = CPPN()
        order1 = compute_order(cppn1.activate(np.linspace(-1, 1, 32), np.linspace(-1, 1, 32)))

        # Create slight perturbation
        cppn2 = CPPN(nodes=cppn1.nodes.copy(), connections=[
            __import__('copy').deepcopy(c) for c in cppn1.connections
        ])

        # Perturb one weight
        if cppn2.connections:
            cppn2.connections[0].weight += np.random.randn() * 0.1

        # Compute step size (weight space distance)
        weights1 = np.array([c.weight for c in cppn1.connections])
        weights2 = np.array([c.weight for c in cppn2.connections])
        step = np.linalg.norm(weights2 - weights1)

        order2 = compute_order(cppn2.activate(np.linspace(-1, 1, 32), np.linspace(-1, 1, 32)))
        avg_order = (order1 + order2) / 2

        step_sizes.append(step)
        orders.append(avg_order)

    return np.array(orders), np.array(step_sizes)


def main():
    np.random.seed(42)

    n_trajectories = 50
    all_orders = []
    all_steps = []

    print("Simulating ESS trajectories...")
    for traj in range(n_trajectories):
        orders, steps = simulate_ess_trajectory(n_steps=100)
        all_orders.extend(orders)
        all_steps.extend(steps)

    all_orders = np.array(all_orders)
    all_steps = np.array(all_steps)

    # Correlation: as order increases, step size should decrease
    corr, p_corr = stats.spearmanr(all_orders, all_steps)

    # Bin analysis
    order_bins = [
        all_steps[all_orders < 0.3],
        all_steps[(all_orders >= 0.3) & (all_orders < 0.6)],
        all_steps[all_orders >= 0.6]
    ]

    bin_means = [np.mean(b) for b in order_bins if len(b) > 0]

    # Effect size
    if len(order_bins[0]) > 0 and len(order_bins[-1]) > 0:
        pooled_std = np.sqrt((np.std(order_bins[0])**2 + np.std(order_bins[-1])**2) / 2)
        effect_size = (np.mean(order_bins[0]) - np.mean(order_bins[-1])) / (pooled_std + 1e-10)
    else:
        effect_size = 0.0

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Correlation (order vs step size): rho={corr:.3f}, p={p_corr:.2e}")
    print(f"Step sizes by order bin:")
    for i, mean in enumerate(bin_means):
        print(f"  Bin {i}: {mean:.4f}")
    print(f"Effect size (Cohen's d): {effect_size:.2f}")

    validated = corr < -0.5 and p_corr < 0.01
    status = 'validated' if validated else 'refuted'
    print(f"\nSTATUS: {status}")

    return validated, abs(corr), p_corr


if __name__ == '__main__':
    validated, effect_size, p_value = main()
    print(f"\nFinal: rho={effect_size:.3f}, p={p_value:.2e}")
