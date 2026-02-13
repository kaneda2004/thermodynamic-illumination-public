#!/usr/bin/env python3
"""
RES-061: Extended test of output activation effect.

Testing more activations and more complex CPPNs (with hidden nodes).
"""

import numpy as np
from scipy import stats
from dataclasses import dataclass
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import (
    CPPN, Node, Connection, ACTIVATIONS, PRIOR_SIGMA,
    order_multiplicative
)


def create_cppn_with_hidden_and_output(output_activation: str, n_hidden: int = 2) -> CPPN:
    """Create CPPN with hidden nodes and specified output activation."""
    # Input nodes: 0,1,2,3 (x,y,r,bias)
    # Hidden nodes: 5,6,...
    # Output: 4

    nodes = [
        Node(0, 'identity', 0.0),
        Node(1, 'identity', 0.0),
        Node(2, 'identity', 0.0),
        Node(3, 'identity', 0.0),
        Node(4, output_activation, np.random.randn() * PRIOR_SIGMA),
    ]

    connections = []

    # Add hidden nodes with periodic activations
    hidden_acts = ['sin', 'gauss', 'cos', 'tanh', 'abs']
    for i in range(n_hidden):
        hidden_id = 5 + i
        act = hidden_acts[i % len(hidden_acts)]
        nodes.append(Node(hidden_id, act, np.random.randn() * PRIOR_SIGMA))

        # Connect inputs to hidden
        for inp in [0, 1, 2, 3]:
            if np.random.rand() > 0.3:  # 70% connection probability
                connections.append(Connection(inp, hidden_id, np.random.randn() * PRIOR_SIGMA))

        # Connect hidden to output
        connections.append(Connection(hidden_id, 4, np.random.randn() * PRIOR_SIGMA))

    # Also some direct input-to-output connections
    for inp in [0, 1, 2, 3]:
        if np.random.rand() > 0.5:
            connections.append(Connection(inp, 4, np.random.randn() * PRIOR_SIGMA))

    return CPPN(nodes=nodes, connections=connections)


def hill_climb_order(cppn: CPPN, steps: int = 100, step_size: float = 0.1) -> float:
    """Simple hill climbing to maximize order."""
    weights = cppn.get_weights()
    best_order = order_multiplicative(cppn.render(32))

    for _ in range(steps):
        new_weights = weights + np.random.randn(len(weights)) * step_size
        cppn.set_weights(new_weights)
        new_order = order_multiplicative(cppn.render(32))

        if new_order > best_order:
            best_order = new_order
            weights = new_weights.copy()
        else:
            cppn.set_weights(weights)

    return best_order


def run_experiment():
    """Run extended output activation experiment."""

    # More output activations to test
    activations = ['sigmoid', 'tanh', 'identity', 'sin', 'gauss', 'abs', 'relu']

    n_samples = 50
    hill_climb_steps = 300
    n_hidden = 3

    results = {act: [] for act in activations}

    print(f"Testing {len(activations)} output activations")
    print(f"CPPN architecture: 4 inputs -> {n_hidden} hidden -> 1 output")
    print(f"Samples per activation: {n_samples}")
    print(f"Hill climbing steps: {hill_climb_steps}")
    print("-" * 60)

    for act in activations:
        print(f"\nTesting output activation: {act}")
        for i in range(n_samples):
            np.random.seed(1000 + i)  # Different seed range
            cppn = create_cppn_with_hidden_and_output(act, n_hidden=n_hidden)
            max_order = hill_climb_order(cppn, steps=hill_climb_steps)
            results[act].append(max_order)

        print(f"  Mean: {np.mean(results[act]):.4f} +/- {np.std(results[act]):.4f}")

    # Analysis
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    for act in activations:
        data = results[act]
        print(f"{act:10s}: {np.mean(data):.4f} +/- {np.std(data):.4f} (max={np.max(data):.4f})")

    # ANOVA
    f_stat, p_anova = stats.f_oneway(*[results[a] for a in activations])
    print(f"\nANOVA: F={f_stat:.4f}, p={p_anova:.6f}")

    # Find best and worst
    means = {act: np.mean(results[act]) for act in activations}
    best_act = max(means, key=means.get)
    worst_act = min(means, key=means.get)

    # Effect size best vs worst
    pooled_std = np.sqrt((np.var(results[best_act]) + np.var(results[worst_act])) / 2)
    cohens_d = abs(means[best_act] - means[worst_act]) / (pooled_std + 1e-10)

    t_stat, p_best_worst = stats.ttest_ind(results[best_act], results[worst_act])

    print(f"\nBest: {best_act} (mean={means[best_act]:.4f})")
    print(f"Worst: {worst_act} (mean={means[worst_act]:.4f})")
    print(f"Effect size (best vs worst): d={cohens_d:.4f}")
    print(f"T-test (best vs worst): p={p_best_worst:.6f}")

    # Focus on the original three: sigmoid, tanh, identity
    print("\n" + "-" * 60)
    print("Original hypothesis (sigmoid vs tanh vs identity):")
    orig = ['sigmoid', 'tanh', 'identity']
    f_orig, p_orig = stats.f_oneway(*[results[a] for a in orig])
    print(f"ANOVA: F={f_orig:.4f}, p={p_orig:.6f}")

    # Verdict
    if p_anova < 0.01 and cohens_d > 0.5:
        verdict = "VALIDATED"
    elif p_anova < 0.01 and cohens_d > 0.2:
        verdict = "VALIDATED (medium effect)"
    elif p_anova >= 0.01:
        verdict = "REFUTED"
    else:
        verdict = "INCONCLUSIVE"

    print(f"\nVERDICT: {verdict}")
    print(f"ANOVA p={p_anova:.6f}, effect size d={cohens_d:.4f}")

    return {
        'means': means,
        'p_anova': p_anova,
        'effect_size': cohens_d,
        'verdict': verdict
    }


if __name__ == '__main__':
    np.random.seed(42)
    run_experiment()
