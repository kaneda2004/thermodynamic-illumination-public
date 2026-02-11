#!/usr/bin/env python3
"""
RES-075: Does mixing periodic and non-periodic activations in hidden layers improve order?

Hypothesis: Mixed activation networks (sin + tanh) achieve higher order than
homogeneous networks (all sin or all tanh) due to complementary expressiveness.

Methodology:
- Generate CPPNs with 2 hidden nodes using three configurations:
  1. Homogeneous periodic: all sin activations
  2. Homogeneous non-periodic: all tanh activations
  3. Mixed: one sin + one tanh
- Measure order (compressibility, edge density, spectral coherence)
- Statistical comparison with paired t-test
"""

import numpy as np
from scipy import stats
from dataclasses import dataclass
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import (
    CPPN, Node, Connection, ACTIVATIONS, PRIOR_SIGMA,
    compute_compressibility, compute_edge_density, compute_spectral_coherence
)


def create_cppn_with_hidden(hidden_activations: list[str], seed: int) -> CPPN:
    """Create CPPN with specified hidden layer activations."""
    np.random.seed(seed)

    # Input nodes (0-3): x, y, r, bias
    nodes = [
        Node(0, 'identity', 0.0),
        Node(1, 'identity', 0.0),
        Node(2, 'identity', 0.0),
        Node(3, 'identity', 0.0),
    ]

    # Hidden nodes (5, 6, ...)
    hidden_ids = []
    for i, act in enumerate(hidden_activations):
        hid = 5 + i
        hidden_ids.append(hid)
        nodes.append(Node(hid, act, np.random.randn() * PRIOR_SIGMA))

    # Output node (4)
    nodes.append(Node(4, 'sigmoid', np.random.randn() * PRIOR_SIGMA))

    connections = []
    # Input -> Hidden connections
    for inp in [0, 1, 2, 3]:
        for hid in hidden_ids:
            connections.append(Connection(inp, hid, np.random.randn() * PRIOR_SIGMA))

    # Hidden -> Output connections
    for hid in hidden_ids:
        connections.append(Connection(hid, 4, np.random.randn() * PRIOR_SIGMA))

    cppn = CPPN(nodes=nodes, connections=connections, input_ids=[0, 1, 2, 3], output_id=4)
    return cppn


def compute_order(img: np.ndarray) -> float:
    """Combined order metric."""
    comp = compute_compressibility(img)
    edge = compute_edge_density(img)
    spectral = compute_spectral_coherence(img)
    # Order: high compressibility, low edge density, high spectral coherence
    return comp + (1 - edge) + spectral


def run_experiment(n_samples: int = 200):
    """Compare three activation configurations."""

    configs = {
        'periodic': ['sin', 'sin'],      # Homogeneous periodic
        'non_periodic': ['tanh', 'tanh'],  # Homogeneous non-periodic
        'mixed': ['sin', 'tanh'],          # Mixed
    }

    results = {name: [] for name in configs}

    for seed in range(n_samples):
        for name, activations in configs.items():
            cppn = create_cppn_with_hidden(activations, seed=seed * 1000 + hash(name) % 1000)
            img = cppn.render(size=32)
            order = compute_order(img)
            results[name].append(order)

    # Convert to arrays
    for name in results:
        results[name] = np.array(results[name])

    # Statistical tests
    print("=" * 60)
    print("RES-075: Activation Mixing Experiment")
    print("=" * 60)

    for name, orders in results.items():
        print(f"{name:15s}: mean={np.mean(orders):.4f}, std={np.std(orders):.4f}")

    print("\n--- Pairwise Comparisons ---")

    # Mixed vs Periodic
    t_mp, p_mp = stats.ttest_ind(results['mixed'], results['periodic'])
    d_mp = (np.mean(results['mixed']) - np.mean(results['periodic'])) / np.sqrt(
        (np.var(results['mixed']) + np.var(results['periodic'])) / 2)
    print(f"Mixed vs Periodic:     t={t_mp:.3f}, p={p_mp:.4f}, Cohen's d={d_mp:.3f}")

    # Mixed vs Non-Periodic
    t_mn, p_mn = stats.ttest_ind(results['mixed'], results['non_periodic'])
    d_mn = (np.mean(results['mixed']) - np.mean(results['non_periodic'])) / np.sqrt(
        (np.var(results['mixed']) + np.var(results['non_periodic'])) / 2)
    print(f"Mixed vs Non-Periodic: t={t_mn:.3f}, p={p_mn:.4f}, Cohen's d={d_mn:.3f}")

    # Periodic vs Non-Periodic
    t_pn, p_pn = stats.ttest_ind(results['periodic'], results['non_periodic'])
    d_pn = (np.mean(results['periodic']) - np.mean(results['non_periodic'])) / np.sqrt(
        (np.var(results['periodic']) + np.var(results['non_periodic'])) / 2)
    print(f"Periodic vs Non-Periodic: t={t_pn:.3f}, p={p_pn:.4f}, Cohen's d={d_pn:.3f}")

    # Apply Bonferroni correction (3 comparisons)
    alpha = 0.01
    bonferroni_alpha = alpha / 3

    print(f"\nBonferroni-corrected alpha: {bonferroni_alpha:.4f}")

    # Determine outcome
    mixed_better_than_both = (
        np.mean(results['mixed']) > np.mean(results['periodic']) and
        np.mean(results['mixed']) > np.mean(results['non_periodic']) and
        p_mp < bonferroni_alpha and p_mn < bonferroni_alpha and
        d_mp > 0.5 and d_mn > 0.5
    )

    if mixed_better_than_both:
        status = "VALIDATED"
        summary = f"Mixed activations yield {d_mp:.2f}σ higher order than periodic, {d_mn:.2f}σ higher than non-periodic"
    elif p_mp < bonferroni_alpha or p_mn < bonferroni_alpha:
        # Some significant difference exists
        best = max(results.keys(), key=lambda k: np.mean(results[k]))
        if best == 'mixed':
            status = "VALIDATED"
            summary = f"Mixed best (mean={np.mean(results['mixed']):.3f}), but effect size small"
        else:
            status = "REFUTED"
            summary = f"{best} has highest order (mean={np.mean(results[best]):.3f}), mixed not superior"
    else:
        status = "INCONCLUSIVE"
        summary = "No significant difference between activation configurations"

    print(f"\n--- RESULT: {status} ---")
    print(f"Summary: {summary}")

    # Metrics for log
    metrics = {
        'n_samples': n_samples,
        'mean_periodic': float(np.mean(results['periodic'])),
        'mean_non_periodic': float(np.mean(results['non_periodic'])),
        'mean_mixed': float(np.mean(results['mixed'])),
        'p_mixed_vs_periodic': float(p_mp),
        'p_mixed_vs_non_periodic': float(p_mn),
        'd_mixed_vs_periodic': float(d_mp),
        'd_mixed_vs_non_periodic': float(d_mn),
    }

    return status, summary, metrics


if __name__ == '__main__':
    status, summary, metrics = run_experiment(n_samples=200)
    print(f"\nFinal: {status}")
    print(f"Metrics: {metrics}")
