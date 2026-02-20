#!/usr/bin/env python3
"""
RES-061: Deep investigation of output activation effect.

The sin activation achieving perfect order seems suspicious.
Let's investigate the actual images being produced.
"""

import numpy as np
from scipy import stats
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import (
    CPPN, Node, Connection, ACTIVATIONS, PRIOR_SIGMA,
    order_multiplicative, compute_compressibility, compute_edge_density,
    compute_spectral_coherence, compute_symmetry
)


def create_minimal_cppn(output_activation: str) -> CPPN:
    """Create minimal CPPN: inputs -> output directly."""
    nodes = [
        Node(0, 'identity', 0.0),
        Node(1, 'identity', 0.0),
        Node(2, 'identity', 0.0),
        Node(3, 'identity', 0.0),
        Node(4, output_activation, np.random.randn() * PRIOR_SIGMA),
    ]
    connections = [
        Connection(0, 4, np.random.randn() * PRIOR_SIGMA),
        Connection(1, 4, np.random.randn() * PRIOR_SIGMA),
        Connection(2, 4, np.random.randn() * PRIOR_SIGMA),
        Connection(3, 4, np.random.randn() * PRIOR_SIGMA),
    ]
    return CPPN(nodes=nodes, connections=connections)


def analyze_image(img: np.ndarray) -> dict:
    """Get detailed metrics for an image."""
    return {
        'order': order_multiplicative(img),
        'density': np.mean(img),
        'compressibility': compute_compressibility(img),
        'edge_density': compute_edge_density(img),
        'coherence': compute_spectral_coherence(img),
        'symmetry': compute_symmetry(img),
    }


def run_investigation():
    """Investigate why sin achieves perfect order."""

    activations = ['sigmoid', 'tanh', 'identity', 'sin', 'gauss', 'abs']

    print("Investigation: Why does sin output achieve perfect order?")
    print("=" * 70)

    # Test with random seeds
    for act in activations:
        print(f"\n{act.upper()} output activation:")
        print("-" * 40)

        densities = []
        orders = []

        for seed in range(10):
            np.random.seed(seed)
            cppn = create_minimal_cppn(act)
            img = cppn.render(32)
            metrics = analyze_image(img)
            densities.append(metrics['density'])
            orders.append(metrics['order'])

            if seed < 3:
                print(f"  Seed {seed}: density={metrics['density']:.3f}, "
                      f"order={metrics['order']:.4f}, "
                      f"compress={metrics['compressibility']:.3f}")

        print(f"  Summary: density={np.mean(densities):.3f}+/-{np.std(densities):.3f}, "
              f"order={np.mean(orders):.4f}+/-{np.std(orders):.4f}")

    # The issue: different activations have different output ranges!
    print("\n" + "=" * 70)
    print("KEY INSIGHT: Output range analysis")
    print("=" * 70)

    x = np.linspace(-3, 3, 100)
    for act in activations:
        y = ACTIVATIONS[act](x)
        print(f"{act:10s}: range=[{np.min(y):.3f}, {np.max(y):.3f}], "
              f"mean={np.mean(y):.3f}, "
              f"P(>0.5)={np.mean(y > 0.5):.3f}")

    # The threshold is 0.5 in render()!
    print("\n" + "=" * 70)
    print("ISSUE: The > 0.5 threshold in render() interacts with output range!")
    print("=" * 70)

    # For fair comparison, we need to normalize or use different thresholds
    print("\nCorrected test: use median threshold per activation")
    print("-" * 70)

    def render_with_adaptive_threshold(cppn: CPPN, size: int = 32) -> np.ndarray:
        """Render with adaptive threshold (median of output)."""
        coords = np.linspace(-1, 1, size)
        x, y = np.meshgrid(coords, coords)
        output = cppn.activate(x, y)
        threshold = np.median(output)
        return (output > threshold).astype(np.uint8)

    for act in activations:
        print(f"\n{act.upper()} with adaptive threshold:")
        orders = []
        densities = []

        for seed in range(30):
            np.random.seed(seed)
            cppn = create_minimal_cppn(act)
            img = render_with_adaptive_threshold(cppn)
            order = order_multiplicative(img)
            orders.append(order)
            densities.append(np.mean(img))

        print(f"  Mean order: {np.mean(orders):.4f} +/- {np.std(orders):.4f}")
        print(f"  Mean density: {np.mean(densities):.4f}")

    # Final rigorous test with adaptive threshold
    print("\n" + "=" * 70)
    print("FINAL TEST: 100 samples with adaptive threshold")
    print("=" * 70)

    results = {act: [] for act in activations}
    n_samples = 100

    for act in activations:
        for i in range(n_samples):
            np.random.seed(2000 + i)
            cppn = create_minimal_cppn(act)
            img = render_with_adaptive_threshold(cppn)
            order = order_multiplicative(img)
            results[act].append(order)

        print(f"{act:10s}: {np.mean(results[act]):.4f} +/- {np.std(results[act]):.4f}")

    # ANOVA on corrected results
    f_stat, p_val = stats.f_oneway(*[results[a] for a in activations])
    print(f"\nANOVA: F={f_stat:.4f}, p={p_val:.6f}")

    # Focus on original three
    orig = ['sigmoid', 'tanh', 'identity']
    f_orig, p_orig = stats.f_oneway(*[results[a] for a in orig])
    print(f"Original three (sigmoid/tanh/identity): F={f_orig:.4f}, p={p_orig:.6f}")

    # Effect size
    means = {a: np.mean(results[a]) for a in activations}
    best = max(means, key=means.get)
    worst = min(means, key=means.get)
    pooled_std = np.sqrt((np.var(results[best]) + np.var(results[worst])) / 2)
    d = abs(means[best] - means[worst]) / (pooled_std + 1e-10)

    print(f"\nBest: {best} ({means[best]:.4f})")
    print(f"Worst: {worst} ({means[worst]:.4f})")
    print(f"Effect size: d={d:.4f}")

    # Verdict
    if p_val < 0.01 and d > 0.5:
        verdict = "VALIDATED"
    elif p_val >= 0.01:
        verdict = "REFUTED"
    else:
        verdict = "INCONCLUSIVE"

    print(f"\nVERDICT: {verdict}")

    return results


if __name__ == '__main__':
    run_investigation()
