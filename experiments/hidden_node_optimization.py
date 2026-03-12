#!/usr/bin/env python3
"""
RES-074: Optimal Hidden Node Count for Order Generation

Hypothesis: There exists an optimal number of hidden nodes for generating high-order
images. Too few nodes limits expressiveness; too many adds noise without benefit.

Methodology:
- Create CPPNs with varying hidden node counts (0, 1, 2, 4, 8, 16, 32)
- Sample N images from each configuration
- Measure order statistics (mean, max, variance)
- Test for diminishing returns via regression analysis
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from dataclasses import dataclass, field
from core.thermo_sampler_v3 import (
    CPPN, Node, Connection, order_multiplicative, ACTIVATIONS, PRIOR_SIGMA
)
from scipy import stats
import random


def create_cppn_with_hidden_nodes(n_hidden: int) -> CPPN:
    """Create a CPPN with specified number of hidden nodes."""
    # Input nodes: x, y, r, bias (ids 0-3)
    # Output node: id 4
    # Hidden nodes: ids 5, 6, ...

    nodes = [
        Node(0, 'identity', 0.0),  # x
        Node(1, 'identity', 0.0),  # y
        Node(2, 'identity', 0.0),  # r
        Node(3, 'identity', 0.0),  # bias
        Node(4, 'sigmoid', np.random.randn() * PRIOR_SIGMA),  # output
    ]

    connections = []

    # Add hidden nodes with random activations
    activations = list(ACTIVATIONS.keys())
    hidden_ids = []
    for i in range(n_hidden):
        hid = 5 + i
        hidden_ids.append(hid)
        act = random.choice(activations)
        nodes.append(Node(hid, act, np.random.randn() * PRIOR_SIGMA))

    if n_hidden == 0:
        # Direct connections from inputs to output
        for inp in [0, 1, 2, 3]:
            connections.append(Connection(inp, 4, np.random.randn() * PRIOR_SIGMA))
    else:
        # Connections: inputs -> hidden layer(s) -> output
        # For simplicity, use a layered architecture
        if n_hidden <= 4:
            # Single hidden layer
            for inp in [0, 1, 2, 3]:
                for hid in hidden_ids:
                    if random.random() < 0.7:  # 70% connection probability
                        connections.append(Connection(inp, hid, np.random.randn() * PRIOR_SIGMA))
            for hid in hidden_ids:
                connections.append(Connection(hid, 4, np.random.randn() * PRIOR_SIGMA))
        else:
            # Two hidden layers for larger networks
            layer1 = hidden_ids[:len(hidden_ids)//2]
            layer2 = hidden_ids[len(hidden_ids)//2:]

            # Input -> layer1
            for inp in [0, 1, 2, 3]:
                for hid in layer1:
                    if random.random() < 0.5:
                        connections.append(Connection(inp, hid, np.random.randn() * PRIOR_SIGMA))

            # Layer1 -> layer2
            for h1 in layer1:
                for h2 in layer2:
                    if random.random() < 0.5:
                        connections.append(Connection(h1, h2, np.random.randn() * PRIOR_SIGMA))

            # Layer2 -> output
            for hid in layer2:
                connections.append(Connection(hid, 4, np.random.randn() * PRIOR_SIGMA))

    return CPPN(nodes=nodes, connections=connections)


def sample_order_for_hidden_count(n_hidden: int, n_samples: int = 200, image_size: int = 32) -> dict:
    """Sample images from CPPNs with given hidden node count and measure order."""
    orders = []

    for _ in range(n_samples):
        cppn = create_cppn_with_hidden_nodes(n_hidden)
        img = cppn.render(image_size)
        order = order_multiplicative(img)
        orders.append(order)

    orders = np.array(orders)
    return {
        'n_hidden': n_hidden,
        'mean': float(np.mean(orders)),
        'std': float(np.std(orders)),
        'max': float(np.max(orders)),
        'median': float(np.median(orders)),
        'p75': float(np.percentile(orders, 75)),
        'p90': float(np.percentile(orders, 90)),
        'orders': orders
    }


def main():
    print("=" * 70)
    print("RES-074: Optimal Hidden Node Count for Order Generation")
    print("=" * 70)

    np.random.seed(42)
    random.seed(42)

    # Test hidden node counts
    hidden_counts = [0, 1, 2, 4, 8, 16, 32]
    n_samples = 300

    results = []
    print(f"\nSampling {n_samples} images per hidden node count...")
    print("-" * 70)

    for n_hidden in hidden_counts:
        print(f"Testing n_hidden={n_hidden}...", end=" ", flush=True)
        result = sample_order_for_hidden_count(n_hidden, n_samples)
        results.append(result)
        print(f"mean={result['mean']:.4f}, max={result['max']:.4f}, p90={result['p90']:.4f}")

    # Analysis
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'n_hidden':>10} | {'Mean':>8} | {'Std':>8} | {'Max':>8} | {'P90':>8}")
    print("-" * 50)

    for r in results:
        print(f"{r['n_hidden']:>10} | {r['mean']:>8.4f} | {r['std']:>8.4f} | {r['max']:>8.4f} | {r['p90']:>8.4f}")

    # Statistical tests
    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS")
    print("=" * 70)

    # 1. ANOVA test across all groups
    all_orders = [r['orders'] for r in results]
    f_stat, p_anova = stats.f_oneway(*all_orders)
    print(f"\nANOVA across all groups: F={f_stat:.2f}, p={p_anova:.2e}")

    # 2. Find optimal hidden count (highest mean order)
    means = [r['mean'] for r in results]
    optimal_idx = np.argmax(means)
    optimal_hidden = results[optimal_idx]['n_hidden']
    print(f"\nOptimal hidden count by mean order: {optimal_hidden}")

    # 3. Test for diminishing returns using log regression
    # Fit: order ~ log(n_hidden + 1)
    x_log = np.log(np.array(hidden_counts) + 1)
    y_means = np.array(means)

    slope, intercept, r_value, p_value, std_err = stats.linregress(x_log, y_means)
    print(f"\nLog regression (order ~ log(n_hidden+1)):")
    print(f"  Slope: {slope:.4f} (positive = diminishing returns pattern)")
    print(f"  R^2: {r_value**2:.4f}")
    print(f"  p-value: {p_value:.2e}")

    # 4. Pairwise comparisons with baseline (n_hidden=0)
    print("\nPairwise t-tests vs baseline (n_hidden=0):")
    baseline_orders = results[0]['orders']
    for r in results[1:]:
        t_stat, p_val = stats.ttest_ind(r['orders'], baseline_orders)
        effect_size = (r['mean'] - results[0]['mean']) / np.sqrt((r['std']**2 + results[0]['std']**2) / 2)
        sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else ""))
        print(f"  n_hidden={r['n_hidden']:>2}: d={effect_size:>6.3f}, p={p_val:.2e} {sig}")

    # 5. Test for plateau/peak pattern
    # Check if there's a significant drop after optimal
    print("\nPlateau/Peak Analysis:")
    if optimal_idx > 0 and optimal_idx < len(results) - 1:
        before = results[optimal_idx - 1]['orders']
        at_opt = results[optimal_idx]['orders']
        after = results[optimal_idx + 1]['orders']

        t_before, p_before = stats.ttest_ind(at_opt, before)
        t_after, p_after = stats.ttest_ind(at_opt, after)

        print(f"  Optimal ({optimal_hidden}) vs before ({results[optimal_idx-1]['n_hidden']}): t={t_before:.2f}, p={p_before:.3f}")
        print(f"  Optimal ({optimal_hidden}) vs after ({results[optimal_idx+1]['n_hidden']}): t={t_after:.2f}, p={p_after:.3f}")

    # Determine verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    # Check effect sizes for hidden nodes vs no hidden
    max_effect = max([
        (r['mean'] - results[0]['mean']) / np.sqrt((r['std']**2 + results[0]['std']**2) / 2)
        for r in results[1:]
    ])

    if p_anova < 0.01 and max_effect > 0.5:
        # Significant difference and meaningful effect
        if slope > 0 and r_value**2 > 0.5:
            verdict = "VALIDATED"
            summary = f"Optimal hidden count exists with diminishing returns. Peak at {optimal_hidden} nodes, log-linear fit R^2={r_value**2:.3f}."
        else:
            verdict = "VALIDATED"
            summary = f"Optimal hidden count is {optimal_hidden}. Effect size d={max_effect:.2f} vs baseline."
    elif p_anova < 0.05:
        verdict = "INCONCLUSIVE"
        summary = f"Weak evidence for optimal hidden count. ANOVA p={p_anova:.3f}, effect size d={max_effect:.2f}."
    else:
        verdict = "REFUTED"
        summary = f"No significant difference across hidden counts. ANOVA p={p_anova:.3f}."

    print(f"\nStatus: {verdict}")
    print(f"Summary: {summary}")

    # Metrics for log
    metrics = {
        'optimal_hidden': int(optimal_hidden),
        'anova_f': float(f_stat),
        'anova_p': float(p_anova),
        'max_effect_size': float(max_effect),
        'log_slope': float(slope),
        'log_r2': float(r_value**2),
        'n_samples': n_samples
    }

    print(f"\nMetrics: {metrics}")

    return verdict, summary, metrics


if __name__ == "__main__":
    main()
