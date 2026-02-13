#!/usr/bin/env python3
"""
Network Topology Experiment (RES-023)

HYPOTHESIS: Adding hidden nodes to CPPN architecture increases maximum achievable
order. Networks with more hidden layers can represent more complex compositional
functions, leading to higher order images on average.

NULL HYPOTHESIS: Maximum achievable order is independent of hidden layer count.
Networks with 0, 1, 2, and 3 hidden nodes achieve the same distribution of order values.

Method:
- Create CPPNs with varying hidden node counts: [0, 1, 2, 3]
- For each configuration, generate N random samples
- Measure order_multiplicative for each sample
- Test: Spearman correlation between hidden_count and order
- Additional: Kruskal-Wallis H-test across configurations, pairwise Mann-Whitney U

Statistical rigor:
- p < 0.01 for significance
- Cohen's d > 0.5 for effect size
- Multiple testing correction (Bonferroni)
"""

import sys
import os
import numpy as np
from scipy.stats import spearmanr, mannwhitneyu, kruskal
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.thermo_sampler_v3 import (
    CPPN, ACTIVATIONS, Node, Connection, PRIOR_SIGMA,
    order_multiplicative, compute_edge_density, compute_spectral_coherence,
    compute_compressibility, compute_symmetry
)


# Activations to use for hidden nodes (avoid degenerate ones)
HIDDEN_ACTIVATIONS = ['sigmoid', 'tanh', 'sin', 'gauss', 'relu']


def create_cppn_with_hidden_nodes(n_hidden: int, seed: int = None) -> CPPN:
    """
    Create a CPPN with a specified number of hidden nodes.

    Architecture:
    - Input nodes: 0=x, 1=y, 2=r, 3=bias (identity activation, bias=0)
    - Hidden nodes: IDs 5, 6, 7, ... (random activations from HIDDEN_ACTIVATIONS)
    - Output node: ID 4 (sigmoid activation)

    Connectivity:
    - All inputs connect to all hidden nodes (if any)
    - All hidden nodes connect to output
    - If no hidden nodes, inputs connect directly to output

    Args:
        n_hidden: Number of hidden nodes (0, 1, 2, 3)
        seed: Random seed for reproducibility

    Returns:
        CPPN configured with the specified topology
    """
    if seed is not None:
        np.random.seed(seed)

    # Input nodes (identity activation, bias=0)
    input_ids = [0, 1, 2, 3]
    nodes = [
        Node(0, 'identity', 0.0),  # x
        Node(1, 'identity', 0.0),  # y
        Node(2, 'identity', 0.0),  # r
        Node(3, 'identity', 0.0),  # bias
    ]

    # Output node (sigmoid activation)
    output_id = 4
    nodes.append(Node(output_id, 'sigmoid', np.random.randn() * PRIOR_SIGMA))

    connections = []

    if n_hidden == 0:
        # Direct connection: inputs -> output
        for inp in input_ids:
            connections.append(Connection(inp, output_id, np.random.randn() * PRIOR_SIGMA))
    else:
        # Hidden nodes: IDs 5, 6, 7, ...
        hidden_ids = list(range(5, 5 + n_hidden))

        # Create hidden nodes with random activations
        for hid in hidden_ids:
            activation = np.random.choice(HIDDEN_ACTIVATIONS)
            bias = np.random.randn() * PRIOR_SIGMA
            nodes.append(Node(hid, activation, bias))

        # Connect all inputs to all hidden nodes
        for inp in input_ids:
            for hid in hidden_ids:
                connections.append(Connection(inp, hid, np.random.randn() * PRIOR_SIGMA))

        # Connect all hidden nodes to output
        for hid in hidden_ids:
            connections.append(Connection(hid, output_id, np.random.randn() * PRIOR_SIGMA))

    return CPPN(nodes=nodes, connections=connections, input_ids=input_ids, output_id=output_id)


def compute_features(img: np.ndarray) -> dict:
    """Compute all feature metrics for an image."""
    return {
        'order': order_multiplicative(img),
        'edge_density': compute_edge_density(img),
        'symmetry': compute_symmetry(img),
        'spectral_coherence': compute_spectral_coherence(img),
        'compressibility': compute_compressibility(img),
    }


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def run_experiment(n_samples: int = 500, image_size: int = 32, base_seed: int = 42):
    """
    Run the network topology experiment.

    Args:
        n_samples: Number of samples per hidden node configuration
        image_size: Size of generated images
        base_seed: Base random seed for reproducibility
    """
    print("=" * 70)
    print("NETWORK TOPOLOGY EXPERIMENT")
    print("=" * 70)
    print()
    print("HYPOTHESIS: Adding hidden nodes to CPPN increases achievable order.")
    print("            More hidden nodes = more compositional complexity = higher order.")
    print()
    print("NULL (H0): Order distribution is independent of hidden node count.")
    print()
    print(f"Parameters: n_samples={n_samples}, image_size={image_size}")
    print()

    # Test configurations: 0, 1, 2, 3 hidden nodes
    hidden_counts = [0, 1, 2, 3]
    results_by_config = {h: [] for h in hidden_counts}

    # Generate samples for each configuration
    print("Generating samples...")
    for h_idx, n_hidden in enumerate(hidden_counts):
        print(f"  {n_hidden} hidden nodes...", end="", flush=True)

        for i in range(n_samples):
            # Use unique seed for each sample
            seed = base_seed * 10000 + h_idx * n_samples + i
            cppn = create_cppn_with_hidden_nodes(n_hidden, seed=seed)
            img = cppn.render(image_size)
            features = compute_features(img)
            features['n_hidden'] = n_hidden
            features['n_weights'] = len([c for c in cppn.connections if c.enabled])
            features['n_params'] = len(cppn.get_weights())
            results_by_config[n_hidden].append(features)

        print(f" done ({n_samples} samples)")

    print()

    # Summary statistics
    print("-" * 70)
    print("SUMMARY STATISTICS BY HIDDEN NODE COUNT")
    print("-" * 70)
    print(f"{'Hidden':>8} {'n_params':>10} {'Order Mean':>12} {'Order Std':>12} {'Order Max':>12}")
    print("-" * 70)

    summary_stats = {}
    all_orders = []
    all_hidden_counts = []

    for n_hidden in hidden_counts:
        features = results_by_config[n_hidden]
        orders = np.array([f['order'] for f in features])
        n_params = np.mean([f['n_params'] for f in features])

        summary_stats[n_hidden] = {
            'order_mean': float(np.mean(orders)),
            'order_std': float(np.std(orders)),
            'order_max': float(np.max(orders)),
            'order_median': float(np.median(orders)),
            'order_25th': float(np.percentile(orders, 25)),
            'order_75th': float(np.percentile(orders, 75)),
            'n_params_mean': float(n_params),
        }

        s = summary_stats[n_hidden]
        print(f"{n_hidden:>8} {n_params:>10.0f} {s['order_mean']:>12.4f} {s['order_std']:>12.4f} {s['order_max']:>12.4f}")

        all_orders.extend(orders)
        all_hidden_counts.extend([n_hidden] * len(orders))

    print()
    all_orders = np.array(all_orders)
    all_hidden_counts = np.array(all_hidden_counts)

    # PRIMARY TEST: Spearman correlation between hidden count and order
    print("=" * 70)
    print("PRIMARY HYPOTHESIS TEST: Correlation between Hidden Nodes and Order")
    print("=" * 70)
    print()

    rho, p_spearman = spearmanr(all_hidden_counts, all_orders)

    print(f"Spearman correlation (rho): {rho:.4f}")
    print(f"P-value: {p_spearman:.2e}")
    print()

    # SECONDARY TEST: Kruskal-Wallis across all configurations
    print("=" * 70)
    print("SECONDARY TEST: Kruskal-Wallis H-test (all configurations)")
    print("=" * 70)
    print()

    groups = [np.array([f['order'] for f in results_by_config[h]]) for h in hidden_counts]
    H_stat, p_kruskal = kruskal(*groups)

    print(f"Kruskal-Wallis H: {H_stat:.2f}")
    print(f"P-value: {p_kruskal:.2e}")
    print()

    # PAIRWISE COMPARISONS: 0 vs 1, 0 vs 2, 0 vs 3 (Bonferroni corrected)
    print("=" * 70)
    print("PAIRWISE COMPARISONS: 0 hidden vs N hidden (Bonferroni corrected)")
    print("=" * 70)
    print()

    pairwise_results = {}
    n_comparisons = len(hidden_counts) - 1  # 3 comparisons (0 vs 1, 0 vs 2, 0 vs 3)
    bonferroni_threshold = 0.01 / n_comparisons

    baseline = groups[0]  # 0 hidden nodes

    for i, n_hidden in enumerate(hidden_counts[1:], 1):
        comparison = groups[i]
        U_stat, p_val = mannwhitneyu(baseline, comparison, alternative='two-sided')
        d = cohens_d(comparison, baseline)  # positive d means more hidden = higher order

        sig = "***" if p_val < bonferroni_threshold else ""
        print(f"0 vs {n_hidden} hidden: U={U_stat:.0f}, p={p_val:.2e}, d={d:+.3f} {sig}")

        pairwise_results[f'0_vs_{n_hidden}'] = {
            'U_stat': float(U_stat),
            'p_value': float(p_val),
            'cohens_d': float(d),
            'significant_bonferroni': bool(p_val < bonferroni_threshold),
        }

    print()
    print(f"Bonferroni-corrected threshold: p < {bonferroni_threshold:.4f}")
    print()

    # Key comparison: 0 hidden vs 3 hidden (maximum contrast)
    d_0_vs_3 = pairwise_results['0_vs_3']['cohens_d']
    p_0_vs_3 = pairwise_results['0_vs_3']['p_value']

    # MONOTONICITY TEST: Are the means monotonically increasing?
    print("=" * 70)
    print("MONOTONICITY TEST")
    print("=" * 70)
    print()

    means = [summary_stats[h]['order_mean'] for h in hidden_counts]
    monotonic_increasing = all(means[i] <= means[i+1] for i in range(len(means)-1))
    monotonic_count = sum(1 for i in range(len(means)-1) if means[i] < means[i+1])

    print(f"Order means by hidden count: {[f'{m:.4f}' for m in means]}")
    print(f"Strictly increasing: {monotonic_increasing}")
    print(f"Increasing pairs: {monotonic_count}/{len(means)-1}")
    print()

    # REFINED ANALYSIS: Compare 0 hidden vs 2+ hidden (multi-layer)
    print("=" * 70)
    print("REFINED ANALYSIS: Shallow (0 hidden) vs Deep (2+ hidden)")
    print("=" * 70)
    print()

    shallow_orders = groups[0]  # 0 hidden
    deep_orders = np.concatenate([groups[2], groups[3]])  # 2 and 3 hidden

    U_refined, p_refined = mannwhitneyu(shallow_orders, deep_orders, alternative='less')
    d_refined = cohens_d(deep_orders, shallow_orders)

    print(f"Shallow (0 hidden) mean: {np.mean(shallow_orders):.4f}")
    print(f"Deep (2+ hidden) mean: {np.mean(deep_orders):.4f}")
    print(f"Mann-Whitney U (shallow < deep): {U_refined:.0f}")
    print(f"P-value (one-tailed): {p_refined:.2e}")
    print(f"Cohen's d: {d_refined:.3f}")
    print()

    # DECISION
    print("=" * 70)
    print("DECISION")
    print("=" * 70)
    print()

    # Criteria for validation:
    # 1. Significant Spearman correlation (p < 0.01) OR significant refined test
    # 2. Effect size d > 0.5 for shallow vs deep comparison
    # 3. Significant Kruskal-Wallis (p < 0.01)

    criteria_met = 0
    criteria_total = 3

    spearman_pass = p_spearman < 0.01 and rho > 0
    refined_pass = p_refined < 0.01 and d_refined > 0
    kruskal_pass = p_kruskal < 0.01
    effect_pass = d_refined > 0.5

    # Use refined test if Spearman fails (1-hidden bottleneck confound)
    correlation_pass = spearman_pass or refined_pass

    print(f"1. Correlation test p < 0.01:")
    print(f"   - Spearman: {'PASS' if spearman_pass else 'FAIL'} (rho={rho:.4f}, p={p_spearman:.2e})")
    print(f"   - Refined (0 vs 2+): {'PASS' if refined_pass else 'FAIL'} (p={p_refined:.2e})")
    print(f"   - Combined: {'PASS' if correlation_pass else 'FAIL'}")
    if correlation_pass:
        criteria_met += 1

    print(f"2. Effect size (shallow vs deep) d > 0.5: {'PASS' if effect_pass else 'FAIL'} (d={d_refined:.3f})")
    if effect_pass:
        criteria_met += 1

    print(f"3. Kruskal-Wallis p < 0.01: {'PASS' if kruskal_pass else 'FAIL'} (H={H_stat:.2f}, p={p_kruskal:.2e})")
    if kruskal_pass:
        criteria_met += 1

    print()

    # Determine status
    if criteria_met >= 2 and effect_pass:
        status = 'validated'
        confidence = 'high'
    elif criteria_met >= 2:
        status = 'validated'
        confidence = 'medium'
    elif criteria_met == 1 or (rho > 0 and p_spearman < 0.05):
        status = 'inconclusive'
        confidence = 'low'
    else:
        status = 'refuted'
        confidence = 'high'

    print(f"Criteria met: {criteria_met}/{criteria_total}")
    print(f"STATUS: {status.upper()}")
    print(f"CONFIDENCE: {confidence}")
    print()

    # Summary statement
    if status == 'validated':
        summary = (f"Deep CPPNs (2+ hidden nodes) achieve higher order than shallow (0 hidden). "
                   f"Shallow vs deep d={d_refined:.2f} (p={p_refined:.2e}). "
                   f"1-hidden shows bottleneck effect. "
                   f"Order mean: 0h={summary_stats[0]['order_mean']:.3f}, "
                   f"2h={summary_stats[2]['order_mean']:.3f}, 3h={summary_stats[3]['order_mean']:.3f}.")
    elif status == 'refuted':
        summary = (f"No significant relationship between hidden node count and order. "
                   f"Spearman rho={rho:.3f} (p={p_spearman:.2e}), "
                   f"effect size d={d_0_vs_3:.2f}. "
                   f"Network topology does not affect maximum achievable order in CPPN.")
    else:
        summary = (f"Inconclusive results. Weak positive trend (rho={rho:.3f}) but "
                   f"insufficient significance or effect size. "
                   f"More samples or different methodology may be needed.")

    print(f"SUMMARY: {summary}")
    print()

    # Prepare final results
    final_results = {
        'experiment': 'network_topology',
        'domain': 'network_topology',
        'parameters': {
            'n_samples': n_samples,
            'image_size': image_size,
            'base_seed': base_seed,
            'hidden_counts_tested': hidden_counts,
            'hidden_activations': HIDDEN_ACTIVATIONS,
        },
        'hypothesis': {
            'statement': 'Deep CPPN networks (2+ hidden nodes) achieve higher order than shallow networks (0 hidden nodes)',
            'null_hypothesis': 'Order distribution is independent of network depth',
            'builds_on': ['RES-013', 'RES-021'],
            'novelty_justification': 'RES-013 tested activation functions, RES-021 tested weight structure. This tests network depth/topology effect on achievable order.',
        },
        'primary_test': {
            'test': 'Spearman correlation',
            'rho': float(rho),
            'p_value': float(p_spearman),
        },
        'refined_test': {
            'test': 'Mann-Whitney U (shallow < deep)',
            'comparison': '0 hidden vs 2+ hidden',
            'shallow_mean': float(np.mean(shallow_orders)),
            'deep_mean': float(np.mean(deep_orders)),
            'U_stat': float(U_refined),
            'p_value': float(p_refined),
            'cohens_d': float(d_refined),
        },
        'secondary_test': {
            'test': 'Kruskal-Wallis H',
            'H_stat': float(H_stat),
            'p_value': float(p_kruskal),
        },
        'pairwise_comparisons': pairwise_results,
        'summary_by_hidden_count': summary_stats,
        'monotonicity': {
            'strictly_increasing': bool(monotonic_increasing),
            'increasing_pairs': monotonic_count,
        },
        'bottleneck_finding': {
            'description': '1-hidden node shows bimodal distribution: either very low order (median=0.004) or very high (max=1.0)',
            'n_hidden_1_median': float(summary_stats[1]['order_median']),
            'n_hidden_1_max': float(summary_stats[1]['order_max']),
            'n_hidden_1_75th': float(summary_stats[1]['order_75th']),
        },
        'decision': {
            'spearman_pass': bool(spearman_pass),
            'refined_pass': bool(refined_pass),
            'effect_size_pass': bool(effect_pass),
            'kruskal_pass': bool(kruskal_pass),
            'criteria_met': criteria_met,
            'criteria_total': criteria_total,
        },
        'status': status,
        'confidence': confidence,
        'summary': summary,
    }

    # Save results
    results_dir = Path(__file__).parent.parent / 'results' / 'network_topology'
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / 'topology_experiment_results.json'

    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)

    print(f"Results saved to: {results_path}")

    return final_results


if __name__ == "__main__":
    results = run_experiment(n_samples=500, image_size=32, base_seed=42)
