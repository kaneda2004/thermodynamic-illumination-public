#!/usr/bin/env python3
"""
RES-061: Test if output activation function affects achievable order in CPPNs.

Hypothesis: Output activation (sigmoid vs tanh vs identity) significantly
affects maximum achievable order in CPPN-generated images.

Methodology:
- Generate N CPPNs with each output activation
- For each CPPN, use hill-climbing to maximize order
- Compare maximum achieved orders across activation types
- Statistical test: ANOVA + pairwise t-tests with Bonferroni correction
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


def create_cppn_with_output_activation(output_activation: str) -> CPPN:
    """Create CPPN with specified output activation."""
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


def hill_climb_order(cppn: CPPN, steps: int = 100, step_size: float = 0.1) -> float:
    """Simple hill climbing to maximize order."""
    weights = cppn.get_weights()
    best_order = order_multiplicative(cppn.render(32))

    for _ in range(steps):
        # Try small perturbation
        new_weights = weights + np.random.randn(len(weights)) * step_size
        cppn.set_weights(new_weights)
        new_order = order_multiplicative(cppn.render(32))

        if new_order > best_order:
            best_order = new_order
            weights = new_weights.copy()
        else:
            cppn.set_weights(weights)

    return best_order


def run_experiment(n_samples: int = 100, hill_climb_steps: int = 200):
    """Run the output activation experiment."""

    # Output activations to test
    activations = ['sigmoid', 'tanh', 'identity']

    results = {act: [] for act in activations}

    print(f"Testing {len(activations)} output activations with {n_samples} samples each")
    print(f"Hill climbing: {hill_climb_steps} steps per sample")
    print("-" * 60)

    for act in activations:
        print(f"\nTesting output activation: {act}")
        for i in range(n_samples):
            np.random.seed(i)  # Reproducible random init
            cppn = create_cppn_with_output_activation(act)
            max_order = hill_climb_order(cppn, steps=hill_climb_steps)
            results[act].append(max_order)
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{n_samples}: mean order = {np.mean(results[act]):.4f}")

    # Statistical analysis
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    for act in activations:
        data = results[act]
        print(f"\n{act}:")
        print(f"  Mean order:   {np.mean(data):.4f} +/- {np.std(data):.4f}")
        print(f"  Max achieved: {np.max(data):.4f}")
        print(f"  Min achieved: {np.min(data):.4f}")

    # ANOVA test
    print("\n" + "-" * 60)
    print("Statistical Tests")
    print("-" * 60)

    f_stat, p_anova = stats.f_oneway(*[results[a] for a in activations])
    print(f"\nOne-way ANOVA: F={f_stat:.4f}, p={p_anova:.6f}")

    # Pairwise t-tests with Bonferroni correction
    n_comparisons = 3  # 3 choose 2 = 3
    alpha = 0.01 / n_comparisons  # Bonferroni correction

    print(f"\nPairwise t-tests (Bonferroni corrected alpha = {alpha:.4f}):")

    pairs = [('sigmoid', 'tanh'), ('sigmoid', 'identity'), ('tanh', 'identity')]
    effect_sizes = {}

    for a1, a2 in pairs:
        t_stat, p_val = stats.ttest_ind(results[a1], results[a2])
        # Cohen's d effect size
        pooled_std = np.sqrt((np.var(results[a1]) + np.var(results[a2])) / 2)
        cohens_d = abs(np.mean(results[a1]) - np.mean(results[a2])) / (pooled_std + 1e-10)
        effect_sizes[(a1, a2)] = cohens_d

        sig = "***" if p_val < alpha else ""
        print(f"  {a1} vs {a2}: t={t_stat:.4f}, p={p_val:.6f}, d={cohens_d:.4f} {sig}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Find best and worst
    means = {act: np.mean(results[act]) for act in activations}
    best_act = max(means, key=means.get)
    worst_act = min(means, key=means.get)

    # Overall effect size: best vs worst
    overall_d = effect_sizes.get((best_act, worst_act)) or effect_sizes.get((worst_act, best_act))

    print(f"\nBest activation:  {best_act} (mean = {means[best_act]:.4f})")
    print(f"Worst activation: {worst_act} (mean = {means[worst_act]:.4f})")
    print(f"Overall effect size (Cohen's d): {overall_d:.4f}")
    print(f"ANOVA p-value: {p_anova:.6f}")

    # Verdict
    if p_anova < 0.01 and overall_d > 0.5:
        verdict = "VALIDATED"
        reason = f"Significant difference (p={p_anova:.6f}) with large effect (d={overall_d:.2f})"
    elif p_anova < 0.01 and overall_d > 0.2:
        verdict = "VALIDATED"
        reason = f"Significant difference (p={p_anova:.6f}) with medium effect (d={overall_d:.2f})"
    elif p_anova >= 0.01:
        verdict = "REFUTED"
        reason = f"No significant difference (p={p_anova:.6f})"
    else:
        verdict = "INCONCLUSIVE"
        reason = f"Significant but small effect (p={p_anova:.6f}, d={overall_d:.2f})"

    print(f"\nVERDICT: {verdict}")
    print(f"Reason: {reason}")

    return {
        'results': results,
        'means': means,
        'p_anova': p_anova,
        'effect_size': overall_d,
        'verdict': verdict,
        'best_activation': best_act,
        'worst_activation': worst_act
    }


if __name__ == '__main__':
    np.random.seed(42)
    run_experiment(n_samples=100, hill_climb_steps=200)
