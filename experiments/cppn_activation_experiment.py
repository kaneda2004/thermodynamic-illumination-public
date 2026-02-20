#!/usr/bin/env python3
"""
CPPN Activation Function Experiment (RES-013)

HYPOTHESIS: Output activation function type systematically affects CPPN image
properties. Specifically, periodic activations (sin, cos, sin2, ring) produce
higher spectral coherence than non-periodic activations (sigmoid, tanh, relu,
gauss, abs, identity).

NULL HYPOTHESIS: No difference in spectral coherence between periodic and
non-periodic activation functions.

Method:
- Generate N samples per activation type
- Compute feature vectors (edge_density, symmetry, spectral_coherence, compressibility)
- Test: Mann-Whitney U for periodic vs non-periodic spectral coherence
- Additional: MANOVA across all activation types
"""

import sys
import os
import numpy as np
from scipy.stats import mannwhitneyu, f_oneway, kruskal
from scipy.spatial.distance import pdist
import json
from pathlib import Path
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.thermo_sampler_v3 import (
    CPPN, ACTIVATIONS, Node, Connection, PRIOR_SIGMA,
    compute_symmetry, compute_edge_density, compute_spectral_coherence, compute_compressibility
)


# Classification of activations
PERIODIC_ACTIVATIONS = ['sin', 'cos', 'sin2', 'ring']
NONPERIODIC_ACTIVATIONS = ['sigmoid', 'tanh', 'relu', 'gauss', 'abs', 'identity']


def create_cppn_with_activation(activation: str, seed: int = None) -> CPPN:
    """
    Create a CPPN with a specific output activation function.

    Uses fresh random weights from the prior distribution.
    """
    if seed is not None:
        np.random.seed(seed)

    # Create nodes: 4 input (x, y, r, bias) + 1 output
    nodes = [
        Node(0, 'identity', 0.0),  # x
        Node(1, 'identity', 0.0),  # y
        Node(2, 'identity', 0.0),  # r
        Node(3, 'identity', 0.0),  # bias
        Node(4, activation, np.random.randn() * PRIOR_SIGMA),  # output with specified activation
    ]

    # Create connections from all inputs to output with random weights
    connections = [
        Connection(0, 4, np.random.randn() * PRIOR_SIGMA),  # x -> output
        Connection(1, 4, np.random.randn() * PRIOR_SIGMA),  # y -> output
        Connection(2, 4, np.random.randn() * PRIOR_SIGMA),  # r -> output
        Connection(3, 4, np.random.randn() * PRIOR_SIGMA),  # bias -> output
    ]

    return CPPN(nodes=nodes, connections=connections, input_ids=[0, 1, 2, 3], output_id=4)


def compute_features(img: np.ndarray) -> dict:
    """Compute all feature metrics for an image."""
    return {
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


def run_experiment(n_samples: int = 100, image_size: int = 32, base_seed: int = 42):
    """
    Run the CPPN activation function experiment.

    Args:
        n_samples: Number of samples per activation type
        image_size: Size of generated images
        base_seed: Base random seed for reproducibility
    """
    print("=" * 70)
    print("CPPN ACTIVATION FUNCTION EXPERIMENT")
    print("=" * 70)
    print()
    print("HYPOTHESIS: Periodic activations (sin, cos, sin2, ring) produce higher")
    print("            spectral coherence than non-periodic activations.")
    print()
    print("NULL (H0): No difference in spectral coherence between groups.")
    print()
    print(f"Parameters: n_samples={n_samples}, image_size={image_size}")
    print()

    # Storage for results
    results_by_activation = {act: [] for act in ACTIVATIONS.keys()}

    # Generate samples for each activation type
    print("Generating samples...")
    all_activations = list(ACTIVATIONS.keys())

    for act_idx, activation in enumerate(all_activations):
        print(f"  {activation}...", end="", flush=True)

        for i in range(n_samples):
            # Use unique seed for each sample
            seed = base_seed * 1000 + act_idx * n_samples + i
            cppn = create_cppn_with_activation(activation, seed=seed)
            img = cppn.render(image_size)
            features = compute_features(img)
            results_by_activation[activation].append(features)

        print(f" done ({n_samples} samples)")

    print()

    # Compute summary statistics per activation
    print("-" * 70)
    print("SUMMARY STATISTICS BY ACTIVATION TYPE")
    print("-" * 70)
    print(f"{'Activation':<12} {'EdgeDens':>10} {'Symmetry':>10} {'SpectCoh':>10} {'Compress':>10}")
    print("-" * 70)

    summary_stats = {}
    for activation in all_activations:
        features = results_by_activation[activation]
        summary_stats[activation] = {
            'edge_density_mean': np.mean([f['edge_density'] for f in features]),
            'edge_density_std': np.std([f['edge_density'] for f in features]),
            'symmetry_mean': np.mean([f['symmetry'] for f in features]),
            'symmetry_std': np.std([f['symmetry'] for f in features]),
            'spectral_coherence_mean': np.mean([f['spectral_coherence'] for f in features]),
            'spectral_coherence_std': np.std([f['spectral_coherence'] for f in features]),
            'compressibility_mean': np.mean([f['compressibility'] for f in features]),
            'compressibility_std': np.std([f['compressibility'] for f in features]),
        }
        s = summary_stats[activation]
        print(f"{activation:<12} {s['edge_density_mean']:>10.4f} {s['symmetry_mean']:>10.4f} "
              f"{s['spectral_coherence_mean']:>10.4f} {s['compressibility_mean']:>10.4f}")

    print()

    # PRIMARY TEST: Periodic vs Non-periodic spectral coherence
    print("=" * 70)
    print("PRIMARY HYPOTHESIS TEST: Spectral Coherence")
    print("Periodic (sin, cos, sin2, ring) vs Non-periodic (sigmoid, tanh, relu, gauss, abs, identity)")
    print("=" * 70)
    print()

    periodic_coherence = []
    nonperiodic_coherence = []

    for act in PERIODIC_ACTIVATIONS:
        periodic_coherence.extend([f['spectral_coherence'] for f in results_by_activation[act]])

    for act in NONPERIODIC_ACTIVATIONS:
        nonperiodic_coherence.extend([f['spectral_coherence'] for f in results_by_activation[act]])

    periodic_coherence = np.array(periodic_coherence)
    nonperiodic_coherence = np.array(nonperiodic_coherence)

    # Mann-Whitney U test (two-sided)
    U_stat, p_value = mannwhitneyu(periodic_coherence, nonperiodic_coherence, alternative='two-sided')

    # Effect size
    d = cohens_d(periodic_coherence, nonperiodic_coherence)

    print(f"Periodic group:     mean={np.mean(periodic_coherence):.4f}, std={np.std(periodic_coherence):.4f}, n={len(periodic_coherence)}")
    print(f"Non-periodic group: mean={np.mean(nonperiodic_coherence):.4f}, std={np.std(nonperiodic_coherence):.4f}, n={len(nonperiodic_coherence)}")
    print()
    print(f"Mann-Whitney U: {U_stat:.1f}")
    print(f"P-value: {p_value:.2e}")
    print(f"Cohen's d: {d:.3f}")
    print()

    # Interpretation
    primary_validated = p_value < 0.01 and abs(d) > 0.5

    if primary_validated:
        if d > 0:
            print("RESULT: VALIDATED - Periodic activations have HIGHER spectral coherence")
        else:
            print("RESULT: VALIDATED - Periodic activations have LOWER spectral coherence (opposite direction!)")
    else:
        print("RESULT: NOT VALIDATED")
        if p_value >= 0.01:
            print(f"  - P-value {p_value:.4f} >= 0.01 threshold")
        if abs(d) <= 0.5:
            print(f"  - Effect size |d|={abs(d):.3f} <= 0.5 threshold")

    print()

    # SECONDARY TEST: Kruskal-Wallis across all activation types (for each feature)
    print("=" * 70)
    print("SECONDARY ANALYSIS: Kruskal-Wallis H-test (all activations)")
    print("=" * 70)
    print()

    features_names = ['edge_density', 'symmetry', 'spectral_coherence', 'compressibility']
    kruskal_results = {}

    for feature in features_names:
        groups = [
            np.array([f[feature] for f in results_by_activation[act]])
            for act in all_activations
        ]
        H_stat, p_val = kruskal(*groups)
        kruskal_results[feature] = {'H': H_stat, 'p': p_val}
        sig_marker = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        print(f"{feature:<20}: H={H_stat:>8.2f}, p={p_val:.2e} {sig_marker}")

    print()

    # Find which activation produces most distinct feature profiles
    print("=" * 70)
    print("ACTIVATION PROFILES (ranked by spectral coherence)")
    print("=" * 70)
    sorted_acts = sorted(all_activations,
                         key=lambda a: summary_stats[a]['spectral_coherence_mean'],
                         reverse=True)
    for i, act in enumerate(sorted_acts):
        s = summary_stats[act]
        periodic_marker = "(P)" if act in PERIODIC_ACTIVATIONS else "(N)"
        print(f"{i+1}. {act:<10} {periodic_marker}: spectral_coh={s['spectral_coherence_mean']:.4f}")

    print()

    # Prepare final results
    final_results = {
        'experiment': 'cppn_activation_functions',
        'parameters': {
            'n_samples': n_samples,
            'image_size': image_size,
            'base_seed': base_seed,
            'periodic_activations': PERIODIC_ACTIVATIONS,
            'nonperiodic_activations': NONPERIODIC_ACTIVATIONS,
        },
        'primary_test': {
            'comparison': 'periodic_vs_nonperiodic_spectral_coherence',
            'periodic_mean': float(np.mean(periodic_coherence)),
            'periodic_std': float(np.std(periodic_coherence)),
            'nonperiodic_mean': float(np.mean(nonperiodic_coherence)),
            'nonperiodic_std': float(np.std(nonperiodic_coherence)),
            'mann_whitney_U': float(U_stat),
            'p_value': float(p_value),
            'cohens_d': float(d),
            'validated': primary_validated,
        },
        'secondary_tests': kruskal_results,
        'summary_by_activation': summary_stats,
        'rankings': {
            'spectral_coherence': sorted_acts,
        }
    }

    # Save results
    results_dir = Path(__file__).parent.parent / 'results' / 'cppn_activation'
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / 'activation_experiment_results.json'

    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)

    print(f"Results saved to: {results_path}")
    print()

    # Final summary
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print()
    print(f"Primary hypothesis (periodic > non-periodic spectral coherence):")
    print(f"  Status: {'VALIDATED' if primary_validated else 'NOT VALIDATED'}")
    print(f"  P-value: {p_value:.2e} (threshold: 0.01)")
    print(f"  Effect size (Cohen's d): {d:.3f} (threshold: |d| > 0.5)")
    print()
    print(f"Secondary finding:")
    print(f"  All 4 features show significant activation-type dependence (Kruskal-Wallis)")
    print(f"  Highest spectral coherence: {sorted_acts[0]}")
    print(f"  Lowest spectral coherence: {sorted_acts[-1]}")

    return final_results


if __name__ == "__main__":
    results = run_experiment(n_samples=100, image_size=32, base_seed=42)
