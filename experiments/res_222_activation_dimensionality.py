#!/usr/bin/env python3
"""
RES-222: Periodic Activations Cause High Initial Dimensionality
Tests whether sine/periodic activations create high-dimensional weight space.

Hypothesis: Sine activations in CPPNs create higher effective dimensionality.
Replacing with ReLU/tanh reduces initial eff_dim by ≥2×.

Method:
1. Initialize 40 CPPNs with sine activations (baseline)
2. Initialize 40 CPPNs with ReLU activations (same structure)
3. Initialize 40 CPPNs with tanh activations (same structure)
4. Measure effective dimensionality via PCA on weight space
5. Compare and validate if sine >= 2x higher than ReLU/tanh
"""

import sys
import os
from pathlib import Path
import json
import numpy as np
from typing import Tuple, Dict, List
import traceback

# Ensure project root in path
project_root = Path('/Users/matt/Development/monochrome_noise_converger')
sys.path.insert(0, str(project_root))
os.chdir(str(project_root))

# Import CPPN from core
from core.thermo_sampler_v3 import CPPN, Node, Connection, PRIOR_SIGMA, set_global_seed


def create_cppn_with_activation(activation: str, seed: int = None) -> CPPN:
    """
    Create a simple CPPN with specified activation function.

    Creates a small network: 4 inputs -> 2 hidden nodes -> 1 output
    All connections initialized from N(0, PRIOR_SIGMA).

    Args:
        activation: 'sin', 'relu', 'tanh', etc. (must be in ACTIVATIONS dict)
        seed: random seed for reproducibility

    Returns:
        CPPN instance
    """
    if seed is not None:
        set_global_seed(seed)

    # Create nodes: 4 inputs (0,1,2,3) + 2 hidden (5,6) + 1 output (4)
    nodes = [
        Node(0, 'identity', 0.0),   # x coordinate
        Node(1, 'identity', 0.0),   # y coordinate
        Node(2, 'identity', 0.0),   # radius
        Node(3, 'identity', 0.0),   # bias
        Node(4, 'sigmoid', np.random.randn() * PRIOR_SIGMA),  # output
        Node(5, activation, np.random.randn() * PRIOR_SIGMA),  # hidden 1
        Node(6, activation, np.random.randn() * PRIOR_SIGMA),  # hidden 2
    ]

    # Create connections
    connections = []

    # Input -> hidden connections
    for in_id in [0, 1, 2, 3]:
        for hid_id in [5, 6]:
            w = np.random.randn() * PRIOR_SIGMA
            connections.append(Connection(in_id, hid_id, w, enabled=True))

    # Hidden -> output connections
    for hid_id in [5, 6]:
        w = np.random.randn() * PRIOR_SIGMA
        connections.append(Connection(hid_id, 4, w, enabled=True))

    # Direct input -> output skip connections
    for in_id in [0, 1, 2, 3]:
        w = np.random.randn() * PRIOR_SIGMA
        connections.append(Connection(in_id, 4, w, enabled=True))

    cppn = CPPN(nodes=nodes, connections=connections, input_ids=[0, 1, 2, 3], output_id=4)
    return cppn


def estimate_weight_space_dimensionality(activation_type: str, n_cppes_per_sample: int = 10) -> float:
    """
    Estimate effective dimensionality using weight space PCA.

    Key insight: For a given activation, sample many random CPPNs and
    compute PCA on their weight vectors. The effective dimensionality
    indicates how much of the weight space is actually being used.

    Args:
        activation_type: 'sin', 'relu', 'tanh', etc.
        n_cppes_per_sample: number of CPPNs to sample for PCA

    Returns:
        Effective dimensionality (number of PCA components for 90% variance)
    """
    # Generate weight vectors for multiple random CPPNs
    weight_vectors = []

    for i in range(n_cppes_per_sample):
        set_global_seed(42 + i)
        cppn = create_cppn_with_activation(activation_type, seed=42 + i)
        weights = cppn.get_weights()
        weight_vectors.append(weights)

    if not weight_vectors:
        return 6.0

    # Stack into matrix (n_cppns x n_weights)
    weight_matrix = np.array(weight_vectors)

    # Center
    weight_matrix = weight_matrix - weight_matrix.mean(axis=0)

    # SVD
    U, S, Vt = np.linalg.svd(weight_matrix, full_matrices=False)

    # Compute cumulative variance explained
    explained_variance = (S ** 2) / (np.sum(S ** 2) + 1e-10)
    cumsum_variance = np.cumsum(explained_variance)

    # Find how many components needed for 90% variance
    variance_threshold = 0.90
    if np.max(cumsum_variance) < variance_threshold:
        # Insufficient variance, return full dimension
        return float(len(weight_vectors[0]))

    n_components = np.argmax(cumsum_variance >= variance_threshold) + 1

    return float(n_components)


def measure_activation_dimensionality(
    activation: str,
    n_trials: int = 4
) -> Tuple[List[float], float, float]:
    """
    Measure effective dimensionality for CPPNs with given activation.

    Runs multiple trials, each sampling a batch of CPPNs and computing
    PCA on their weight space.

    Args:
        activation: activation type
        n_trials: number of independent measurement trials

    Returns:
        (individual_eff_dims, mean_eff_dim, std_eff_dim)
    """
    eff_dims = []

    for trial in range(n_trials):
        try:
            # Each trial samples 10 CPPNs and computes PCA
            eff_dim = estimate_weight_space_dimensionality(
                activation,
                n_cppes_per_sample=10
            )
            eff_dims.append(eff_dim)
        except Exception as e:
            print(f"  Warning: Failed to measure activation {activation}, trial {trial}: {e}")
            traceback.print_exc()
            continue

    if not eff_dims:
        return [], 0.0, 0.0

    return eff_dims, np.mean(eff_dims), np.std(eff_dims)


def run_experiment() -> Dict:
    """
    Run the full RES-222 experiment.

    Returns:
        Dictionary with results
    """
    print("RES-222: Periodic Activations Cause High Initial Dimensionality")
    print("=" * 60)

    # Parameters - 4 trials x 10 CPPNs per trial = 40 CPPNs total per activation
    n_trials = 4
    n_cppns_per_trial = 10

    print(f"\nConfiguration: {n_trials} trials x {n_cppns_per_trial} CPPNs = {n_trials * n_cppns_per_trial} total per activation")
    print("Network structure: 4 inputs -> 2 hidden + skip -> 1 output (16 total weights)")
    print("Method: PCA on weight space (90% variance threshold)")

    # Test each activation
    print("\n[1/3] Testing SINE activation...")
    sine_eff_dims, sine_mean, sine_std = measure_activation_dimensionality(
        'sin', n_trials=n_trials
    )
    print(f"  Sine: mean={sine_mean:.4f} ± {sine_std:.4f} (n={len(sine_eff_dims)} trials)")

    print("[2/3] Testing ReLU activation...")
    relu_eff_dims, relu_mean, relu_std = measure_activation_dimensionality(
        'relu', n_trials=n_trials
    )
    print(f"  ReLU: mean={relu_mean:.4f} ± {relu_std:.4f} (n={len(relu_eff_dims)} trials)")

    print("[3/3] Testing tanh activation...")
    tanh_eff_dims, tanh_mean, tanh_std = measure_activation_dimensionality(
        'tanh', n_trials=n_trials
    )
    print(f"  tanh: mean={tanh_mean:.4f} ± {tanh_std:.4f} (n={len(tanh_eff_dims)} trials)")

    # Calculate ratios
    sine_to_relu_ratio = sine_mean / relu_mean if relu_mean > 0 else 0
    sine_to_tanh_ratio = sine_mean / tanh_mean if tanh_mean > 0 else 0

    # Statistical test: Mann-Whitney U test
    from scipy import stats

    u_statistic_relu, p_value_relu = stats.mannwhitneyu(sine_eff_dims, relu_eff_dims, alternative='greater')
    u_statistic_tanh, p_value_tanh = stats.mannwhitneyu(sine_eff_dims, tanh_eff_dims, alternative='greater')

    # Cohen's d effect size
    pooled_std_relu = np.sqrt((np.std(sine_eff_dims)**2 + np.std(relu_eff_dims)**2) / 2)
    cohens_d_relu = (sine_mean - relu_mean) / pooled_std_relu if pooled_std_relu > 0 else 0

    pooled_std_tanh = np.sqrt((np.std(sine_eff_dims)**2 + np.std(tanh_eff_dims)**2) / 2)
    cohens_d_tanh = (sine_mean - tanh_mean) / pooled_std_tanh if pooled_std_tanh > 0 else 0

    # Determine validation status
    # Hypothesis: sine >= 2x higher than ReLU/tanh AND statistically significant
    validate_relu = (sine_to_relu_ratio >= 2.0) and (p_value_relu < 0.05)
    validate_tanh = (sine_to_tanh_ratio >= 2.0) and (p_value_tanh < 0.05)

    conclusion = "validate" if (validate_relu or validate_tanh) else "refute"

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Sine effective dim:   {sine_mean:.4f} ± {sine_std:.4f}")
    print(f"ReLU effective dim:   {relu_mean:.4f} ± {relu_std:.4f}")
    print(f"tanh effective dim:   {tanh_mean:.4f} ± {tanh_std:.4f}")
    print()
    print(f"Sine/ReLU ratio:      {sine_to_relu_ratio:.4f}x (p={p_value_relu:.6f}, Cohen's d={cohens_d_relu:.4f})")
    print(f"Sine/tanh ratio:      {sine_to_tanh_ratio:.4f}x (p={p_value_tanh:.6f}, Cohen's d={cohens_d_tanh:.4f})")
    print()
    print(f"Hypothesis validation (≥2x AND p<0.05):")
    print(f"  ReLU comparison: {'VALIDATE' if validate_relu else 'REFUTE'}")
    print(f"  tanh comparison: {'VALIDATE' if validate_tanh else 'REFUTE'}")
    print()
    print(f"CONCLUSION: {conclusion.upper()}")

    if conclusion == "validate":
        print("✓ Periodic activations DO create higher initial dimensionality")
    else:
        print("✗ Periodic activations do NOT create significantly higher dimensionality (ratio < 2x or p > 0.05)")

    # Prepare results
    total_cppns_per_activation = n_trials * n_cppns_per_trial
    results = {
        "method": "Effective dimensionality comparison across activation functions",
        "configuration": {
            "n_trials": n_trials,
            "n_cppns_per_trial": n_cppns_per_trial,
            "n_total_cppns_per_activation": total_cppns_per_activation,
            "network_structure": "4 inputs -> 2 hidden + skip -> 1 output",
            "total_weights": 16,
            "variance_threshold": 0.90
        },
        "measurements": {
            "sine_eff_dim": float(sine_mean),
            "sine_eff_dim_std": float(sine_std),
            "relu_eff_dim": float(relu_mean),
            "relu_eff_dim_std": float(relu_std),
            "tanh_eff_dim": float(tanh_mean),
            "tanh_eff_dim_std": float(tanh_std)
        },
        "comparisons": {
            "sine_to_relu_ratio": float(sine_to_relu_ratio),
            "sine_to_tanh_ratio": float(sine_to_tanh_ratio),
            "sine_vs_relu_p_value": float(p_value_relu),
            "sine_vs_tanh_p_value": float(p_value_tanh),
            "sine_vs_relu_cohens_d": float(cohens_d_relu),
            "sine_vs_tanh_cohens_d": float(cohens_d_tanh)
        },
        "validation": {
            "relu_validates_hypothesis": bool(validate_relu),
            "tanh_validates_hypothesis": bool(validate_tanh),
            "threshold_ratio": 2.0,
            "threshold_p_value": 0.05
        },
        "conclusion": conclusion,
        "summary": f"Sine activations show {'significantly higher' if conclusion == 'validate' else 'similar'} effective dimensionality compared to ReLU/tanh. Sine/ReLU ratio={sine_to_relu_ratio:.4f}x, Sine/tanh ratio={sine_to_tanh_ratio:.4f}x."
    }

    return results


def main():
    """Execute experiment and save results."""
    # Create results directory
    results_dir = Path('/Users/matt/Development/monochrome_noise_converger/results/activation_function_dimensionality')
    results_dir.mkdir(parents=True, exist_ok=True)

    # Run experiment
    results = run_experiment()

    # Save results
    output_file = results_dir / 'res_222_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {output_file}")

    # Return summary for research log
    return results


if __name__ == '__main__':
    results = main()

    # Print concise summary for research log
    print("\n" + "=" * 60)
    print("FOR RESEARCH LOG UPDATE:")
    print("=" * 60)
    print(f"RES-222 | activation_function_dimensionality | {results['conclusion']} | sine_ratio={results['comparisons']['sine_to_relu_ratio']:.2f}x")
