#!/usr/bin/env python3
"""
RES-090: Weight-space averaging vs output-space averaging for CPPN ensembles.

Hypothesis: Averaging multiple CPPN weight vectors before generating an image
preserves order better than averaging outputs (RES-022 approach).

Rationale:
- RES-022 found that output-space averaging fails because different CPPNs
  create incompatible spatial patterns that destructively interfere
- Weight-space averaging creates a single coherent CPPN that generates
  one pattern, avoiding the interference problem
- The averaged weight vector lies on a lower-dimensional manifold in
  weight space (central limit theorem), potentially in a more "generic" region
"""

import numpy as np
from scipy import stats
import json
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import CPPN, order_multiplicative


def create_random_cppn(seed: int = None) -> CPPN:
    """Create a random CPPN with fresh weights."""
    if seed is not None:
        np.random.seed(seed)
    return CPPN()


def output_space_average(cppns: list, size: int = 32) -> np.ndarray:
    """
    Average continuous outputs before thresholding.
    This is the RES-022 approach that was refuted.
    """
    coords = np.linspace(-1, 1, size)
    x, y = np.meshgrid(coords, coords)

    outputs = []
    for cppn in cppns:
        out = cppn.activate(x, y)  # Returns continuous [0, 1] after sigmoid
        outputs.append(out)

    # Average continuous outputs, then threshold
    avg_output = np.mean(outputs, axis=0)
    return (avg_output > 0.5).astype(np.uint8)


def weight_space_average(cppns: list, size: int = 32) -> np.ndarray:
    """
    Average weight vectors, create new CPPN with averaged weights, then render.
    """
    # Get weight vectors from all CPPNs
    weight_vectors = [cppn.get_weights() for cppn in cppns]
    avg_weights = np.mean(weight_vectors, axis=0)

    # Create new CPPN and set averaged weights
    # Use the first CPPN as template for architecture
    avg_cppn = cppns[0].copy()
    avg_cppn.set_weights(avg_weights)

    return avg_cppn.render(size)


def run_experiment(
    n_trials: int = 500,
    ensemble_sizes: list = [2, 3, 5, 8],
    size: int = 32,
    seed: int = 42
) -> dict:
    """
    Compare weight-space vs output-space averaging.

    For each trial:
    1. Generate K independent CPPNs
    2. Compute order of each individual
    3. Compute order of output-space average
    4. Compute order of weight-space average
    """
    np.random.seed(seed)

    results = {
        'n_trials': n_trials,
        'ensemble_sizes': ensemble_sizes,
        'size': size,
        'by_k': {}
    }

    for k in ensemble_sizes:
        print(f"\nTesting ensemble size K={k}...")

        individual_orders = []
        output_avg_orders = []
        weight_avg_orders = []

        for trial in range(n_trials):
            if trial % 100 == 0:
                print(f"  Trial {trial}/{n_trials}")

            # Generate K independent CPPNs
            cppns = [create_random_cppn() for _ in range(k)]

            # Individual orders
            indiv_orders = [order_multiplicative(cppn.render(size)) for cppn in cppns]
            individual_orders.append(np.mean(indiv_orders))

            # Output-space averaging (RES-022 approach)
            output_img = output_space_average(cppns, size)
            output_avg_orders.append(order_multiplicative(output_img))

            # Weight-space averaging (our hypothesis)
            weight_img = weight_space_average(cppns, size)
            weight_avg_orders.append(order_multiplicative(weight_img))

        individual_orders = np.array(individual_orders)
        output_avg_orders = np.array(output_avg_orders)
        weight_avg_orders = np.array(weight_avg_orders)

        # Statistical tests: weight-space vs output-space
        t_stat, p_value = stats.ttest_rel(weight_avg_orders, output_avg_orders)

        # Effect size (Cohen's d for paired samples)
        diff = weight_avg_orders - output_avg_orders
        cohens_d = np.mean(diff) / (np.std(diff) + 1e-10)

        # Win rates
        weight_beats_output = np.mean(weight_avg_orders > output_avg_orders)
        weight_beats_individual = np.mean(weight_avg_orders > individual_orders)
        output_beats_individual = np.mean(output_avg_orders > individual_orders)

        results['by_k'][str(k)] = {
            'individual_mean': float(np.mean(individual_orders)),
            'individual_std': float(np.std(individual_orders)),
            'output_avg_mean': float(np.mean(output_avg_orders)),
            'output_avg_std': float(np.std(output_avg_orders)),
            'weight_avg_mean': float(np.mean(weight_avg_orders)),
            'weight_avg_std': float(np.std(weight_avg_orders)),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'cohens_d': float(cohens_d),
            'weight_beats_output': float(weight_beats_output),
            'weight_beats_individual': float(weight_beats_individual),
            'output_beats_individual': float(output_beats_individual)
        }

        print(f"  K={k}: weight_avg={np.mean(weight_avg_orders):.4f}, "
              f"output_avg={np.mean(output_avg_orders):.4f}, "
              f"individual={np.mean(individual_orders):.4f}")
        print(f"       Cohen's d = {cohens_d:.3f}, p = {p_value:.2e}")
        print(f"       Weight beats output: {weight_beats_output:.1%}")

    # Overall summary
    k5_data = results['by_k']['5']
    results['summary'] = {
        'validated': k5_data['cohens_d'] > 0.5 and k5_data['p_value'] < 0.01,
        'primary_effect_size': k5_data['cohens_d'],
        'primary_p_value': k5_data['p_value'],
        'weight_avg_advantage': k5_data['weight_avg_mean'] - k5_data['output_avg_mean']
    }

    return results


def analyze_why_weight_averaging_works(n_samples: int = 100, k: int = 5, size: int = 32):
    """
    Investigate the mechanism: does weight averaging produce less variance?
    """
    np.random.seed(42)

    weight_variances = []
    output_pixel_variances = []

    for _ in range(n_samples):
        cppns = [create_random_cppn() for _ in range(k)]

        # Variance in weight space
        weight_vectors = np.array([cppn.get_weights() for cppn in cppns])
        weight_var = np.mean(np.var(weight_vectors, axis=0))
        weight_variances.append(weight_var)

        # Variance in output space (per-pixel)
        coords = np.linspace(-1, 1, size)
        x, y = np.meshgrid(coords, coords)
        outputs = np.array([cppn.activate(x, y) for cppn in cppns])
        pixel_var = np.mean(np.var(outputs, axis=0))
        output_pixel_variances.append(pixel_var)

    return {
        'mean_weight_variance': float(np.mean(weight_variances)),
        'mean_pixel_variance': float(np.mean(output_pixel_variances)),
        'variance_ratio': float(np.mean(output_pixel_variances) / np.mean(weight_variances))
    }


if __name__ == '__main__':
    print("=" * 60)
    print("RES-090: Weight-space vs Output-space CPPN Averaging")
    print("=" * 60)

    # Main experiment
    results = run_experiment(n_trials=500, ensemble_sizes=[2, 3, 5, 8])

    # Mechanism analysis
    print("\n" + "=" * 60)
    print("Mechanism Analysis")
    print("=" * 60)
    mechanism = analyze_why_weight_averaging_works()
    results['mechanism'] = mechanism
    print(f"Mean weight variance: {mechanism['mean_weight_variance']:.4f}")
    print(f"Mean pixel variance: {mechanism['mean_pixel_variance']:.4f}")
    print(f"Variance amplification ratio: {mechanism['variance_ratio']:.2f}x")

    # Save results
    import os
    os.makedirs('/Users/matt/Development/monochrome_noise_converger/results/weight_space_ensemble', exist_ok=True)
    with open('/Users/matt/Development/monochrome_noise_converger/results/weight_space_ensemble/results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if results['summary']['validated']:
        print("VALIDATED: Weight-space averaging significantly outperforms output-space averaging")
    else:
        print("NOT VALIDATED: Effect size or significance threshold not met")

    print(f"Cohen's d = {results['summary']['primary_effect_size']:.3f}")
    print(f"p-value = {results['summary']['primary_p_value']:.2e}")
    print(f"Advantage = {results['summary']['weight_avg_advantage']:.4f}")
