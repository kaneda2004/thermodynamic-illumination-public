#!/usr/bin/env python3
"""
RES-104: CPPN Architecture Complexity vs Convergence Rate

Hypothesis: CPPN architecture complexity (hidden node count) correlates with
convergence speed to high-order regions in nested sampling. Simpler architectures
converge faster due to lower-dimensional weight space.

Methodology:
- Create CPPNs with varying hidden node counts (0, 2, 4, 8)
- Run mini-nested sampling with each architecture type
- Measure iterations to reach order threshold (convergence speed)
- Correlate architecture complexity with convergence metrics
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from dataclasses import dataclass, field
from typing import List, Tuple
from core.thermo_sampler_v3 import (
    CPPN, Node, Connection, order_multiplicative, ACTIVATIONS, PRIOR_SIGMA,
    elliptical_slice_sample, log_prior
)
from scipy import stats
import random


def create_cppn_with_hidden_nodes(n_hidden: int) -> CPPN:
    """Create a CPPN with specified number of hidden nodes."""
    nodes = [
        Node(0, 'identity', 0.0),  # x
        Node(1, 'identity', 0.0),  # y
        Node(2, 'identity', 0.0),  # r
        Node(3, 'identity', 0.0),  # bias
        Node(4, 'sigmoid', np.random.randn() * PRIOR_SIGMA),  # output
    ]

    connections = []
    activations = list(ACTIVATIONS.keys())
    hidden_ids = []

    for i in range(n_hidden):
        hid = 5 + i
        hidden_ids.append(hid)
        act = random.choice(activations)
        nodes.append(Node(hid, act, np.random.randn() * PRIOR_SIGMA))

    if n_hidden == 0:
        for inp in [0, 1, 2, 3]:
            connections.append(Connection(inp, 4, np.random.randn() * PRIOR_SIGMA))
    else:
        # Single hidden layer architecture
        for inp in [0, 1, 2, 3]:
            for hid in hidden_ids:
                if random.random() < 0.7:
                    connections.append(Connection(inp, hid, np.random.randn() * PRIOR_SIGMA))
        for hid in hidden_ids:
            connections.append(Connection(hid, 4, np.random.randn() * PRIOR_SIGMA))

    return CPPN(nodes=nodes, connections=connections)


def get_weight_dimension(cppn: CPPN) -> int:
    """Get the number of free parameters (weight space dimension)."""
    return len(cppn.get_weights())


def mini_nested_sampling(
    n_hidden: int,
    n_live: int = 30,
    max_iterations: int = 300,
    target_order: float = 0.3,
    image_size: int = 32
) -> dict:
    """
    Run nested sampling with a specific architecture and measure convergence.

    Returns dict with convergence metrics.
    """
    # Initialize live points with this architecture
    live_points = []

    for _ in range(n_live):
        cppn = create_cppn_with_hidden_nodes(n_hidden)
        img = cppn.render(image_size)
        order = order_multiplicative(img)
        live_points.append({
            'cppn': cppn,
            'image': img,
            'order': order,
            'log_prior': log_prior(cppn)
        })

    initial_orders = [lp['order'] for lp in live_points]
    weight_dim = get_weight_dimension(live_points[0]['cppn'])

    # Track order progression
    order_history = [np.mean(initial_orders)]
    threshold_history = [min(initial_orders)]

    iterations_to_threshold = None

    for iteration in range(max_iterations):
        # Find worst (lowest order)
        worst_idx = min(range(n_live), key=lambda i: live_points[i]['order'])
        threshold = live_points[worst_idx]['order']
        threshold_history.append(threshold)

        # Check if we've reached target
        if threshold >= target_order and iterations_to_threshold is None:
            iterations_to_threshold = iteration

        # Select seed (any point except worst)
        valid_seeds = [i for i in range(n_live) if i != worst_idx]
        seed_idx = random.choice(valid_seeds)
        seed_cppn = live_points[seed_idx]['cppn']

        # Try to replace worst with ESS
        new_cppn, new_img, new_order, _, _, success = elliptical_slice_sample(
            seed_cppn,
            threshold,
            image_size,
            order_multiplicative,
            max_contractions=100,
            max_restarts=5
        )

        if new_order >= threshold:
            live_points[worst_idx] = {
                'cppn': new_cppn,
                'image': new_img,
                'order': new_order,
                'log_prior': log_prior(new_cppn)
            }
        else:
            # Rejection - create new random point meeting threshold
            for _ in range(100):
                new_cppn = create_cppn_with_hidden_nodes(n_hidden)
                new_img = new_cppn.render(image_size)
                new_order = order_multiplicative(new_img)
                if new_order >= threshold:
                    live_points[worst_idx] = {
                        'cppn': new_cppn,
                        'image': new_img,
                        'order': new_order,
                        'log_prior': log_prior(new_cppn)
                    }
                    break

        order_history.append(np.mean([lp['order'] for lp in live_points]))

    # Final metrics
    final_orders = [lp['order'] for lp in live_points]

    # Compute convergence speed as slope of threshold increase
    # Use first half to avoid saturation effects
    half = len(threshold_history) // 2
    x = np.arange(half)
    y = np.array(threshold_history[:half])
    if len(x) > 1:
        slope, _, r_value, _, _ = stats.linregress(x, y)
        convergence_rate = slope  # Order increase per iteration
    else:
        convergence_rate = 0
        r_value = 0

    return {
        'n_hidden': n_hidden,
        'weight_dim': weight_dim,
        'initial_mean_order': float(np.mean(initial_orders)),
        'final_mean_order': float(np.mean(final_orders)),
        'final_threshold': float(threshold_history[-1]),
        'iterations_to_target': iterations_to_threshold,
        'convergence_rate': float(convergence_rate),
        'convergence_r2': float(r_value**2),
        'order_history': order_history,
        'threshold_history': threshold_history
    }


def main():
    print("=" * 70)
    print("RES-104: CPPN Architecture Complexity vs Convergence Rate")
    print("=" * 70)

    np.random.seed(42)
    random.seed(42)

    # Test architectures
    hidden_counts = [0, 2, 4, 8]
    n_trials = 10  # Multiple trials per architecture

    print(f"\nRunning {n_trials} trials per architecture...")
    print(f"Target order threshold: 0.3")
    print("-" * 70)

    all_results = {n: [] for n in hidden_counts}

    for trial in range(n_trials):
        print(f"\nTrial {trial + 1}/{n_trials}")
        for n_hidden in hidden_counts:
            print(f"  n_hidden={n_hidden}...", end=" ", flush=True)
            result = mini_nested_sampling(n_hidden, n_live=30, max_iterations=200, target_order=0.3)
            all_results[n_hidden].append(result)
            print(f"rate={result['convergence_rate']:.6f}, final={result['final_threshold']:.4f}")

    # Aggregate results
    print("\n" + "=" * 70)
    print("AGGREGATED RESULTS")
    print("=" * 70)

    aggregated = []
    for n_hidden in hidden_counts:
        trials = all_results[n_hidden]
        weight_dims = [t['weight_dim'] for t in trials]
        rates = [t['convergence_rate'] for t in trials]
        final_thresholds = [t['final_threshold'] for t in trials]

        aggregated.append({
            'n_hidden': n_hidden,
            'weight_dim': np.mean(weight_dims),
            'mean_rate': np.mean(rates),
            'std_rate': np.std(rates),
            'mean_final': np.mean(final_thresholds),
            'std_final': np.std(final_thresholds),
            'rates': rates
        })

    print(f"\n{'n_hidden':>10} | {'Weight Dim':>12} | {'Conv Rate':>12} | {'Final Order':>12}")
    print("-" * 55)
    for a in aggregated:
        print(f"{a['n_hidden']:>10} | {a['weight_dim']:>12.1f} | {a['mean_rate']:>12.6f} | {a['mean_final']:>12.4f}")

    # Statistical analysis
    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS")
    print("=" * 70)

    # 1. Correlation: weight_dim vs convergence_rate
    weight_dims = []
    rates = []
    for n_hidden in hidden_counts:
        for trial in all_results[n_hidden]:
            weight_dims.append(trial['weight_dim'])
            rates.append(trial['convergence_rate'])

    weight_dims = np.array(weight_dims)
    rates = np.array(rates)

    r_corr, p_corr = stats.pearsonr(weight_dims, rates)
    print(f"\nCorrelation (weight_dim vs convergence_rate):")
    print(f"  Pearson r = {r_corr:.4f}")
    print(f"  p-value = {p_corr:.2e}")

    # 2. Linear regression
    slope, intercept, r_val, p_val, std_err = stats.linregress(weight_dims, rates)
    print(f"\nLinear regression (rate ~ weight_dim):")
    print(f"  Slope = {slope:.6f}")
    print(f"  R^2 = {r_val**2:.4f}")
    print(f"  p-value = {p_val:.2e}")

    # 3. ANOVA across architectures
    all_rates = [[t['convergence_rate'] for t in all_results[n]] for n in hidden_counts]
    f_stat, p_anova = stats.f_oneway(*all_rates)
    print(f"\nANOVA across architectures:")
    print(f"  F-statistic = {f_stat:.2f}")
    print(f"  p-value = {p_anova:.2e}")

    # 4. Effect size: simplest vs most complex
    simplest_rates = np.array([t['convergence_rate'] for t in all_results[hidden_counts[0]]])
    complex_rates = np.array([t['convergence_rate'] for t in all_results[hidden_counts[-1]]])

    pooled_std = np.sqrt((np.var(simplest_rates) + np.var(complex_rates)) / 2)
    effect_size = (np.mean(simplest_rates) - np.mean(complex_rates)) / pooled_std if pooled_std > 0 else 0

    t_stat, p_ttest = stats.ttest_ind(simplest_rates, complex_rates)

    print(f"\nSimplest (n_hidden={hidden_counts[0]}) vs Most Complex (n_hidden={hidden_counts[-1]}):")
    print(f"  Effect size (Cohen's d) = {effect_size:.3f}")
    print(f"  t-statistic = {t_stat:.2f}")
    print(f"  p-value = {p_ttest:.2e}")

    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    # Check if simpler architectures converge faster (negative correlation with dimension)
    if p_corr < 0.01 and r_corr < -0.3:
        verdict = "VALIDATED"
        direction = "simpler architectures converge FASTER"
    elif p_corr < 0.01 and r_corr > 0.3:
        verdict = "REFUTED"
        direction = "complex architectures converge FASTER (opposite of hypothesis)"
    elif p_anova < 0.01 and abs(effect_size) > 0.5:
        if effect_size > 0:
            verdict = "VALIDATED"
            direction = "simpler architectures converge faster"
        else:
            verdict = "REFUTED"
            direction = "complex architectures converge faster"
    elif p_corr < 0.05 or p_anova < 0.05:
        verdict = "INCONCLUSIVE"
        direction = f"weak effect (r={r_corr:.3f})"
    else:
        verdict = "REFUTED"
        direction = "no significant relationship"

    summary = f"Weight dimension {'negatively' if r_corr < 0 else 'positively'} correlates with convergence rate (r={r_corr:.3f}, p={p_corr:.2e}). {direction}."

    print(f"\nStatus: {verdict}")
    print(f"Summary: {summary}")

    metrics = {
        'correlation_r': float(r_corr),
        'correlation_p': float(p_corr),
        'regression_slope': float(slope),
        'regression_r2': float(r_val**2),
        'anova_f': float(f_stat),
        'anova_p': float(p_anova),
        'effect_size': float(effect_size),
        'ttest_p': float(p_ttest)
    }

    print(f"\nMetrics: {metrics}")

    return verdict, summary, metrics


if __name__ == "__main__":
    main()
