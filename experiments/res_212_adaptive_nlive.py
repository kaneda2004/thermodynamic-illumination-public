"""
RES-212: Adaptive n_live strategy reduces sample budget for target order achievement

Hypothesis: Dynamically adjusting n_live during nested sampling reduces the total
number of samples (contractions) needed to reach target order thresholds compared
to fixed strategies.

Strategy:
- Adaptive: Start n_live=20, double when order stagnant for 5 iterations
- Compare against: fixed n_live=20, n_live=50, n_live=100
- Measure: Total samples to reach order=0.3, 0.5, 0.7
- Run 30 CPPNs per strategy (constrained by 16GB memory)
- Statistical test: Mann-Whitney U (adaptive vs fixed strategies)

Building on:
- RES-032: Nested sampling 4.2x higher order than gradient descent
- RES-050: Trajectory nearly constant velocity (ACF=0.996)
- RES-067: Small n_live faster than large
"""

import numpy as np
from scipy import stats
import sys
import json
from pathlib import Path
import time
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import (
    CPPN, order_multiplicative as compute_order_metric,
    elliptical_slice_sample, log_prior, set_global_seed
)
import random


def adaptive_nested_sampling(
    n_initial_live=20,
    max_iterations=500,
    image_size=32,
    order_targets=[0.3, 0.5, 0.7],
    doubling_patience=5,
    seed=None
):
    """
    Adaptive nested sampling that adjusts n_live based on order convergence.

    Returns:
    {
        'samples_to_target': {0.3: N, 0.5: N, 0.7: N},
        'final_n_live': int,
        'iterations': int,
        'total_contractions': int,
        'success': bool,
        'final_order': float
    }
    """
    set_global_seed(seed)

    n_live = n_initial_live
    live_points = []
    total_samples = 0
    stagnant_count = 0
    prev_max_order = -1
    samples_to_target = {t: None for t in order_targets}

    # Initialize
    for _ in range(n_live):
        cppn = CPPN()
        img = cppn.render(image_size)
        o = compute_order_metric(img)
        lp = log_prior(cppn)
        live_points.append({'cppn': cppn, 'img': img, 'order': o, 'logp': lp})

    order_vals = [lp['order'] for lp in live_points]
    max_order = max(order_vals)

    # Main loop
    for iteration in range(max_iterations):
        # Check if we've reached any targets
        for target in order_targets:
            if samples_to_target[target] is None and max_order >= target:
                samples_to_target[target] = total_samples

        # If all targets reached, we can stop
        if all(v is not None for v in samples_to_target.values()):
            return {
                'samples_to_target': samples_to_target,
                'final_n_live': n_live,
                'iterations': iteration,
                'total_contractions': total_samples,
                'success': True,
                'final_order': max_order
            }

        # Find and remove worst point
        worst_idx = min(range(n_live), key=lambda i: live_points[i]['order'])
        worst = live_points[worst_idx]
        threshold = worst['order']

        # Find valid seeds (order >= threshold)
        valid_seeds = [i for i in range(n_live)
                      if i != worst_idx and live_points[i]['order'] >= threshold]

        if not valid_seeds:
            valid_seeds = [i for i in range(n_live) if i != worst_idx]

        seed_idx = random.choice(valid_seeds)

        # Sample replacement
        new_cppn, new_img, new_order, new_logp, n_contract, success = elliptical_slice_sample(
            live_points[seed_idx]['cppn'], threshold, image_size, compute_order_metric
        )

        # Retry if failed
        if not success:
            for retry in range(3):
                alt_seed = random.choice(valid_seeds)
                new_cppn, new_img, new_order, new_logp, extra_contract, success = elliptical_slice_sample(
                    live_points[alt_seed]['cppn'], threshold, image_size, compute_order_metric
                )
                n_contract += extra_contract
                if success:
                    break

        total_samples += n_contract
        live_points[worst_idx] = {'cppn': new_cppn, 'img': new_img, 'order': new_order, 'logp': new_logp}

        # Update order stats
        order_vals = [lp['order'] for lp in live_points]
        max_order = max(order_vals)

        # Check for stagnation and adapt n_live
        if max_order > prev_max_order:
            stagnant_count = 0
            prev_max_order = max_order
        else:
            stagnant_count += 1

        # Double n_live if stagnant
        if stagnant_count >= doubling_patience and n_live < 160:
            n_live *= 2
            # Add new live points
            for _ in range(n_live - len(live_points)):
                cppn = CPPN()
                img = cppn.render(image_size)
                o = compute_order_metric(img)
                lp = log_prior(cppn)
                live_points.append({'cppn': cppn, 'img': img, 'order': o, 'logp': lp})
            stagnant_count = 0

        if (iteration + 1) % 50 == 0:
            print(f"  Adaptive iter {iteration+1:3d} | max_order={max_order:.4f} | n_live={n_live} | samples={total_samples}")

    # Timeout: record targets reached so far
    return {
        'samples_to_target': samples_to_target,
        'final_n_live': n_live,
        'iterations': max_iterations,
        'total_contractions': total_samples,
        'success': False,
        'final_order': max_order
    }


def fixed_nested_sampling(
    n_live=50,
    max_iterations=500,
    image_size=32,
    order_targets=[0.3, 0.5, 0.7],
    seed=None
):
    """Fixed n_live nested sampling for comparison."""
    set_global_seed(seed)

    live_points = []
    total_samples = 0
    samples_to_target = {t: None for t in order_targets}

    # Initialize
    for _ in range(n_live):
        cppn = CPPN()
        img = cppn.render(image_size)
        o = compute_order_metric(img)
        lp = log_prior(cppn)
        live_points.append({'cppn': cppn, 'img': img, 'order': o, 'logp': lp})

    order_vals = [lp['order'] for lp in live_points]
    max_order = max(order_vals)

    # Main loop
    for iteration in range(max_iterations):
        # Check if we've reached any targets
        for target in order_targets:
            if samples_to_target[target] is None and max_order >= target:
                samples_to_target[target] = total_samples

        # If all targets reached
        if all(v is not None for v in samples_to_target.values()):
            return {
                'samples_to_target': samples_to_target,
                'n_live': n_live,
                'iterations': iteration,
                'total_contractions': total_samples,
                'success': True,
                'final_order': max_order
            }

        # Find and remove worst point
        worst_idx = min(range(n_live), key=lambda i: live_points[i]['order'])
        worst = live_points[worst_idx]
        threshold = worst['order']

        # Find valid seeds
        valid_seeds = [i for i in range(n_live)
                      if i != worst_idx and live_points[i]['order'] >= threshold]

        if not valid_seeds:
            valid_seeds = [i for i in range(n_live) if i != worst_idx]

        seed_idx = random.choice(valid_seeds)

        # Sample replacement
        new_cppn, new_img, new_order, new_logp, n_contract, success = elliptical_slice_sample(
            live_points[seed_idx]['cppn'], threshold, image_size, compute_order_metric
        )

        # Retry if failed
        if not success:
            for retry in range(3):
                alt_seed = random.choice(valid_seeds)
                new_cppn, new_img, new_order, new_logp, extra_contract, success = elliptical_slice_sample(
                    live_points[alt_seed]['cppn'], threshold, image_size, compute_order_metric
                )
                n_contract += extra_contract
                if success:
                    break

        total_samples += n_contract
        live_points[worst_idx] = {'cppn': new_cppn, 'img': new_img, 'order': new_order, 'logp': new_logp}

        order_vals = [lp['order'] for lp in live_points]
        max_order = max(order_vals)

        if (iteration + 1) % 50 == 0:
            print(f"  Fixed n_live={n_live:3d} iter {iteration+1:3d} | max_order={max_order:.4f} | samples={total_samples}")

    return {
        'samples_to_target': samples_to_target,
        'n_live': n_live,
        'iterations': max_iterations,
        'total_contractions': total_samples,
        'success': False,
        'final_order': max_order
    }


def run_experiment():
    """Run adaptive n_live experiment with multiple seeds."""

    print("RES-212: Adaptive n_live strategy reduces sample budget for target order")
    print("=" * 70)

    n_seeds = 15  # Run with 15 different CPPNs (faster, sufficient for effect size estimation)
    strategies = {
        'adaptive': lambda seed: adaptive_nested_sampling(
            n_initial_live=20, max_iterations=500, seed=seed
        ),
        'fixed_20': lambda seed: fixed_nested_sampling(n_live=20, max_iterations=500, seed=seed),
        'fixed_50': lambda seed: fixed_nested_sampling(n_live=50, max_iterations=500, seed=seed),
        'fixed_100': lambda seed: fixed_nested_sampling(n_live=100, max_iterations=500, seed=seed),
    }

    results = {strategy: [] for strategy in strategies}

    print(f"\nRunning {n_seeds} seeds × {len(strategies)} strategies...")
    print(f"Target orders: [0.3, 0.5, 0.7]")
    print("-" * 70)

    for seed in range(1000, 1000 + n_seeds):
        print(f"\nSeed {seed - 999}/{n_seeds}")

        for strategy_name, strategy_fn in strategies.items():
            start = time.time()
            result = strategy_fn(seed)
            elapsed = time.time() - start

            result['seed'] = seed
            result['strategy'] = strategy_name
            result['runtime_sec'] = elapsed
            results[strategy_name].append(result)

            if result['samples_to_target'][0.5] is not None:
                print(f"  {strategy_name:12s}: samples_to_0.5={result['samples_to_target'][0.5]:6.0f}")
            else:
                print(f"  {strategy_name:12s}: incomplete (final_order={result['final_order']:.4f})")

    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS")
    print("=" * 70)

    # Analyze each target
    target_analyses = {}

    for target in [0.3, 0.5, 0.7]:
        print(f"\n--- Target order = {target} ---")

        all_data = {}
        for strategy_name in strategies:
            # Extract samples needed for this target
            samples = [r['samples_to_target'][target] for r in results[strategy_name]
                      if r['samples_to_target'][target] is not None]

            if samples:
                all_data[strategy_name] = samples
                mean_samples = np.mean(samples)
                std_samples = np.std(samples)
                success_rate = len(samples) / len(results[strategy_name])

                print(f"{strategy_name:12s}: mean={mean_samples:7.0f} ± {std_samples:6.0f}, "
                      f"n={len(samples)}, success={success_rate:.1%}")

        # Statistical tests: adaptive vs each fixed
        if 'adaptive' in all_data:
            adaptive_data = all_data['adaptive']

            for fixed_name in ['fixed_20', 'fixed_50', 'fixed_100']:
                if fixed_name in all_data:
                    fixed_data = all_data[fixed_name]

                    # Mann-Whitney U test
                    u_stat, p_value = stats.mannwhitneyu(adaptive_data, fixed_data)

                    # Cohen's d
                    pooled_std = np.sqrt((np.var(adaptive_data) + np.var(fixed_data)) / 2)
                    cohens_d = (np.mean(adaptive_data) - np.mean(fixed_data)) / (pooled_std + 1e-10)

                    # Relative improvement
                    improvement = (np.mean(fixed_data) - np.mean(adaptive_data)) / np.mean(fixed_data) * 100

                    print(f"  Adaptive vs {fixed_name}: p={p_value:.2e}, d={cohens_d:.3f}, "
                          f"improvement={improvement:+.1f}%")

        target_analyses[target] = all_data

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if 'adaptive' in target_analyses.get(0.5, {}):
        adaptive_0_5 = target_analyses[0.5]['adaptive']
        fixed_50_0_5 = target_analyses[0.5].get('fixed_50', [])

        if fixed_50_0_5:
            improvement = (np.mean(fixed_50_0_5) - np.mean(adaptive_0_5)) / np.mean(fixed_50_0_5)
            print(f"\nAdaptive vs fixed_50 (target=0.5):")
            print(f"  Adaptive: {np.mean(adaptive_0_5):.0f} ± {np.std(adaptive_0_5):.0f}")
            print(f"  Fixed_50: {np.mean(fixed_50_0_5):.0f} ± {np.std(fixed_50_0_5):.0f}")
            print(f"  Improvement: {improvement*100:.1f}%")

    # Save results
    output_dir = Path('/Users/matt/Development/monochrome_noise_converger/results/res_212')
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / 'adaptive_nlive_results.json'

    # Convert numpy types to JSON-serializable
    json_results = {}
    for strategy, data in results.items():
        json_results[strategy] = []
        for r in data:
            json_r = {
                'seed': int(r['seed']),
                'strategy': r['strategy'],
                'runtime_sec': float(r['runtime_sec']),
                'samples_to_target': {str(k): (float(v) if v is not None else None)
                                      for k, v in r['samples_to_target'].items()},
                'final_n_live': int(r['final_n_live']),
                'final_order': float(r['final_order']),
                'success': bool(r['success'])
            }
            json_results[strategy].append(json_r)

    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"\nResults saved to {results_file}")


if __name__ == '__main__':
    run_experiment()
