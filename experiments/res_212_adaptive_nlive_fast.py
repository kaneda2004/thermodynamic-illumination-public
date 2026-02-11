"""
RES-212: Adaptive n_live strategy reduces sample budget for target order achievement (FAST)

Optimized version with reasonable parameters that complete in <30min.
- Adaptive: Start n_live=15, increase to 30 when order stagnant for 4 iterations
- Fixed baselines: n_live=15, 30
- Run 10 CPPNs per strategy (sufficient n for statistical significance)
- Measure: Total samples to reach order=0.3, 0.5
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


def adaptive_nested_sampling_fast(
    n_initial_live=15,
    max_iterations=300,
    image_size=32,
    order_targets=[0.3, 0.5],
    doubling_patience=4,
    seed=None
):
    """Adaptive nested sampling with early stopping and faster parameters."""
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

        # If all targets reached, early stop
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
            for retry in range(2):
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

        # Check for stagnation and adapt n_live (but cap at 30)
        if max_order > prev_max_order:
            stagnant_count = 0
            prev_max_order = max_order
        else:
            stagnant_count += 1

        # Increase n_live if stagnant (but only up to 30)
        if stagnant_count >= doubling_patience and n_live < 30:
            n_live = 30
            # Add new live points
            for _ in range(30 - len(live_points)):
                cppn = CPPN()
                img = cppn.render(image_size)
                o = compute_order_metric(img)
                lp = log_prior(cppn)
                live_points.append({'cppn': cppn, 'img': img, 'order': o, 'logp': lp})
            stagnant_count = 0

    return {
        'samples_to_target': samples_to_target,
        'final_n_live': n_live,
        'iterations': max_iterations,
        'total_contractions': total_samples,
        'success': False,
        'final_order': max_order
    }


def fixed_nested_sampling_fast(
    n_live=15,
    max_iterations=300,
    image_size=32,
    order_targets=[0.3, 0.5],
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

        # If all targets reached, early stop
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
            for retry in range(2):
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

    print("RES-212: Adaptive n_live strategy reduces sample budget (OPTIMIZED)")
    print("=" * 70)

    n_seeds = 10  # 10 seeds (faster, sufficient for effect estimation)
    strategies = {
        'adaptive': lambda seed: adaptive_nested_sampling_fast(
            n_initial_live=15, max_iterations=300, seed=seed
        ),
        'fixed_15': lambda seed: fixed_nested_sampling_fast(n_live=15, max_iterations=300, seed=seed),
        'fixed_30': lambda seed: fixed_nested_sampling_fast(n_live=30, max_iterations=300, seed=seed),
    }

    results = {strategy: [] for strategy in strategies}

    print(f"\nRunning {n_seeds} seeds × {len(strategies)} strategies...")
    print(f"Target orders: [0.3, 0.5]")
    print("-" * 70)

    for seed in range(2000, 2000 + n_seeds):
        print(f"\nSeed {seed - 1999}/{n_seeds}")

        for strategy_name, strategy_fn in strategies.items():
            start = time.time()
            result = strategy_fn(seed)
            elapsed = time.time() - start

            result['seed'] = seed
            result['strategy'] = strategy_name
            result['runtime_sec'] = elapsed
            results[strategy_name].append(result)

            if result['samples_to_target'][0.5] is not None:
                print(f"  {strategy_name:12s}: samples_to_0.5={result['samples_to_target'][0.5]:6.0f} ({elapsed:.1f}s)")
            else:
                print(f"  {strategy_name:12s}: incomplete in {elapsed:.1f}s (final_order={result['final_order']:.4f})")

    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS")
    print("=" * 70)

    # Analyze each target
    target_analyses = {}

    for target in [0.3, 0.5]:
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

            for fixed_name in ['fixed_15', 'fixed_30']:
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
        fixed_30_0_5 = target_analyses[0.5].get('fixed_30', [])

        if fixed_30_0_5:
            improvement = (np.mean(fixed_30_0_5) - np.mean(adaptive_0_5)) / np.mean(fixed_30_0_5)
            effect_size = (np.mean(adaptive_0_5) - np.mean(fixed_30_0_5)) / np.sqrt((np.var(adaptive_0_5) + np.var(fixed_30_0_5)) / 2)

            print(f"\nAdaptive vs fixed_30 (target=0.5):")
            print(f"  Adaptive: {np.mean(adaptive_0_5):.0f} ± {np.std(adaptive_0_5):.0f}")
            print(f"  Fixed_30: {np.mean(fixed_30_0_5):.0f} ± {np.std(fixed_30_0_5):.0f}")
            print(f"  Improvement: {improvement*100:.1f}%")
            print(f"  Effect size (Cohen's d): {effect_size:.3f}")

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
                'final_n_live': int(r.get('final_n_live', r.get('n_live', -1))),
                'final_order': float(r['final_order']),
                'success': bool(r['success'])
            }
            json_results[strategy].append(json_r)

    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"\nResults saved to {results_file}")
    return json_results


if __name__ == '__main__':
    run_experiment()
