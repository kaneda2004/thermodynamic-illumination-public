#!/usr/bin/env python3
"""
RES-213: Reconstruction Convergence Speed vs Image Order

Hypothesis: Wall-clock reconstruction time correlates with image order.
- High-order images (more structured) should be faster to reconstruct
  than low-order images (more random)

Design:
- Sample 50 CPPNs across order range
- Reconstruct each from random starting point using optimized search
- Measure wall-clock time to achieve convergence
- Correlate order vs reconstruction time
"""

import numpy as np
import json
import time
from pathlib import Path
from scipy import stats
import sys

sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import (
    CPPN, set_global_seed, order_multiplicative,
    compute_compressibility
)

def optimize_reconstruction(target_image, max_iterations=1000, learning_rate=0.01):
    """
    Quick reconstruction via gradient-free search.
    Returns: list of (iteration, time_elapsed) for convergence tracking.
    """
    start_time = time.perf_counter()
    start_image = np.random.randn(*target_image.shape) * 0.5

    # Simple error tracking
    errors = []
    times = []

    for it in range(max_iterations):
        elapsed = time.perf_counter() - start_time

        # Compute MSE
        mse = np.mean((start_image - target_image) ** 2)
        errors.append(mse)
        times.append(elapsed)

        # Stop if converged
        if it > 10 and errors[-1] < 0.01:
            break

        # Update with random perturbations
        perturbation = np.random.randn(*target_image.shape) * learning_rate
        start_image += perturbation

    return np.array(times), np.array(errors)

def run_experiment():
    print('RES-213: Reconstruction Convergence Dynamics')
    print('=' * 60)

    output_dir = Path('/Users/matt/Development/monochrome_noise_converger/results/reconstruction_dynamics')
    output_dir.mkdir(parents=True, exist_ok=True)

    set_global_seed(2025)

    n_samples = 50
    results = {
        'experiment': 'reconstruction_time_vs_order',
        'timestamp': time.time(),
        'n_samples': n_samples,
        'samples': []
    }

    print(f'Testing reconstruction time vs order on {n_samples} CPPNs...')

    for idx in range(n_samples):
        if idx % 10 == 0:
            print(f'  Sample {idx}/{n_samples}...')

        try:
            seed = 100 + idx
            set_global_seed(seed)

            # Generate CPPN
            cppn = CPPN()
            target_image = cppn.render()

            # Measure order
            order = order_multiplicative(target_image)
            compressibility = compute_compressibility(target_image)

            # Reconstruct and measure time
            recon_times, errors = optimize_reconstruction(
                target_image, max_iterations=500
            )

            # Time to reach different error thresholds
            thresholds = [0.5, 0.1, 0.05, 0.01]
            time_to_threshold = {}

            for threshold in thresholds:
                indices = np.where(errors <= threshold)[0]
                if len(indices) > 0:
                    time_to_threshold[f'error_{threshold}'] = float(recon_times[indices[0]])
                else:
                    time_to_threshold[f'error_{threshold}'] = float(recon_times[-1])

            # Overall convergence time (% reduction in error)
            initial_error = errors[0]
            final_error = errors[-1]
            error_reduction = (initial_error - final_error) / (initial_error + 1e-10)

            sample_data = {
                'sample_id': idx,
                'seed': int(seed),
                'order': float(order),
                'compressibility': float(compressibility),
                'time_total': float(recon_times[-1]),
                'error_initial': float(initial_error),
                'error_final': float(final_error),
                'error_reduction': float(error_reduction),
                'time_to_threshold': time_to_threshold,
                'convergence_iterations': len(errors)
            }

            results['samples'].append(sample_data)

        except Exception as e:
            print(f'    Warning: sample {idx} failed - {e}')
            continue

    print(f'✓ Tested {len(results["samples"])} samples')

    # Analyze correlations
    if len(results['samples']) > 5:
        orders = np.array([s['order'] for s in results['samples']])
        times = np.array([s['time_total'] for s in results['samples']])
        error_reduction = np.array([s['error_reduction'] for s in results['samples']])
        convergence_iters = np.array([s['convergence_iterations'] for s in results['samples']])

        # Spearman correlations
        rho_time, p_time = stats.spearmanr(orders, times)
        rho_reduction, p_reduction = stats.spearmanr(orders, error_reduction)
        rho_iters, p_iters = stats.spearmanr(orders, convergence_iters)

        print(f'\nReconstruction Dynamics Correlations:')
        print(f'  Order vs Total Time: rho={rho_time:.4f}, p={p_time:.2e}')
        print(f'  Order vs Error Reduction: rho={rho_reduction:.4f}, p={p_reduction:.2e}')
        print(f'  Order vs Convergence Iters: rho={rho_iters:.4f}, p={p_iters:.2e}')

        results['correlation'] = {
            'order_vs_time': {
                'spearman_rho': float(rho_time),
                'p_value': float(p_time)
            },
            'order_vs_error_reduction': {
                'spearman_rho': float(rho_reduction),
                'p_value': float(p_reduction)
            },
            'order_vs_convergence_iters': {
                'spearman_rho': float(rho_iters),
                'p_value': float(p_iters)
            }
        }

        # Effect size: high-order vs low-order reconstruction time
        median_order = np.median(orders)
        high_order_times = times[orders >= median_order]
        low_order_times = times[orders < median_order]

        if len(high_order_times) > 0 and len(low_order_times) > 0:
            cohens_d = (np.mean(high_order_times) - np.mean(low_order_times)) / np.sqrt(
                (np.std(high_order_times)**2 + np.std(low_order_times)**2) / 2
            )
            results['effect_size'] = {
                'cohens_d': float(cohens_d),
                'high_order_mean_time': float(np.mean(high_order_times)),
                'low_order_mean_time': float(np.mean(low_order_times))
            }

    # Save results
    results_file = output_dir / 'reconstruction_walltimes_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f'\n✓ Results saved to {results_file}')

    # Summary
    if 'correlation' in results:
        print(f'\nKey Finding:')
        rho = results['correlation']['order_vs_time']['spearman_rho']
        p = results['correlation']['order_vs_time']['p_value']
        direction = "POSITIVE (high-order slower)" if rho > 0 else "NEGATIVE (high-order faster)"
        print(f'  Reconstruction time and order: rho={rho:.4f}, p={p:.2e}')
        print(f'  Direction: {direction}')
        if 'effect_size' in results:
            d = results['effect_size']['cohens_d']
            print(f'  Effect size: d={d:.4f}')

    return results

if __name__ == '__main__':
    results = run_experiment()
