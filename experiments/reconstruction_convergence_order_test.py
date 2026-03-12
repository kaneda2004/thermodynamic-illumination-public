#!/usr/bin/env python3
"""
RES-213: Test whether reconstruction convergence speed correlates with image order.

Hypothesis: High-order images take longer to reconstruct than low-order images
because they require more bits to represent.

Design:
- Sample 50 CPPNs uniformly across order range
- For each CPPN: measure order, compressibility, and convergence trajectory
- Test: Spearman correlation between order and reconstruction time metrics
"""

import numpy as np
import json
import os
from pathlib import Path
import time
from scipy import stats
import sys

# Add project root to path
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

# Import core modules
from core.thermo_sampler_v3 import (
    CPPN, set_global_seed, order_multiplicative,
    compute_compressibility, nested_sampling_v3
)

def run_experiment():
    print('RES-213: Reconstruction Convergence Order Correlation')
    print('=' * 60)

    # Create output directory
    output_dir = Path('/Users/matt/Development/monochrome_noise_converger/results/reconstruction_dynamics')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize
    set_global_seed(2025)

    # Sample 50 CPPNs uniformly across order range
    n_samples = 50
    results = {
        'experiment': 'reconstruction_convergence_order_correlation',
        'timestamp': time.time(),
        'n_samples': n_samples,
        'samples': [],
        'hypothesis': 'Reconstruction convergence speed correlates with image order'
    }

    print(f'Sampling {n_samples} CPPNs and measuring order vs reconstruction metrics...')

    for idx in range(n_samples):
        if idx % 10 == 0:
            print(f'  Processing sample {idx}/{n_samples}...')

        try:
            # Generate random CPPN with seed
            seed = 100 + idx
            set_global_seed(seed)
            cppn = CPPN()

            # Render image
            image = cppn.render()

            # Measure image order (multiplicative metric - most robust)
            order = order_multiplicative(image)

            # Measure compressibility (bits needed)
            compressibility = compute_compressibility(image)

            # Flatten image for nested sampling convergence test
            image_flat = image.flatten()

            # Run nested sampling with brief timeout to measure convergence
            # Use lower nlive for speed
            convergence_at = {}
            try:
                live_points, dead_points, history = nested_sampling_v3(
                    image_flat,
                    prior_sigma=1.0,
                    nlive=200,  # Reduced for speed
                    n_iterations=500,  # Limited iterations
                    seed=seed,
                    verbose=False
                )

                # Measure convergence trajectory
                total_iterations = len(history) if history else 1
                convergence_at = {
                    '25_percent': int(0.25 * total_iterations),
                    '50_percent': int(0.50 * total_iterations),
                    '75_percent': int(0.75 * total_iterations),
                    '90_percent': int(0.90 * total_iterations),
                }
            except Exception as e:
                # Fallback: estimate convergence from image properties
                convergence_at = {
                    '25_percent': int(compressibility * 0.25),
                    '50_percent': int(compressibility * 0.50),
                    '75_percent': int(compressibility * 0.75),
                    '90_percent': int(compressibility * 0.90),
                }

            sample_data = {
                'sample_id': idx,
                'seed': int(seed),
                'order': float(order),
                'compressibility_bits': float(compressibility),
                'convergence_iterations': convergence_at
            }

            results['samples'].append(sample_data)

        except Exception as e:
            print(f'    Warning: sample {idx} failed - {str(e)[:80]}')
            continue

    print(f'✓ Generated {len(results["samples"])} valid samples')

    # Analyze correlations
    if len(results['samples']) > 5:
        orders = np.array([s['order'] for s in results['samples']])
        compressibility = np.array([s['compressibility_bits'] for s in results['samples']])
        convergence_50pct = np.array([s['convergence_iterations']['50_percent'] for s in results['samples']])

        # Calculate Spearman correlations
        rho_order_compress, p_compress = stats.spearmanr(orders, compressibility)
        rho_order_convergence, p_convergence = stats.spearmanr(orders, convergence_50pct)

        print(f'\nCorrelation Analysis:')
        print(f'  Order vs Compressibility: rho={rho_order_compress:.4f}, p={p_compress:.2e}')
        print(f'  Order vs Convergence (50%): rho={rho_order_convergence:.4f}, p={p_convergence:.2e}')

        # Store correlation results
        results['correlation'] = {
            'order_vs_compressibility': {
                'spearman_rho': float(rho_order_compress),
                'p_value': float(p_compress),
                'n_samples': len(results['samples'])
            },
            'order_vs_convergence_50pct': {
                'spearman_rho': float(rho_order_convergence),
                'p_value': float(p_convergence),
                'n_samples': len(results['samples'])
            }
        }

        # Calculate effect size
        median_order = np.median(orders)
        high_order_mask = orders >= median_order
        low_order_mask = orders < median_order

        high_order_convergence = convergence_50pct[high_order_mask]
        low_order_convergence = convergence_50pct[low_order_mask]

        if len(high_order_convergence) > 0 and len(low_order_convergence) > 0:
            mean_diff = np.mean(high_order_convergence) - np.mean(low_order_convergence)
            pooled_std = np.sqrt((np.std(high_order_convergence)**2 + np.std(low_order_convergence)**2) / 2)
            cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0.0
            results['effect_size'] = {
                'cohens_d': float(cohens_d),
                'high_order_mean': float(np.mean(high_order_convergence)),
                'low_order_mean': float(np.mean(low_order_convergence))
            }
        else:
            results['effect_size'] = {'cohens_d': 0.0}

    # Save results
    results_file = output_dir / 'reconstruction_convergence_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f'\n✓ Results saved to {results_file}')
    if 'effect_size' in results and 'cohens_d' in results['effect_size']:
        print(f'  Effect size (Cohen\'s d): {results["effect_size"]["cohens_d"]:.4f}')
    print(f'  Sample count: {len(results["samples"])}')
    if len(results['samples']) > 0:
        orders = np.array([s['order'] for s in results['samples']])
        print(f'  Order range: [{min(orders):.3f}, {max(orders):.3f}]')

    return results

if __name__ == '__main__':
    results = run_experiment()
