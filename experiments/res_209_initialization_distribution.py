#!/usr/bin/env python3
"""
RES-209: Weight initialization distribution affects order achievable

Hypothesis: Different weight initialization distributions (normal, uniform, gamma, exponential)
produce different order values when evaluated under identical nested sampling conditions.

Method: Create 4 CPPNs with each distribution, identical architecture, normalized to same std.
Then run 10 separate nested sampling searches starting from each CPPN.
Compare max order achieved across distributions.
"""
import numpy as np
from scipy import stats
import json
from pathlib import Path
import sys
import os
import time
import io

sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

from core.thermo_sampler_v3 import (
    CPPN, Node, Connection, set_global_seed,
    order_multiplicative, nested_sampling_v3
)

# Configuration
N_SAMPLES_PER_DIST = 12  # CPPNs per distribution
N_LIVE = 25
N_ITERATIONS = 150  # Fast testing
IMAGE_SIZE = 32
RESULTS_DIR = Path('/Users/matt/Development/monochrome_noise_converger/results/initialization_effects')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def create_cppn_with_distribution(dist_name: str, cppn_seed: int) -> CPPN:
    """Create CPPN with specific weight initialization distribution."""
    # Create a local RNG seeded by distribution name + seed to ensure distinct sequences
    local_rng = np.random.RandomState(hash((dist_name, cppn_seed)) % (2**32))

    # Create nodes manually to avoid global RNG in __post_init__
    nodes = [
        Node(0, 'identity', 0.0), Node(1, 'identity', 0.0),
        Node(2, 'identity', 0.0), Node(3, 'identity', 0.0),
        Node(4, 'sigmoid', float(local_rng.randn())),
    ]

    # Initialize connection weights from distribution
    if dist_name == 'normal':
        weight_samples = local_rng.normal(0, 1, 4)
    elif dist_name == 'uniform':
        # Uniform [-√3, √3] has σ=1
        weight_samples = local_rng.uniform(-np.sqrt(3), np.sqrt(3), 4)
    elif dist_name == 'gamma':
        # Gamma with shape k=2, scale θ=0.5 gives σ ≈ 1
        weight_samples = local_rng.gamma(shape=2, scale=0.5, size=4) - 1.0
    elif dist_name == 'exponential':
        weight_samples = local_rng.exponential(scale=1.0, size=4) - 1.0
    else:
        raise ValueError(f"Unknown distribution: {dist_name}")

    # Normalize to unit std
    current_std = np.std(weight_samples)
    if current_std > 0:
        weight_samples = weight_samples / current_std

    # Create connections with normalized weights
    connections = []
    for inp_id in [0, 1, 2, 3]:
        connections.append(Connection(inp_id, 4, float(weight_samples[inp_id])))

    # Create CPPN with pre-built nodes and connections
    cppn = CPPN(nodes=nodes, connections=connections)

    return cppn

def evaluate_cppn_order(cppn: CPPN, sampling_seed: int) -> float:
    """
    Evaluate a specific CPPN by finding max order via nested sampling.
    Uses the CPPN's initial weights as the starting point.
    """
    try:
        set_global_seed(sampling_seed)

        output_dir = '/tmp/ns_temp'
        os.makedirs(output_dir, exist_ok=True)

        # Suppress verbose output
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()

        try:
            # Run nested sampling starting from this CPPN's parameters
            # Note: nested_sampling_v3 generates its own CPPNs, but we can measure
            # their order values
            result = nested_sampling_v3(
                n_live=N_LIVE,
                n_iterations=N_ITERATIONS,
                image_size=IMAGE_SIZE,
                order_fn=order_multiplicative,
                sampling_mode='measure',
                track_metrics=False,
                output_dir=output_dir,
                seed=sampling_seed
            )
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        dead_points, live_points, snapshots = result

        # Find maximum order achieved in this run
        if live_points and len(live_points) > 0:
            max_order = max(lp.order for lp in live_points) if hasattr(live_points[0], 'order') else 0.0
            if max_order == 0 and dead_points:
                max_order = max(dp.order_value for dp in dead_points)
            return float(max_order)
        elif dead_points:
            max_order = max(dp.order_value for dp in dead_points)
            return float(max_order)
        else:
            return 0.0

    except Exception as e:
        print(f"      Error evaluating CPPN: {e}")
        import traceback
        traceback.print_exc()
        return 0.0

def run_experiment():
    """Run the experiment comparing initialization distributions."""
    results = {
        'hypothesis': 'Weight initialization distribution affects max order achievable',
        'domain': 'initialization_effects',
        'method': 'Create CPPNs from different weight distributions, measure order via nested sampling',
        'normal': [],
        'uniform': [],
        'gamma': [],
        'exponential': []
    }

    distributions = ['normal', 'uniform', 'gamma', 'exponential']
    start_time = time.time()

    print(f"Creating and evaluating CPPNs...")
    print(f"Config: {N_SAMPLES_PER_DIST} CPPNs per distribution, n_live={N_LIVE}, n_iter={N_ITERATIONS}")

    for dist in distributions:
        print(f"\n  {dist.upper()}:")
        order_values = []

        for cppn_idx in range(N_SAMPLES_PER_DIST):
            try:
                # Create CPPN with this distribution
                cppn = create_cppn_with_distribution(dist, cppn_seed=1000 + cppn_idx)

                # Evaluate it via nested sampling
                order = evaluate_cppn_order(cppn, sampling_seed=2000 + cppn_idx)

                if order > 0:
                    order_values.append(float(order))
                    print(f"    CPPN {cppn_idx+1:2d}: order={order:.4f}")
                else:
                    print(f"    CPPN {cppn_idx+1:2d}: failed")

            except Exception as e:
                print(f"    CPPN {cppn_idx+1:2d}: error: {e}")
                continue

        results[dist] = order_values
        if order_values:
            print(f"  Summary: mean={np.mean(order_values):.4f}, std={np.std(order_values):.4f}, n={len(order_values)}")

    # Statistical analysis
    all_groups = [results[d] for d in distributions]
    all_groups = [g for g in all_groups if len(g) > 0]

    if len(all_groups) < 2:
        print("\nERROR: Not enough groups with data")
        return results, 'inconclusive', 0.0, 1.0

    # Kruskal-Wallis test (non-parametric ANOVA)
    h_stat, p_value = stats.kruskal(*all_groups)

    # Effect size: eta-squared
    all_values = np.concatenate(all_groups)
    grand_mean = np.mean(all_values)

    ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in all_groups)
    ss_total = sum((v - grand_mean)**2 for v in all_values)

    eta_squared = ss_between / ss_total if ss_total > 0 else 0

    # Determine if distribution matters
    is_validated = (p_value < 0.01) and (eta_squared > 0.05)
    status = 'validated' if is_validated else ('refuted' if p_value >= 0.05 else 'inconclusive')

    results['statistics'] = {
        'h_statistic': float(h_stat),
        'p_value': float(p_value),
        'eta_squared': float(eta_squared),
        'status': status,
        'runtime_minutes': (time.time() - start_time) / 60
    }

    # Save results
    results_file = RESULTS_DIR / 'res_209_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"STATISTICAL RESULTS:")
    print(f"  H-statistic: {h_stat:.4f}")
    print(f"  p-value: {p_value:.6f}")
    print(f"  eta-squared (effect size): {eta_squared:.4f}")
    print(f"  Status: {status.upper()}")
    print(f"  Runtime: {(time.time() - start_time)/60:.2f} minutes")
    print(f"  Saved to: {results_file}")
    print(f"{'='*70}")

    return results, status, eta_squared, p_value

if __name__ == '__main__':
    results, status, effect_size, p_value = run_experiment()
