"""
RES-177: NS order progression accelerates during low-to-mid order transition phase

HYPOTHESIS: NS order progression accelerates during low-to-mid order transition phase

DOMAIN: convergence_dynamics

RATIONALE:
- RES-145: gradient magnitude starts very low (0.07), increases 37x during NS
- RES-154: mid-order images have highest sensitivity (10.8x variation)
- Combined: order should be slow in low-order (flat gradient), accelerate in mid-order
  (steep gradient), then potentially slow again in high-order (lower gradient than mid)

METHOD:
1. Run multiple NS trajectories, recording (iteration, order) pairs
2. Segment trajectories into phases by order level: low (<0.1), mid (0.1-0.5), high (>0.5)
3. Compute order velocity (d_order/d_iteration) in each phase
4. Test: mid-phase velocity > low-phase velocity (with d>0.5, p<0.01)
"""

import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')

import numpy as np
import json
import os
from scipy import stats
from core.thermo_sampler_v3 import (
    CPPN, order_multiplicative, elliptical_slice_sample, log_prior
)


def run_ns_trajectory(n_iterations=300, n_live=50, image_size=32, seed=None):
    """Run a simplified NS and return (iteration, min_order) trajectory."""
    if seed is not None:
        np.random.seed(seed)

    # Initialize live points
    live_points = []
    for _ in range(n_live):
        cppn = CPPN()
        img = cppn.render(image_size)
        order = order_multiplicative(img)
        live_points.append({'cppn': cppn, 'order': order})

    trajectory = []

    for iteration in range(n_iterations):
        # Get current minimum (threshold)
        orders = [lp['order'] for lp in live_points]
        min_idx = np.argmin(orders)
        min_order = orders[min_idx]

        trajectory.append({
            'iteration': iteration,
            'min_order': min_order,
            'mean_order': np.mean(orders),
            'max_order': max(orders)
        })

        # Replace worst point via ESS
        # Pick a random seed (not the worst one)
        valid_seeds = [i for i in range(n_live) if i != min_idx]
        seed_idx = np.random.choice(valid_seeds)
        seed_cppn = live_points[seed_idx]['cppn']

        # ESS to find replacement above threshold
        new_cppn, new_img, new_order, _, _, success = elliptical_slice_sample(
            seed_cppn, min_order, image_size, order_multiplicative,
            max_contractions=50, max_restarts=3
        )

        if success:
            live_points[min_idx] = {'cppn': new_cppn, 'order': new_order}

    return trajectory


def compute_phase_velocities(trajectory):
    """
    Compute order velocity in three phases:
    - low: order < 0.1
    - mid: 0.1 <= order < 0.5
    - high: order >= 0.5

    Velocity = (order_end - order_start) / (iter_end - iter_start) for each phase
    """
    orders = np.array([t['min_order'] for t in trajectory])
    iterations = np.array([t['iteration'] for t in trajectory])

    # Find phase boundaries
    low_mask = orders < 0.1
    mid_mask = (orders >= 0.1) & (orders < 0.5)
    high_mask = orders >= 0.5

    phases = {}

    # Compute velocity as slope of order vs iteration in each phase
    for phase_name, mask in [('low', low_mask), ('mid', mid_mask), ('high', high_mask)]:
        if np.sum(mask) >= 5:  # Need at least 5 points
            phase_orders = orders[mask]
            phase_iters = iterations[mask]

            # Use regression slope for velocity
            slope, _, r_value, p_value, _ = stats.linregress(phase_iters, phase_orders)

            phases[phase_name] = {
                'n_points': int(np.sum(mask)),
                'velocity': slope,
                'order_start': float(phase_orders[0]),
                'order_end': float(phase_orders[-1]),
                'iter_span': int(phase_iters[-1] - phase_iters[0]) if len(phase_iters) > 1 else 0,
                'r2': r_value**2,
                'p_value': p_value
            }
        else:
            phases[phase_name] = {
                'n_points': int(np.sum(mask)),
                'velocity': np.nan,
                'order_start': np.nan,
                'order_end': np.nan,
                'iter_span': 0,
                'r2': np.nan,
                'p_value': np.nan
            }

    return phases


def main():
    print("="*70)
    print("RES-177: NS Order Acceleration Study")
    print("="*70)

    n_runs = 30  # Multiple trajectories for statistics
    n_iterations = 400  # Enough to span low->mid->high

    all_low_velocities = []
    all_mid_velocities = []
    all_high_velocities = []
    all_runs = []

    print(f"\nRunning {n_runs} NS trajectories...")

    for run_idx in range(n_runs):
        if (run_idx + 1) % 5 == 0:
            print(f"  Run {run_idx + 1}/{n_runs}")

        trajectory = run_ns_trajectory(n_iterations=n_iterations, seed=42 + run_idx)
        phases = compute_phase_velocities(trajectory)

        all_runs.append({
            'run': run_idx,
            'phases': phases,
            'final_order': trajectory[-1]['min_order']
        })

        if not np.isnan(phases['low']['velocity']):
            all_low_velocities.append(phases['low']['velocity'])
        if not np.isnan(phases['mid']['velocity']):
            all_mid_velocities.append(phases['mid']['velocity'])
        if not np.isnan(phases['high']['velocity']):
            all_high_velocities.append(phases['high']['velocity'])

    print(f"\nPhase velocity statistics:")
    print(f"  Low phase:  n={len(all_low_velocities)}, mean={np.mean(all_low_velocities):.6f}, std={np.std(all_low_velocities):.6f}")
    print(f"  Mid phase:  n={len(all_mid_velocities)}, mean={np.mean(all_mid_velocities):.6f}, std={np.std(all_mid_velocities):.6f}")
    print(f"  High phase: n={len(all_high_velocities)}, mean={np.mean(all_high_velocities):.6f}, std={np.std(all_high_velocities):.6f}")

    # Statistical tests
    # Primary test: mid > low
    if len(all_mid_velocities) >= 5 and len(all_low_velocities) >= 5:
        t_mid_vs_low, p_mid_vs_low = stats.ttest_ind(all_mid_velocities, all_low_velocities)
        d_mid_vs_low = (np.mean(all_mid_velocities) - np.mean(all_low_velocities)) / \
                       np.sqrt((np.std(all_mid_velocities)**2 + np.std(all_low_velocities)**2) / 2)

        print(f"\nMid vs Low phase:")
        print(f"  t-statistic: {t_mid_vs_low:.3f}")
        print(f"  p-value: {p_mid_vs_low:.2e}")
        print(f"  Cohen's d: {d_mid_vs_low:.3f}")
    else:
        p_mid_vs_low, d_mid_vs_low = 1.0, 0.0
        print("\nInsufficient data for mid vs low comparison")

    # Secondary test: mid vs high
    if len(all_mid_velocities) >= 5 and len(all_high_velocities) >= 5:
        t_mid_vs_high, p_mid_vs_high = stats.ttest_ind(all_mid_velocities, all_high_velocities)
        d_mid_vs_high = (np.mean(all_mid_velocities) - np.mean(all_high_velocities)) / \
                        np.sqrt((np.std(all_mid_velocities)**2 + np.std(all_high_velocities)**2) / 2)

        print(f"\nMid vs High phase:")
        print(f"  t-statistic: {t_mid_vs_high:.3f}")
        print(f"  p-value: {p_mid_vs_high:.2e}")
        print(f"  Cohen's d: {d_mid_vs_high:.3f}")
    else:
        p_mid_vs_high, d_mid_vs_high = 1.0, 0.0
        print("\nInsufficient data for mid vs high comparison")

    # Check velocity monotonicity (acceleration pattern)
    # If hypothesis is correct: low < mid (acceleration), possibly mid > high (deceleration)
    mean_low = np.mean(all_low_velocities) if all_low_velocities else 0
    mean_mid = np.mean(all_mid_velocities) if all_mid_velocities else 0
    mean_high = np.mean(all_high_velocities) if all_high_velocities else 0

    acceleration_ratio = mean_mid / mean_low if mean_low > 0 else float('inf')

    print(f"\nAcceleration ratio (mid/low): {acceleration_ratio:.2f}x")

    # Determine result
    validated = (d_mid_vs_low > 0.5 and p_mid_vs_low < 0.01 and mean_mid > mean_low)

    status = "validated" if validated else "refuted"

    if d_mid_vs_low > 0.5 and p_mid_vs_low < 0.01 and mean_mid < mean_low:
        # Opposite direction - mid is slower than low
        summary = f"Mid-phase shows LOWER velocity than low-phase (d={d_mid_vs_low:.2f}, p={p_mid_vs_low:.2e}). " \
                  f"Low: {mean_low:.5f}, Mid: {mean_mid:.5f}, High: {mean_high:.5f}. " \
                  f"Order progression decelerates after leaving low-order regime, opposite to hypothesis."
    elif validated:
        summary = f"Mid-phase shows {acceleration_ratio:.1f}x higher velocity (d={d_mid_vs_low:.2f}, p={p_mid_vs_low:.2e}). " \
                  f"Low: {mean_low:.5f}, Mid: {mean_mid:.5f}, High: {mean_high:.5f}. " \
                  f"NS order progression accelerates during low-to-mid transition."
    else:
        summary = f"No significant acceleration pattern (d={d_mid_vs_low:.2f}, p={p_mid_vs_low:.2e}). " \
                  f"Low: {mean_low:.5f}, Mid: {mean_mid:.5f}, High: {mean_high:.5f}. " \
                  f"Order velocity is similar across phases."

    print(f"\n{'='*70}")
    print(f"STATUS: {status.upper()}")
    print(f"SUMMARY: {summary}")
    print(f"{'='*70}")

    # Save results
    results = {
        'experiment': 'RES-177',
        'domain': 'convergence_dynamics',
        'hypothesis': 'NS order progression accelerates during low-to-mid order transition phase',
        'status': status,
        'metrics': {
            'n_runs': n_runs,
            'n_iterations': n_iterations,
            'velocities': {
                'low_mean': float(mean_low),
                'low_std': float(np.std(all_low_velocities)) if all_low_velocities else 0,
                'mid_mean': float(mean_mid),
                'mid_std': float(np.std(all_mid_velocities)) if all_mid_velocities else 0,
                'high_mean': float(mean_high),
                'high_std': float(np.std(all_high_velocities)) if all_high_velocities else 0,
            },
            'mid_vs_low': {
                'effect_size': float(d_mid_vs_low),
                'p_value': float(p_mid_vs_low),
            },
            'mid_vs_high': {
                'effect_size': float(d_mid_vs_high),
                'p_value': float(p_mid_vs_high),
            },
            'acceleration_ratio': float(acceleration_ratio) if not np.isinf(acceleration_ratio) else None,
        },
        'summary': summary
    }

    os.makedirs('/Users/matt/Development/monochrome_noise_converger/results/res_177', exist_ok=True)
    with open('/Users/matt/Development/monochrome_noise_converger/results/res_177/results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to results/res_177/results.json")

    return results


if __name__ == '__main__':
    main()
