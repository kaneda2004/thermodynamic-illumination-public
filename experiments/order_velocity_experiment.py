"""
Experiment RES-050: Order Velocity Power-Law Pattern

Hypothesis: Order velocity (dO/dt) during nested sampling follows a power-law
deceleration pattern, with faster gains at low order that decelerate as high
order is approached.

Approach:
1. Run multiple nested sampling trajectories
2. Compute order velocity dO/dt at each iteration
3. Fit power-law model: dO/dt ~ O^(-alpha)
4. Test if alpha > 0 (deceleration at high order)
"""

import numpy as np
import json
import os
from scipy import stats
from scipy.optimize import curve_fit
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.thermo_sampler_v3 import (
    CPPN, order_multiplicative, elliptical_slice_sample
)

def run_sampling_trajectory(n_live=50, n_iterations=400, image_size=32, seed=None):
    """Run nested sampling and return order trajectory."""
    if seed is not None:
        np.random.seed(seed)

    order_fn = order_multiplicative

    # Initialize live points
    live_points = []
    for _ in range(n_live):
        cppn = CPPN()
        img = cppn.render(image_size)
        o = order_fn(img)
        live_points.append({'cppn': cppn, 'img': img, 'order': o})

    # Track order trajectory (threshold at each iteration)
    order_trajectory = []

    for iteration in range(n_iterations):
        # Find worst (lowest order)
        worst_idx = min(range(n_live), key=lambda i: live_points[i]['order'])
        threshold = live_points[worst_idx]['order']
        order_trajectory.append(threshold)

        # Select seed
        valid_seeds = [i for i in range(n_live) if i != worst_idx]
        seed_idx = np.random.choice(valid_seeds)

        # Replace using ESS
        new_cppn, new_img, new_order, _, _, success = elliptical_slice_sample(
            live_points[seed_idx]['cppn'], threshold, image_size, order_fn
        )

        if not success:
            # If ESS fails, just duplicate a valid point
            seed_idx = np.random.choice(valid_seeds)
            new_cppn = live_points[seed_idx]['cppn'].copy()
            new_img = live_points[seed_idx]['img'].copy()
            new_order = live_points[seed_idx]['order']

        live_points[worst_idx] = {'cppn': new_cppn, 'img': new_img, 'order': new_order}

    return np.array(order_trajectory)


def compute_velocity(trajectory, window=5):
    """Compute smoothed velocity (finite difference)."""
    # Use central difference with smoothing
    velocities = []
    orders = []

    for i in range(window, len(trajectory) - window):
        # Central difference: dO/dt = (O[i+w] - O[i-w]) / (2*w)
        dO = trajectory[i + window] - trajectory[i - window]
        dt = 2 * window
        velocities.append(dO / dt)
        orders.append(trajectory[i])

    return np.array(orders), np.array(velocities)


def power_law(x, alpha, c):
    """Power law: y = c * x^(-alpha)"""
    return c * np.power(x + 0.01, -alpha)  # Add small offset to avoid division by zero


def main():
    print("=" * 70)
    print("EXPERIMENT RES-050: Order Velocity Power-Law Pattern")
    print("=" * 70)

    n_trajectories = 30
    n_live = 50
    n_iterations = 400
    image_size = 32

    print(f"\nParameters:")
    print(f"  n_trajectories: {n_trajectories}")
    print(f"  n_live: {n_live}")
    print(f"  n_iterations: {n_iterations}")
    print(f"  image_size: {image_size}")

    # Collect trajectories
    print("\nRunning sampling trajectories...")
    all_orders = []
    all_velocities = []
    trajectory_alphas = []

    for i in range(n_trajectories):
        if (i + 1) % 10 == 0:
            print(f"  Trajectory {i+1}/{n_trajectories}")

        trajectory = run_sampling_trajectory(
            n_live=n_live,
            n_iterations=n_iterations,
            image_size=image_size,
            seed=i * 42
        )

        orders, velocities = compute_velocity(trajectory, window=5)

        # Only keep positive velocities (order increases)
        mask = velocities > 0
        orders_pos = orders[mask]
        velocities_pos = velocities[mask]

        all_orders.extend(orders_pos)
        all_velocities.extend(velocities_pos)

        # Fit power law to this trajectory
        if len(orders_pos) > 20:
            try:
                popt, _ = curve_fit(power_law, orders_pos, velocities_pos,
                                   p0=[0.5, 0.01], maxfev=5000,
                                   bounds=([-2, 0], [5, 1]))
                trajectory_alphas.append(popt[0])
            except:
                pass

    all_orders = np.array(all_orders)
    all_velocities = np.array(all_velocities)
    trajectory_alphas = np.array(trajectory_alphas)

    print(f"\nData points collected: {len(all_orders)}")
    print(f"Successful power-law fits: {len(trajectory_alphas)}")

    # Analysis 1: Fit combined power law
    print("\n" + "-" * 70)
    print("ANALYSIS 1: Combined Power-Law Fit")
    print("-" * 70)

    try:
        popt, pcov = curve_fit(power_law, all_orders, all_velocities,
                               p0=[0.5, 0.01], maxfev=10000,
                               bounds=([-2, 0], [5, 1]))
        alpha_combined = popt[0]
        c_combined = popt[1]
        alpha_se = np.sqrt(pcov[0, 0])

        print(f"Combined fit: dO/dt ~ {c_combined:.4f} * O^(-{alpha_combined:.3f})")
        print(f"Alpha = {alpha_combined:.3f} +/- {alpha_se:.3f}")

        # Predict and compute R^2
        predicted = power_law(all_orders, *popt)
        ss_res = np.sum((all_velocities - predicted) ** 2)
        ss_tot = np.sum((all_velocities - np.mean(all_velocities)) ** 2)
        r_squared = 1 - ss_res / ss_tot
        print(f"R^2 = {r_squared:.4f}")
    except Exception as e:
        print(f"Combined fit failed: {e}")
        alpha_combined = np.nan
        r_squared = np.nan

    # Analysis 2: Per-trajectory alpha distribution
    print("\n" + "-" * 70)
    print("ANALYSIS 2: Per-Trajectory Alpha Distribution")
    print("-" * 70)

    mean_alpha = np.mean(trajectory_alphas)
    std_alpha = np.std(trajectory_alphas)
    sem_alpha = std_alpha / np.sqrt(len(trajectory_alphas))

    print(f"Alpha per trajectory: {mean_alpha:.3f} +/- {sem_alpha:.3f}")
    print(f"  Std: {std_alpha:.3f}")
    print(f"  Range: [{np.min(trajectory_alphas):.3f}, {np.max(trajectory_alphas):.3f}]")

    # Test H0: alpha = 0 (no deceleration)
    t_stat, p_value = stats.ttest_1samp(trajectory_alphas, 0)
    cohens_d = mean_alpha / std_alpha if std_alpha > 0 else 0

    print(f"\nTest H0: alpha = 0 (no deceleration)")
    print(f"  t-statistic: {t_stat:.2f}")
    print(f"  p-value: {p_value:.2e}")
    print(f"  Cohen's d: {cohens_d:.2f}")

    # Analysis 3: Binned velocity comparison (low vs high order)
    print("\n" + "-" * 70)
    print("ANALYSIS 3: Low vs High Order Velocity")
    print("-" * 70)

    median_order = np.median(all_orders)
    low_mask = all_orders < median_order
    high_mask = all_orders >= median_order

    low_velocity = all_velocities[low_mask]
    high_velocity = all_velocities[high_mask]

    print(f"Low order (<{median_order:.3f}): mean velocity = {np.mean(low_velocity):.5f}")
    print(f"High order (>={median_order:.3f}): mean velocity = {np.mean(high_velocity):.5f}")
    print(f"Ratio (low/high): {np.mean(low_velocity) / np.mean(high_velocity):.2f}x")

    # Mann-Whitney U test
    u_stat, p_value_mw = stats.mannwhitneyu(low_velocity, high_velocity, alternative='greater')
    print(f"\nMann-Whitney U test (low > high)")
    print(f"  U-statistic: {u_stat:.0f}")
    print(f"  p-value: {p_value_mw:.2e}")

    # Effect size (rank-biserial correlation)
    n1, n2 = len(low_velocity), len(high_velocity)
    r_rb = 1 - (2 * u_stat) / (n1 * n2)
    print(f"  Rank-biserial r: {r_rb:.3f}")

    # Analysis 4: Alternative model - exponential decay
    print("\n" + "-" * 70)
    print("ANALYSIS 4: Model Comparison (Power-Law vs Exponential)")
    print("-" * 70)

    def exp_decay(x, beta, c):
        return c * np.exp(-beta * x)

    try:
        popt_exp, _ = curve_fit(exp_decay, all_orders, all_velocities,
                                p0=[1.0, 0.01], maxfev=10000,
                                bounds=([0, 0], [10, 1]))
        predicted_exp = exp_decay(all_orders, *popt_exp)
        ss_res_exp = np.sum((all_velocities - predicted_exp) ** 2)
        r_squared_exp = 1 - ss_res_exp / ss_tot
        print(f"Exponential fit: dO/dt ~ {popt_exp[1]:.4f} * exp(-{popt_exp[0]:.2f} * O)")
        print(f"R^2 (exponential): {r_squared_exp:.4f}")
        print(f"R^2 (power-law): {r_squared:.4f}")
        print(f"Better model: {'Power-law' if r_squared > r_squared_exp else 'Exponential'}")
    except Exception as e:
        print(f"Exponential fit failed: {e}")
        r_squared_exp = np.nan

    # Determine verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    # Criteria: p < 0.01, effect size > 0.5, consistent direction
    if p_value < 0.01 and cohens_d > 0.5 and mean_alpha > 0:
        verdict = "VALIDATED"
        confidence = "high"
        summary = (f"Order velocity follows power-law deceleration with alpha={mean_alpha:.3f} "
                   f"(d={cohens_d:.1f}, p={p_value:.2e}). Low-order regions show "
                   f"{np.mean(low_velocity)/np.mean(high_velocity):.1f}x faster velocity.")
    elif p_value < 0.05 and mean_alpha > 0:
        verdict = "INCONCLUSIVE"
        confidence = "medium"
        summary = (f"Trend toward deceleration (alpha={mean_alpha:.3f}, p={p_value:.2e}) "
                   f"but effect size insufficient (d={cohens_d:.2f} < 0.5).")
    else:
        verdict = "REFUTED"
        confidence = "medium"
        summary = (f"No significant power-law deceleration detected "
                   f"(alpha={mean_alpha:.3f}, p={p_value:.2e}).")

    print(f"Status: {verdict}")
    print(f"Confidence: {confidence}")
    print(f"\n{summary}")

    # Save results
    results_dir = "results/order_velocity"
    os.makedirs(results_dir, exist_ok=True)

    results = {
        'experiment_id': 'RES-050',
        'hypothesis': 'Order velocity follows power-law deceleration',
        'status': verdict.lower(),
        'confidence': confidence,
        'metrics': {
            'alpha_mean': float(mean_alpha),
            'alpha_std': float(std_alpha),
            'alpha_sem': float(sem_alpha),
            'cohens_d': float(cohens_d),
            'p_value': float(p_value),
            't_statistic': float(t_stat),
            'r_squared_powerlaw': float(r_squared) if not np.isnan(r_squared) else None,
            'r_squared_exponential': float(r_squared_exp) if not np.isnan(r_squared_exp) else None,
            'velocity_ratio_low_high': float(np.mean(low_velocity) / np.mean(high_velocity)),
            'n_trajectories': n_trajectories,
            'n_data_points': len(all_orders)
        },
        'summary': summary
    }

    results_path = os.path.join(results_dir, 'velocity_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    return results


if __name__ == '__main__':
    main()
