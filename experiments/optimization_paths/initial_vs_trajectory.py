"""
RES-194: Initial weight configuration predicts ESS path final order better than trajectory length

Hypothesis: The starting weights (initial L2 norm, weight ratio, etc.) predict final order
better than trajectory features (cumulative displacement, acceptance rate).

Rationale:
- RES-193 showed cumulative displacement doesn't predict success
- RES-021 showed connection-to-bias weight ratio predicts order
- Perhaps path outcome is determined at initialization, not during optimization

Approach:
1. Run ESS optimization paths from random starts
2. Compute initial features (L2 norm, connection-to-bias ratio, max abs weight)
3. Compute trajectory features (displacement, acceptance rate)
4. Compare predictive power using R^2 of linear models

Effect size threshold: R^2 difference > 0.1, confirmed by cross-validation
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
import json

from core.thermo_sampler_v3 import CPPN, order_multiplicative, set_global_seed

# Configuration
N_PATHS = 300  # Number of optimization paths
N_STEPS = 50   # Steps per path

set_global_seed(42)


def elliptical_slice_step(cppn: CPPN, current_order: float, order_func=order_multiplicative):
    """One step of ESS. Returns: (new_order, weight_change_norm, accepted)"""
    current_weights = cppn.get_weights()
    n_weights = len(current_weights)

    nu = np.random.randn(n_weights)
    angle = np.random.uniform(0, 2 * np.pi)
    angle_min = angle - 2 * np.pi
    angle_max = angle
    threshold = current_order

    max_iterations = 100
    for _ in range(max_iterations):
        new_weights = current_weights * np.cos(angle) + nu * np.sin(angle)
        cppn.set_weights(new_weights)
        img = cppn.render(32)
        new_order = order_func(img)

        if new_order > threshold:
            weight_change = np.linalg.norm(new_weights - current_weights)
            return new_order, weight_change, True

        if angle < 0:
            angle_min = angle
        else:
            angle_max = angle
        angle = np.random.uniform(angle_min, angle_max)

    cppn.set_weights(current_weights)
    return current_order, 0.0, False


def compute_initial_features(weights):
    """Compute features from initial weight configuration."""
    n_connections = 4  # 4 input connections (x, y, r, bias)
    n_biases = 1       # 1 output bias

    connection_weights = weights[:n_connections]
    bias_weights = weights[n_connections:]

    return {
        'l2_norm': np.linalg.norm(weights),
        'max_abs_weight': np.max(np.abs(weights)),
        'connection_l2': np.linalg.norm(connection_weights),
        'bias_l2': np.linalg.norm(bias_weights) if len(bias_weights) > 0 else 0,
        'connection_bias_ratio': np.linalg.norm(connection_weights) / (np.linalg.norm(bias_weights) + 1e-10),
        'r_weight_magnitude': np.abs(weights[2]),  # Weight on radial input
        'xy_weight_magnitude': np.sqrt(weights[0]**2 + weights[1]**2),  # x,y inputs combined
        'weight_variance': np.var(weights),
        'sign_uniformity': np.abs(np.sum(np.sign(weights))) / len(weights),
    }


def run_ess_path(seed):
    """Run one ESS path and return features."""
    np.random.seed(seed)
    cppn = CPPN()

    # Initial features
    initial_weights = cppn.get_weights().copy()
    initial_features = compute_initial_features(initial_weights)

    # Initial order
    img = cppn.render(32)
    initial_order = order_multiplicative(img)
    initial_features['initial_order'] = initial_order

    # Run ESS
    orders = [initial_order]
    current_order = initial_order
    cumulative_displacement = 0.0
    n_accepted = 0

    for step in range(N_STEPS):
        new_order, step_disp, accepted = elliptical_slice_step(cppn, current_order)
        current_order = new_order
        orders.append(current_order)
        cumulative_displacement += step_disp
        if accepted:
            n_accepted += 1

    final_weights = cppn.get_weights()
    total_displacement = np.linalg.norm(final_weights - initial_weights)

    # Trajectory features
    trajectory_features = {
        'cumulative_displacement': cumulative_displacement,
        'total_displacement': total_displacement,
        'acceptance_rate': n_accepted / N_STEPS,
        'mean_step_displacement': cumulative_displacement / max(1, n_accepted),
        'order_gain': current_order - initial_order,
    }

    return {
        'final_order': current_order,
        'initial_features': initial_features,
        'trajectory_features': trajectory_features,
    }


def main():
    print("RES-194: Initial weight configuration predicts ESS path final order better than trajectory")
    print("="*70)
    print(f"Running {N_PATHS} ESS paths, {N_STEPS} steps each")
    print()

    results = []
    for i in range(N_PATHS):
        if (i+1) % 50 == 0:
            print(f"  Progress: {i+1}/{N_PATHS}")
        result = run_ess_path(seed=i*13 + 789)
        results.append(result)

    # Build feature matrices
    final_orders = np.array([r['final_order'] for r in results])

    # Initial features matrix
    init_feature_names = list(results[0]['initial_features'].keys())
    X_initial = np.array([[r['initial_features'][k] for k in init_feature_names] for r in results])

    # Trajectory features matrix
    traj_feature_names = list(results[0]['trajectory_features'].keys())
    X_trajectory = np.array([[r['trajectory_features'][k] for k in traj_feature_names] for r in results])

    # Combined features
    X_combined = np.hstack([X_initial, X_trajectory])

    print(f"\nFeature dimensions:")
    print(f"  Initial features: {X_initial.shape[1]} ({init_feature_names})")
    print(f"  Trajectory features: {X_trajectory.shape[1]} ({traj_feature_names})")
    print(f"  Combined features: {X_combined.shape[1]}")

    # Cross-validated R^2
    ridge = Ridge(alpha=1.0)

    cv_initial = cross_val_score(ridge, X_initial, final_orders, cv=5, scoring='r2')
    cv_trajectory = cross_val_score(ridge, X_trajectory, final_orders, cv=5, scoring='r2')
    cv_combined = cross_val_score(ridge, X_combined, final_orders, cv=5, scoring='r2')

    r2_initial = np.mean(cv_initial)
    r2_trajectory = np.mean(cv_trajectory)
    r2_combined = np.mean(cv_combined)

    r2_diff = r2_initial - r2_trajectory

    print(f"\n{'='*70}")
    print("RESULTS - Cross-validated R^2 scores")
    print(f"{'='*70}")
    print(f"\nInitial features only:     R^2 = {r2_initial:.4f} (+/- {np.std(cv_initial):.4f})")
    print(f"Trajectory features only:  R^2 = {r2_trajectory:.4f} (+/- {np.std(cv_trajectory):.4f})")
    print(f"Combined features:         R^2 = {r2_combined:.4f} (+/- {np.std(cv_combined):.4f})")

    print(f"\nR^2 difference (initial - trajectory): {r2_diff:.4f}")

    # Paired t-test on CV folds
    t_stat, p_value = stats.ttest_rel(cv_initial, cv_trajectory)
    print(f"Paired t-test: t={t_stat:.3f}, p={p_value:.4f}")

    # Effect size (Cohen's d for paired samples)
    diff = cv_initial - cv_trajectory
    cohens_d = np.mean(diff) / (np.std(diff) + 1e-10)
    print(f"Cohen's d: {cohens_d:.3f}")

    # Individual feature correlations
    print(f"\n{'='*70}")
    print("Individual feature correlations with final order")
    print(f"{'='*70}")

    print("\nInitial features:")
    for i, name in enumerate(init_feature_names):
        rho, p = stats.spearmanr(X_initial[:, i], final_orders)
        print(f"  {name:25s}: rho={rho:+.3f} (p={p:.2e})")

    print("\nTrajectory features:")
    for i, name in enumerate(traj_feature_names):
        rho, p = stats.spearmanr(X_trajectory[:, i], final_orders)
        print(f"  {name:25s}: rho={rho:+.3f} (p={p:.2e})")

    # Determine status
    if p_value < 0.01 and abs(r2_diff) > 0.1:
        if r2_diff > 0:
            status = 'validated'
            conclusion = f"Initial features predict final order BETTER than trajectory (R^2: {r2_initial:.3f} vs {r2_trajectory:.3f})"
        else:
            status = 'refuted'
            conclusion = f"Trajectory features predict BETTER than initial (opposite direction)"
    elif abs(r2_diff) < 0.05:
        status = 'refuted'
        conclusion = f"Initial and trajectory features have similar predictive power (R^2 diff={r2_diff:.3f})"
    else:
        status = 'inconclusive'
        conclusion = f"Trend exists but p={p_value:.3f} or effect size below threshold"

    print(f"\n{'='*70}")
    print(f"CONCLUSION: {status.upper()}")
    print(f"{conclusion}")
    print(f"{'='*70}")

    # Build summary
    summary = {
        'experiment_id': 'RES-194',
        'hypothesis': 'Initial weight configuration predicts ESS path final order better than trajectory length',
        'status': status,
        'conclusion': conclusion,
        'n_paths': N_PATHS,
        'n_steps': N_STEPS,
        'metrics': {
            'r2_initial': float(r2_initial),
            'r2_trajectory': float(r2_trajectory),
            'r2_combined': float(r2_combined),
            'r2_diff': float(r2_diff),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'cohens_d': float(cohens_d),
        },
        'cv_results': {
            'initial_cv_scores': [float(x) for x in cv_initial],
            'trajectory_cv_scores': [float(x) for x in cv_trajectory],
            'combined_cv_scores': [float(x) for x in cv_combined],
        },
        'feature_names': {
            'initial': init_feature_names,
            'trajectory': traj_feature_names,
        }
    }

    # Save results
    output_path = os.path.join(os.path.dirname(__file__), '../../results/optimization_paths/initial_vs_trajectory_results.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return summary


if __name__ == '__main__':
    main()
