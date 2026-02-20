"""
RES-165: NS trajectory effective dimension decreases as sampling approaches high-order regions

Hypothesis: The effective dimensionality of nested sampling trajectory decreases
during progression - the sampling explores fewer independent directions as it
approaches high-order regions.

Rationale:
- RES-145: Gradient magnitude increases 37x during NS (landscape becomes steeper)
- RES-161: Trajectory curvature stays constant (~2.1 rad)
- RES-141: High-order optima are separated by low-order barriers

If high-order regions are narrow peaks/ridges in weight space, the effective
number of directions sampled should decrease. We measure this using:
1. Local PCA on trajectory windows - variance explained by top k components
2. Correlation dimension of trajectory segments
3. Direction entropy - how spread out are consecutive displacements?
"""

import numpy as np
import json
from pathlib import Path
from scipy import stats
from sklearn.decomposition import PCA
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.thermo_sampler_v3 import (
    CPPN, order_multiplicative, set_global_seed
)

def run_nested_sampling_with_trajectory(
    n_live: int = 50,
    max_iter: int = 300,
    threshold: float = 0.3,
    seed: int = 42
) -> dict:
    """Run nested sampling and record full trajectory for analysis."""
    set_global_seed(seed)

    # Initialize live points
    live_points = [CPPN() for _ in range(n_live)]
    live_weights = [lp.get_weights() for lp in live_points]
    live_orders = [order_multiplicative(lp.render(32)) for lp in live_points]

    trajectory = []  # List of (iteration, weights, order) tuples
    weight_dim = len(live_weights[0])

    for it in range(max_iter):
        # Find worst point
        worst_idx = np.argmin(live_orders)
        worst_order = live_orders[worst_idx]

        if worst_order >= threshold:
            break

        # Record all live points' state
        for w, o in zip(live_weights, live_orders):
            trajectory.append({
                'iteration': it,
                'weights': w.copy(),
                'order': o
            })

        # ESS to replace worst point
        # Pick random live point (not worst) as reference
        ref_idx = np.random.randint(n_live)
        while ref_idx == worst_idx:
            ref_idx = np.random.randint(n_live)

        ref_weights = live_weights[ref_idx].copy()

        # Elliptical slice sampling
        nu = np.random.randn(weight_dim) * 1.0  # Auxiliary from prior
        phi = np.random.uniform(0, 2 * np.pi)
        phi_min, phi_max = phi - 2 * np.pi, phi

        accepted = False
        contractions = 0
        for _ in range(50):
            new_weights = ref_weights * np.cos(phi) + nu * np.sin(phi)
            test_cppn = CPPN()
            test_cppn.set_weights(new_weights)
            new_order = order_multiplicative(test_cppn.render(32))

            if new_order > worst_order:
                live_points[worst_idx] = test_cppn
                live_weights[worst_idx] = new_weights
                live_orders[worst_idx] = new_order
                accepted = True
                break

            # Contract bracket
            contractions += 1
            if phi < 0:
                phi_min = phi
            else:
                phi_max = phi
            phi = np.random.uniform(phi_min, phi_max)

        if not accepted:
            # Fall back to random perturbation
            new_weights = ref_weights + np.random.randn(weight_dim) * 0.1
            test_cppn = CPPN()
            test_cppn.set_weights(new_weights)
            new_order = order_multiplicative(test_cppn.render(32))
            if new_order > worst_order:
                live_points[worst_idx] = test_cppn
                live_weights[worst_idx] = new_weights
                live_orders[worst_idx] = new_order

    return {
        'trajectory': trajectory,
        'final_orders': live_orders,
        'weight_dim': weight_dim
    }


def compute_trajectory_effective_dim(trajectory: list, window_size: int = 50) -> list:
    """
    Compute effective dimension metrics for sliding windows along trajectory.

    Returns list of (mean_order, effective_dim_90, effective_dim_95, direction_entropy)
    """
    results = []

    for start in range(0, len(trajectory) - window_size, window_size // 2):
        window = trajectory[start:start + window_size]
        weights = np.array([w['weights'] for w in window])
        orders = [w['order'] for w in window]

        # 1. PCA effective dimension (variance explained)
        pca = PCA()
        pca.fit(weights)
        cumvar = np.cumsum(pca.explained_variance_ratio_)

        # Effective dim at 90% and 95% variance explained
        eff_dim_90 = np.searchsorted(cumvar, 0.90) + 1
        eff_dim_95 = np.searchsorted(cumvar, 0.95) + 1

        # Participation ratio (inverse Herfindahl index)
        # PR = 1 / sum(p_i^2) where p_i is variance fraction
        var_ratios = pca.explained_variance_ratio_
        var_ratios = var_ratios[var_ratios > 0]
        participation_ratio = 1.0 / np.sum(var_ratios ** 2)

        # 2. Direction entropy for consecutive displacements
        displacements = np.diff(weights, axis=0)
        norms = np.linalg.norm(displacements, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1e-10
        unit_displacements = displacements / norms

        # Compute pairwise cosine similarities
        cos_sims = unit_displacements @ unit_displacements.T
        # Direction "spread" - low mean cosine = more diverse directions
        direction_coherence = np.mean(np.abs(cos_sims))

        results.append({
            'mean_order': np.mean(orders),
            'max_order': np.max(orders),
            'eff_dim_90': int(eff_dim_90),
            'eff_dim_95': int(eff_dim_95),
            'participation_ratio': float(participation_ratio),
            'direction_coherence': float(direction_coherence),
            'top_3_variance': float(cumvar[2] if len(cumvar) > 2 else cumvar[-1])
        })

    return results


def main():
    print("RES-165: NS trajectory effective dimension experiment")
    print("=" * 60)

    # Run multiple NS trajectories
    n_runs = 20
    all_window_results = []

    for seed in range(n_runs):
        print(f"  Run {seed+1}/{n_runs}...", end=" ", flush=True)

        result = run_nested_sampling_with_trajectory(
            n_live=50, max_iter=400, threshold=0.4, seed=seed * 123
        )

        trajectory = result['trajectory']
        if len(trajectory) < 100:
            print(f"short trajectory ({len(trajectory)} points), skipping")
            continue

        # Compute effective dimension in windows
        window_results = compute_trajectory_effective_dim(trajectory, window_size=100)
        all_window_results.extend(window_results)

        print(f"OK ({len(window_results)} windows)")

    # Analyze results
    print(f"\nTotal windows analyzed: {len(all_window_results)}")

    # Extract arrays for correlation
    orders = np.array([w['mean_order'] for w in all_window_results])
    eff_dim_90 = np.array([w['eff_dim_90'] for w in all_window_results])
    eff_dim_95 = np.array([w['eff_dim_95'] for w in all_window_results])
    participation_ratio = np.array([w['participation_ratio'] for w in all_window_results])
    direction_coherence = np.array([w['direction_coherence'] for w in all_window_results])
    top_3_var = np.array([w['top_3_variance'] for w in all_window_results])

    # Split into early (low order) and late (high order) groups
    order_median = np.median(orders)
    early_mask = orders < order_median
    late_mask = orders >= order_median

    print(f"\nOrder range: {orders.min():.4f} to {orders.max():.4f}")
    print(f"Order median: {order_median:.4f}")
    print(f"Early windows (low order): {np.sum(early_mask)}")
    print(f"Late windows (high order): {np.sum(late_mask)}")

    # Correlation: order vs effective dimension
    corr_dim90, p_dim90 = stats.spearmanr(orders, eff_dim_90)
    corr_dim95, p_dim95 = stats.spearmanr(orders, eff_dim_95)
    corr_pr, p_pr = stats.spearmanr(orders, participation_ratio)
    corr_coh, p_coh = stats.spearmanr(orders, direction_coherence)
    corr_top3, p_top3 = stats.spearmanr(orders, top_3_var)

    print("\n--- Correlation with Order ---")
    print(f"eff_dim_90 vs order: rho={corr_dim90:.3f}, p={p_dim90:.2e}")
    print(f"eff_dim_95 vs order: rho={corr_dim95:.3f}, p={p_dim95:.2e}")
    print(f"participation_ratio vs order: rho={corr_pr:.3f}, p={p_pr:.2e}")
    print(f"direction_coherence vs order: rho={corr_coh:.3f}, p={p_coh:.2e}")
    print(f"top_3_variance vs order: rho={corr_top3:.3f}, p={p_top3:.2e}")

    # Effect size: compare early vs late
    early_dim90 = eff_dim_90[early_mask]
    late_dim90 = eff_dim_90[late_mask]
    early_pr = participation_ratio[early_mask]
    late_pr = participation_ratio[late_mask]

    # Cohen's d for participation ratio (our main metric)
    pooled_std = np.sqrt((np.std(early_pr)**2 + np.std(late_pr)**2) / 2)
    cohens_d_pr = (np.mean(late_pr) - np.mean(early_pr)) / pooled_std if pooled_std > 0 else 0

    # Mann-Whitney U test
    stat_pr, p_mw_pr = stats.mannwhitneyu(early_pr, late_pr, alternative='two-sided')

    print("\n--- Early vs Late Comparison ---")
    print(f"Early eff_dim_90: mean={np.mean(early_dim90):.1f}, std={np.std(early_dim90):.1f}")
    print(f"Late eff_dim_90: mean={np.mean(late_dim90):.1f}, std={np.std(late_dim90):.1f}")
    print(f"Early participation_ratio: mean={np.mean(early_pr):.2f}, std={np.std(early_pr):.2f}")
    print(f"Late participation_ratio: mean={np.mean(late_pr):.2f}, std={np.std(late_pr):.2f}")
    print(f"Cohen's d (participation ratio): {cohens_d_pr:.3f}")
    print(f"Mann-Whitney U p-value: {p_mw_pr:.2e}")

    # Validation criteria
    # Hypothesis: effective dimension DECREASES (negative correlation with order)
    validated = (corr_pr < -0.2 and p_pr < 0.01 and abs(cohens_d_pr) > 0.5)

    print("\n" + "=" * 60)
    print("VALIDATION CRITERIA")
    print(f"  Correlation < -0.2: {corr_pr:.3f} {'PASS' if corr_pr < -0.2 else 'FAIL'}")
    print(f"  p < 0.01: {p_pr:.2e} {'PASS' if p_pr < 0.01 else 'FAIL'}")
    print(f"  |Cohen's d| > 0.5: {abs(cohens_d_pr):.3f} {'PASS' if abs(cohens_d_pr) > 0.5 else 'FAIL'}")

    status = 'validated' if validated else 'refuted'
    print(f"\nSTATUS: {status.upper()}")

    # Save results
    results = {
        'experiment_id': 'RES-165',
        'hypothesis': 'NS trajectory effective dimension decreases as sampling approaches high-order regions',
        'status': status,
        'n_runs': n_runs,
        'n_windows': len(all_window_results),
        'metrics': {
            'corr_participation_ratio': float(corr_pr),
            'p_participation_ratio': float(p_pr),
            'corr_eff_dim_90': float(corr_dim90),
            'p_eff_dim_90': float(p_dim90),
            'cohens_d_pr': float(cohens_d_pr),
            'early_pr_mean': float(np.mean(early_pr)),
            'late_pr_mean': float(np.mean(late_pr)),
            'early_dim90_mean': float(np.mean(early_dim90)),
            'late_dim90_mean': float(np.mean(late_dim90)),
        },
        'summary': f"Participation ratio {'decreases' if corr_pr < 0 else 'increases'} with order (rho={corr_pr:.3f}, p={p_pr:.2e}). " +
                   f"Early (low-order) windows: PR={np.mean(early_pr):.2f}, Late (high-order): PR={np.mean(late_pr):.2f}. " +
                   f"Effect size d={cohens_d_pr:.3f}. " +
                   (f"Effective dimension {'decreases' if cohens_d_pr < 0 else 'does not decrease'} as sampling progresses." if validated else
                    "Effect size below threshold or wrong direction - hypothesis refuted.")
    }

    output_dir = Path(__file__).parent.parent / 'results' / 'res_165_trajectory_dimension'
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir / 'results.json'}")

    return results


if __name__ == '__main__':
    main()
