"""
RES-161: NS trajectory curvature (turning rate) increases as sampling progresses to higher order

Hypothesis: The nested sampling trajectory exhibits increasing curvature (sharper turns)
as it progresses toward higher-order regions.

Rationale: RES-127 showed NS follows tortuous paths. RES-145 showed gradient magnitude
increases. If the landscape becomes steeper and more peaked (RES-129), the trajectory
should need to "turn" more often to navigate ridges and valleys.

Metric: Curvature approximated as angle between consecutive displacement vectors in
weight space. Higher curvature = sharper turns.
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/matt/Development/monochrome_noise_converger')
from core.thermo_sampler_v3 import CPPN, order_multiplicative, PRIOR_SIGMA
from scipy import stats

def run_ns_with_trajectory(n_iterations=200, n_live=20, image_size=32, seed=None):
    """Run nested sampling and record weight trajectory."""
    if seed is not None:
        np.random.seed(seed)

    # Initialize live points
    live_points = []
    for _ in range(n_live):
        cppn = CPPN()
        img = cppn.render(image_size)
        order = order_multiplicative(img)
        live_points.append({
            'weights': cppn.get_weights().copy(),
            'order': order
        })

    trajectory = []  # (weights, order) pairs in order of removal/replacement

    for it in range(n_iterations):
        # Find worst point
        orders = [lp['order'] for lp in live_points]
        worst_idx = np.argmin(orders)
        threshold = orders[worst_idx]

        # Record removed point
        trajectory.append({
            'weights': live_points[worst_idx]['weights'].copy(),
            'order': threshold,
            'iteration': it
        })

        # Select seed (not the worst)
        candidates = [i for i in range(n_live) if i != worst_idx]
        seed_idx = np.random.choice(candidates)
        seed_weights = live_points[seed_idx]['weights'].copy()

        # Simple ESS replacement
        found = False
        for _ in range(50):  # max tries
            nu = np.random.randn(len(seed_weights)) * PRIOR_SIGMA
            phi = np.random.uniform(0, 2 * np.pi)
            new_weights = seed_weights * np.cos(phi) + nu * np.sin(phi)

            new_cppn = CPPN()
            new_cppn.set_weights(new_weights)
            new_img = new_cppn.render(image_size)
            new_order = order_multiplicative(new_img)

            if new_order >= threshold:
                live_points[worst_idx] = {
                    'weights': new_weights.copy(),
                    'order': new_order
                }
                found = True
                break

        if not found:
            # Keep old point (sampling stuck)
            pass

    return trajectory

def compute_trajectory_curvature(trajectory):
    """
    Compute curvature at each point as angle between consecutive displacement vectors.
    Returns list of (order, curvature) pairs.
    """
    if len(trajectory) < 3:
        return []

    results = []
    weights = [t['weights'] for t in trajectory]
    orders = [t['order'] for t in trajectory]

    for i in range(1, len(trajectory) - 1):
        # Displacement vectors
        v1 = weights[i] - weights[i-1]
        v2 = weights[i+1] - weights[i]

        # Normalize
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)

        if n1 > 1e-10 and n2 > 1e-10:
            v1_unit = v1 / n1
            v2_unit = v2 / n2

            # Angle between vectors (curvature proxy)
            cos_angle = np.clip(np.dot(v1_unit, v2_unit), -1, 1)
            angle = np.arccos(cos_angle)  # radians, 0 = straight, pi = full reversal

            results.append({
                'order': orders[i],
                'curvature': angle,
                'iteration': i
            })

    return results

def main():
    print("RES-161: Testing if NS trajectory curvature increases with order")
    print("=" * 70)

    n_runs = 15
    all_curvatures = []

    for run in range(n_runs):
        print(f"\nRun {run+1}/{n_runs}...")
        trajectory = run_ns_with_trajectory(n_iterations=150, n_live=15, seed=run*42)
        curvature_data = compute_trajectory_curvature(trajectory)
        all_curvatures.extend(curvature_data)

        if curvature_data:
            orders = [c['order'] for c in curvature_data]
            curvs = [c['curvature'] for c in curvature_data]
            r, p = stats.spearmanr(orders, curvs)
            print(f"  Points: {len(curvature_data)}, Order range: [{min(orders):.3f}, {max(orders):.3f}]")
            print(f"  Within-run correlation: r={r:.3f}, p={p:.4f}")

    # Aggregate analysis
    print("\n" + "=" * 70)
    print("AGGREGATE ANALYSIS")
    print("=" * 70)

    orders = np.array([c['order'] for c in all_curvatures])
    curvs = np.array([c['curvature'] for c in all_curvatures])
    iterations = np.array([c['iteration'] for c in all_curvatures])

    # Spearman correlation (order vs curvature)
    r_order, p_order = stats.spearmanr(orders, curvs)
    print(f"\nOrder vs Curvature: r={r_order:.3f}, p={p_order:.2e}")

    # Iteration vs curvature (does curvature increase over time?)
    r_iter, p_iter = stats.spearmanr(iterations, curvs)
    print(f"Iteration vs Curvature: r={r_iter:.3f}, p={p_iter:.2e}")

    # Split into early/late phases
    median_iter = np.median(iterations)
    early_curvs = curvs[iterations < median_iter]
    late_curvs = curvs[iterations >= median_iter]

    print(f"\nEarly phase (iter < {median_iter:.0f}): mean curvature = {early_curvs.mean():.3f} (n={len(early_curvs)})")
    print(f"Late phase (iter >= {median_iter:.0f}): mean curvature = {late_curvs.mean():.3f} (n={len(late_curvs)})")

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((early_curvs.var() + late_curvs.var()) / 2)
    cohens_d = (late_curvs.mean() - early_curvs.mean()) / pooled_std if pooled_std > 0 else 0

    # t-test
    t_stat, p_ttest = stats.ttest_ind(late_curvs, early_curvs)

    print(f"\nCohen's d (late vs early): {cohens_d:.3f}")
    print(f"t-test: t={t_stat:.2f}, p={p_ttest:.2e}")

    # Split by order terciles
    order_low = np.percentile(orders, 33)
    order_high = np.percentile(orders, 67)

    low_order_curvs = curvs[orders <= order_low]
    mid_order_curvs = curvs[(orders > order_low) & (orders <= order_high)]
    high_order_curvs = curvs[orders > order_high]

    print(f"\nCurvature by order tercile:")
    print(f"  Low order (<=  {order_low:.3f}): {low_order_curvs.mean():.3f} +/- {low_order_curvs.std():.3f}")
    print(f"  Mid order ({order_low:.3f}-{order_high:.3f}): {mid_order_curvs.mean():.3f} +/- {mid_order_curvs.std():.3f}")
    print(f"  High order (> {order_high:.3f}): {high_order_curvs.mean():.3f} +/- {high_order_curvs.std():.3f}")

    # Low vs high Cohen's d
    if len(low_order_curvs) > 0 and len(high_order_curvs) > 0:
        pooled_std_lh = np.sqrt((low_order_curvs.var() + high_order_curvs.var()) / 2)
        cohens_d_lh = (high_order_curvs.mean() - low_order_curvs.mean()) / pooled_std_lh if pooled_std_lh > 0 else 0
        t_lh, p_lh = stats.ttest_ind(high_order_curvs, low_order_curvs)
        print(f"\n  High vs Low order curvature:")
        print(f"  Cohen's d = {cohens_d_lh:.3f}")
        print(f"  t-test: t={t_lh:.2f}, p={p_lh:.2e}")

    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    validated = abs(cohens_d) > 0.5 and p_ttest < 0.01
    direction = "INCREASES" if cohens_d > 0 else "DECREASES"

    if validated:
        print(f"VALIDATED: Trajectory curvature {direction} with sampling progression")
        print(f"  Effect size d={cohens_d:.2f} exceeds threshold (0.5)")
        print(f"  p={p_ttest:.2e} < 0.01")
        status = "validated"
    else:
        if p_ttest < 0.05 and abs(cohens_d) > 0.3:
            print(f"INCONCLUSIVE: Trend toward {direction} but effect size weak")
            status = "inconclusive"
        else:
            print(f"REFUTED: No significant change in trajectory curvature")
            status = "refuted"

    # Summary for log
    summary = f"NS trajectory curvature shows r={r_order:.2f} with order (p={p_order:.2e}). "
    summary += f"Late phase has {direction.lower()} curvature vs early (d={cohens_d:.2f}, p={p_ttest:.2e}). "
    summary += f"Mean curvature: early={early_curvs.mean():.3f}, late={late_curvs.mean():.3f}."

    print(f"\nSUMMARY: {summary}")
    print(f"STATUS: {status}")

    return status, summary, cohens_d, p_ttest

if __name__ == "__main__":
    main()
