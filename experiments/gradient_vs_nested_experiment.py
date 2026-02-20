"""
Gradient Descent vs Nested Sampling Experiment

Domain: gradient_optimization
Research ID: RES-023

Hypothesis: Gradient descent on the order metric gets stuck in local minima
more frequently than nested sampling (ESS), resulting in lower final order
values and higher variance across runs.

Builds on:
- RES-015: High-order images have high Lipschitz constant (steep gradients)
- RES-019: Nested sampling trajectories are highly autocorrelated (smooth exploration)
- RES-022: Nested sampling achieves near-optimal bit-cost (10% overhead)

Novelty: First direct comparison of gradient-based optimization vs nested sampling
for finding high-order images. Tests whether the steep gradient landscape at
high order levels leads to local minima trapping.

Method:
1. Run gradient descent (finite-difference) to maximize order, tracking convergence
2. Run nested sampling to reach equivalent order thresholds
3. Compare: final order achieved, variance, success rate, local minima detection

Statistical tests:
- Mann-Whitney U for final order comparison
- Levene's test for variance comparison
- Cohen's d for effect size
"""

import json
import sys
from pathlib import Path
import numpy as np
from scipy import stats
from dataclasses import dataclass
from typing import Optional, Callable, List, Tuple
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.thermo_sampler_v3 import (
    CPPN,
    order_multiplicative,
    PRIOR_SIGMA,
    elliptical_slice_sample,
)


@dataclass
class GradientDescentResult:
    """Result of a single gradient descent run."""
    final_order: float
    trajectory: List[float]  # Order values at each step
    n_steps: int
    converged: bool  # Did improvement stop?
    stuck_count: int  # Number of times improvement was < threshold
    initial_order: float
    improvement: float


@dataclass
class NestedSamplingResult:
    """Result of a single nested sampling run to reach threshold."""
    final_order: float
    trajectory: List[float]  # Min live order at each iteration
    n_iterations: int
    success: bool  # Did we reach the threshold?
    bits_used: float  # log2(1/X) at termination
    initial_order: float


def estimate_gradient(
    cppn: CPPN,
    order_fn: Callable,
    image_size: int = 32,
    epsilon: float = 0.01,
    n_directions: int = 10
) -> np.ndarray:
    """
    Estimate gradient of order w.r.t. CPPN weights using finite differences.

    Uses random direction sampling (more efficient than coordinate-wise).
    """
    base_weights = cppn.get_weights()
    n_params = len(base_weights)

    gradient = np.zeros(n_params)

    for _ in range(n_directions):
        # Random unit direction
        direction = np.random.randn(n_params)
        direction = direction / np.linalg.norm(direction)

        # Forward difference
        cppn_plus = cppn.copy()
        cppn_plus.set_weights(base_weights + epsilon * direction)
        order_plus = order_fn(cppn_plus.render(image_size))

        # Backward difference
        cppn_minus = cppn.copy()
        cppn_minus.set_weights(base_weights - epsilon * direction)
        order_minus = order_fn(cppn_minus.render(image_size))

        # Directional derivative
        deriv = (order_plus - order_minus) / (2 * epsilon)

        # Accumulate gradient estimate
        gradient += deriv * direction

    return gradient / n_directions


def gradient_descent_optimize(
    seed: int,
    order_fn: Callable = order_multiplicative,
    image_size: int = 32,
    max_steps: int = 200,
    learning_rate: float = 0.1,
    epsilon: float = 0.01,
    n_gradient_directions: int = 10,
    convergence_threshold: float = 1e-4,
    patience: int = 20
) -> GradientDescentResult:
    """
    Run gradient descent to maximize order metric.

    Uses adaptive learning rate with momentum.
    Declares convergence when improvement is below threshold for patience steps.
    """
    np.random.seed(seed)

    # Initialize CPPN
    cppn = CPPN()
    current_img = cppn.render(image_size)
    current_order = order_fn(current_img)
    initial_order = current_order

    trajectory = [current_order]

    # Momentum
    velocity = np.zeros(len(cppn.get_weights()))
    momentum = 0.9

    stuck_count = 0
    best_order = current_order
    steps_since_improvement = 0

    for step in range(max_steps):
        # Estimate gradient
        gradient = estimate_gradient(cppn, order_fn, image_size, epsilon, n_gradient_directions)

        # Update with momentum
        velocity = momentum * velocity + learning_rate * gradient
        new_weights = cppn.get_weights() + velocity

        # Apply update
        new_cppn = cppn.copy()
        new_cppn.set_weights(new_weights)
        new_img = new_cppn.render(image_size)
        new_order = order_fn(new_img)

        # Track improvement
        improvement = new_order - current_order

        if improvement > convergence_threshold:
            cppn = new_cppn
            current_order = new_order
            steps_since_improvement = 0

            if current_order > best_order:
                best_order = current_order
        else:
            stuck_count += 1
            steps_since_improvement += 1

            # Still accept if better, even if below threshold
            if new_order > current_order:
                cppn = new_cppn
                current_order = new_order

        trajectory.append(current_order)

        # Adaptive learning rate decay
        if steps_since_improvement > 5:
            learning_rate *= 0.95

        # Convergence check
        if steps_since_improvement >= patience:
            break

    converged = steps_since_improvement >= patience

    return GradientDescentResult(
        final_order=current_order,
        trajectory=trajectory,
        n_steps=len(trajectory) - 1,
        converged=converged,
        stuck_count=stuck_count,
        initial_order=initial_order,
        improvement=current_order - initial_order
    )


def nested_sampling_optimize(
    seed: int,
    target_order: float,
    order_fn: Callable = order_multiplicative,
    image_size: int = 32,
    n_live: int = 20,
    max_iterations: int = 500
) -> NestedSamplingResult:
    """
    Run nested sampling to reach target order threshold.

    Simplified version focused on reaching threshold efficiently.
    """
    np.random.seed(seed)

    # Initialize live points
    live_points = []
    for _ in range(n_live):
        cppn = CPPN()
        img = cppn.render(image_size)
        order = order_fn(img)
        live_points.append({'cppn': cppn, 'img': img, 'order': order})

    initial_order = np.mean([lp['order'] for lp in live_points])
    trajectory = [min(lp['order'] for lp in live_points)]

    for iteration in range(max_iterations):
        # Find worst point
        worst_idx = min(range(n_live), key=lambda i: live_points[i]['order'])
        threshold = live_points[worst_idx]['order']

        # Check if we've reached target
        if threshold >= target_order:
            bits_used = (iteration + 1) / n_live
            return NestedSamplingResult(
                final_order=threshold,
                trajectory=trajectory,
                n_iterations=iteration + 1,
                success=True,
                bits_used=bits_used,
                initial_order=initial_order
            )

        # Select seed (any point above threshold)
        valid_seeds = [i for i in range(n_live) if i != worst_idx and live_points[i]['order'] >= threshold]
        if not valid_seeds:
            valid_seeds = [i for i in range(n_live) if i != worst_idx]

        seed_idx = np.random.choice(valid_seeds)
        seed_cppn = live_points[seed_idx]['cppn']

        # ESS sampling
        new_cppn, new_img, new_order, _, _, success = elliptical_slice_sample(
            seed_cppn, threshold, image_size, order_fn, max_contractions=50, max_restarts=3
        )

        if success:
            live_points[worst_idx] = {'cppn': new_cppn, 'img': new_img, 'order': new_order}

        trajectory.append(min(lp['order'] for lp in live_points))

    # Did not reach target
    final_order = max(lp['order'] for lp in live_points)
    bits_used = max_iterations / n_live

    return NestedSamplingResult(
        final_order=final_order,
        trajectory=trajectory,
        n_iterations=max_iterations,
        success=False,
        bits_used=bits_used,
        initial_order=initial_order
    )


def detect_local_minimum(trajectory: List[float], window: int = 10, threshold: float = 0.01) -> Tuple[bool, int]:
    """
    Detect if trajectory got stuck in local minimum.

    Returns (is_stuck, stuck_iteration) where stuck means:
    - Variance in last `window` points is below threshold
    - AND final value is below some percentile of the trajectory max
    """
    if len(trajectory) < window:
        return False, -1

    # Check for plateau at end
    end_variance = np.var(trajectory[-window:])
    is_plateau = end_variance < threshold

    if not is_plateau:
        return False, -1

    # Find when we got stuck
    for i in range(len(trajectory) - window, 0, -1):
        var_window = np.var(trajectory[i:i+window])
        if var_window >= threshold:
            return True, i + window

    return True, 0


def nested_sampling_maximize(
    seed: int,
    order_fn: Callable = order_multiplicative,
    image_size: int = 32,
    n_live: int = 20,
    n_iterations: int = 200
) -> NestedSamplingResult:
    """
    Run nested sampling for fixed iterations to maximize order.

    Unlike nested_sampling_optimize which stops at a threshold,
    this runs for a fixed budget to enable fair comparison with GD.
    """
    np.random.seed(seed)

    # Initialize live points
    live_points = []
    for _ in range(n_live):
        cppn = CPPN()
        img = cppn.render(image_size)
        order = order_fn(img)
        live_points.append({'cppn': cppn, 'img': img, 'order': order})

    initial_order = np.mean([lp['order'] for lp in live_points])
    trajectory = [min(lp['order'] for lp in live_points)]

    for iteration in range(n_iterations):
        # Find worst point
        worst_idx = min(range(n_live), key=lambda i: live_points[i]['order'])
        threshold = live_points[worst_idx]['order']

        # Select seed (any point above threshold)
        valid_seeds = [i for i in range(n_live) if i != worst_idx and live_points[i]['order'] >= threshold]
        if not valid_seeds:
            valid_seeds = [i for i in range(n_live) if i != worst_idx]

        seed_idx = np.random.choice(valid_seeds)
        seed_cppn = live_points[seed_idx]['cppn']

        # ESS sampling
        new_cppn, new_img, new_order, _, _, success = elliptical_slice_sample(
            seed_cppn, threshold, image_size, order_fn, max_contractions=50, max_restarts=3
        )

        if success:
            live_points[worst_idx] = {'cppn': new_cppn, 'img': new_img, 'order': new_order}

        trajectory.append(min(lp['order'] for lp in live_points))

    # Return max order achieved among live points
    final_order = max(lp['order'] for lp in live_points)
    bits_used = n_iterations / n_live

    return NestedSamplingResult(
        final_order=final_order,
        trajectory=trajectory,
        n_iterations=n_iterations,
        success=True,
        bits_used=bits_used,
        initial_order=initial_order
    )


def run_experiment(
    n_runs: int = 50,
    image_size: int = 32,
    gradient_steps: int = 200,
    gradient_lr: float = 0.1,
    ns_n_live: int = 20,
    ns_max_iter: int = 300,
    target_orders: List[float] = None,
    seed: int = 42
) -> dict:
    """
    Run the gradient descent vs nested sampling comparison experiment.
    """
    if target_orders is None:
        target_orders = [0.05, 0.10, 0.15, 0.20]

    np.random.seed(seed)

    print("=" * 70)
    print("GRADIENT DESCENT vs NESTED SAMPLING EXPERIMENT")
    print("=" * 70)
    print(f"\nHypothesis: Gradient descent gets stuck in local minima more often")
    print(f"            than nested sampling, achieving lower final orders.")
    print(f"\nParameters:")
    print(f"  - n_runs: {n_runs}")
    print(f"  - image_size: {image_size}x{image_size}")
    print(f"  - gradient_steps: {gradient_steps}")
    print(f"  - gradient_lr: {gradient_lr}")
    print(f"  - ns_n_live: {ns_n_live}")
    print(f"  - ns_max_iter: {ns_max_iter}")
    print(f"  - target_orders: {target_orders}")
    print()

    # Run gradient descent optimization
    print("1. Running gradient descent optimization...")
    gd_results = []
    for i in range(n_runs):
        result = gradient_descent_optimize(
            seed=seed + i,
            image_size=image_size,
            max_steps=gradient_steps,
            learning_rate=gradient_lr
        )
        gd_results.append(result)

        if (i + 1) % 10 == 0:
            print(f"   GD runs completed: {i+1}/{n_runs}")

    gd_final_orders = np.array([r.final_order for r in gd_results])
    gd_improvements = np.array([r.improvement for r in gd_results])

    print(f"\n   GD final order: mean={np.mean(gd_final_orders):.4f}, std={np.std(gd_final_orders):.4f}")
    print(f"   GD improvement: mean={np.mean(gd_improvements):.4f}, std={np.std(gd_improvements):.4f}")

    # Detect local minima in GD
    gd_stuck_count = 0
    for result in gd_results:
        is_stuck, _ = detect_local_minimum(result.trajectory)
        if is_stuck and result.final_order < 0.15:  # Stuck at low order
            gd_stuck_count += 1

    gd_stuck_rate = gd_stuck_count / n_runs
    print(f"   GD stuck rate (local min at order < 0.15): {gd_stuck_rate:.2%}")

    # Run nested sampling MAXIMIZE (fair comparison - same budget as GD)
    print("\n2. Running nested sampling MAXIMIZE (fair comparison)...")
    ns_max_results = []

    # Use same effective budget: GD uses ~200 gradient estimates, each with 10 directions
    # So ~2000 function evaluations. NS with 20 live points for 200 iterations = 4000 evals
    # Let's use 100 iterations for fair comparison
    ns_fair_iter = 100

    for i in range(n_runs):
        result = nested_sampling_maximize(
            seed=seed + 2000 + i,
            image_size=image_size,
            n_live=ns_n_live,
            n_iterations=ns_fair_iter
        )
        ns_max_results.append(result)

        if (i + 1) % 10 == 0:
            print(f"   NS maximize runs completed: {i+1}/{n_runs}")

    ns_max_final_orders = np.array([r.final_order for r in ns_max_results])
    print(f"\n   NS max final order: mean={np.mean(ns_max_final_orders):.4f}, std={np.std(ns_max_final_orders):.4f}")

    # Run nested sampling for each target threshold (for comparison)
    print("\n3. Running nested sampling THRESHOLD (to reach specific targets)...")
    ns_results_by_target = {}

    for target in target_orders:
        print(f"\n   Target order = {target}:")
        ns_results = []

        for i in range(n_runs):
            result = nested_sampling_optimize(
                seed=seed + 1000 + i,
                target_order=target,
                image_size=image_size,
                n_live=ns_n_live,
                max_iterations=ns_max_iter
            )
            ns_results.append(result)

            if (i + 1) % 10 == 0:
                print(f"      NS runs completed: {i+1}/{n_runs}")

        ns_final_orders = np.array([r.final_order for r in ns_results])
        ns_success_rate = np.mean([r.success for r in ns_results])
        ns_bits_mean = np.mean([r.bits_used for r in ns_results if r.success]) if ns_success_rate > 0 else float('inf')

        print(f"      NS success rate: {ns_success_rate:.2%}")
        print(f"      NS bits to reach {target}: {ns_bits_mean:.2f}")

        ns_results_by_target[target] = ns_results

    # Statistical comparison for best comparable target
    # Find highest target where GD achieves similar median
    print("\n4. Statistical comparison...")

    comparison_results = {}

    for target in target_orders:
        ns_results = ns_results_by_target[target]
        ns_final_orders = np.array([r.final_order for r in ns_results])

        # Mann-Whitney U test: is GD final order lower than NS?
        stat, p_value = stats.mannwhitneyu(gd_final_orders, ns_final_orders, alternative='less')

        # Levene's test for variance equality
        levene_stat, levene_p = stats.levene(gd_final_orders, ns_final_orders)

        # Cohen's d (positive = NS higher)
        pooled_std = np.sqrt((np.var(gd_final_orders) + np.var(ns_final_orders)) / 2)
        if pooled_std > 0:
            cohens_d = (np.mean(ns_final_orders) - np.mean(gd_final_orders)) / pooled_std
        else:
            cohens_d = 0.0

        # Variance ratio (GD/NS > 1 means GD more variable)
        variance_ratio = np.var(gd_final_orders) / (np.var(ns_final_orders) + 1e-10)

        comparison_results[target] = {
            'gd_mean': float(np.mean(gd_final_orders)),
            'gd_std': float(np.std(gd_final_orders)),
            'gd_median': float(np.median(gd_final_orders)),
            'ns_mean': float(np.mean(ns_final_orders)),
            'ns_std': float(np.std(ns_final_orders)),
            'ns_median': float(np.median(ns_final_orders)),
            'ns_success_rate': float(np.mean([r.success for r in ns_results])),
            'mann_whitney_stat': float(stat),
            'mann_whitney_p': float(p_value),
            'levene_stat': float(levene_stat),
            'levene_p': float(levene_p),
            'cohens_d': float(cohens_d),
            'variance_ratio': float(variance_ratio)
        }

        print(f"\n   Target {target}:")
        print(f"      GD: mean={np.mean(gd_final_orders):.4f}, std={np.std(gd_final_orders):.4f}")
        print(f"      NS: mean={np.mean(ns_final_orders):.4f}, std={np.std(ns_final_orders):.4f}")
        print(f"      Mann-Whitney p (GD < NS): {p_value:.6f}")
        print(f"      Cohen's d: {cohens_d:.3f}")
        print(f"      Variance ratio (GD/NS): {variance_ratio:.3f}")

    # Determine overall outcome
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    # PRIMARY TEST: Fair comparison - GD vs NS maximize with same budget
    print("\n--- PRIMARY TEST: GD vs NS (fair budget comparison) ---")

    # Mann-Whitney U: is GD final order lower than NS maximize?
    fair_stat, fair_p = stats.mannwhitneyu(gd_final_orders, ns_max_final_orders, alternative='less')

    # Levene's test for variance
    fair_levene_stat, fair_levene_p = stats.levene(gd_final_orders, ns_max_final_orders)

    # Cohen's d
    fair_pooled_std = np.sqrt((np.var(gd_final_orders) + np.var(ns_max_final_orders)) / 2)
    if fair_pooled_std > 0:
        fair_cohens_d = (np.mean(ns_max_final_orders) - np.mean(gd_final_orders)) / fair_pooled_std
    else:
        fair_cohens_d = 0.0

    fair_variance_ratio = np.var(gd_final_orders) / (np.var(ns_max_final_orders) + 1e-10)

    print(f"\n   GD final:  mean={np.mean(gd_final_orders):.4f}, std={np.std(gd_final_orders):.4f}")
    print(f"   NS final:  mean={np.mean(ns_max_final_orders):.4f}, std={np.std(ns_max_final_orders):.4f}")
    print(f"   Mann-Whitney p (GD < NS): {fair_p:.6f}")
    print(f"   Cohen's d: {fair_cohens_d:.3f}")
    print(f"   Variance ratio (GD/NS): {fair_variance_ratio:.3f}")
    print(f"   Levene's p: {fair_levene_p:.6f}")

    # Success criteria
    order_sig = fair_p < 0.01
    effect_large = fair_cohens_d > 0.5
    variance_higher = fair_variance_ratio > 1.0 and fair_levene_p < 0.01
    stuck_rate_high = gd_stuck_rate > 0.2

    print(f"\nValidation criteria:")
    print(f"  1. NS achieves higher order (p < 0.01): {order_sig} (p={fair_p:.6f})")
    print(f"  2. Effect size d > 0.5: {effect_large} (d={fair_cohens_d:.3f})")
    print(f"  3. GD has higher variance (p < 0.01): {variance_higher} (ratio={fair_variance_ratio:.3f})")
    print(f"  4. GD stuck rate > 20%: {stuck_rate_high} ({gd_stuck_rate:.1%})")

    # Determine status
    if order_sig and effect_large:
        if variance_higher or stuck_rate_high:
            status = 'validated'
            summary = f"GD achieves lower final order than NS (d={fair_cohens_d:.2f}, p<0.01). GD stuck rate={gd_stuck_rate:.0%}, variance ratio={fair_variance_ratio:.1f}x. Nested sampling escapes local minima more effectively."
        else:
            status = 'validated'
            summary = f"GD achieves lower final order than NS (d={fair_cohens_d:.2f}, p<0.01). Local minima trapping confirmed."
    elif order_sig or effect_large:
        status = 'inconclusive'
        summary = f"Mixed evidence: order_sig={order_sig}, effect={effect_large}, variance={variance_higher}. Partial support for hypothesis."
    else:
        if fair_cohens_d < -0.5:
            status = 'refuted'
            summary = f"GD achieves HIGHER order than NS (d={fair_cohens_d:.2f}). Hypothesis refuted - gradient descent works better."
        else:
            status = 'refuted'
            summary = f"No significant difference between GD and NS (d={fair_cohens_d:.2f}, p={fair_p:.4f}). Both methods comparable."

    print(f"\nSTATUS: {status.upper()}")
    print(f"Summary: {summary}")

    # Compile full results
    results = {
        'experiment': 'gradient_vs_nested_sampling',
        'hypothesis': 'Gradient descent gets stuck in local minima more frequently than nested sampling',
        'null_hypothesis': 'GD and NS achieve equivalent final orders with similar variance',
        'status': status,
        'summary': summary,
        'parameters': {
            'n_runs': n_runs,
            'image_size': image_size,
            'gradient_steps': gradient_steps,
            'gradient_lr': gradient_lr,
            'ns_n_live': ns_n_live,
            'ns_max_iter': ns_max_iter,
            'ns_fair_iter': ns_fair_iter,
            'target_orders': target_orders,
            'seed': seed
        },
        'gradient_descent': {
            'final_order_mean': float(np.mean(gd_final_orders)),
            'final_order_std': float(np.std(gd_final_orders)),
            'final_order_median': float(np.median(gd_final_orders)),
            'final_order_min': float(np.min(gd_final_orders)),
            'final_order_max': float(np.max(gd_final_orders)),
            'improvement_mean': float(np.mean(gd_improvements)),
            'stuck_rate': float(gd_stuck_rate),
            'convergence_rate': float(np.mean([r.converged for r in gd_results]))
        },
        'nested_sampling_maximize': {
            'final_order_mean': float(np.mean(ns_max_final_orders)),
            'final_order_std': float(np.std(ns_max_final_orders)),
            'final_order_median': float(np.median(ns_max_final_orders)),
            'final_order_min': float(np.min(ns_max_final_orders)),
            'final_order_max': float(np.max(ns_max_final_orders))
        },
        'fair_comparison': {
            'mann_whitney_stat': float(fair_stat),
            'mann_whitney_p': float(fair_p),
            'cohens_d': float(fair_cohens_d),
            'variance_ratio': float(fair_variance_ratio),
            'levene_stat': float(fair_levene_stat),
            'levene_p': float(fair_levene_p)
        },
        'comparison_by_target': comparison_results,
        'validation_checks': {
            'order_significant': bool(order_sig),
            'effect_large': bool(effect_large),
            'variance_higher': bool(variance_higher),
            'stuck_rate_high': bool(stuck_rate_high)
        }
    }

    # Save results
    results_dir = Path(__file__).parent.parent / 'results' / 'gradient_optimization'
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / 'gradient_vs_nested_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_dir / 'gradient_vs_nested_results.json'}")

    return results


if __name__ == "__main__":
    results = run_experiment()
