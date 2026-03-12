#!/usr/bin/env python3
"""
RES-259: Iteration Efficiency Across Thresholds (Fixed)
Properly implement threshold-dependent sampling by using percentile-based targets.
"""

import os
import sys
import json
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

def generate_diverse_cppn(seed: int, use_full: bool = False, size: int = 32):
    """Generate a CPPN-based image."""
    np.random.seed(seed)

    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(x, y)

    phase_x = np.random.uniform(0, 2*np.pi)
    phase_y = np.random.uniform(0, 2*np.pi)
    freq = np.random.uniform(2, 8)
    r = np.sqrt(xx**2 + yy**2)
    freq_r = np.random.uniform(1, 4)

    image = np.sin(freq * np.pi * xx + phase_x) * np.cos(freq * np.pi * yy + phase_y)
    image = image + 0.5 * np.cos(freq_r * np.pi * r)

    if use_full:
        image = image + 0.2 * np.sin(3*np.pi * xx * yy)
        image = image + 0.15 * np.cos(4*np.pi * xx**2)
        image = image + 0.15 * np.sin(4*np.pi * yy**2)
        image = image + 0.1 * np.sin(np.pi * (xx**2 + yy**2))

    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    return image

def compute_order(image: np.ndarray) -> float:
    """Measure spatial order using gradient magnitude."""
    gy = np.diff(image, axis=0, prepend=image[0:1])
    gx = np.diff(image, axis=1, prepend=image[:, 0:1])
    grad_mag = np.sqrt(gx**2 + gy**2)
    return float(np.mean(grad_mag))

def nested_sampling_fixed(
    use_full: bool,
    target_percentile: float,
    n_live: int = 12,
    seed_base: int = 0
) -> dict:
    """
    Nested sampling with percentile-based targets.
    target_percentile: aim for this percentile of initial samples (0.2=20th percentile, etc.)
    """
    np.random.seed(seed_base)

    # Initialize live set
    live_images = []
    live_orders = []

    for i in range(n_live):
        img = generate_diverse_cppn(seed_base + i, use_full=use_full)
        order = compute_order(img)
        live_images.append(img)
        live_orders.append(order)

    # Determine target based on initial sample distribution
    initial_orders = np.array(live_orders)
    target_order = np.percentile(initial_orders, 100 * target_percentile)

    iteration_data = []
    samples_drawn = n_live
    target_reached_at = None

    max_iters = 80

    for iteration in range(max_iters):
        live_orders_arr = np.array(live_orders)
        mean_order = np.mean(live_orders_arr)
        worst_idx = np.argmin(live_orders_arr)
        worst_order = live_orders_arr[worst_idx]

        # Track if target reached (mean exceeds target)
        if mean_order >= target_order and target_reached_at is None:
            target_reached_at = iteration

        # Acceptance rate
        accepted = 0
        proposed = 0

        for attempt in range(50):
            new_img = generate_diverse_cppn(
                seed_base + iteration * 100 + attempt,
                use_full=use_full
            )
            new_order = compute_order(new_img)
            proposed += 1
            samples_drawn += 1

            if new_order > worst_order:
                live_images[worst_idx] = new_img
                live_orders[worst_idx] = new_order
                accepted += 1
                break

        acceptance_rate = accepted / max(proposed, 1)
        order_variance = float(np.var(live_orders_arr))

        iteration_data.append({
            'iteration': iteration,
            'mean_order': float(mean_order),
            'worst_order': float(worst_order),
            'acceptance_rate': float(acceptance_rate),
            'order_variance': float(order_variance),
            'samples_this_iter': proposed,
            'cumulative_samples': samples_drawn
        })

        # Stop: target reached and convergence
        if target_reached_at is not None and iteration > target_reached_at + 4:
            recent_acc = np.mean([iteration_data[i]['acceptance_rate']
                                 for i in range(max(0, len(iteration_data)-4), len(iteration_data))])
            if recent_acc < 0.002:
                break

    return {
        'target_order': float(target_order),
        'target_percentile': float(target_percentile),
        'total_iterations': len(iteration_data),
        'total_samples': samples_drawn,
        'target_reached_at': target_reached_at if target_reached_at is not None else len(iteration_data),
        'mean_acceptance_rate': float(np.mean([d['acceptance_rate'] for d in iteration_data])) if iteration_data else 0.0,
        'mean_order_variance': float(np.mean([d['order_variance'] for d in iteration_data])) if iteration_data else 0.0,
        'final_mean_order': float(iteration_data[-1]['mean_order']) if iteration_data else 0.0,
    }

def run_percentile_test(percentile: float, n_trials: int = 8) -> dict:
    """Test at a specific target percentile."""
    baseline_trials = []
    full_trials = []

    for trial_id in range(n_trials):
        baseline = nested_sampling_fixed(
            use_full=False,
            target_percentile=percentile,
            seed_base=trial_id * 1000
        )
        baseline_trials.append(baseline)

        full = nested_sampling_fixed(
            use_full=True,
            target_percentile=percentile,
            seed_base=trial_id * 1000 + 100000
        )
        full_trials.append(full)

    # Aggregate
    baseline_acc = np.mean([t['mean_acceptance_rate'] for t in baseline_trials])
    full_acc = np.mean([t['mean_acceptance_rate'] for t in full_trials])

    baseline_ess = np.mean([t['mean_order_variance'] for t in baseline_trials])
    full_ess = np.mean([t['mean_order_variance'] for t in full_trials])

    baseline_iters = np.mean([t['total_iterations'] for t in baseline_trials])
    full_iters = np.mean([t['total_iterations'] for t in full_trials])

    baseline_samples = np.mean([t['total_samples'] for t in baseline_trials])
    full_samples = np.mean([t['total_samples'] for t in full_trials])

    baseline_reached = np.mean([t['target_reached_at'] for t in baseline_trials])
    full_reached = np.mean([t['target_reached_at'] for t in full_trials])

    # Improvements
    acc_improvement = 100 * (full_acc - baseline_acc) / (baseline_acc + 1e-8)
    ess_improvement = 100 * (full_ess - baseline_ess) / (baseline_ess + 1e-8)
    iter_efficiency = baseline_iters / (full_iters + 1e-8)
    sample_efficiency = baseline_samples / (full_samples + 1e-8)
    convergence_speedup = baseline_reached / (full_reached + 1e-8)

    return {
        'target_percentile': float(percentile),
        'baseline_acceptance_rate': float(baseline_acc),
        'full_acceptance_rate': float(full_acc),
        'acceptance_improvement_pct': float(acc_improvement),
        'baseline_ess_per_iter': float(baseline_ess),
        'full_ess_per_iter': float(full_ess),
        'ess_improvement_pct': float(ess_improvement),
        'baseline_iterations': float(baseline_iters),
        'full_iterations': float(full_iters),
        'iteration_efficiency': float(iter_efficiency),
        'baseline_samples': float(baseline_samples),
        'full_samples': float(full_samples),
        'sample_efficiency': float(sample_efficiency),
        'baseline_convergence_iters': float(baseline_reached),
        'full_convergence_iters': float(full_reached),
        'convergence_speedup': float(convergence_speedup)
    }

def main():
    print("RES-259: Iteration Efficiency Across Percentile Thresholds")
    print("=" * 60)

    # Test at three percentile levels
    percentiles = [0.2, 0.5, 0.8]  # 20th, 50th, 80th percentile
    all_results = {}

    for percentile in percentiles:
        print(f"\n  Testing 100*{percentile}th percentile target...")
        result = run_percentile_test(percentile, n_trials=8)
        all_results[str(percentile)] = result

        print(f"    Baseline: {result['baseline_acceptance_rate']:.4f} acc, "
              f"{result['baseline_iterations']:.1f} iters, "
              f"{result['baseline_samples']:.0f} samples")
        print(f"    Full:     {result['full_acceptance_rate']:.4f} acc, "
              f"{result['full_iterations']:.1f} iters, "
              f"{result['full_samples']:.0f} samples")
        print(f"    Acc improvement: {result['acceptance_improvement_pct']:+.1f}%, "
              f"Iter efficiency: {result['iteration_efficiency']:.2f}x")

    # Save
    results_dir = Path('/Users/matt/Development/monochrome_noise_converger/results/entropy_reduction')
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / 'res_259_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✓ Results saved")

    # Analysis
    percentiles_str = [str(p) for p in percentiles]
    acc_improvements = [all_results[p]['acceptance_improvement_pct'] for p in percentiles_str]
    iter_efficiencies = [all_results[p]['iteration_efficiency'] for p in percentiles_str]
    ess_improvements = [all_results[p]['ess_improvement_pct'] for p in percentiles_str]

    print(f"\nAcceptance improvements: {[f'{x:+.1f}%' for x in acc_improvements]}")
    print(f"Iteration efficiencies: {[f'{x:.2f}x' for x in iter_efficiencies]}")
    print(f"ESS improvements: {[f'{x:+.1f}%' for x in ess_improvements]}")

    # Status determination
    positive_acc = sum(1 for a in acc_improvements if a > 5)
    positive_iter = sum(1 for e in iter_efficiencies if e > 1.05)
    threshold_depend = len(set(x > 0 for x in acc_improvements)) > 1

    if positive_acc >= 2 and positive_iter >= 2:
        status = "validated"
        summary = "Richer features consistently accelerate per-iteration convergence"
    elif threshold_depend and (positive_acc >= 1 or positive_iter >= 1):
        status = "inconclusive"
        summary = "Per-iteration efficiency gains are threshold-dependent"
    else:
        status = "refuted"
        summary = "Richer features do not improve per-iteration convergence"

    print(f"\nStatus: {status.upper()}")
    print(f"Summary: {summary}")

    detail = (
        f"Per-iteration convergence efficiency by percentile target: "
        f"P=20%: {acc_improvements[0]:+.1f}% acc, {iter_efficiencies[0]:.2f}x; "
        f"P=50%: {acc_improvements[1]:+.1f}% acc, {iter_efficiencies[1]:.2f}x; "
        f"P=80%: {acc_improvements[2]:+.1f}% acc, {iter_efficiencies[2]:.2f}x. "
        f"{summary}"
    )

    os.system(f'uv run python -m research_system.log_manager complete RES-259 {status} "{detail}"')

    return status

if __name__ == '__main__':
    status = main()
    sys.exit(0 if status == "validated" else 1)
