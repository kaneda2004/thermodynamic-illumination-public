#!/usr/bin/env python3
"""
RES-263: Iteration Efficiency Across Thresholds (Final)
Proper implementation: continue until reaching target order, track per-iteration efficiency.
"""

import os
import sys
import json
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

def generate_cppn_image(seed: int, use_full: bool = False, size: int = 32):
    """Generate diverse CPPN images."""
    np.random.seed(seed)

    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(x, y)

    scale_x = np.random.uniform(0.5, 2.5)
    scale_y = np.random.uniform(0.5, 2.5)
    scale_r = np.random.uniform(0.8, 2.0)
    bias = np.random.uniform(-0.3, 0.3)

    r = np.sqrt((xx * scale_x)**2 + (yy * scale_y)**2) / scale_r

    # Baseline
    image = (
        np.sin(4 * np.pi * xx) * np.cos(4 * np.pi * yy) +
        0.5 * np.sin(np.pi * r) +
        bias
    )

    if use_full:
        xy = (xx * scale_x) * (yy * scale_y)
        x2 = (xx * scale_x)**2
        y2 = (yy * scale_y)**2
        image += (
            0.3 * np.sin(2*np.pi * xy) +
            0.2 * np.cos(4*np.pi * x2) +
            0.2 * np.sin(4*np.pi * y2) +
            0.15 * np.cos(2*np.pi * (x2 + y2))
        )

    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    return image

def measure_order(image: np.ndarray) -> float:
    """Measure order via gradient magnitude (spatial structure)."""
    gy = np.gradient(image, axis=0)
    gx = np.gradient(image, axis=1)
    grad = np.sqrt(gx**2 + gy**2)
    return float(np.mean(grad))

def run_nested_sampling_to_convergence(
    use_full: bool,
    target_order: float,
    n_live: int = 16,
    seed_offset: int = 0
):
    """
    Run nested sampling until reaching target order.
    Track per-iteration metrics including ESS.
    Returns: iterations, samples, per-iteration ESS values
    """
    np.random.seed(seed_offset)

    live_images = []
    live_orders = []
    total_samples = 0
    iteration_ess_values = []  # ESS for each iteration
    iteration_acceptance_rates = []

    # Initialize
    for i in range(n_live):
        img = generate_cppn_image(seed_offset + i, use_full=use_full)
        order = measure_order(img)
        live_images.append(img)
        live_orders.append(order)
        total_samples += 1

    max_iterations = 100
    target_reached_iter = -1

    for iteration in range(max_iterations):
        live_orders_arr = np.array(live_orders)
        mean_order = np.mean(live_orders_arr)
        worst_idx = np.argmin(live_orders_arr)
        worst_order = live_orders_arr[worst_idx]

        # Check if target reached
        if mean_order >= target_order and target_reached_iter < 0:
            target_reached_iter = iteration

        # Compute ESS (effective sample size from live points)
        # ESS = (sum(weights))^2 / sum(weights^2)
        # Use relative order differences as weights
        relative_orders = live_orders_arr - worst_order
        weights = np.exp(5 * np.maximum(0, relative_orders))  # Higher order = higher weight
        w_norm = weights / np.sum(weights)
        ess = np.sum(w_norm)**2 / np.sum(w_norm**2)
        iteration_ess_values.append(ess)

        # Try to improve worst point
        accepted = 0
        attempted = 0

        for attempt in range(100):
            new_seed = seed_offset + iteration * 100 + attempt
            new_img = generate_cppn_image(new_seed, use_full=use_full)
            new_order = measure_order(new_img)
            attempted += 1
            total_samples += 1

            if new_order > worst_order:
                live_images[worst_idx] = new_img
                live_orders[worst_idx] = new_order
                accepted += 1
                break

        acceptance_rate = accepted / max(attempted, 1)
        iteration_acceptance_rates.append(acceptance_rate)

        # Early stopping: target reached AND acceptance rate has dropped
        if target_reached_iter >= 0 and iteration > target_reached_iter + 2:
            if np.mean(iteration_acceptance_rates[-3:]) < 0.02:
                break

    return {
        'total_iterations': len(iteration_ess_values),
        'total_samples': total_samples,
        'target_reached_iter': max(0, target_reached_iter),
        'mean_ess_per_iter': float(np.mean(iteration_ess_values)) if iteration_ess_values else 0.0,
        'mean_acceptance_rate': float(np.mean(iteration_acceptance_rates)) if iteration_acceptance_rates else 0.0,
        'ess_values': iteration_ess_values,
        'acceptance_rates': iteration_acceptance_rates
    }

def run_threshold_campaign(threshold: float, n_trials: int = 8):
    """Run multiple trials at a single threshold."""
    baseline_results = []
    full_results = []

    for trial in range(n_trials):
        baseline = run_nested_sampling_to_convergence(
            use_full=False,
            target_order=threshold,
            seed_offset=trial * 1000
        )
        baseline_results.append(baseline)

        full = run_nested_sampling_to_convergence(
            use_full=True,
            target_order=threshold,
            seed_offset=trial * 1000 + 50000
        )
        full_results.append(full)

    # Aggregate
    baseline_ess = np.mean([r['mean_ess_per_iter'] for r in baseline_results])
    full_ess = np.mean([r['mean_ess_per_iter'] for r in full_results])

    baseline_acc = np.mean([r['mean_acceptance_rate'] for r in baseline_results])
    full_acc = np.mean([r['mean_acceptance_rate'] for r in full_results])

    baseline_iters = np.mean([r['total_iterations'] for r in baseline_results])
    full_iters = np.mean([r['total_iterations'] for r in full_results])

    baseline_samples = np.mean([r['total_samples'] for r in baseline_results])
    full_samples = np.mean([r['total_samples'] for r in full_results])

    baseline_reached = np.mean([r['target_reached_iter'] for r in baseline_results])
    full_reached = np.mean([r['target_reached_iter'] for r in full_results])

    # Compute improvements
    ess_improvement = 100 * (full_ess - baseline_ess) / (baseline_ess + 1e-8)
    acc_improvement = 100 * (full_acc - baseline_acc) / (baseline_acc + 1e-8)
    iter_speedup = baseline_iters / (full_iters + 1e-8)
    sample_speedup = baseline_samples / (full_samples + 1e-8)
    convergence_speedup = baseline_reached / (full_reached + 1e-8)

    return {
        'threshold': float(threshold),
        'baseline_ess_per_iter': float(baseline_ess),
        'full_ess_per_iter': float(full_ess),
        'ess_improvement_pct': float(ess_improvement),
        'baseline_acceptance_rate': float(baseline_acc),
        'full_acceptance_rate': float(full_acc),
        'acceptance_improvement_pct': float(acc_improvement),
        'baseline_iterations': float(baseline_iters),
        'full_iterations': float(full_iters),
        'iteration_speedup': float(iter_speedup),
        'baseline_samples': float(baseline_samples),
        'full_samples': float(full_samples),
        'sample_speedup': float(sample_speedup),
        'baseline_convergence_iters': float(baseline_reached),
        'full_convergence_iters': float(full_reached),
        'convergence_speedup': float(convergence_speedup),
        'n_trials': n_trials
    }

def main():
    print("RES-263: Iteration Efficiency Across Thresholds (Final)")
    print("=" * 60)

    thresholds = [0.04, 0.08, 0.12]  # Order thresholds (gradient magnitude)
    all_results = {}

    for threshold in thresholds:
        print(f"\nRunning threshold = {threshold}...")
        result = run_threshold_campaign(threshold, n_trials=8)
        all_results[str(threshold)] = result

        print(f"\n  Threshold = {threshold}")
        print(f"    Baseline ESS/iter: {result['baseline_ess_per_iter']:.4f}")
        print(f"    Full ESS/iter:     {result['full_ess_per_iter']:.4f}")
        print(f"    ESS improvement:   {result['ess_improvement_pct']:+.1f}%")
        print(f"    Baseline acceptance rate: {result['baseline_acceptance_rate']:.3f}")
        print(f"    Full acceptance rate:     {result['full_acceptance_rate']:.3f}")
        print(f"    Acceptance improvement:   {result['acceptance_improvement_pct']:+.1f}%")
        print(f"    Baseline convergence iters: {result['baseline_convergence_iters']:.1f}")
        print(f"    Full convergence iters:     {result['full_convergence_iters']:.1f}")
        print(f"    Convergence speedup: {result['convergence_speedup']:.2f}x")
        print(f"    Iteration speedup:   {result['iteration_speedup']:.2f}x")
        print(f"    Sample speedup:      {result['sample_speedup']:.2f}x")

    # Save results
    results_dir = Path('/Users/matt/Development/monochrome_noise_converger/results/entropy_reduction')
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / 'res_263_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✓ Results saved")

    # Analyze for status
    thresholds_str = [str(t) for t in thresholds]
    ess_improvements = [all_results[t]['ess_improvement_pct'] for t in thresholds_str]
    acc_improvements = [all_results[t]['acceptance_improvement_pct'] for t in thresholds_str]
    iter_speedups = [all_results[t]['iteration_speedup'] for t in thresholds_str]

    print(f"\nESS improvements across thresholds: {[f'{x:.1f}%' for x in ess_improvements]}")
    print(f"Acceptance improvements: {[f'{x:.1f}%' for x in acc_improvements]}")
    print(f"Iteration speedups: {[f'{x:.2f}x' for x in iter_speedups]}")

    # Hypothesis: per-iteration efficiency explains speedup
    # Evidence: acceptance rate improvements and/or ESS per iteration improvements
    positive_ess = sum(1 for e in ess_improvements if e > 3)
    positive_acc = sum(1 for a in acc_improvements if a > 5)
    positive_iter = sum(1 for s in iter_speedups if s > 1.05)

    if positive_acc >= 2 and positive_iter >= 2:
        status = "validated"
        summary = "Richer features significantly accelerate per-iteration convergence across thresholds"
    elif positive_acc >= 1 or positive_iter >= 1:
        status = "inconclusive"
        summary = "Per-iteration efficiency improvements are present but threshold-dependent"
    else:
        status = "refuted"
        summary = "Richer features do not improve per-iteration convergence efficiency"

    print(f"\nStatus: {status.upper()}")
    print(f"Summary: {summary}")

    detail = (
        f"Threshold-dependent iteration efficiency: "
        f"T=0.04: {ess_improvements[0]:.1f}% ESS, {acc_improvements[0]:.1f}% acceptance, {iter_speedups[0]:.2f}x iters; "
        f"T=0.08: {ess_improvements[1]:.1f}% ESS, {acc_improvements[1]:.1f}% acceptance, {iter_speedups[1]:.2f}x iters; "
        f"T=0.12: {ess_improvements[2]:.1f}% ESS, {acc_improvements[2]:.1f}% acceptance, {iter_speedups[2]:.2f}x iters. "
        f"{summary}"
    )

    os.system(f'uv run python -m research_system.log_manager complete RES-263 {status} "{detail}"')

    return status

if __name__ == '__main__':
    status = main()
    sys.exit(0 if status == "validated" else 1)
