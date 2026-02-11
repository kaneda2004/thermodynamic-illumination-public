#!/usr/bin/env python3
"""
RES-259: Iteration Efficiency Across Thresholds
Measure per-iteration convergence speed and ESS across order thresholds.
Hypothesis: Richer features enable faster per-iteration convergence.
"""

import os
import sys
import json
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

def generate_diverse_cppn(seed: int, use_full: bool = False, size: int = 32):
    """Generate a CPPN-based image with seeded randomness."""
    np.random.seed(seed)

    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(x, y)

    # Random parameters for diversity
    phase_x = np.random.uniform(0, 2*np.pi)
    phase_y = np.random.uniform(0, 2*np.pi)
    freq = np.random.uniform(2, 8)
    r = np.sqrt(xx**2 + yy**2)
    freq_r = np.random.uniform(1, 4)

    # Core sinusoid features
    image = np.sin(freq * np.pi * xx + phase_x) * np.cos(freq * np.pi * yy + phase_y)
    image = image + 0.5 * np.cos(freq_r * np.pi * r)

    # Additional features for full version
    if use_full:
        # Multiplicative interactions
        image = image + 0.2 * np.sin(3*np.pi * xx * yy)
        # Polynomial features
        image = image + 0.15 * np.cos(4*np.pi * xx**2)
        image = image + 0.15 * np.sin(4*np.pi * yy**2)
        # Radial features
        image = image + 0.1 * np.sin(np.pi * (xx**2 + yy**2))

    # Normalize
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    return image

def compute_spatial_order(image: np.ndarray) -> float:
    """
    Measure spatial order using gradient magnitude.
    High gradient = more spatial structure = higher order.
    """
    # Compute Sobel-like gradients
    gy = np.diff(image, axis=0, prepend=image[0:1])
    gx = np.diff(image, axis=1, prepend=image[:, 0:1])
    grad_mag = np.sqrt(gx**2 + gy**2)
    return float(np.mean(grad_mag))

def nested_sampling_trial(
    use_full: bool,
    target_order: float,
    n_live: int = 12,
    seed_base: int = 0
) -> dict:
    """
    Single nested sampling trial.
    Returns metrics tracking efficiency and convergence.
    """
    np.random.seed(seed_base)

    # Initialize live set
    live_images = []
    live_orders = []

    for i in range(n_live):
        img = generate_diverse_cppn(seed_base + i, use_full=use_full)
        order = compute_spatial_order(img)
        live_images.append(img)
        live_orders.append(order)

    # Iteration metrics
    iteration_data = []
    samples_drawn = n_live
    target_reached_at = None

    max_iters = 60

    for iteration in range(max_iters):
        live_orders_arr = np.array(live_orders)
        mean_order = np.mean(live_orders_arr)
        worst_idx = np.argmin(live_orders_arr)
        worst_order = live_orders_arr[worst_idx]

        # Track if target reached
        if mean_order >= target_order and target_reached_at is None:
            target_reached_at = iteration

        # Acceptance ratio = probability of improving worst point
        accepted = 0
        proposed = 0

        for attempt in range(50):
            new_img = generate_diverse_cppn(
                seed_base + iteration * 100 + attempt,
                use_full=use_full
            )
            new_order = compute_spatial_order(new_img)
            proposed += 1
            samples_drawn += 1

            if new_order > worst_order:
                live_images[worst_idx] = new_img
                live_orders[worst_idx] = new_order
                accepted += 1
                break

        acceptance_rate = accepted / max(proposed, 1)

        # Compute ESS from live set
        # Higher variance in live orders = higher diversity = higher ESS
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

        # Stop if converged: target reached and acceptance very low
        if target_reached_at is not None and iteration > target_reached_at + 3:
            recent_acc = np.mean([iteration_data[i]['acceptance_rate']
                                 for i in range(max(0, len(iteration_data)-3), len(iteration_data))])
            if recent_acc < 0.01:
                break

    return {
        'total_iterations': len(iteration_data),
        'total_samples': samples_drawn,
        'target_reached_at': target_reached_at if target_reached_at is not None else len(iteration_data),
        'mean_acceptance_rate': float(np.mean([d['acceptance_rate'] for d in iteration_data])),
        'mean_order_variance': float(np.mean([d['order_variance'] for d in iteration_data])),
        'final_mean_order': float(iteration_data[-1]['mean_order']) if iteration_data else 0.0,
        'iterations_data': iteration_data
    }

def run_threshold_test(threshold: float, n_trials: int = 8) -> dict:
    """Run multiple trials at a single threshold."""
    baseline_trials = []
    full_trials = []

    print(f"    Testing threshold {threshold:.3f} with {n_trials} trials...")

    for trial_id in range(n_trials):
        baseline = nested_sampling_trial(
            use_full=False,
            target_order=threshold,
            seed_base=trial_id * 1000
        )
        baseline_trials.append(baseline)

        full = nested_sampling_trial(
            use_full=True,
            target_order=threshold,
            seed_base=trial_id * 1000 + 100000
        )
        full_trials.append(full)

    # Aggregate metrics
    baseline_acc = np.mean([t['mean_acceptance_rate'] for t in baseline_trials])
    full_acc = np.mean([t['mean_acceptance_rate'] for t in full_trials])

    baseline_ess = np.mean([t['mean_order_variance'] for t in baseline_trials])
    full_ess = np.mean([t['mean_order_variance'] for t in full_trials])

    baseline_iters = np.mean([t['total_iterations'] for t in baseline_trials])
    full_iters = np.mean([t['total_iterations'] for t in full_trials])

    baseline_samples = np.mean([t['total_samples'] for t in baseline_trials])
    full_samples = np.mean([t['total_samples'] for t in full_trials])

    baseline_convergence = np.mean([t['target_reached_at'] for t in baseline_trials])
    full_convergence = np.mean([t['target_reached_at'] for t in full_trials])

    # Compute improvements
    acc_improvement = 100 * (full_acc - baseline_acc) / (baseline_acc + 1e-8)
    ess_improvement = 100 * (full_ess - baseline_ess) / (baseline_ess + 1e-8)
    iter_efficiency = baseline_iters / (full_iters + 1e-8)
    convergence_speedup = baseline_convergence / (full_convergence + 1e-8)
    sample_efficiency = baseline_samples / (full_samples + 1e-8)

    # Per-iteration efficiency: samples per iteration
    baseline_samples_per_iter = baseline_samples / max(baseline_iters, 1)
    full_samples_per_iter = full_samples / max(full_iters, 1)
    per_iter_efficiency = baseline_samples_per_iter / (full_samples_per_iter + 1e-8)

    return {
        'threshold': float(threshold),
        'baseline_acceptance_rate': float(baseline_acc),
        'full_acceptance_rate': float(full_acc),
        'acceptance_improvement_pct': float(acc_improvement),
        'baseline_ess_per_iter': float(baseline_ess),
        'full_ess_per_iter': float(full_ess),
        'ess_improvement_pct': float(ess_improvement),
        'baseline_iterations': float(baseline_iters),
        'full_iterations': float(full_iters),
        'iteration_efficiency_speedup': float(iter_efficiency),
        'baseline_convergence_iters': float(baseline_convergence),
        'full_convergence_iters': float(full_convergence),
        'convergence_speedup': float(convergence_speedup),
        'baseline_total_samples': float(baseline_samples),
        'full_total_samples': float(full_samples),
        'per_iter_efficiency_speedup': float(per_iter_efficiency),
        'baseline_samples_per_iter': float(baseline_samples_per_iter),
        'full_samples_per_iter': float(full_samples_per_iter)
    }

def main():
    print("RES-259: Iteration Efficiency Across Thresholds")
    print("=" * 60)

    # Test at three representative order thresholds
    thresholds = [0.02, 0.05, 0.10]
    all_results = {}

    for threshold in thresholds:
        print(f"\n  Threshold = {threshold}")
        result = run_threshold_test(threshold, n_trials=8)
        all_results[str(threshold)] = result

        print(f"    Baseline: {result['baseline_acceptance_rate']:.3f} acc, "
              f"{result['baseline_iterations']:.1f} iters")
        print(f"    Full:     {result['full_acceptance_rate']:.3f} acc, "
              f"{result['full_iterations']:.1f} iters")
        print(f"    Improvements: {result['acceptance_improvement_pct']:+.1f}% acc, "
              f"{result['iteration_efficiency_speedup']:.2f}x iter efficiency")

    # Save results
    results_dir = Path('/Users/matt/Development/monochrome_noise_converger/results/entropy_reduction')
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / 'res_259_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✓ Results saved to results/entropy_reduction/res_259_results.json")

    # Analysis
    thresholds_str = [str(t) for t in thresholds]
    acc_improvements = [all_results[t]['acceptance_improvement_pct'] for t in thresholds_str]
    iter_speedups = [all_results[t]['iteration_efficiency_speedup'] for t in thresholds_str]
    ess_improvements = [all_results[t]['ess_improvement_pct'] for t in thresholds_str]

    print(f"\nAcceptance rate improvements: {[f'{x:+.1f}%' for x in acc_improvements]}")
    print(f"Iteration efficiency speedups: {[f'{x:.2f}x' for x in iter_speedups]}")
    print(f"ESS improvements: {[f'{x:+.1f}%' for x in ess_improvements]}")

    # Determine status
    # Hypothesis: richer features enable faster per-iteration convergence
    # Evidence: improvement in acceptance rate (acceptance per proposal) across thresholds
    positive_thresholds = sum(1 for acc in acc_improvements if acc > 5)
    positive_iter_thresholds = sum(1 for sp in iter_speedups if sp > 1.05)

    if positive_thresholds >= 2 and positive_iter_thresholds >= 2:
        status = "validated"
        summary = "Richer features consistently accelerate per-iteration convergence across thresholds"
    elif positive_thresholds >= 1 or positive_iter_thresholds >= 1:
        status = "inconclusive"
        summary = "Per-iteration efficiency improvements are present but threshold-dependent and modest"
    else:
        status = "refuted"
        summary = "Richer features do not improve per-iteration convergence efficiency"

    print(f"\nStatus: {status.upper()}")
    print(f"Summary: {summary}")

    detail = (
        f"Per-iteration convergence across thresholds: "
        f"T=0.02: {acc_improvements[0]:+.1f}% acc, {iter_speedups[0]:.2f}x; "
        f"T=0.05: {acc_improvements[1]:+.1f}% acc, {iter_speedups[1]:.2f}x; "
        f"T=0.10: {acc_improvements[2]:+.1f}% acc, {iter_speedups[2]:.2f}x. "
        f"{summary}"
    )

    os.system(f'uv run python -m research_system.log_manager complete RES-259 {status} "{detail}"')

    return status

if __name__ == '__main__':
    status = main()
    sys.exit(0 if status == "validated" else 1)
