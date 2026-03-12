"""
Experiment: n_live Efficiency Scaling in Nested Sampling

Hypothesis: There exists an optimal n_live that minimizes total computational cost
(measured as total ESS contractions) to reach a given order threshold.

Theoretical background:
- Nested sampling compresses prior volume by factor (n_live-1)/n_live per iteration
- More live points = faster volume compression per iteration = fewer iterations
- But more live points = more diverse population = potentially easier ESS sampling
- Trade-off: iterations decrease, contractions per iteration may also decrease
- Total cost = iterations * mean_contractions_per_iteration

Null hypotheses:
H0_1: Contractions per iteration is independent of n_live
H0_2: There is no efficiency sweet spot (total contractions monotonic in n_live)

Method:
- Run nested sampling at n_live in [25, 50, 100, 200, 400]
- Fixed: image_size=32, threshold=0.1, seed varies
- Track: iterations to threshold, mean contractions per iteration
- Compute: total contractions = iterations * mean_contractions

Statistical tests:
- Spearman correlation for contractions_per_iter vs n_live
- Bootstrap CI for optimal n_live
- Cohen's d for effect size
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy import stats
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import random

from core.thermo_sampler_v3 import (
    CPPN, order_multiplicative, elliptical_slice_sample,
    log_prior, set_global_seed, LivePoint, PRIOR_SIGMA
)


@dataclass
class SamplingResult:
    """Result of a single nested sampling run."""
    n_live: int
    seed: int
    iterations_to_threshold: int
    total_contractions: int
    mean_contractions_per_iter: float
    final_order: float
    success: bool


def run_nested_sampling_efficiency(
    n_live: int,
    threshold: float,
    image_size: int,
    max_iterations: int,
    seed: int
) -> SamplingResult:
    """
    Run nested sampling and track efficiency metrics.

    Returns iteration count and contraction statistics when threshold is reached.
    """
    set_global_seed(seed)

    # Initialize live points
    live_points = []
    for _ in range(n_live):
        cppn = CPPN()
        img = cppn.render(image_size)
        order = order_multiplicative(img)
        lp = log_prior(cppn)
        live_points.append(LivePoint(cppn, img, order, lp, {}))

    total_contractions = 0
    iteration_contractions = []

    for iteration in range(max_iterations):
        # Find worst point
        worst_idx = min(range(n_live), key=lambda i: live_points[i].order_value)
        worst = live_points[worst_idx]
        current_threshold = worst.order_value

        # Check if we've reached target threshold
        if current_threshold >= threshold:
            mean_contractions = np.mean(iteration_contractions) if iteration_contractions else 0
            return SamplingResult(
                n_live=n_live,
                seed=seed,
                iterations_to_threshold=iteration,
                total_contractions=total_contractions,
                mean_contractions_per_iter=mean_contractions,
                final_order=current_threshold,
                success=True
            )

        # Select seed (valid points that satisfy constraint)
        valid_seeds = [i for i in range(n_live)
                       if i != worst_idx and live_points[i].order_value >= current_threshold]

        if not valid_seeds:
            valid_seeds = [i for i in range(n_live) if i != worst_idx]

        seed_idx = random.choice(valid_seeds)

        # ESS sampling
        new_cppn, new_img, new_order, new_logp, n_contract, success = elliptical_slice_sample(
            live_points[seed_idx].cppn, current_threshold, image_size, order_multiplicative
        )

        # Retry if failed
        if not success:
            for _ in range(3):
                alt_seed = random.choice(valid_seeds)
                new_cppn, new_img, new_order, new_logp, extra_contract, success = elliptical_slice_sample(
                    live_points[alt_seed].cppn, current_threshold, image_size, order_multiplicative
                )
                n_contract += extra_contract
                if success:
                    break

        total_contractions += n_contract
        iteration_contractions.append(n_contract)

        # Replace worst
        live_points[worst_idx] = LivePoint(new_cppn, new_img, new_order, new_logp, {})

    # Didn't reach threshold
    mean_contractions = np.mean(iteration_contractions) if iteration_contractions else 0
    final_order = min(lp.order_value for lp in live_points)
    return SamplingResult(
        n_live=n_live,
        seed=seed,
        iterations_to_threshold=max_iterations,
        total_contractions=total_contractions,
        mean_contractions_per_iter=mean_contractions,
        final_order=final_order,
        success=False
    )


def run_experiment():
    """
    Main experiment: vary n_live and measure efficiency.
    """
    print("=" * 70)
    print("N_LIVE EFFICIENCY SCALING EXPERIMENT")
    print("=" * 70)

    # Parameters
    n_live_values = [25, 50, 100, 200, 400]
    n_seeds = 8
    threshold = 0.1
    image_size = 32
    max_iterations = 2000  # Safety limit

    print(f"n_live values: {n_live_values}")
    print(f"Seeds per n_live: {n_seeds}")
    print(f"Threshold: {threshold}")
    print(f"Image size: {image_size}")
    print()

    results = []

    for n_live in n_live_values:
        print(f"\nTesting n_live = {n_live}...")
        n_live_results = []

        for seed in range(n_seeds):
            result = run_nested_sampling_efficiency(
                n_live=n_live,
                threshold=threshold,
                image_size=image_size,
                max_iterations=max_iterations,
                seed=seed * 1000 + n_live  # Unique seeds
            )
            n_live_results.append(result)

            status = "OK" if result.success else "TIMEOUT"
            print(f"  Seed {seed}: {result.iterations_to_threshold} iters, "
                  f"{result.mean_contractions_per_iter:.1f} contr/iter, "
                  f"{result.total_contractions} total [{status}]")

        results.extend(n_live_results)

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    # Group by n_live
    by_n_live = {}
    for r in results:
        if r.n_live not in by_n_live:
            by_n_live[r.n_live] = []
        by_n_live[r.n_live].append(r)

    # Summary statistics
    print("\nSummary by n_live:")
    print(f"{'n_live':>8} | {'Iters':>10} | {'Contr/Iter':>12} | {'Total Contr':>12} | {'Bits':>8}")
    print("-" * 65)

    summary_data = {}
    for n_live in sorted(by_n_live.keys()):
        runs = by_n_live[n_live]
        iters = [r.iterations_to_threshold for r in runs if r.success]
        contr_per = [r.mean_contractions_per_iter for r in runs if r.success]
        total_contr = [r.total_contractions for r in runs if r.success]

        if iters:
            # Bits = iterations / n_live (standard nested sampling)
            bits = [i / n_live for i in iters]

            summary_data[n_live] = {
                'mean_iters': np.mean(iters),
                'std_iters': np.std(iters),
                'mean_contr_per_iter': np.mean(contr_per),
                'std_contr_per_iter': np.std(contr_per),
                'mean_total_contr': np.mean(total_contr),
                'std_total_contr': np.std(total_contr),
                'mean_bits': np.mean(bits),
                'std_bits': np.std(bits),
                'n_success': len(iters)
            }

            print(f"{n_live:>8} | {np.mean(iters):>10.1f} | {np.mean(contr_per):>12.2f} | "
                  f"{np.mean(total_contr):>12.1f} | {np.mean(bits):>8.2f}")

    # Statistical tests
    print("\n" + "=" * 70)
    print("STATISTICAL TESTS")
    print("=" * 70)

    # Test 1: Spearman correlation for contractions_per_iter vs n_live
    n_live_arr = []
    contr_per_arr = []
    for r in results:
        if r.success:
            n_live_arr.append(r.n_live)
            contr_per_arr.append(r.mean_contractions_per_iter)

    spearman_contr, p_contr = stats.spearmanr(n_live_arr, contr_per_arr)
    print(f"\n1. Contractions/iteration vs n_live:")
    print(f"   Spearman r = {spearman_contr:.4f}, p = {p_contr:.2e}")

    # Effect size: compare n_live=25 vs n_live=400
    contr_25 = [r.mean_contractions_per_iter for r in by_n_live[25] if r.success]
    contr_400 = [r.mean_contractions_per_iter for r in by_n_live[400] if r.success]

    if contr_25 and contr_400:
        cohens_d_contr = (np.mean(contr_25) - np.mean(contr_400)) / np.sqrt(
            (np.var(contr_25) + np.var(contr_400)) / 2
        )
        print(f"   Cohen's d (n=25 vs n=400): {cohens_d_contr:.3f}")
    else:
        cohens_d_contr = 0

    # Test 2: Is there a sweet spot in total contractions?
    n_live_for_total = []
    total_contr_arr = []
    for r in results:
        if r.success:
            n_live_for_total.append(r.n_live)
            total_contr_arr.append(r.total_contractions)

    # Quadratic regression to detect U-shape
    log_n = np.log(n_live_for_total)
    X = np.column_stack([np.ones_like(log_n), log_n, log_n**2])
    y = np.array(total_contr_arr)

    try:
        beta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
        y_pred = X @ beta
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - ss_res / ss_tot

        # Quadratic coefficient significance
        # If beta[2] > 0, U-shape; if beta[2] < 0, inverted U
        print(f"\n2. Total contractions vs log(n_live) quadratic fit:")
        print(f"   Intercept: {beta[0]:.2f}")
        print(f"   Linear: {beta[1]:.2f}")
        print(f"   Quadratic: {beta[2]:.2f}")
        print(f"   R^2 = {r_squared:.4f}")

        # Optimal n_live (minimum of quadratic)
        if abs(beta[2]) > 1e-6:
            optimal_log_n = -beta[1] / (2 * beta[2])
            optimal_n_live = np.exp(optimal_log_n)
            print(f"   Optimal n_live (if U-shape): {optimal_n_live:.1f}")
        else:
            optimal_n_live = None
            print("   No clear optimal (quadratic coefficient ~0)")
    except Exception as e:
        print(f"   Quadratic fit failed: {e}")
        r_squared = 0
        beta = [0, 0, 0]
        optimal_n_live = None

    # Test 3: Does bits-to-threshold depend on n_live?
    bits_arr = [r.iterations_to_threshold / r.n_live for r in results if r.success]
    n_live_bits = [r.n_live for r in results if r.success]

    spearman_bits, p_bits = stats.spearmanr(n_live_bits, bits_arr)
    print(f"\n3. Bits to threshold vs n_live:")
    print(f"   Spearman r = {spearman_bits:.4f}, p = {p_bits:.2e}")

    bits_25 = [r.iterations_to_threshold / r.n_live for r in by_n_live[25] if r.success]
    bits_400 = [r.iterations_to_threshold / r.n_live for r in by_n_live[400] if r.success]

    if bits_25 and bits_400:
        cohens_d_bits = (np.mean(bits_25) - np.mean(bits_400)) / np.sqrt(
            (np.var(bits_25) + np.var(bits_400)) / 2
        )
        print(f"   Cohen's d (n=25 vs n=400): {cohens_d_bits:.3f}")
    else:
        cohens_d_bits = 0

    # Primary hypothesis test: contractions_per_iter decreases with n_live
    print("\n" + "=" * 70)
    print("HYPOTHESIS EVALUATION")
    print("=" * 70)

    # H0_1: contractions_per_iter independent of n_live
    h0_1_rejected = p_contr < 0.01 and abs(cohens_d_contr) > 0.5
    print(f"\nH0_1 (contr/iter independent of n_live):")
    print(f"  p = {p_contr:.2e} {'< 0.01 REJECT' if p_contr < 0.01 else '>= 0.01 FAIL TO REJECT'}")
    print(f"  |d| = {abs(cohens_d_contr):.3f} {'> 0.5 LARGE' if abs(cohens_d_contr) > 0.5 else '<= 0.5 SMALL'}")
    print(f"  Direction: {'decreases' if spearman_contr < 0 else 'increases'} with n_live")

    # Find optimal n_live by minimum mean total contractions
    min_total_n_live = min(summary_data.keys(), key=lambda k: summary_data[k]['mean_total_contr'])
    print(f"\nEmpirical optimal n_live (min total contractions): {min_total_n_live}")
    print(f"  Total contractions at optimum: {summary_data[min_total_n_live]['mean_total_contr']:.1f}")

    # Save results
    output_dir = Path("results/nlive_efficiency")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_dict = {
        'parameters': {
            'n_live_values': n_live_values,
            'n_seeds': n_seeds,
            'threshold': threshold,
            'image_size': image_size
        },
        'summary': {str(k): v for k, v in summary_data.items()},
        'statistical_tests': {
            'contractions_vs_nlive': {
                'spearman_r': float(spearman_contr),
                'p_value': float(p_contr),
                'cohens_d': float(cohens_d_contr) if cohens_d_contr else None,
                'direction': 'decreases' if spearman_contr < 0 else 'increases'
            },
            'quadratic_fit': {
                'coefficients': [float(b) for b in beta],
                'r_squared': float(r_squared),
                'optimal_n_live': float(optimal_n_live) if optimal_n_live else None
            },
            'bits_vs_nlive': {
                'spearman_r': float(spearman_bits),
                'p_value': float(p_bits),
                'cohens_d': float(cohens_d_bits) if cohens_d_bits else None
            }
        },
        'conclusion': {
            'h0_1_rejected': bool(h0_1_rejected),
            'empirical_optimal_n_live': int(min_total_n_live),
            'status': 'validated' if h0_1_rejected else 'inconclusive'
        }
    }

    with open(output_dir / "efficiency_results.json", 'w') as f:
        json.dump(results_dict, f, indent=2)

    print(f"\nResults saved to {output_dir / 'efficiency_results.json'}")

    return results_dict


if __name__ == "__main__":
    results = run_experiment()
