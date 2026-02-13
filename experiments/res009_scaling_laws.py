#!/usr/bin/env python3
"""
RES-009: Size Scaling Laws Experiment

Hypothesis: Bit-cost scales sub-linearly with image size: B(N) ~ N^beta where beta < 1
Status: REFUTED - Actual findings show beta=1.455 > 1 (super-linear scaling)

This script runs the full experiment and saves results to results/scaling_laws/results.json
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy import stats
from dataclasses import dataclass, field
from typing import Optional, Callable
import json
import time

from core.thermo_sampler_v3 import (
    CPPN, order_multiplicative, set_global_seed, PRIOR_SIGMA
)


@dataclass
class SizeScalingResult:
    """Results for a single size."""
    size: int
    bits_mean: float
    bits_std: float
    bits_values: list
    n_seeds: int
    converged_count: int


@dataclass
class LivePoint:
    """Live point in nested sampling."""
    cppn: CPPN
    image: np.ndarray
    order_value: float


def elliptical_slice_sample(
    cppn: CPPN,
    threshold: float,
    image_size: int,
    order_fn: Callable,
    max_contractions: int = 100,
    max_restarts: int = 5
) -> tuple:
    """
    Elliptical slice sampling for constrained prior sampling.
    Returns: (new_cppn, new_img, new_order, n_contractions, success)
    """
    current_w = cppn.get_weights()
    n_params = len(current_w)
    total_contractions = 0

    for restart in range(max_restarts):
        # Draw auxiliary vector from prior (defines ellipse)
        nu = np.random.randn(n_params) * PRIOR_SIGMA

        # Initial angle and bracket
        phi = np.random.uniform(0, 2 * np.pi)
        phi_min = phi - 2 * np.pi
        phi_max = phi

        n_contractions = 0
        while n_contractions < max_contractions:
            # Proposal on ellipse: w' = w*cos(phi) + nu*sin(phi)
            proposal_w = current_w * np.cos(phi) + nu * np.sin(phi)

            proposal_cppn = cppn.copy()
            proposal_cppn.set_weights(proposal_w)
            proposal_img = proposal_cppn.render(image_size)
            proposal_order = order_fn(proposal_img)

            if proposal_order >= threshold:
                return proposal_cppn, proposal_img, proposal_order, total_contractions + n_contractions, True

            # Shrink bracket
            if phi < 0:
                phi_min = phi
            else:
                phi_max = phi

            phi = np.random.uniform(phi_min, phi_max)
            n_contractions += 1

            if phi_max - phi_min < 1e-10:
                break

        total_contractions += n_contractions

    # Failed - return current
    current_img = cppn.render(image_size)
    return cppn, current_img, order_fn(current_img), total_contractions, False


def run_nested_sampling_for_size(
    size: int,
    threshold: float = 0.1,
    n_live: int = 50,
    n_iterations: int = 1000,
    seed: Optional[int] = None
) -> Optional[float]:
    """
    Run nested sampling with ESS to find bits-to-threshold for given image size.

    Returns bits if threshold reached, None if failed.
    """
    set_global_seed(seed)

    # Initialize live points
    live_points = []
    for _ in range(n_live):
        cppn = CPPN()  # Random init from prior
        img = cppn.render(size)
        order = order_multiplicative(img)
        live_points.append(LivePoint(cppn=cppn, image=img, order_value=order))

    # Nested sampling loop
    for iteration in range(n_iterations):
        # Sort by order (ascending)
        live_points.sort(key=lambda lp: lp.order_value)

        # Worst point (lowest order)
        worst_order = live_points[0].order_value

        # Check if all points above threshold
        if worst_order >= threshold:
            # Bits = iteration (number of points removed) / n_live + 1
            bits = (iteration + 1) / n_live
            return bits

        # Select seed from better points
        seed_idx = np.random.randint(1, len(live_points))
        seed_point = live_points[seed_idx]

        # ESS to get new point above worst_order
        new_cppn, new_img, new_order, _, success = elliptical_slice_sample(
            cppn=seed_point.cppn,
            threshold=worst_order,
            image_size=size,
            order_fn=order_multiplicative,
            max_contractions=50,
            max_restarts=3
        )

        if success and new_order > worst_order:
            # Replace worst point
            live_points[0] = LivePoint(cppn=new_cppn, image=new_img, order_value=new_order)
        else:
            # Fallback: sample from prior
            for _ in range(50):
                new_cppn = CPPN()
                new_img = new_cppn.render(size)
                new_order = order_multiplicative(new_img)
                if new_order > worst_order:
                    live_points[0] = LivePoint(cppn=new_cppn, image=new_img, order_value=new_order)
                    break

    # Check final state
    live_points.sort(key=lambda lp: lp.order_value)
    if live_points[0].order_value >= threshold:
        bits = n_iterations / n_live
        return bits

    return None  # Failed to reach threshold


def run_size_scaling_experiment(
    sizes: list = [8, 12, 16, 24, 32, 48],
    threshold: float = 0.1,
    n_seeds: int = 8,
    n_live: int = 50,
    base_iterations: int = 1000
) -> dict:
    """
    Run the full size scaling experiment according to RES-009 parameters.
    """
    print("=" * 70)
    print("RES-009: SIZE SCALING LAWS EXPERIMENT")
    print("=" * 70)
    print(f"\nHypothesis: B(N) ~ N^beta with beta < 1 (sub-linear scaling)")
    print(f"Null: beta >= 1 (linear or super-linear)")
    print(f"\nSizes: {sizes}")
    print(f"Seeds per size: {n_seeds}")
    print(f"Threshold: {threshold}")
    print(f"Live points: {n_live}")
    print()

    results = []

    for size in sizes:
        # Scale iterations with size
        n_iterations = int(base_iterations * max(1, size / 16))

        print(f"Size {size}x{size} ({n_iterations} iterations)...")

        bits_values = []
        converged = 0

        for seed in range(n_seeds):
            bits = run_nested_sampling_for_size(
                size=size,
                threshold=threshold,
                n_live=n_live,
                n_iterations=n_iterations,
                seed=42 + seed * 1000 + size
            )

            if bits is not None:
                bits_values.append(bits)
                converged += 1

        if bits_values:
            result = SizeScalingResult(
                size=size,
                bits_mean=np.mean(bits_values),
                bits_std=np.std(bits_values),
                bits_values=bits_values,
                n_seeds=n_seeds,
                converged_count=converged
            )
            results.append(result)
            print(f"  -> bits = {result.bits_mean:.2f} +/- {result.bits_std:.2f} "
                  f"({converged}/{n_seeds} converged)")
        else:
            print(f"  -> FAILED (0/{n_seeds} converged)")

    # Analyze scaling
    print("\n" + "=" * 70)
    print("POWER LAW ANALYSIS")
    print("=" * 70)

    if len(results) < 3:
        print("ERROR: Not enough data points for analysis")
        return {"status": "failed", "reason": "insufficient_data"}

    # Extract data
    sizes_arr = np.array([r.size for r in results])
    bits_arr = np.array([r.bits_mean for r in results])
    bits_std_arr = np.array([r.bits_std for r in results])

    # Log-log fit
    log_sizes = np.log(sizes_arr)
    log_bits = np.log(bits_arr)

    # Linear regression
    slope, intercept, r_value, p_value_fit, std_err = stats.linregress(log_sizes, log_bits)
    beta = slope
    r_squared = r_value ** 2

    print(f"\nFitted: log(bits) = {beta:.4f} * log(size) + {intercept:.4f}")
    print(f"Beta (exponent) = {beta:.4f} +/- {std_err:.4f}")
    print(f"R-squared = {r_squared:.4f}")

    # Bootstrap for CI
    print("\nBootstrap analysis (1000 samples)...")
    n_bootstrap = 1000
    beta_samples = []

    np.random.seed(42)
    for _ in range(n_bootstrap):
        # Resample bits values for each size
        boot_bits = []
        for r in results:
            if len(r.bits_values) > 0:
                boot_sample = np.random.choice(r.bits_values, size=len(r.bits_values), replace=True)
                boot_bits.append(np.mean(boot_sample))
            else:
                boot_bits.append(r.bits_mean)

        boot_bits = np.array(boot_bits)
        log_boot_bits = np.log(boot_bits)

        boot_slope, _, _, _, _ = stats.linregress(log_sizes, log_boot_bits)
        beta_samples.append(boot_slope)

    beta_samples = np.array(beta_samples)
    beta_ci_low = np.percentile(beta_samples, 2.5)
    beta_ci_high = np.percentile(beta_samples, 97.5)
    beta_mean = np.mean(beta_samples)
    beta_bootstrap_std = np.std(beta_samples)

    print(f"Bootstrap beta = {beta_mean:.4f} +/- {beta_bootstrap_std:.4f}")
    print(f"95% CI: [{beta_ci_low:.4f}, {beta_ci_high:.4f}]")

    # Statistical test: H0: beta < 1
    # One-sided test for sub-linearity
    z_stat = (1.0 - beta_mean) / beta_bootstrap_std
    p_value = stats.norm.cdf(z_stat)  # P(beta < 1)

    # Effect size: Cohen's d (distance from 1 in units of std)
    cohens_d = (1.0 - beta_mean) / beta_bootstrap_std

    print(f"\nTest H0: beta < 1 (sub-linear)")
    print(f"Z-statistic = {z_stat:.4f}")
    print(f"p-value (two-sided) = {2*(1 - stats.norm.cdf(abs(z_stat))):.6f}")
    print(f"Cohen's d = {cohens_d:.4f}")

    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    # Determine if we validate sub-linear hypothesis
    # For validation: beta must be significantly < 1
    is_sublinear = beta_ci_high < 1.0

    if is_sublinear:
        status = "validated"
        print(f"\n*** HYPOTHESIS VALIDATED ***")
        print(f"Bit-cost scales sub-linearly with image size (beta = {beta_mean:.3f})")
    elif beta_ci_low >= 1.0:
        status = "refuted"
        print(f"\n*** HYPOTHESIS REFUTED: beta CI [{beta_ci_low:.3f}, {beta_ci_high:.3f}] is entirely >= 1.0 ***")
    else:
        status = "inconclusive"
        print(f"\n*** INCONCLUSIVE: beta CI [{beta_ci_low:.3f}, {beta_ci_high:.3f}] straddles 1.0 ***")

    # Summary data
    result_data = {
        "hypothesis": "Bit-cost scales sub-linearly with image size: B(N) ~ N^beta where beta < 1",
        "null_hypothesis": "Bit-cost scales linearly or super-linearly with image size (beta >= 1)",
        "sizes": sizes,
        "threshold": threshold,
        "n_seeds": n_seeds,
        "n_live": n_live,
        "results_per_size": [
            {
                "size": r.size,
                "bits_mean": r.bits_mean,
                "bits_std": r.bits_std,
                "bits_values": r.bits_values,
                "converged": r.converged_count
            }
            for r in results
        ],
        "analysis": {
            "beta": float(beta_mean),
            "beta_std": float(beta_bootstrap_std) if not np.isinf(beta_bootstrap_std) else 0.0,
            "beta_ci": [float(beta_ci_low), float(beta_ci_high)],
            "r_squared": float(r_squared),
            "z_statistic": float(z_stat) if not np.isinf(z_stat) else 999.0,
            "p_value": float(p_value),
            "cohens_d": float(cohens_d) if not np.isinf(cohens_d) else 999.0
        },
        "status": status
    }

    return result_data


def main(thresholds=None, sizes=None, n_seeds=8, n_live=50):
    """
    Run size scaling experiment for one or more thresholds.

    Args:
        thresholds: List of thresholds to test, or None for [0.25]
        sizes: List of sizes to test, or None for [8, 12, 16, 24, 32, 48]
        n_seeds: Number of seeds per size
        n_live: Number of live points in nested sampling
    """
    if thresholds is None:
        thresholds = [0.25]
    if sizes is None:
        sizes = [8, 12, 16, 24, 32, 48]

    # If single threshold, run normally
    if len(thresholds) == 1:
        start_time = time.time()

        results = run_size_scaling_experiment(
            sizes=sizes,
            threshold=thresholds[0],
            n_seeds=n_seeds,
            n_live=n_live,
            base_iterations=1000
        )

        elapsed = time.time() - start_time
        results["elapsed_seconds"] = elapsed

        print(f"\nTotal time: {elapsed:.1f} seconds")

        # Save results to scaling_laws directory
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "scaling_laws")
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, "results.json")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {output_path}")

        return results

    # If multiple thresholds, run all and aggregate
    else:
        all_results = {}
        for threshold in thresholds:
            print(f"\n{'=' * 70}")
            print(f"Running threshold={threshold}")
            print(f"{'=' * 70}\n")

            start_time = time.time()

            results = run_size_scaling_experiment(
                sizes=sizes,
                threshold=threshold,
                n_seeds=n_seeds,
                n_live=n_live,
                base_iterations=1000
            )

            elapsed = time.time() - start_time
            results["elapsed_seconds"] = elapsed

            all_results[f"threshold_{threshold}"] = results

        # Save aggregated results
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "threshold_sweep")
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, "multi_threshold_results.json")
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2)

        print(f"\n{'=' * 70}")
        print(f"All thresholds completed")
        print(f"Results saved to: {output_path}")
        print(f"{'=' * 70}\n")

        return all_results


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="RES-009: Size Scaling Laws with configurable thresholds")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.25],
                        help="Thresholds to test (default: [0.25])")
    parser.add_argument("--sizes", type=int, nargs="+", default=[8, 12, 16, 24, 32, 48],
                        help="Image sizes to test (default: [8, 12, 16, 24, 32, 48])")
    parser.add_argument("--seeds", type=int, default=8, help="Seeds per size (default: 8)")
    parser.add_argument("--n-live", type=int, default=50, help="Live points (default: 50)")

    args = parser.parse_args()

    main(thresholds=args.thresholds, sizes=args.sizes, n_seeds=args.seeds, n_live=args.n_live)
