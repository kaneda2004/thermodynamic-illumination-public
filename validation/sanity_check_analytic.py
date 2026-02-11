#!/usr/bin/env python3
"""
Tier 0 Sanity Check: Validate Nested Sampling Against Analytic Ground Truth

This script tests whether our nested sampling estimator correctly recovers
known probabilities. If this fails, we cannot trust any other results.

Two metrics with computable analytic probabilities:

1. Mean-pixel threshold:
   - O(x) = mean(x) for binary images
   - Pr[mean(x) >= T] = sum_{k>=ceil(T*n)}^n Binomial(n,k) / 2^n
   - Exact for any n and T

2. Perfect vertical symmetry:
   - O(x) = 1 if x == flip_horizontal(x) else 0
   - For n×n image, left half determines right half
   - p = 2^(-n*n/2), so B = n*n/2 bits exactly

Usage:
    uv run python sanity_check_analytic.py
"""

import numpy as np
from scipy.stats import binom
import sys
from typing import Callable

# Import from main framework
sys.path.insert(0, '..')
from core.thermo_sampler_v3 import nested_sampling_with_prior


def analytic_mean_threshold_probability(n_pixels: int, threshold: float) -> float:
    """
    Compute exact Pr[mean(x) >= threshold] for uniform binary images.

    For n binary pixels, mean >= T requires at least ceil(T*n) ones.
    Pr[k ones] = C(n,k) / 2^n
    """
    k_min = int(np.ceil(threshold * n_pixels))
    if k_min > n_pixels:
        return 0.0
    if k_min <= 0:
        return 1.0

    # Sum binomial probabilities from k_min to n
    # Using survival function: Pr[X >= k] = 1 - CDF(k-1)
    prob = 1.0 - binom.cdf(k_min - 1, n_pixels, 0.5)
    return prob


def analytic_mean_threshold_bits(n_pixels: int, threshold: float) -> float:
    """Compute exact bits for mean threshold."""
    p = analytic_mean_threshold_probability(n_pixels, threshold)
    if p <= 0:
        return float('inf')
    return -np.log2(p)


def order_mean(img: np.ndarray) -> float:
    """Order metric: mean pixel value."""
    return float(np.mean(img))


def order_vertical_symmetry(img: np.ndarray) -> float:
    """Order metric: 1.0 if perfectly vertically symmetric, else 0.0."""
    flipped = np.fliplr(img)
    return 1.0 if np.array_equal(img, flipped) else 0.0


def analytic_symmetry_bits(image_size: int) -> float:
    """
    Exact bits for perfect vertical symmetry.

    For n×n image, the left half (n × n/2 pixels) determines the right half.
    So p = 2^(-n*n/2) and B = n*n/2 bits.
    """
    n_constrained = image_size * (image_size // 2)
    return float(n_constrained)


def run_sanity_check_mean_threshold(
    image_size: int = 8,
    thresholds: list[float] = [0.55, 0.6, 0.65, 0.7],
    n_live: int = 50,
    n_iterations: int = 500,
    n_runs: int = 3,
):
    """
    Sanity check: Mean threshold metric.

    Compare estimated bits from nested sampling to analytic ground truth.
    """
    print("=" * 70)
    print("SANITY CHECK: Mean-Pixel Threshold")
    print("=" * 70)
    print(f"Image size: {image_size}x{image_size} ({image_size**2} pixels)")
    print(f"n_live: {n_live}, n_iterations: {n_iterations}, n_runs: {n_runs}")
    print()

    n_pixels = image_size ** 2

    # Compute analytic ground truth
    print("Analytic ground truth:")
    print(f"{'Threshold':<12} {'Probability':<15} {'Bits':<10}")
    print("-" * 40)
    for T in thresholds:
        p = analytic_mean_threshold_probability(n_pixels, T)
        b = analytic_mean_threshold_bits(n_pixels, T)
        print(f"{T:<12.2f} {p:<15.6e} {b:<10.2f}")
    print()

    # Run nested sampling
    print("Running nested sampling...")

    results = {T: [] for T in thresholds}

    for run in range(n_runs):
        print(f"  Run {run+1}/{n_runs}...", end=" ", flush=True)

        dead_points, live_points, _ = nested_sampling_with_prior(
            prior_type="uniform",
            n_live=n_live,
            n_iterations=n_iterations,
            image_size=image_size,
            order_fn=order_mean,
            verbose=False,
        )

        # Extract bits at each threshold
        for T in thresholds:
            bits_found = None
            for d in dead_points:
                if d['order'] >= T:
                    bits_found = -d['log_X'] / np.log(2)
                    break

            if bits_found is None:
                # Threshold not reached - report lower bound
                max_bits = n_iterations / (n_live * np.log(2))
                results[T].append((max_bits, False))  # (bits, reached)
            else:
                results[T].append((bits_found, True))

        max_order = max(d['order'] for d in dead_points) if dead_points else 0
        print(f"max_order={max_order:.3f}")

    # Compare results
    print()
    print("=" * 70)
    print("RESULTS: Estimated vs Analytic Bits")
    print("=" * 70)
    print()
    print(f"{'Threshold':<10} {'Analytic':<12} {'Estimated':<15} {'Error':<12} {'Status'}")
    print("-" * 70)

    all_passed = True

    for T in thresholds:
        analytic_bits = analytic_mean_threshold_bits(n_pixels, T)

        # Average over runs
        reached_runs = [r for r in results[T] if r[1]]
        unreached_runs = [r for r in results[T] if not r[1]]

        if reached_runs:
            est_bits = np.mean([r[0] for r in reached_runs])
            est_std = np.std([r[0] for r in reached_runs]) if len(reached_runs) > 1 else 0
            error = est_bits - analytic_bits
            error_pct = 100 * error / analytic_bits if analytic_bits > 0 else 0

            # Pass if within 2 bits or 30% (generous for small sample)
            passed = abs(error) < 2.0 or abs(error_pct) < 30
            status = "PASS" if passed else "FAIL"
            if not passed:
                all_passed = False

            est_str = f"{est_bits:.2f} ± {est_std:.2f}"
            error_str = f"{error:+.2f} ({error_pct:+.1f}%)"
        else:
            est_bits = np.mean([r[0] for r in unreached_runs])
            est_str = f"≥{est_bits:.2f}"

            if analytic_bits > est_bits:
                status = "OK (not reached, but analytic is higher)"
                error_str = "N/A"
            else:
                status = "FAIL (should have reached)"
                error_str = "N/A"
                all_passed = False

        print(f"{T:<10.2f} {analytic_bits:<12.2f} {est_str:<15} {error_str:<12} {status}")

    print()
    if all_passed:
        print("*** ALL CHECKS PASSED ***")
        print("Nested sampling estimator is working correctly for this metric.")
    else:
        print("*** SOME CHECKS FAILED ***")
        print("Investigate: n_live, n_iterations, or sampler issues.")

    return all_passed


def run_sanity_check_symmetry(
    image_size: int = 8,
    n_live: int = 50,
    n_iterations: int = 200,
    n_runs: int = 3,
):
    """
    Sanity check: Perfect symmetry metric.

    This is a HARD test - perfect symmetry requires exactly matching pixels.
    For 8x8 image, B = 32 bits (very hard to reach with rejection sampling).
    For 4x4 image, B = 8 bits (feasible).
    """
    print()
    print("=" * 70)
    print("SANITY CHECK: Perfect Vertical Symmetry")
    print("=" * 70)
    print(f"Image size: {image_size}x{image_size}")
    print(f"n_live: {n_live}, n_iterations: {n_iterations}, n_runs: {n_runs}")
    print()

    analytic_bits = analytic_symmetry_bits(image_size)
    print(f"Analytic ground truth: B = {analytic_bits:.1f} bits")
    print(f"(Left half determines right half: {image_size}×{image_size//2} = {image_size * (image_size//2)} constrained bits)")
    print()

    max_probed_bits = n_iterations / (n_live * np.log(2))
    print(f"Maximum bits we can probe: {max_probed_bits:.1f}")

    if max_probed_bits < analytic_bits:
        print(f"WARNING: Cannot reach analytic bits ({analytic_bits:.1f}) with current budget.")
        print(f"         This test will only verify we get a consistent lower bound.")
    print()

    print("Running nested sampling...")

    reached_count = 0
    bits_when_reached = []

    for run in range(n_runs):
        print(f"  Run {run+1}/{n_runs}...", end=" ", flush=True)

        dead_points, live_points, _ = nested_sampling_with_prior(
            prior_type="uniform",
            n_live=n_live,
            n_iterations=n_iterations,
            image_size=image_size,
            order_fn=order_vertical_symmetry,
            verbose=False,
        )

        # Check if we reached symmetry
        reached = False
        for d in dead_points:
            if d['order'] >= 0.99:  # Perfect symmetry
                bits = -d['log_X'] / np.log(2)
                bits_when_reached.append(bits)
                reached = True
                reached_count += 1
                print(f"reached at {bits:.1f} bits")
                break

        if not reached:
            print("not reached")

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Reached perfect symmetry: {reached_count}/{n_runs} runs")

    if reached_count > 0:
        mean_bits = np.mean(bits_when_reached)
        error = mean_bits - analytic_bits
        print(f"Estimated bits: {mean_bits:.2f} (analytic: {analytic_bits:.1f})")
        print(f"Error: {error:+.2f} bits")

        if abs(error) < 3.0:
            print("*** PASS: Within 3 bits of analytic ***")
            return True
        else:
            print("*** FAIL: Error too large ***")
            return False
    else:
        print(f"Lower bound: ≥{max_probed_bits:.1f} bits")
        if analytic_bits > max_probed_bits:
            print("*** OK: Analytic is beyond our probe budget, consistent ***")
            return True
        else:
            print("*** FAIL: Should have reached but didn't ***")
            return False


def main():
    print("=" * 70)
    print("TIER 0 SANITY CHECKS: Validating the Estimator")
    print("=" * 70)
    print()
    print("These tests verify that nested sampling correctly estimates")
    print("probabilities by comparing to metrics with known analytic values.")
    print()
    print("If these fail, we cannot trust any downstream results.")
    print()

    # Test 1: Mean threshold (easier, should definitely pass)
    passed_mean = run_sanity_check_mean_threshold(
        image_size=8,  # 64 pixels
        thresholds=[0.55, 0.6, 0.65, 0.7],
        n_live=50,
        n_iterations=500,
        n_runs=3,
    )

    # Test 2: Symmetry (harder, mainly checks consistency)
    passed_sym = run_sanity_check_symmetry(
        image_size=4,  # 16 pixels, B=8 bits (feasible)
        n_live=50,
        n_iterations=300,
        n_runs=3,
    )

    print()
    print("=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)
    print(f"Mean threshold test: {'PASS' if passed_mean else 'FAIL'}")
    print(f"Symmetry test: {'PASS' if passed_sym else 'FAIL'}")

    if passed_mean and passed_sym:
        print()
        print("*** ALL SANITY CHECKS PASSED ***")
        print("The nested sampling estimator is working correctly.")
        print("Downstream results can be trusted (at least for the estimator).")
        return 0
    else:
        print()
        print("*** SANITY CHECKS FAILED ***")
        print("The estimator may have bugs. Investigate before trusting other results.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
